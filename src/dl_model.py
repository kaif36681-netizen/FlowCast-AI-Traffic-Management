"""M6 — Deep-Learning Engine.

A recurrent sequence forecaster, designed and trained from scratch. No
pre-trained weights, no transfer learning, no off-the-shelf forecasting package.

Why a recurrent model at all, when XGBoost already has lag features? Because the
tree models see the recent past as a bag of independent columns — lag1, lag2,
lag3 are just three numbers with no notion that they form a trajectory. An LSTM
consumes them as an ordered sequence and carries a hidden state that can encode
"volume has been climbing for the last two hours", which is exactly the shape of
a congestion onset. Whether that structural advantage is worth the extra
complexity on this corridor is an empirical question, and Section 12.4 of the
PRD requires that the answer be stated plainly either way. It is, in the
benchmark table this module writes.

Architecture
------------
    per-step input   16 dynamic channels (sensor + weather + clock), scaled
    static context   segment embedding + capacity, concatenated after the LSTM
    core             2 stacked LSTM layers, hidden 64, dropout between them
    heads            (a) linear -> next-window volume        [MSE]
                     (b) linear -> 4 congestion logits       [cross-entropy]

The two heads share the recurrent trunk. Congestion is a banding of volume, so
the tasks are closely related and the shared representation is a genuine
multi-task setup rather than two models glued together.

Uncertainty
-----------
Dropout is left active at inference and the model is sampled 30 times. The
spread of those samples is the predictive confidence shown on the dashboard
(FR-11). This is Monte-Carlo dropout, which approximates a Bayesian posterior
over the weights — it is an approximation, not a calibrated interval, and the
report says so.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluate import classification_metrics, regression_metrics
from features import CONGESTION_ORDER
from utils import (get_logger, load_config, load_json, read_table, save_json,
                   set_seed, time_split_bounds)

# Channels fed to the recurrent trunk, one value per time step. Deliberately
# narrow: the LSTM's job is to read the trajectory, not to re-derive the wide
# engineered feature set the tree models already exploit.
SEQ_CHANNELS = [
    "traffic_volume", "avg_speed", "occupancy", "vc_ratio",
    "temperature", "rainfall", "visibility",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_peak", "is_weekend", "public_holiday", "event_flag", "roadwork_flag",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class FlowCastLSTM(nn.Module):
    """Multi-task LSTM: next-window volume regression + congestion classification."""

    def __init__(self, n_channels: int, n_segments: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.25,
                 embed_dim: int = 8, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        # A learned embedding per segment lets one global model hold 25 different
        # baseline behaviours without fitting 25 separate models.
        self.segment_embed = nn.Embedding(n_segments, embed_dim)
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        head_in = out_dim + embed_dim + 1        # + capacity scalar
        self.volume_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.congestion_head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 4))

    def forward(self, x_seq, segment_idx, capacity):
        out, _ = self.lstm(x_seq)
        last = out[:, -1, :]                      # final hidden state of the window
        ctx = torch.cat([self.dropout(last),
                         self.segment_embed(segment_idx),
                         capacity.unsqueeze(1)], dim=1)
        return self.volume_head(ctx).squeeze(1), self.congestion_head(ctx)


# --------------------------------------------------------------------------- #
# Sequence construction
# --------------------------------------------------------------------------- #

def build_sequence_tensors(df: pd.DataFrame, cfg: dict, log):
    """Reshape the flat table into (segment, time, channel) and cut the windows.

    The rebuilt uniform grid from M2 is what makes this safe: every segment has
    the same number of evenly spaced time steps, so a window of N rows is always
    exactly N * 30 minutes and never straddles a sensor dropout.
    """
    N = cfg["models"]["deep"]["sequence_length"]
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    segments = sorted(df["road_id"].unique())
    seg_to_idx = {s: i for i, s in enumerate(segments)}
    n_seg = len(segments)
    n_time = len(df) // n_seg
    assert n_seg * n_time == len(df), "grid is ragged — M2 should have squared it"

    channels = np.stack(
        [df[c].to_numpy(dtype=np.float32).reshape(n_seg, n_time) for c in SEQ_CHANNELS],
        axis=-1)                                   # (seg, time, channel)
    volume = df["y_volume"].to_numpy(dtype=np.float32).reshape(n_seg, n_time)
    congestion = df["y_congestion_code"].to_numpy(dtype=np.int64).reshape(n_seg, n_time)
    capacity = df["road_capacity"].to_numpy(dtype=np.float32).reshape(n_seg, n_time)
    timestamps = df["timestamp"].to_numpy().reshape(n_seg, n_time)[0]

    log.info("sequences: %d segments x %d time steps x %d channels, window N=%d",
             n_seg, n_time, len(SEQ_CHANNELS), N)
    return channels, volume, congestion, capacity, timestamps, segments, seg_to_idx, N


def split_indices(timestamps, cfg, N, log):
    """Time-based train/val/test cut on the *target* index, mirroring M5."""
    ts = pd.Series(pd.to_datetime(timestamps))
    train_end, val_end = time_split_bounds(ts, cfg["split"]["train_frac"],
                                           cfg["split"]["val_frac"])
    idx = np.arange(N, len(ts))                    # first N steps have no history
    tr = idx[ts.iloc[idx].values <= np.datetime64(pd.Timestamp(train_end))]
    va = idx[(ts.iloc[idx].values > np.datetime64(pd.Timestamp(train_end))) &
             (ts.iloc[idx].values <= np.datetime64(pd.Timestamp(val_end)))]
    te = idx[ts.iloc[idx].values > np.datetime64(pd.Timestamp(val_end))]
    log.info("sequence split: %d train / %d val / %d test target windows per segment",
             len(tr), len(va), len(te))
    return tr, va, te


class SequenceBatcher:
    """Yields mini-batches of windows without materialising every window.

    Storing all 170k windows as an explicit (n, 12, 16) array would copy the data
    twelve times over. Slicing the (segment, time, channel) array per batch keeps
    memory flat and costs nothing measurable.
    """

    def __init__(self, channels, volume, congestion, capacity, seg_ids,
                 time_idx, N, batch_size, shuffle, rng):
        self.channels, self.volume = channels, volume
        self.congestion, self.capacity = congestion, capacity
        self.N, self.batch_size, self.shuffle, self.rng = N, batch_size, shuffle, rng
        # Cartesian product of (segment, target time index).
        self.pairs = np.array([(s, t) for s in seg_ids for t in time_idx], dtype=np.int64)

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __iter__(self):
        order = self.rng.permutation(len(self.pairs)) if self.shuffle \
            else np.arange(len(self.pairs))
        for start in range(0, len(order), self.batch_size):
            sel = self.pairs[order[start:start + self.batch_size]]
            s, t = sel[:, 0], sel[:, 1]
            # Window is [t-N, t) — strictly before the target window.
            window = np.stack([self.channels[si, ti - self.N:ti, :]
                               for si, ti in zip(s, t)])
            yield (torch.from_numpy(window),
                   torch.from_numpy(s),
                   torch.from_numpy(self.capacity[s, t]),
                   torch.from_numpy(self.volume[s, t]),
                   torch.from_numpy(self.congestion[s, t]))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_model(model, train_batcher, val_batcher, cfg, log, norm, tag: str,
                max_epochs: int | None = None):
    """Train with early stopping on validation loss, restoring the best weights."""
    dl = cfg["models"]["deep"]
    opt = torch.optim.Adam(model.parameters(), lr=dl["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

    best_loss, best_state, bad_epochs = np.inf, None, 0
    history = []

    for epoch in range(max_epochs or dl["epochs"]):
        model.train()
        t0, running, n_batches = time.time(), 0.0, 0
        for xb, sb, cb, yv, yc in train_batcher:
            xb = ((xb - norm["mu"]) / norm["sigma"]).to(DEVICE)
            yv_s = ((yv - norm["y_mu"]) / norm["y_sigma"]).to(DEVICE)
            opt.zero_grad()
            pv, pc = model(xb, sb.to(DEVICE), (cb / norm["cap_scale"]).to(DEVICE))
            # Equal weighting: both heads operate on standardised scales, so
            # neither dominates the shared trunk's gradient.
            loss = mse(pv, yv_s) + ce(pc, yc.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1

        val_loss, val_rmse = evaluate_loss(model, val_batcher, norm, mse, ce)
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": running / max(1, n_batches),
                        "val_loss": val_loss, "val_rmse_volume": val_rmse,
                        "seconds": round(time.time() - t0, 1)})
        log.info("  [%s] epoch %2d | train %.4f | val %.4f | val volume RMSE %6.2f | %.0fs",
                 tag, epoch, history[-1]["train_loss"], val_loss, val_rmse,
                 history[-1]["seconds"])

        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= dl["patience"]:
                log.info("  [%s] early stop at epoch %d (best val %.4f)",
                         tag, epoch, best_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_loss(model, batcher, norm, mse, ce):
    model.eval()
    total, n, sq_err, count = 0.0, 0, 0.0, 0
    for xb, sb, cb, yv, yc in batcher:
        xb = ((xb - norm["mu"]) / norm["sigma"]).to(DEVICE)
        yv_s = ((yv - norm["y_mu"]) / norm["y_sigma"]).to(DEVICE)
        pv, pc = model(xb, sb.to(DEVICE), (cb / norm["cap_scale"]).to(DEVICE))
        total += float(mse(pv, yv_s) + ce(pc, yc.to(DEVICE)))
        n += 1
        pred = pv.cpu().numpy() * norm["y_sigma"].item() + norm["y_mu"].item()
        sq_err += float(np.sum((pred - yv.numpy()) ** 2))
        count += len(yv)
    return total / max(1, n), float(np.sqrt(sq_err / max(1, count)))


@torch.no_grad()
def predict(model, batcher, norm, mc_samples: int = 0):
    """Predict volume and congestion. With mc_samples > 0, also return the spread.

    Monte-Carlo dropout: dropout layers are kept active and the forward pass is
    repeated, so each sample comes from a slightly different thinned network.
    The standard deviation across samples is the model's own uncertainty about
    the forecast — wide where the recent trajectory is unusual, narrow where the
    pattern is familiar.
    """
    model.eval()
    vols, congs, actual_v, actual_c, stds = [], [], [], [], []
    for xb, sb, cb, yv, yc in batcher:
        xn = ((xb - norm["mu"]) / norm["sigma"]).to(DEVICE)
        sb_d, cb_d = sb.to(DEVICE), (cb / norm["cap_scale"]).to(DEVICE)
        pv, pc = model(xn, sb_d, cb_d)
        vols.append(pv.cpu().numpy() * norm["y_sigma"].item() + norm["y_mu"].item())
        congs.append(pc.argmax(1).cpu().numpy())
        actual_v.append(yv.numpy())
        actual_c.append(yc.numpy())

        if mc_samples:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            samples = np.stack([
                (model(xn, sb_d, cb_d)[0].cpu().numpy() * norm["y_sigma"].item()
                 + norm["y_mu"].item()) for _ in range(mc_samples)])
            stds.append(samples.std(axis=0))
            model.eval()

    out = {"pred_volume": np.concatenate(vols),
           "pred_congestion": np.concatenate(congs),
           "actual_volume": np.concatenate(actual_v),
           "actual_congestion": np.concatenate(actual_c)}
    if mc_samples:
        out["pred_volume_std"] = np.concatenate(stds)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(cfg: dict | None = None, variants: list[str] | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("dl_model", cfg)
    set_seed(cfg["project"]["random_seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    log.info("=== M6 Deep-Learning Engine (device: %s) ===", DEVICE)

    variants = variants or ["lstm", "bilstm"]
    results_path = Path(cfg["paths"]["reports"]) / "deep_results.json"
    if results_path.exists():
        saved = load_json(results_path)
        results, histories = saved.get("results", {}), saved.get("history", {})
        log.info("resuming — %d variant(s) already trained", len(results))
    else:
        results, histories = {}, {}

    df = read_table(Path(cfg["paths"]["processed"]) / "flowcast_dataset.parquet")
    (channels, volume, congestion, capacity, timestamps, segments,
     seg_to_idx, N) = build_sequence_tensors(df, cfg, log)
    tr_idx, va_idx, te_idx = split_indices(timestamps, cfg, N, log)

    # Normalisation statistics come from the training time steps only.
    train_slice = channels[:, tr_idx, :].reshape(-1, len(SEQ_CHANNELS))
    norm = {
        "mu": torch.tensor(train_slice.mean(axis=0), dtype=torch.float32),
        "sigma": torch.tensor(np.where(train_slice.std(axis=0) == 0, 1.0,
                                       train_slice.std(axis=0)), dtype=torch.float32),
        "y_mu": torch.tensor(float(volume[:, tr_idx].mean())),
        "y_sigma": torch.tensor(float(volume[:, tr_idx].std())),
        "cap_scale": torch.tensor(float(capacity.max())),
    }

    rng = np.random.default_rng(cfg["project"]["random_seed"])
    seg_ids = np.arange(len(segments))
    bs = cfg["models"]["deep"]["batch_size"]

    def mk(idx, shuf):
        return SequenceBatcher(channels, volume, congestion, capacity,
                               seg_ids, idx, N, bs, shuf, rng)

    train_b, val_b, test_b = mk(tr_idx, True), mk(va_idx, False), mk(te_idx, False)
    dl = cfg["models"]["deep"]

    def save():
        save_json({"results": results, "history": histories}, results_path)

    # ---- primary: unidirectional LSTM forecaster ------------------------ #
    if "lstm" in variants and "LSTM" not in results:
        model = FlowCastLSTM(len(SEQ_CHANNELS), len(segments), dl["hidden_size"],
                             dl["num_layers"], dl["dropout"]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("LSTM forecaster: %d trainable parameters", n_params)
        model, hist = train_model(model, train_b, val_b, cfg, log, norm, "LSTM")
        histories["LSTM"] = hist

        preds = predict(model, test_b, norm, mc_samples=dl.get("mc_samples", 20))
        results["LSTM"] = {
            "volume": regression_metrics(preds["actual_volume"], preds["pred_volume"]),
            "congestion": classification_metrics(preds["actual_congestion"],
                                                 preds["pred_congestion"],
                                                 labels=CONGESTION_ORDER),
            "n_parameters": n_params,
            "epochs_run": len(hist),
        }
        log.info("LSTM test: volume RMSE %.2f | MAPE %.2f%% | congestion macro-F1 %.4f",
                 results["LSTM"]["volume"]["rmse"], results["LSTM"]["volume"]["mape"],
                 results["LSTM"]["congestion"]["macro_f1"])
        log.info("MC-dropout mean predictive std: %.1f vehicles",
                 float(np.mean(preds["pred_volume_std"])))

        torch.save({"state_dict": model.state_dict(), "channels": SEQ_CHANNELS,
                    "segments": segments, "config": dl,
                    "norm": {k: v.tolist() for k, v in norm.items()}},
                   Path(cfg["paths"]["models"]) / "lstm_forecaster.pt")

        pairs = np.array([(s, t) for s in seg_ids for t in te_idx])
        pd.DataFrame({
            "road_id": [segments[s] for s in pairs[:, 0]],
            "timestamp": pd.to_datetime(timestamps[pairs[:, 1]]),
            "dl_pred_volume": preds["pred_volume"],
            "dl_pred_volume_std": preds["pred_volume_std"],
            "dl_pred_congestion_code": preds["pred_congestion"],
        }).to_parquet(Path(cfg["paths"]["processed"]) / "dl_predictions.parquet",
                      index=False)
        save()

    # ---- comparison: bidirectional variant ------------------------------ #
    # Bidirectional is *not* a valid forecaster — it reads the window backwards
    # too, which in deployment would mean reading the future. It is trained here
    # only for the historical-analysis view the PRD asks for, where the whole
    # window is already known, and is labelled as such so nobody mistakes its
    # score for a forecasting result.
    key = "BiLSTM (historical analysis only)"
    if "bilstm" in variants and key not in results:
        bi = FlowCastLSTM(len(SEQ_CHANNELS), len(segments), dl["hidden_size"],
                          dl["num_layers"], dl["dropout"],
                          bidirectional=True).to(DEVICE)
        bi, bi_hist = train_model(bi, train_b, val_b, cfg, log, norm, "BiLSTM",
                                  max_epochs=dl.get("bilstm_epochs"))
        histories["BiLSTM"] = bi_hist
        bi_preds = predict(bi, test_b, norm)
        results[key] = {
            "volume": regression_metrics(bi_preds["actual_volume"],
                                         bi_preds["pred_volume"]),
            "congestion": classification_metrics(bi_preds["actual_congestion"],
                                                 bi_preds["pred_congestion"],
                                                 labels=CONGESTION_ORDER),
            "epochs_run": len(bi_hist),
            "note": "reads the window in both directions; not deployable as a forecaster",
        }
        log.info("BiLSTM test: volume RMSE %.2f", results[key]["volume"]["rmse"])
        torch.save({"state_dict": bi.state_dict()},
                   Path(cfg["paths"]["models"]) / "bilstm_analysis.pt")
        save()

    save()
    log.info("M6 complete")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the sequence models.")
    ap.add_argument("--variants", nargs="*", default=None,
                    choices=["lstm", "bilstm"],
                    help="Which sequence models to train.")
    a = ap.parse_args()
    run(variants=a.variants)
