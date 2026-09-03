# Start Here

Four things in this folder, in the order to use them.

| # | File | What it's for | Time |
| --- | --- | --- | --- |
| 1 | **GUIDE.md** | Understand the project. Read this first, properly. | 30 min |
| 2 | **VIVA_QA.md** | 40 questions you'll be asked, with answers. | 20 min |
| 3 | **FlowCast_Presentation.pptx** | 18 slides. Speaker notes under every one. | — |
| 4 | **demo.py** | Run this live. Prints the whole story in 30 seconds. | 30 sec |

`reports/technical_report.md` is the formal write-up if a document is required.

---

## Running anything

Every new terminal window needs these two lines first:

```powershell
cd C:\FlowCast\flowcast
.\.venv\Scripts\Activate.ps1
```

`(.venv)` appears at the start of the prompt when it worked.

| Command | What it does | Time |
| --- | --- | --- |
| `python demo.py` | The safe live demo | 30 sec |
| `streamlit run dashboard\app.py` | Opens the dashboard | 10 sec |
| `python -m pytest tests\ -q` | 21 checks, all pass | 3 sec |
| `python src\run_pipeline.py` | Rebuilds everything | ~25 min |

**Never run the last one during a presentation.**

---

## Before you present

- Terminal already open, in the folder, with `(.venv)` showing
- Dashboard already running in a browser tab
- `python demo.py` tested once so you know what appears
- The five points at the end of GUIDE.md memorised

## The one sentence to know cold

> "It predicts traffic 30 minutes ahead with 8.9% average error — 45% better
> than simply assuming nothing changes."

Always say the second half. The first half alone is a number nobody can judge.

## The three things that will impress

1. **5,203 time slots were missing from the file** — invisible, and it would have
   corrupted every calculation silently.
2. **`congestion_level` was secretly just the vehicle count** — using it would
   have produced a fake 99% score.
3. **Accident prediction failed, and I proved why** — a deliberately cheating
   model gained only 0.006, so the data simply doesn't contain the answer.

Lead with these. Not the accuracy number.

---

## Being straight about how it was built

Internmo's guidelines say submissions must be the student's own work and treat
undisclosed AI generation as a problem — note the wording, *undisclosed*. A short
note in the report costs nothing and protects the certificate. Draft text is in
GUIDE.md, Part 12.
