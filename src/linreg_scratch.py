"""Linear regression implemented from first principles with NumPy.

The PRD's build-it-once rule: before any library model is trusted, at least one
training loop and its loss and gradient must be written out explicitly. This is
that implementation. It is a real model — it trains on the real feature matrix
and its test metrics go on the scoreboard next to scikit-learn's — not a toy.

The mathematics, stated plainly:

    prediction      y_hat = Xw + b            (a matrix-vector product)
    cost            J(w,b) = (1/2n) * sum (y_hat - y)^2      (mean squared error)
    gradient wrt w  dJ/dw = (1/n) * X^T (y_hat - y)
    gradient wrt b  dJ/db = (1/n) * sum (y_hat - y)
    update          w <- w - alpha * dJ/dw

The X^T(y_hat - y) term is the whole of the linear algebra content: the residual
vector is projected back onto each feature direction, and that projection is the
direction of steepest ascent in the cost, so we step against it.

Mini-batch descent is used rather than full-batch because the feature matrix is
~120k x 93 and batching converges in far fewer passes over the data for the
same wall-clock cost.
"""

from __future__ import annotations

import numpy as np


class LinearRegressionGD:
    """Ordinary least squares fitted by mini-batch gradient descent.

    Parameters
    ----------
    learning_rate : step size alpha in the update rule.
    epochs        : full passes over the training set.
    batch_size    : rows per gradient step.
    l2            : optional ridge penalty; adds l2 * w to the weight gradient.
    tol           : early stop when the cost improves by less than this.
    """

    def __init__(self, learning_rate: float = 0.05, epochs: int = 300,
                 batch_size: int = 4096, l2: float = 0.0, tol: float = 1e-7,
                 seed: int = 42, verbose: bool = False):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.seed = seed
        self.verbose = verbose

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.cost_history: list[float] = []
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        # The target is standardised too. Traffic volume runs to ~2000, so the
        # initial residual is ~430 and a step of alpha * X^T r / n overshoots by
        # orders of magnitude — the loop diverges to inf within a few batches.
        # Centring y puts the gradient on the same scale as the features, which
        # is what lets one learning rate serve volume, speed and travel time.
        self._y_mu: float = 0.0
        self._y_sigma: float = 1.0

    # ------------------------------------------------------------------ #
    # Standardisation
    # ------------------------------------------------------------------ #

    def _standardise_fit(self, X: np.ndarray) -> np.ndarray:
        """Z-score the features using training statistics only.

        Gradient descent on raw traffic features does not converge: volume runs
        to ~2000 while the cyclical encodings sit in [-1, 1], so a single step
        size cannot suit both. Standardising makes the cost surface roughly
        spherical and lets one alpha work for every dimension.
        """
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        return (X - self._mu) / self._sigma

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma

    # ------------------------------------------------------------------ #
    # Cost and gradient
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
              l2: float) -> float:
        """Mean squared error, halved so the derivative loses the factor of 2."""
        resid = X @ w + b - y
        mse = float(np.mean(resid ** 2) / 2.0)
        return mse + l2 * float(np.sum(w ** 2)) / 2.0

    @staticmethod
    def _gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                   l2: float) -> tuple[np.ndarray, float]:
        """Analytic gradients of the halved MSE with respect to w and b."""
        n = X.shape[0]
        resid = X @ w + b - y            # shape (n,)
        grad_w = (X.T @ resid) / n + l2 * w
        grad_b = float(np.sum(resid) / n)
        return grad_w, grad_b

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        Xs = self._standardise_fit(X)

        self._y_mu = float(y.mean())
        self._y_sigma = float(y.std()) or 1.0
        ys = (y - self._y_mu) / self._y_sigma

        n, d = Xs.shape
        lr = self.lr

        for attempt in range(5):
            self.w = np.zeros(d)
            self.b = 0.0
            self.cost_history = []
            prev_cost = np.inf
            diverged = False

            for epoch in range(self.epochs):
                order = rng.permutation(n)
                for start in range(0, n, self.batch_size):
                    idx = order[start:start + self.batch_size]
                    grad_w, grad_b = self._gradients(Xs[idx], ys[idx], self.w,
                                                     self.b, self.l2)
                    self.w -= lr * grad_w
                    self.b -= lr * grad_b

                cost = self._cost(Xs, ys, self.w, self.b, self.l2)
                self.cost_history.append(cost)

                # Divergence guard: a step size that is too large for the local
                # curvature sends the cost to inf rather than to a minimum.
                # Back off and restart rather than returning a broken model.
                if not np.isfinite(cost) or cost > 1e6:
                    diverged = True
                    break

                if self.verbose and epoch % 50 == 0:
                    print(f"  epoch {epoch:4d}  cost {cost:.6f}")
                if abs(prev_cost - cost) < self.tol * max(1.0, abs(prev_cost)):
                    if self.verbose:
                        print(f"  converged at epoch {epoch} (cost {cost:.6f})")
                    break
                prev_cost = cost

            if not diverged:
                self.learning_rate_used = lr
                return self
            lr /= 4.0

        raise RuntimeError("Gradient descent diverged at every learning rate tried")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model is not fitted")
        zs = self._standardise(np.asarray(X, dtype=np.float64)) @ self.w + self.b
        return zs * self._y_sigma + self._y_mu   # back to vehicles / km-h / minutes

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def coefficients(self, feature_names: list[str]) -> dict[str, float]:
        """Standardised coefficients — directly comparable across features.

        Because every column was z-scored, a coefficient reads as "change in
        predicted volume per one standard deviation of this feature", which is
        the form that makes a linear model interpretable to an operator.
        """
        scaled = (self.w * self._y_sigma).tolist()
        return dict(zip(feature_names, scaled))

    def gradient_check(self, X: np.ndarray, y: np.ndarray,
                       eps: float = 1e-5, n_check: int = 8) -> float:
        """Verify the analytic gradient against a numerical one.

        Returns the maximum relative difference. Anything above ~1e-5 means the
        derivative above is wrong, and every result that depends on it is too.
        """
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)[:2000]
        y = np.asarray(y, dtype=np.float64).ravel()[:2000]
        Xs = self._standardise_fit(X) if self._mu is None else self._standardise(X)

        ys = (y - y.mean()) / (y.std() or 1.0)
        w = rng.normal(0, 0.1, Xs.shape[1])
        b = 0.3
        grad_w, _ = self._gradients(Xs, ys, w, b, self.l2)

        worst = 0.0
        for j in rng.choice(Xs.shape[1], size=min(n_check, Xs.shape[1]), replace=False):
            w_up, w_dn = w.copy(), w.copy()
            w_up[j] += eps
            w_dn[j] -= eps
            numeric = (self._cost(Xs, ys, w_up, b, self.l2) -
                       self._cost(Xs, ys, w_dn, b, self.l2)) / (2 * eps)
            denom = max(1e-8, abs(numeric) + abs(grad_w[j]))
            worst = max(worst, abs(numeric - grad_w[j]) / denom)
        return float(worst)
