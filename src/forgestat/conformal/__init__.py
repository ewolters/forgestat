"""Split conformal prediction for regression and classification.

Provides finite-sample marginal coverage guarantees under exchangeability
(i.i.d.) — no distributional assumptions required.

Regression: constant-width prediction intervals via absolute residual scores.
Classification: prediction sets via softmax nonconformity scores.

References:
    Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
    Lei et al. (2018). Distribution-Free Predictive Inference for Regression.
    Romano, Sesia, Candès (2020). Classification with Valid Adaptive Coverage.
"""

import numpy as np

__all__ = [
    "conformal_qhat",
    "ConformalRegressor",
    "ConformalClassifier",
    "compute_conformal",
]


def conformal_qhat(scores_sorted, alpha, n_cal):
    """Conformal quantile via order statistic — the coverage guarantee mechanism.

    Given sorted nonconformity scores s_(1) <= ... <= s_(n), compute:
        k = ceil((n+1)(1-alpha))
        qhat = s_(k)   (1-indexed)

    Uses the actual order statistic, NOT np.quantile interpolation.
    This is what gives the finite-sample coverage guarantee.

    Parameters
    ----------
    scores_sorted : ndarray
        Sorted nonconformity scores from calibration set.
    alpha : float
        Miscoverage level (e.g. 0.10 for 90% coverage).
    n_cal : int
        Number of calibration points.

    Returns
    -------
    float
        The conformal quantile qhat.
    """
    k = int(np.ceil((n_cal + 1) * (1 - alpha))) - 1  # 0-indexed
    k = int(np.clip(k, 0, n_cal - 1))
    return float(scores_sorted[k])


class ConformalRegressor:
    """Split conformal prediction intervals for regression.

    Nonconformity score: |y - yhat| (absolute residual).

    Valid marginal coverage guarantee under exchangeability (i.i.d.).
    Intervals are constant-width — honest but may be wider than needed
    in low-noise regions if error variance is heteroscedastic.
    """

    def calibrate(self, y_cal, y_pred_cal, alphas=(0.10, 0.05)):
        """Compute nonconformity scores on calibration set and store qhats.

        Parameters
        ----------
        y_cal : array-like
            True responses on calibration set.
        y_pred_cal : array-like
            Model predictions on calibration set (from X_cal only).
        alphas : tuple of float
            Miscoverage levels to precompute qhats for.
        """
        self.scores_sorted = np.sort(np.abs(np.asarray(y_cal) - np.asarray(y_pred_cal)))
        self.n_cal = len(self.scores_sorted)
        self.qhats = {}
        for alpha in alphas:
            self.qhats[str(alpha)] = conformal_qhat(self.scores_sorted, alpha, self.n_cal)

    def predict_interval(self, y_pred, alpha=0.10):
        """Return (lower, upper) arrays with nominal 1-alpha coverage.

        Parameters
        ----------
        y_pred : array-like
            Point predictions for new data.
        alpha : float
            Miscoverage level.

        Returns
        -------
        lower : ndarray
        upper : ndarray
        """
        qhat = self.qhats.get(str(alpha))
        if qhat is None:
            if self.scores_sorted is not None:
                qhat = conformal_qhat(self.scores_sorted, alpha, self.n_cal)
            else:
                raise ValueError(
                    f"No qhat for alpha={alpha} and scores not available. "
                    f"Available alphas: {list(self.qhats.keys())}"
                )
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return y_pred - qhat, y_pred + qhat

    def get_state(self):
        """Serialize for model persistence. Stores qhats, not full scores."""
        return {
            "type": "regression",
            "method": "split_abs_residual",
            "n_cal": self.n_cal,
            "qhats": self.qhats,
        }

    @classmethod
    def from_state(cls, state):
        """Reconstruct from saved state for inference."""
        obj = cls()
        obj.n_cal = state["n_cal"]
        obj.qhats = state["qhats"]
        obj.scores_sorted = None
        return obj


class ConformalClassifier:
    """Split conformal prediction sets for classification.

    Nonconformity score: 1 - p(true class).
    Prediction set: include class c if p(c|x) >= 1 - qhat.

    Requires a model with predict_proba().
    """

    def calibrate(self, y_cal_int, proba_cal, alphas=(0.10, 0.05)):
        """Compute nonconformity scores on calibration set.

        Parameters
        ----------
        y_cal_int : array-like of int
            Integer-encoded true class labels matching proba_cal column indices.
        proba_cal : ndarray (n_cal, n_classes)
            Class probability predictions on calibration set (from X_cal only).
        alphas : tuple of float
            Miscoverage levels to precompute qhats for.
        """
        y_cal_int = np.asarray(y_cal_int, dtype=int)
        proba_cal = np.asarray(proba_cal)
        scores = np.array([1.0 - proba_cal[i, y_cal_int[i]] for i in range(len(y_cal_int))])
        self.scores_sorted = np.sort(scores)
        self.n_cal = len(self.scores_sorted)
        self.qhats = {}
        for alpha in alphas:
            self.qhats[str(alpha)] = conformal_qhat(self.scores_sorted, alpha, self.n_cal)

    def predict_sets(self, proba_new, alpha=0.10):
        """Return prediction sets — list of class-index lists per sample.

        Parameters
        ----------
        proba_new : ndarray (n_samples, n_classes)
            Class probability predictions for new data.
        alpha : float
            Miscoverage level.

        Returns
        -------
        sets : list of list of int
            For each sample, the indices of classes in the prediction set.
        meta : dict
            Contains 'threshold' and 'qhat'.
        """
        qhat = self.qhats.get(str(alpha))
        if qhat is None:
            if self.scores_sorted is not None:
                qhat = conformal_qhat(self.scores_sorted, alpha, self.n_cal)
            else:
                raise ValueError(
                    f"No qhat for alpha={alpha} and scores not available. "
                    f"Available alphas: {list(self.qhats.keys())}"
                )
        proba_new = np.asarray(proba_new)
        threshold = 1.0 - qhat
        mask = proba_new >= threshold
        sets = [np.where(mask[i])[0].tolist() for i in range(mask.shape[0])]
        return sets, {"threshold": float(threshold), "qhat": float(qhat)}

    def get_state(self):
        """Serialize for model persistence."""
        return {
            "type": "classification",
            "method": "split_softmax",
            "n_cal": self.n_cal,
            "qhats": self.qhats,
        }

    @classmethod
    def from_state(cls, state):
        """Reconstruct from saved state for inference."""
        obj = cls()
        obj.n_cal = state["n_cal"]
        obj.qhats = state["qhats"]
        obj.scores_sorted = None
        return obj


def compute_conformal(model, X_cal, y_cal, task_type, alphas=(0.10, 0.05)):
    """One-call wrapper: predict on X_cal, calibrate, return conformal object.

    Predictions are computed on X_cal only — no leakage from test set.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Must have predict() (regression) or predict_proba() (classification).
    X_cal : DataFrame or ndarray
        Calibration features (never seen during training).
    y_cal : array-like
        Calibration targets.
    task_type : str
        "regression" or "classification".
    alphas : tuple of float
        Miscoverage levels to precompute.

    Returns
    -------
    ConformalRegressor or ConformalClassifier
    """
    if task_type == "regression":
        y_pred_cal = model.predict(X_cal)
        cf = ConformalRegressor()
        cf.calibrate(y_cal, y_pred_cal, alphas=alphas)
        return cf
    else:
        proba_cal = model.predict_proba(X_cal)
        y_cal_int = np.asarray(y_cal).astype(int)
        cf = ConformalClassifier()
        cf.calibrate(y_cal_int, proba_cal, alphas=alphas)
        return cf
