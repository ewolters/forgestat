"""Anytime-Valid Inference — E-processes and Confidence Sequences.

Sequential testing where you can peek at data continuously without inflating
error rates. Based on Grünwald et al. (JRSS-B 2024), Howard et al. (2021),
and Waudby-Smith & Ramdas (2024).

Core objects:
    GaussianMeanEProcess          — known sigma, mixture over mean
    SelfNormalizedMeanEProcess    — unknown sigma, empirical variance
    TwoSampleEProcess             — A/B test wrapper

Key property: each E-process is a supermartingale under H0.
    E[E_t | F_{t-1}] <= E_{t-1}
    -> valid at any data-dependent stopping time (optional stopping).

All arithmetic in log-space for numerical stability.

Dependencies: numpy only (math from stdlib).
"""

import math

__all__ = [
    "GaussianMeanEProcess",
    "SelfNormalizedMeanEProcess",
    "TwoSampleEProcess",
]


class GaussianMeanEProcess:
    """E-process for testing H0: mu = mu0 (known variance).

    Construction: mixture likelihood ratio
        E_t = integral prod_i [f(x_i; mu, sigma^2) / f(x_i; mu0, sigma^2)] pi(dmu)

    with mixing prior mu ~ N(mu0, rho^2).

    Closed form (derivation via completing the square):
        logE_t = -0.5 log(1 + t*rho^2/sigma^2) + rho^2*S_t^2 / (2*sigma^2*(sigma^2 + t*rho^2))

    where S_t = sum_i (x_i - mu0) is the cumulative deviation.

    Parameters
    ----------
    mu0 : float
        Null hypothesis mean.
    sigma : float
        Known standard deviation.
    rho : float
        Scale of the mixing prior — controls sensitivity.
        Larger rho = more sensitive to large effects, less to small.
        Rule of thumb: rho ~ expected effect size * sigma.
    """

    def __init__(self, mu0: float = 0.0, sigma: float = 1.0, rho: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if rho <= 0:
            raise ValueError("rho must be positive")

        self.mu0 = float(mu0)
        self.sigma = float(sigma)
        self.rho = float(rho)

        self.t = 0
        self.S_t = 0.0
        self.sum_x = 0.0
        self._logE = 0.0
        self._history = []

    def update(self, x: float) -> "GaussianMeanEProcess":
        """Process one observation. Updates E_t multiplicatively."""
        self.t += 1
        self.S_t += x - self.mu0
        self.sum_x += x

        sigma2 = self.sigma**2
        rho2 = self.rho**2
        t = self.t

        V_t = t * rho2 / sigma2
        self._logE = -0.5 * math.log1p(V_t) + rho2 * self.S_t**2 / (2 * sigma2 * (sigma2 + t * rho2))

        self._history.append((t, self._logE, self.sum_x / t))
        return self

    def update_batch(self, xs) -> "GaussianMeanEProcess":
        """Process multiple observations sequentially."""
        for x in xs:
            self.update(float(x))
        return self

    @property
    def log_e(self) -> float:
        """Current log e-value. E_t = exp(logE)."""
        return self._logE

    @property
    def e_value(self) -> float:
        """Current e-value (clamped for display)."""
        if self._logE > 34:  # exp(34) ~ 5.8e14
            return 1e15
        return min(math.exp(self._logE), 1e15)

    def decision(self, alpha: float = 0.05) -> bool:
        """Reject H0 if logE_t >= log(1/alpha)."""
        return self._logE >= math.log(1.0 / alpha)

    def cs(self, alpha: float = 0.05) -> tuple:
        """(1-alpha)-confidence sequence: interval [L_t, U_t].

        Constructed by inverting the e-process: find the set of mu0
        for which E_t(mu0) < 1/alpha.

        Returns (L_t, U_t) or (nan, nan) if t=0.
        """
        if self.t == 0:
            return (float("nan"), float("nan"))

        t = self.t
        sigma2 = self.sigma**2
        rho2 = self.rho**2
        x_bar = self.sum_x / t

        threshold = math.log(1.0 / alpha) + 0.5 * math.log1p(t * rho2 / sigma2)
        if threshold <= 0:
            return (float("-inf"), float("inf"))

        half_width = (self.sigma / t) * math.sqrt(2 * (sigma2 + t * rho2) / rho2 * threshold)

        return (x_bar - half_width, x_bar + half_width)

    @property
    def history(self) -> list:
        """List of (t, logE_t, x_bar_t) tuples."""
        return list(self._history)

    def summary(self) -> dict:
        """Return a summary dict of current state."""
        L, U = self.cs()
        return {
            "t": self.t,
            "logE": round(self._logE, 6),
            "E": round(self.e_value, 6),
            "x_bar": round(self.sum_x / self.t, 6) if self.t > 0 else None,
            "S_t": round(self.S_t, 6),
            "cs_lower": round(L, 6) if not math.isnan(L) else None,
            "cs_upper": round(U, 6) if not math.isnan(U) else None,
            "mu0": self.mu0,
            "sigma": self.sigma,
            "rho": self.rho,
        }


class SelfNormalizedMeanEProcess:
    """E-process for testing H0: mu = mu0 (unknown variance).

    Uses empirical variance with a mixture construction.
    Based on Howard et al. (2021) sub-psi normal mixture.

    Note: this is an *approximate* e-process. The supermartingale property
    holds asymptotically. For exact finite-sample validity with unknown sigma,
    see Waudby-Smith & Ramdas (2024).

    Parameters
    ----------
    mu0 : float
        Null hypothesis mean.
    rho : float
        Scale of mixing prior.
    min_obs : int
        Minimum observations before computing (need >= 2 for variance).
    """

    def __init__(self, mu0: float = 0.0, rho: float = 1.0, min_obs: int = 5):
        if rho <= 0:
            raise ValueError("rho must be positive")

        self.mu0 = float(mu0)
        self.rho = float(rho)
        self.min_obs = max(2, int(min_obs))

        self.t = 0
        self.S_t = 0.0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self._logE = 0.0
        self._history = []

    def update(self, x: float) -> "SelfNormalizedMeanEProcess":
        """Process one observation."""
        self.t += 1
        x_f = float(x)
        self.S_t += x_f - self.mu0
        self.sum_x += x_f
        self.sum_x2 += x_f**2

        if self.t < self.min_obs:
            self._history.append((self.t, 0.0, self.sum_x / self.t))
            return self

        t = self.t
        rho2 = self.rho**2
        x_bar = self.sum_x / t

        V_hat = max(self.sum_x2 - t * x_bar**2, 1e-10)
        sigma2_hat = V_hat / t

        info_ratio = t * rho2 / V_hat
        self._logE = -0.5 * math.log1p(info_ratio) + rho2 * self.S_t**2 / (2 * V_hat * (sigma2_hat + rho2))

        self._history.append((self.t, self._logE, x_bar))
        return self

    def update_batch(self, xs) -> "SelfNormalizedMeanEProcess":
        """Process multiple observations sequentially."""
        for x in xs:
            self.update(float(x))
        return self

    @property
    def log_e(self) -> float:
        return self._logE

    @property
    def e_value(self) -> float:
        if self._logE > 34:
            return 1e15
        return min(math.exp(self._logE), 1e15)

    def decision(self, alpha: float = 0.05) -> bool:
        return self._logE >= math.log(1.0 / alpha)

    def cs(self, alpha: float = 0.05) -> tuple:
        """Confidence sequence using empirical variance."""
        if self.t < self.min_obs:
            return (float("nan"), float("nan"))

        t = self.t
        rho2 = self.rho**2
        x_bar = self.sum_x / t
        V_hat = max(self.sum_x2 - t * x_bar**2, 1e-10)
        sigma2_hat = V_hat / t

        threshold = math.log(1.0 / alpha) + 0.5 * math.log1p(t * rho2 / V_hat)
        if threshold <= 0:
            return (float("-inf"), float("inf"))

        half_width = math.sqrt(2 * sigma2_hat * (sigma2_hat + rho2) / rho2 * threshold) / math.sqrt(t)

        return (x_bar - half_width, x_bar + half_width)

    @property
    def history(self) -> list:
        return list(self._history)

    def summary(self) -> dict:
        L, U = self.cs()
        sigma_hat = (
            math.sqrt(max(self.sum_x2 - self.t * (self.sum_x / self.t) ** 2, 0) / self.t) if self.t >= 2 else None
        )
        return {
            "t": self.t,
            "logE": round(self._logE, 6),
            "E": round(self.e_value, 6),
            "x_bar": round(self.sum_x / self.t, 6) if self.t > 0 else None,
            "sigma_hat": round(sigma_hat, 6) if sigma_hat is not None else None,
            "cs_lower": round(L, 6) if not math.isnan(L) else None,
            "cs_upper": round(U, 6) if not math.isnan(U) else None,
            "mu0": self.mu0,
            "rho": self.rho,
        }


class TwoSampleEProcess:
    """Two-sample A/B test via anytime-valid inference.

    Tests H0: mu_A = mu_B by reducing to a one-sample test on
    paired differences d_i = x_{A,i} - x_{B,i}.

    Observations are paired FIFO: the first A pairs with the first B,
    second A with second B, etc. Unpaired observations are buffered.

    Parameters
    ----------
    rho : float
        Scale of mixing prior for the mean difference (expected
        effect size). Rule of thumb: rho ~ anticipated |mu_A - mu_B|.
    min_pairs : int
        Minimum paired observations before computing (need >= 2
        for variance estimation).
    """

    def __init__(self, rho: float = 0.5, min_pairs: int = 5):
        self.rho = float(rho)
        self.min_pairs = max(2, int(min_pairs))

        self._engine = SelfNormalizedMeanEProcess(mu0=0.0, rho=rho, min_obs=self.min_pairs)

        self._buf_a = []
        self._buf_b = []

        self.n_a = 0
        self.n_b = 0
        self.sum_a = 0.0
        self.sum_b = 0.0
        self.sum_a2 = 0.0
        self.sum_b2 = 0.0
        self.n_pairs = 0

        self._history = []

    def update(self, x: float, group: str) -> "TwoSampleEProcess":
        """Process one observation from group 'A' or 'B'."""
        x_f = float(x)
        if group.upper() == "A":
            self.n_a += 1
            self.sum_a += x_f
            self.sum_a2 += x_f**2
            self._buf_a.append(x_f)
        elif group.upper() == "B":
            self.n_b += 1
            self.sum_b += x_f
            self.sum_b2 += x_f**2
            self._buf_b.append(x_f)
        else:
            raise ValueError(f"group must be 'A' or 'B', got '{group}'")

        if self._buf_a and self._buf_b:
            xa = self._buf_a.pop(0)
            xb = self._buf_b.pop(0)
            self._engine.update(xa - xb)
            self.n_pairs += 1

        t = self.n_a + self.n_b
        if self.n_a > 0 and self.n_b > 0:
            diff = self.sum_a / self.n_a - self.sum_b / self.n_b
            if self.n_a > 1 and self.n_b > 1:
                var_a = max(self.sum_a2 / self.n_a - (self.sum_a / self.n_a) ** 2, 1e-10)
                var_b = max(self.sum_b2 / self.n_b - (self.sum_b / self.n_b) ** 2, 1e-10)
                se = math.sqrt(var_a / self.n_a + var_b / self.n_b)
            else:
                se = float("nan")
        else:
            diff, se = 0.0, float("nan")

        self._history.append((t, self._engine.log_e, diff, se))
        return self

    def update_groups(self, xs_a, xs_b) -> "TwoSampleEProcess":
        """Process arrays of observations from groups A and B, interleaved."""
        ia, ib = 0, 0
        while ia < len(xs_a) or ib < len(xs_b):
            if ia < len(xs_a):
                self.update(float(xs_a[ia]), "A")
                ia += 1
            if ib < len(xs_b):
                self.update(float(xs_b[ib]), "B")
                ib += 1
        return self

    @property
    def log_e(self) -> float:
        return self._engine.log_e

    @property
    def e_value(self) -> float:
        return self._engine.e_value

    def decision(self, alpha: float = 0.05) -> bool:
        return self._engine.decision(alpha)

    def cs(self, alpha: float = 0.05) -> tuple:
        """Confidence sequence for mu_A - mu_B."""
        return self._engine.cs(alpha)

    @property
    def history(self) -> list:
        return list(self._history)

    def summary(self) -> dict:
        L, U = self.cs()
        diff = (self.sum_a / self.n_a - self.sum_b / self.n_b) if self.n_a > 0 and self.n_b > 0 else None
        return {
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_pairs": self.n_pairs,
            "mean_a": round(self.sum_a / self.n_a, 6) if self.n_a > 0 else None,
            "mean_b": round(self.sum_b / self.n_b, 6) if self.n_b > 0 else None,
            "diff": round(diff, 6) if diff is not None else None,
            "logE": round(self._engine.log_e, 6),
            "E": round(self._engine.e_value, 6),
            "cs_lower": round(L, 6) if not math.isnan(L) else None,
            "cs_upper": round(U, 6) if not math.isnan(U) else None,
            "rho": self.rho,
        }
