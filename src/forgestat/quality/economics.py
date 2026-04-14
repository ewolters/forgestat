"""Decision-Theoretic Quality Economics.

Bayesian decision theory meets Taguchi loss functions for optimal quality
decisions under uncertainty. Instead of "is the process in control?",
answers "what is the expected cost of each possible action?"

Classes:
    TaguchiLoss        — quadratic loss functions (NIB, STB, LTB, asymmetric)
    ProcessDecision    — Bayesian optimal SPC action (continue/investigate/adjust)
    AcceptanceDecision — economic lot sentencing (accept/reject/screen)
    CostOfQuality      — CoQ breakdown with Bayesian uncertainty

Dependencies: numpy, scipy (for asymmetric loss only).
"""

import math

import numpy as np

__all__ = [
    "TaguchiLoss",
    "ProcessDecision",
    "AcceptanceDecision",
    "CostOfQuality",
]


class TaguchiLoss:
    """Quadratic loss L(y) = k (y - T)^2 for nominal-is-best (NIB).

    Variants:
        NIB:  L = k (y - T)^2           k = A_0 / delta_0^2
        STB:  L = k y^2                 (target = 0)
        LTB:  L = k / y^2              (target = inf)
        ASYM: L = k1 (y-T)^2 [y<T],  k2 (y-T)^2 [y>=T]

    Parameters
    ----------
    loss_type : str
        "nib", "stb", "ltb", or "asymmetric"
    target : float
        Target value (NIB/ASYM). Ignored for STB/LTB.
    delta0 : float
        Customer tolerance (functional limit from target).
    cost_at_limit : float
        Loss incurred when y = T +/- delta0 (e.g. warranty cost A_0).
    k_low, k_high : float, optional
        For asymmetric — separate k below / above target.
    """

    def __init__(
        self,
        loss_type="nib",
        target=0.0,
        delta0=1.0,
        cost_at_limit=100.0,
        k_low=None,
        k_high=None,
    ):
        self.loss_type = loss_type.lower()
        self.target = target
        self.delta0 = delta0
        self.cost_at_limit = cost_at_limit

        if self.loss_type == "asymmetric":
            if k_low is None or k_high is None:
                raise ValueError("asymmetric requires k_low and k_high")
            self.k_low = k_low
            self.k_high = k_high
        else:
            self.k = cost_at_limit / (delta0**2)

    def loss(self, y):
        """Point loss L(y)."""
        y = np.asarray(y, dtype=float)
        if self.loss_type == "nib":
            return self.k * (y - self.target) ** 2
        elif self.loss_type == "stb":
            return self.k * y**2
        elif self.loss_type == "ltb":
            return np.where(np.abs(y) < 1e-12, np.inf, self.k / y**2)
        elif self.loss_type == "asymmetric":
            d = y - self.target
            return np.where(d < 0, self.k_low * d**2, self.k_high * d**2)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def expected_loss(self, mu, sigma):
        """Expected loss E[L(Y)] when Y ~ N(mu, sigma^2).

        NIB:  E[L] = k [sigma^2 + (mu - T)^2]
        STB:  E[L] = k [sigma^2 + mu^2]
        ASYM: Computed via split Gaussian integral.
        LTB:  Approximation for mu >> sigma.
        """
        if self.loss_type == "nib":
            return self.k * (sigma**2 + (mu - self.target) ** 2)
        elif self.loss_type == "stb":
            return self.k * (sigma**2 + mu**2)
        elif self.loss_type == "ltb":
            if abs(mu) < 1e-8:
                return float("inf")
            return self.k * (1.0 / mu**2 + 3 * sigma**2 / mu**4)
        elif self.loss_type == "asymmetric":
            return self._expected_loss_asym(mu, sigma)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _expected_loss_asym(self, mu, sigma):
        """E[L] for asymmetric quadratic via split normal integral."""
        from scipy.stats import norm

        T = self.target
        z = (T - mu) / sigma if sigma > 1e-12 else float("inf")
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)

        var = sigma**2
        d = mu - T
        E_low = var * Phi_z + d**2 * Phi_z - sigma * d * phi_z
        E_high = var * (1 - Phi_z) + d**2 * (1 - Phi_z) + sigma * d * phi_z

        return self.k_low * E_low + self.k_high * E_high

    def to_dict(self):
        d = {
            "type": self.loss_type,
            "target": self.target,
            "delta0": self.delta0,
            "cost_at_limit": self.cost_at_limit,
        }
        if self.loss_type == "asymmetric":
            d["k_low"] = self.k_low
            d["k_high"] = self.k_high
        else:
            d["k"] = self.k
        return d


class ProcessDecision:
    """Bayesian decision for SPC: given posterior belief about process state,
    find the action minimizing expected cost.

    States:
        theta=0  process in control (IC)
        theta=1  process out of control (OOC)

    Actions:
        a=0  Continue (do nothing)
        a=1  Investigate (search for assignable cause)
        a=2  Adjust (reset process to target)

    Parameters
    ----------
    c_miss : float
        Cost per unit of missed defect (continue when OOC).
    c_fa : float
        False alarm investigation cost (investigate when IC).
    c_inv : float
        Investigation cost when truly OOC.
    c_over : float
        Unnecessary adjustment cost (adjust when IC).
    c_adj : float
        Adjustment cost when truly OOC.
    """

    def __init__(self, c_miss=500.0, c_fa=100.0, c_inv=80.0, c_over=120.0, c_adj=150.0):
        self.c_miss = c_miss
        self.c_fa = c_fa
        self.c_inv = c_inv
        self.c_over = c_over
        self.c_adj = c_adj

        self.L = np.array(
            [
                [0.0, c_miss],
                [c_fa, c_inv],
                [c_over, c_adj],
            ]
        )
        self.action_names = ["Continue", "Investigate", "Adjust"]

    def optimal_action(self, p_ooc):
        """Given P(OOC|data), return optimal action and expected costs.

        Parameters
        ----------
        p_ooc : float
            Posterior probability that process is out of control.

        Returns
        -------
        dict with keys: action, action_name, expected_costs, p_ooc, thresholds
        """
        p_ic = 1.0 - p_ooc
        expected = self.L @ np.array([p_ic, p_ooc])

        best = int(np.argmin(expected))

        denom_ci = self.c_fa + self.c_miss - self.c_inv
        thresh_ci = self.c_fa / denom_ci if denom_ci > 0 else 0.5

        denom_ca = self.c_over + self.c_miss - self.c_adj
        thresh_ca = self.c_over / denom_ca if denom_ca > 0 else 0.5

        denom_ia = (self.c_over - self.c_fa) + (self.c_inv - self.c_adj)
        thresh_ia = (self.c_over - self.c_fa) / denom_ia if abs(denom_ia) > 1e-10 else 0.5

        return {
            "action": best,
            "action_name": self.action_names[best],
            "expected_costs": {name: float(ec) for name, ec in zip(self.action_names, expected)},
            "p_ooc": float(p_ooc),
            "thresholds": {
                "continue_vs_investigate": float(thresh_ci),
                "continue_vs_adjust": float(thresh_ca),
                "investigate_vs_adjust": float(thresh_ia),
            },
            "cost_savings": float(np.max(expected) - np.min(expected)),
        }

    def sweep(self, n_points=200):
        """Expected cost curves over P(OOC) in [0, 1]."""
        ps = np.linspace(0, 1, n_points)
        costs = np.array([self.L @ np.array([1 - p, p]) for p in ps])
        optimal = np.argmin(costs, axis=1)
        return {
            "p_ooc": ps.tolist(),
            "continue": costs[:, 0].tolist(),
            "investigate": costs[:, 1].tolist(),
            "adjust": costs[:, 2].tolist(),
            "optimal_action": optimal.tolist(),
        }


class AcceptanceDecision:
    """Economic lot sentencing under uncertainty.

    Given posterior belief about lot defect rate p, compute expected cost
    of Accept / Reject / 100% Screen.

    Parameters
    ----------
    lot_size : int
        Number of units in the lot.
    c_external : float
        Cost per defective reaching the customer.
    c_internal : float
        Cost per defective caught during screening.
    c_inspection : float
        Cost per unit inspected.
    c_reject_lot : float
        Cost of rejecting the entire lot.
    """

    def __init__(
        self,
        lot_size=1000,
        c_external=50.0,
        c_internal=5.0,
        c_inspection=0.5,
        c_reject_lot=200.0,
    ):
        self.N = lot_size
        self.c_ext = c_external
        self.c_int = c_internal
        self.c_insp = c_inspection
        self.c_rej = c_reject_lot

    def expected_costs(self, p_defect):
        """Expected cost for each action given defect rate p."""
        p = np.asarray(p_defect, dtype=float)
        N = self.N

        cost_accept = N * p * self.c_ext
        cost_reject = np.full_like(p, self.c_rej)
        cost_screen = N * self.c_insp + N * p * self.c_int

        return {
            "accept": cost_accept,
            "reject": cost_reject,
            "screen": cost_screen,
        }

    def optimal_action(self, p_defect):
        """Optimal lot decision given posterior defect rate."""
        costs = self.expected_costs(p_defect)
        cost_arr = np.array(
            [
                float(np.mean(costs["accept"])),
                float(np.mean(costs["reject"])),
                float(np.mean(costs["screen"])),
            ]
        )
        names = ["Accept", "Reject", "100% Screen"]
        best = int(np.argmin(cost_arr))

        p_accept_reject = self.c_rej / (self.N * self.c_ext) if self.c_ext > 0 else 1.0
        denom = self.c_ext - self.c_int
        p_accept_screen = self.c_insp / denom if denom > 0 else 1.0

        return {
            "action": best,
            "action_name": names[best],
            "expected_costs": {
                "Accept": float(cost_arr[0]),
                "Reject": float(cost_arr[1]),
                "100% Screen": float(cost_arr[2]),
            },
            "p_defect": float(np.mean(p_defect)),
            "breakeven": {
                "accept_vs_reject": float(min(p_accept_reject, 1.0)),
                "accept_vs_screen": float(min(p_accept_screen, 1.0)),
            },
            "cost_savings": float(np.max(cost_arr) - np.min(cost_arr)),
        }

    def sweep(self, n_points=200):
        """Cost curves over p in [0, max_p]."""
        max_p = min(1.0, 3 * self.c_rej / (self.N * self.c_ext) if self.c_ext > 0 else 0.1)
        max_p = max(max_p, 0.02)
        ps = np.linspace(0, max_p, n_points)
        costs = self.expected_costs(ps)
        return {
            "p_defect": ps.tolist(),
            "accept": costs["accept"].tolist(),
            "reject": costs["reject"].tolist(),
            "screen": costs["screen"].tolist(),
        }


class CostOfQuality:
    """PAF model: Prevention + Appraisal + (Internal + External) Failure.

    Given cost data, computes ratios, optimal prevention investment,
    and total quality cost.

    Parameters
    ----------
    prevention : float
        Prevention spending.
    appraisal : float
        Appraisal (inspection/testing) spending.
    internal_failure : float
        Internal failure costs (scrap, rework).
    external_failure : float
        External failure costs (warranty, returns).
    revenue : float
        Total revenue (for CoQ/revenue ratio).
    """

    def __init__(
        self,
        prevention=0.0,
        appraisal=0.0,
        internal_failure=0.0,
        external_failure=0.0,
        revenue=1.0,
    ):
        self.prevention = prevention
        self.appraisal = appraisal
        self.internal_failure = internal_failure
        self.external_failure = external_failure
        self.revenue = max(revenue, 1.0)

    @property
    def total(self):
        return self.prevention + self.appraisal + self.internal_failure + self.external_failure

    @property
    def conformance_cost(self):
        """Cost of conformance (prevention + appraisal)."""
        return self.prevention + self.appraisal

    @property
    def nonconformance_cost(self):
        """Cost of nonconformance (internal + external failure)."""
        return self.internal_failure + self.external_failure

    def summary(self):
        t = self.total
        rev = self.revenue
        return {
            "prevention": self.prevention,
            "appraisal": self.appraisal,
            "internal_failure": self.internal_failure,
            "external_failure": self.external_failure,
            "total_coq": t,
            "conformance": self.conformance_cost,
            "nonconformance": self.nonconformance_cost,
            "coq_pct_revenue": t / rev * 100 if rev > 0 else 0,
            "conformance_ratio": (self.conformance_cost / t * 100 if t > 0 else 0),
            "nonconformance_ratio": (self.nonconformance_cost / t * 100 if t > 0 else 0),
            "prevention_pct": self.prevention / t * 100 if t > 0 else 0,
            "appraisal_pct": self.appraisal / t * 100 if t > 0 else 0,
            "int_failure_pct": (self.internal_failure / t * 100 if t > 0 else 0),
            "ext_failure_pct": (self.external_failure / t * 100 if t > 0 else 0),
        }

    def optimal_prevention_model(self, n_points=50):
        """Simple economic model: as prevention spending increases,
        failure costs decrease (exponential decay).

        Returns cost curves for plotting.
        """
        base_failure = self.nonconformance_cost
        base_prevention = max(self.prevention, 1.0)
        base_appraisal = self.appraisal

        lam = math.log(2)

        prev_range = np.linspace(0.1 * base_prevention, 5.0 * base_prevention, n_points)
        failure_curve = base_failure * np.exp(-lam * (prev_range / base_prevention - 1))
        appraisal_curve = np.full(n_points, base_appraisal)
        total_curve = prev_range + appraisal_curve + failure_curve

        opt_idx = int(np.argmin(total_curve))

        return {
            "prevention": prev_range.tolist(),
            "failure": failure_curve.tolist(),
            "appraisal": appraisal_curve.tolist(),
            "total": total_curve.tolist(),
            "optimal_prevention": float(prev_range[opt_idx]),
            "optimal_total": float(total_curve[opt_idx]),
            "current_prevention": float(base_prevention),
            "current_total": float(self.total),
        }
