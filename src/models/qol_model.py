import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_reported_qol_pressure_model(
    y,  # shape: (N,) observed counts
    nta_idx,  # shape: (N,) obs -> NTA
    month_idx,  # shape: (N,) obs -> month
    cat_idx,  # shape: (N,) obs -> category
    cat_group_idx,  # shape: (N,) obs -> category group (can equal cat_idx)
    exposure,  # shape: (N,) positive offset
    X_issue_nta,  # shape: (n_nta, p_issue)
    X_reporting_nta,  # shape: (n_nta, p_reporting)
    n_nta,
    n_month,
    n_cat,
    n_cat_group,
    month_of_year=None,  # optional shape: (N,), values 1..12
    coords=None,
    include_local_state=False,
    include_reporting_structural=False,
):
    """
    Hierarchical Negative Binomial model for reported 311 complaint pressure.

    Observation unit:
        one row per (NTA, month, category)

    Interpretation:
        This models reported complaint intensity, not ground-truth QoL.
        The linear predictor separates:
          - issue-generating structural factors
          - reporting-propensity structural factors
          - baseline NTA / month / category effects
          - local NTA x category-group dynamic deviations over time

    Parameters
    ----------
    y : array-like, shape (N,)
        Observed complaint counts.

    nta_idx, month_idx, cat_idx, cat_group_idx : array-like, shape (N,)
        Integer index arrays mapping each observation to its group.

    exposure : array-like, shape (N,)
        Positive exposure term used as a log offset.
        Example: population, households, renter households, etc.

    X_issue_nta : array-like, shape (n_nta, p_issue)
        Standardized NTA-level features associated with actual issue generation.

    X_reporting_nta : array-like, shape (n_nta, p_reporting)
        Standardized NTA-level features associated with reporting propensity.

    month_of_year : array-like, shape (N,), optional
        Integer month-of-year values for seasonality. If provided, a simple
        harmonic seasonal term is added.
    include_local_state : bool, default False
        If True, include an AR(1) local state per (NTA, category_group, month).
        This is more expressive but significantly heavier to sample.
    include_reporting_structural : bool, default False
        If True, include the reporting structural feature block.

    Notes
    -----
    - A category-group dynamic state can be included via AR(1):
          local_state[nta, cat_group, month]
      This is more flexible than one shared NTA-month latent state.
    - If you do not yet have grouped categories, pass:
          cat_group_idx = cat_idx
          n_cat_group = n_cat
    """

    y = np.asarray(y)
    nta_idx = np.asarray(nta_idx)
    month_idx = np.asarray(month_idx)
    cat_idx = np.asarray(cat_idx)
    cat_group_idx = np.asarray(cat_group_idx)
    exposure = np.asarray(exposure)

    X_issue_nta = np.asarray(X_issue_nta)
    X_reporting_nta = np.asarray(X_reporting_nta)

    if np.any(exposure <= 0):
        raise ValueError("All exposure values must be strictly positive.")

    if X_issue_nta.shape[0] != n_nta:
        raise ValueError("X_issue_nta must have shape (n_nta, p_issue).")

    if X_reporting_nta.shape[0] != n_nta:
        raise ValueError("X_reporting_nta must have shape (n_nta, p_reporting).")

    p_issue = X_issue_nta.shape[1]
    p_reporting = X_reporting_nta.shape[1]

    if coords is None:
        coords = {
            "obs": np.arange(len(y)),
            "nta": np.arange(n_nta),
            "month": np.arange(n_month),
            "category": np.arange(n_cat),
            "category_group": np.arange(n_cat_group),
            "issue_feature": np.arange(p_issue),
            "reporting_feature": np.arange(p_reporting),
        }

    with pm.Model(coords=coords) as model:
        # ------------------------------------------------------------------
        # Data containers
        # ------------------------------------------------------------------
        y_data = pm.Data("y_data", y, dims="obs")
        nta_id = pm.Data("nta_id", nta_idx, dims="obs")
        month_id = pm.Data("month_id", month_idx, dims="obs")
        cat_id = pm.Data("cat_id", cat_idx, dims="obs")
        cat_group_id = pm.Data("cat_group_id", cat_group_idx, dims="obs")
        expo = pm.Data("expo", exposure, dims="obs")

        X_issue = pm.Data("X_issue", X_issue_nta, dims=("nta", "issue_feature"))
        X_reporting = pm.Data(
            "X_reporting", X_reporting_nta, dims=("nta", "reporting_feature")
        )

        if month_of_year is not None:
            month_of_year = np.asarray(month_of_year)
            moy = pm.Data("month_of_year", month_of_year, dims="obs")
            # convert to radians for annual monthly seasonality
            theta = 2.0 * np.pi * (moy / 12.0)
        else:
            theta = None

        # ------------------------------------------------------------------
        # Global intercept
        # ------------------------------------------------------------------
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.0)

        # ------------------------------------------------------------------
        # Structural components
        # ------------------------------------------------------------------
        beta_issue = pm.Normal("beta_issue", mu=0.0, sigma=0.5, dims="issue_feature")
        beta_reporting = pm.Normal(
            "beta_reporting", mu=0.0, sigma=0.5, dims="reporting_feature"
        )

        issue_structural = pm.Deterministic(
            "issue_structural",
            X_issue @ beta_issue,
            dims="nta",
        )

        if include_reporting_structural:
            reporting_structural = pm.Deterministic(
                "reporting_structural",
                X_reporting @ beta_reporting,
                dims="nta",
            )
            reporting_structural_term = reporting_structural[nta_id]
        else:
            reporting_structural_term = 0.0

        # ------------------------------------------------------------------
        # Random intercepts
        # ------------------------------------------------------------------
        sigma_nta = pm.Exponential("sigma_nta", 1.0)
        alpha_nta = pm.Normal("alpha_nta", mu=0.0, sigma=sigma_nta, dims="nta")

        sigma_month = pm.Exponential("sigma_month", 1.0)
        gamma_month = pm.Normal("gamma_month", mu=0.0, sigma=sigma_month, dims="month")

        sigma_cat = pm.Exponential("sigma_cat", 1.0)
        delta_cat = pm.Normal("delta_cat", mu=0.0, sigma=sigma_cat, dims="category")

        # ------------------------------------------------------------------
        # Optional global seasonality
        # ------------------------------------------------------------------
        if theta is not None:
            beta_sin = pm.Normal("beta_sin", mu=0.0, sigma=0.5)
            beta_cos = pm.Normal("beta_cos", mu=0.0, sigma=0.5)
            seasonal_term = beta_sin * pt.sin(theta) + beta_cos * pt.cos(theta)
        else:
            seasonal_term = 0.0

        # ------------------------------------------------------------------
        # Optional local dynamic state: NTA x category-group x month
        # AR(1) over the month axis
        # ------------------------------------------------------------------
        if include_local_state:
            rho = pm.Uniform("rho", lower=-0.95, upper=0.95)
            sigma_state = pm.Exponential("sigma_state", 2.0)

            # Time axis should be the last axis for pm.AR
            local_state = pm.AR(
                "local_state",
                rho=rho,
                sigma=sigma_state,
                init_dist=pm.Normal.dist(
                    mu=0.0,
                    sigma=sigma_state / pt.sqrt(1.0 - rho**2),
                ),
                ar_order=1,
                dims=("nta", "category_group", "month"),
            )
            local_state_term = local_state[nta_id, cat_group_id, month_id]
        else:
            local_state_term = 0.0

        # ------------------------------------------------------------------
        # Linear predictor
        # ------------------------------------------------------------------
        eta = (
            beta0
            + pt.log(expo)
            + issue_structural[nta_id]
            + reporting_structural_term
            + alpha_nta[nta_id]
            + gamma_month[month_id]
            + delta_cat[cat_id]
            + local_state_term
            + seasonal_term
        )

        mu = pm.Deterministic("mu", pt.exp(eta), dims="obs")

        # ------------------------------------------------------------------
        # Negative Binomial likelihood
        # ------------------------------------------------------------------
        alpha_nb_cat = pm.Exponential("alpha_nb_cat", 1.0, dims="category")

        pm.NegativeBinomial(
            "y_like",
            mu=mu,
            alpha=alpha_nb_cat[cat_id],
            observed=y_data,
            dims="obs",
        )

    return model
