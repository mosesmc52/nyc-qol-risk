import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_qol_risk_model(
    y,  # shape: (N,) complaint counts
    nta_idx,  # shape: (N,) maps obs -> NTA
    week_idx,  # shape: (N,) maps obs -> week
    cat_idx,  # shape: (N,) maps obs -> category
    exposure,  # shape: (N,) population or housing-unit exposure
    X_nta,  # shape: (n_nta, p) PLUTO features aggregated to NTA
    n_nta,
    n_week,
    n_cat,
    coords=None,
):
    """
    Observation unit: one row per (NTA, week, category)

    y: observed complaint counts
    exposure: positive exposure for offset, same length as y
    X_nta: NTA-level structural features (standardized)
    """

    p = X_nta.shape[1]

    if coords is None:
        coords = {
            "obs": np.arange(len(y)),
            "nta": np.arange(n_nta),
            "week": np.arange(n_week),
            "category": np.arange(n_cat),
            "feature": np.arange(p),
        }

    with pm.Model(coords=coords) as model:
        # data
        y_data = pm.Data("y_data", y, dims="obs")
        nta_id = pm.Data("nta_id", nta_idx, dims="obs")
        week_id = pm.Data("week_id", week_idx, dims="obs")
        cat_id = pm.Data("cat_id", cat_idx, dims="obs")
        expo = pm.Data("expo", exposure, dims="obs")
        X = pm.Data("X", X_nta, dims=("nta", "feature"))

        # global intercept
        beta0 = pm.Normal("beta0", 0.0, 1.0)

        # PLUTO coefficients
        beta = pm.Normal("beta", 0.0, 0.5, dims="feature")

        # NTA random intercept
        sigma_nta = pm.Exponential("sigma_nta", 1.0)
        alpha_nta = pm.Normal("alpha_nta", 0.0, sigma_nta, dims="nta")

        # week effect
        sigma_week = pm.Exponential("sigma_week", 1.0)
        gamma_week = pm.Normal("gamma_week", 0.0, sigma_week, dims="week")

        # category effect
        sigma_cat = pm.Exponential("sigma_cat", 1.0)
        delta_cat = pm.Normal("delta_cat", 0.0, sigma_cat, dims="category")

        # latent dynamic QoL risk: NTA x week random walk
        sigma_rw = pm.Exponential("sigma_rw", 2.0)

        # innovations
        eps = pm.Normal("eps", 0.0, sigma_rw, dims=("nta", "week"))

        # cumulative sum over weeks = random walk
        risk_rw = pm.Deterministic(
            "risk_rw",
            pt.cumsum(eps, axis=1),
            dims=("nta", "week"),
        )

        # NTA structural component from PLUTO
        structural = pm.Deterministic(
            "structural",
            X @ beta,
            dims="nta",
        )

        # linear predictor
        eta = (
            beta0
            + pt.log(expo)
            + structural[nta_id]
            + alpha_nta[nta_id]
            + gamma_week[week_id]
            + delta_cat[cat_id]
            + risk_rw[nta_id, week_id]
        )

        mu = pm.Deterministic("mu", pt.exp(eta), dims="obs")

        # overdispersion
        phi = pm.Exponential("phi", 1.0)

        # likelihood
        pm.NegativeBinomial("y_like", mu=mu, alpha=phi, observed=y_data, dims="obs")

    return model
