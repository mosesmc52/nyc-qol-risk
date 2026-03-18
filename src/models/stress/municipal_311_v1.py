import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_311_stress_model(
    y_log,  # shape (G, T): log(p90_duration_hours) by NTA g and week t
    X=None,  # optional shape (G, T): e.g., standardized log(volume)
    coords=None,
):
    """
    y_log: np.ndarray (G, T) float
    X:     np.ndarray (G, T) float or None
    """

    G, T = y_log.shape
    if coords is None:
        coords = {"geo": np.arange(G), "time": np.arange(T)}

    with pm.Model(coords=coords) as model:
        # ----------------------------
        # Hyperpriors (partial pooling)
        # ----------------------------
        mu0 = pm.Normal("mu0", 0.0, 1.0)  # global baseline (on log scale)
        sigma_geo = pm.Exponential("sigma_geo", 1.0)  # cross-geo baseline spread

        baseline_geo = pm.Normal("baseline_geo", mu=mu0, sigma=sigma_geo, dims="geo")

        # ----------------------------
        # Latent stress dynamics
        # ----------------------------
        sigma_s = pm.Exponential("sigma_s", 1.0)  # innovation scale for stress RW
        S = pm.GaussianRandomWalk("S", sigma=sigma_s, dims=("geo", "time"))

        # Threshold + amplification
        tau = pm.Normal("tau", 0.0, 1.0)  # stress threshold (latent scale)
        A = pm.Deterministic("A", pt.softplus(S - tau))  # amplified stress (>=0)

        # Effect size: how amplified stress moves log duration
        alpha = pm.HalfNormal("alpha", 1.0)

        # Optional shock covariate (e.g., volume)
        if X is not None:
            beta = pm.Normal("beta", 0.0, 0.5)
            cov_term = beta * pm.MutableData("X", X, dims=("geo", "time"))
        else:
            cov_term = 0.0

        # Expected log-duration
        mu = pm.Deterministic(
            "mu",
            baseline_geo[:, None] + alpha * A + cov_term,
            dims=("geo", "time"),
        )

        # Observation noise
        sigma_y = pm.Exponential("sigma_y", 1.0)

        # Likelihood (log durations ~ Normal)
        pm.Normal(
            "y",
            mu=mu,
            sigma=sigma_y,
            observed=y_log,
            dims=("geo", "time"),
        )

    return model
