from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm


@dataclass
class InterventionConfig:
    n_seasons: int = 7
    trend: bool = True
    pre_period: tuple[int, int] | None = None
    post_period: tuple[int, int] | None = None
    mcmc_samples: int = 1000
    tune: int = 500
    target_accept: float = 0.9
    random_seed: int = 42


class BayesianCausalImpact:
    def __init__(self, config: InterventionConfig = InterventionConfig()) -> None:
        self.config = config
        self._model: pm.Model | None = None
        self._idata: az.InferenceData | None = None
        self._n_pre: int = 0

    def fit(self, y_pre: np.ndarray, X_pre: np.ndarray | None = None) -> "BayesianCausalImpact":
        n = len(y_pre)
        self._n_pre = n

        with pm.Model() as model:
            sigma_level = pm.HalfNormal("sigma_level", sigma=1.0)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=np.std(y_pre) + 1e-6)

            level = pm.GaussianRandomWalk(
                "level",
                sigma=sigma_level,
                init_dist=pm.Normal.dist(mu=y_pre[0], sigma=np.std(y_pre) + 1e-6),
                shape=n,
            )

            mu = level
            if self.config.trend:
                slope_init = pm.Normal("slope_init", mu=0, sigma=0.1)
                sigma_slope = pm.HalfNormal("sigma_slope", sigma=0.1)
                slope = pm.GaussianRandomWalk(
                    "slope",
                    sigma=sigma_slope,
                    init_dist=pm.Normal.dist(mu=slope_init, sigma=0.1),
                    shape=n,
                )
                mu = mu + slope

            pm.Normal("obs", mu=mu, sigma=sigma_obs, observed=y_pre)
            self._model = model

        with self._model:
            self._idata = pm.sample(
                draws=self.config.mcmc_samples,
                tune=self.config.tune,
                target_accept=self.config.target_accept,
                random_seed=self.config.random_seed,
                progressbar=True,
            )

        return self

    def predict_counterfactual(self, n_post: int) -> az.InferenceData:
        if self._model is None or self._idata is None:
            raise RuntimeError("Call fit() before predict_counterfactual()")

        with pm.Model():
            sigma_level = pm.HalfNormal("sigma_level", sigma=1.0)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

            level = pm.GaussianRandomWalk(
                "level",
                sigma=sigma_level,
                init_dist=pm.Normal.dist(mu=0, sigma=1.0),
                shape=n_post,
            )
            mu = level
            if self.config.trend:
                sigma_slope = pm.HalfNormal("sigma_slope", sigma=0.1)
                slope = pm.GaussianRandomWalk(
                    "slope",
                    sigma=sigma_slope,
                    init_dist=pm.Normal.dist(mu=0, sigma=0.1),
                    shape=n_post,
                )
                mu = mu + slope

            posterior_level = self._idata.posterior["level"].values
            last_levels = posterior_level[:, :, -1]
            flat_last = last_levels.flatten()
            cf_samples = flat_last[:, None] + np.random.randn(len(flat_last), n_post) * np.exp(
                self._idata.posterior["sigma_obs"].values.flatten()
            ).mean()

        cf_idata = az.convert_to_inference_data(
            {"counterfactual": cf_samples.reshape(1, -1, n_post)}
        )
        return cf_idata

    def estimate_effect(
        self, y_post: np.ndarray, idata: az.InferenceData
    ) -> dict:
        cf = idata.posterior["counterfactual"].values.reshape(-1, len(y_post))
        pointwise = y_post[None, :] - cf
        cumulative = pointwise.sum(axis=1)
        cf_mean = cf.mean(axis=0)
        relative_effect = (y_post.mean() - cf_mean.mean()) / (cf_mean.mean() + 1e-9)
        p_positive = float((cumulative > 0).mean())
        ci_lo, ci_hi = float(np.percentile(cumulative, 2.5)), float(np.percentile(cumulative, 97.5))

        return {
            "cumulative_effect": float(cumulative.mean()),
            "relative_effect_pct": float(relative_effect * 100),
            "p_value": float(1 - p_positive) if p_positive > 0.5 else float(p_positive),
            "credible_interval_95": [ci_lo, ci_hi],
            "posterior_prob_positive": p_positive,
        }

    def plot(
        self,
        y_full: np.ndarray,
        idata: az.InferenceData,
        intervention_date: str,
        output_path: str,
    ) -> None:
        n_pre = self._n_pre
        n_post = len(y_full) - n_pre
        t = np.arange(len(y_full))
        cf = idata.posterior["counterfactual"].values.reshape(-1, n_post)
        cf_mean = cf.mean(axis=0)
        cf_lo = np.percentile(cf, 5, axis=0)
        cf_hi = np.percentile(cf, 95, axis=0)

        pointwise = y_full[n_pre:] - cf
        pw_mean = pointwise.mean(axis=0)
        pw_lo = np.percentile(pointwise, 5, axis=0)
        pw_hi = np.percentile(pointwise, 95, axis=0)

        cum_mean = np.cumsum(pw_mean)
        cum_lo = np.cumsum(np.percentile(pointwise, 5, axis=0))
        cum_hi = np.cumsum(np.percentile(pointwise, 95, axis=0))

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

        ax0 = axes[0]
        ax0.plot(t[:n_pre], y_full[:n_pre], color="steelblue", label="Observed (pre)")
        ax0.plot(t[n_pre:], y_full[n_pre:], color="steelblue", label="Observed (post)")
        ax0.plot(t[n_pre:], cf_mean, color="orangered", linestyle="--", label="Counterfactual")
        ax0.fill_between(t[n_pre:], cf_lo, cf_hi, color="orangered", alpha=0.2)
        ax0.axvline(n_pre, color="black", linestyle=":", label=f"Intervention ({intervention_date})")
        ax0.set_title("Observed vs Counterfactual")
        ax0.legend(fontsize=8)

        ax1 = axes[1]
        t_post = np.arange(n_post)
        ax1.plot(t_post, pw_mean, color="green")
        ax1.fill_between(t_post, pw_lo, pw_hi, color="green", alpha=0.2)
        ax1.axhline(0, color="black", linestyle="--")
        ax1.set_title("Pointwise Effect (Observed − Counterfactual)")

        ax2 = axes[2]
        ax2.plot(t_post, cum_mean, color="purple")
        ax2.fill_between(t_post, cum_lo, cum_hi, color="purple", alpha=0.2)
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_title("Cumulative Effect")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
