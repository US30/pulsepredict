"""
Bayesian CausalImpact for intervention/promotion-lift estimation.

Uses a low-dimensional Bayesian structural model (intercept + trend +
weekly seasonality) fit on the pre-intervention period, then generates
posterior-predictive counterfactual forecasts for the post-period.
The causal effect is estimated as observed − counterfactual.

This design keeps the parameter space small (~10 dims) so NUTS converges
fast even without a C compiler (PyTensor Python backend on Windows).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm


@dataclass
class InterventionConfig:
    n_seasons: int = 7
    trend: bool = True
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
        self._y_pre_std: float = 1.0
        self._y_pre_mean: float = 0.0

    def fit(self, y_pre: np.ndarray) -> "BayesianCausalImpact":
        n = len(y_pre)
        self._n_pre = n
        self._y_pre_mean = float(np.mean(y_pre))
        self._y_pre_std = float(np.std(y_pre)) + 1e-6

        t = np.arange(n, dtype=np.float64) / n
        dow = np.arange(n) % self.config.n_seasons

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=self._y_pre_mean, sigma=self._y_pre_std * 2)
            sigma = pm.HalfNormal("sigma", sigma=self._y_pre_std)

            mu = alpha

            if self.config.trend:
                beta = pm.Normal("beta", mu=0, sigma=self._y_pre_std)
                mu = mu + beta * t

            if self.config.n_seasons > 1:
                season_effect = pm.Normal(
                    "season", mu=0, sigma=self._y_pre_std * 0.5,
                    shape=self.config.n_seasons,
                )
                mu = mu + season_effect[dow]

            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_pre)
            self._model = model

        with self._model:
            try:
                import nutpie
                compiled = nutpie.compile_pymc_model(self._model)
                self._idata = nutpie.sample(
                    compiled,
                    draws=self.config.mcmc_samples,
                    tune=self.config.tune,
                    chains=2,
                    seed=self.config.random_seed,
                    progress_bar=True,
                )
            except (ImportError, Exception):
                self._idata = pm.sample(
                    draws=self.config.mcmc_samples,
                    tune=self.config.tune,
                    target_accept=self.config.target_accept,
                    random_seed=self.config.random_seed,
                    cores=1,
                    chains=2,
                    progressbar=True,
                )

        return self

    def predict_counterfactual(self, n_post: int) -> az.InferenceData:
        if self._idata is None:
            raise RuntimeError("Call fit() before predict_counterfactual()")

        n_pre = self._n_pre
        t_post = np.arange(n_pre, n_pre + n_post, dtype=np.float64) / n_pre
        dow_post = np.arange(n_pre, n_pre + n_post) % self.config.n_seasons

        posterior = self._idata.posterior
        alpha = posterior["alpha"].values.flatten()
        sigma = posterior["sigma"].values.flatten()
        n_samples = len(alpha)

        mu = alpha[:, None] * np.ones((1, n_post))

        if self.config.trend and "beta" in posterior:
            beta = posterior["beta"].values.flatten()
            mu = mu + beta[:, None] * t_post[None, :]

        if self.config.n_seasons > 1 and "season" in posterior:
            season = posterior["season"].values.reshape(n_samples, self.config.n_seasons)
            mu = mu + season[:, dow_post]

        rng = np.random.default_rng(self.config.random_seed)
        cf_samples = mu + rng.normal(0, 1, mu.shape) * sigma[:, None]

        cf_idata = az.convert_to_inference_data(
            {"counterfactual": cf_samples.reshape(1, n_samples, n_post)}
        )
        return cf_idata

    def estimate_effect(
        self, y_post: np.ndarray, idata: az.InferenceData
    ) -> dict:
        cf = idata.posterior["counterfactual"].values.reshape(-1, len(y_post))
        pointwise = y_post[None, :] - cf
        cumulative = pointwise.sum(axis=1)
        cf_mean = cf.mean(axis=0)
        relative_effect = (y_post.mean() - cf_mean.mean()) / (abs(cf_mean.mean()) + 1e-9)
        p_positive = float((cumulative > 0).mean())
        ci_lo = float(np.percentile(cumulative, 2.5))
        ci_hi = float(np.percentile(cumulative, 97.5))

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
        ax1.set_title("Pointwise Effect (Observed - Counterfactual)")

        ax2 = axes[2]
        ax2.plot(t_post, cum_mean, color="purple")
        ax2.fill_between(t_post, cum_lo, cum_hi, color="purple", alpha=0.2)
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_title("Cumulative Effect")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
