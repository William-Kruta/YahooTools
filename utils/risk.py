import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from typing import Any, Dict, Optional


def black_scholes_call_price(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """
    Black-Scholes price of a European call option with continuous dividend yield q.
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def simulate_row(
    row: dict, historical_prices: pl.Series, risk_free_rate: float, n_sims: int = 10_000
):
    return simulate_option_risk(
        hist_prices=historical_prices,
        contractSymbol=row["contractSymbol"],
        S0=row["stock_price"],
        K=row["strike"],
        days_to_expiry=int(row["dte"]),
        r=risk_free_rate,
        option_type=row["type"],
        n_sims=n_sims,  # smaller for testing
    )["summary"]


def simulate_option_risk(
    hist_prices: pl.Series,  # can also be pd.Series
    contractSymbol: str,
    S0: float,
    K: float,
    days_to_expiry: int,
    r: float,
    n_sims: int = 100_000,
    method: str = "both",  # "gbm", "bootstrap", or "both"
    premium: Optional[float] = None,
    trading_days_per_year: int = 252,
    seed: Optional[int] = None,
    option_type: str = "call",
    q: float = 0.0,
) -> Dict[str, Any]:

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if seed is not None:
        np.random.seed(seed)

    # ensure NumPy array for returns computation
    if isinstance(hist_prices, pl.Series):
        hist_prices = hist_prices.to_numpy()
    elif isinstance(hist_prices, pd.Series):
        hist_prices = hist_prices.to_numpy()
    hist_prices = np.asarray(hist_prices, dtype=float)

    if len(hist_prices) < 2:
        raise ValueError("hist_prices must contain at least 2 values")

    # log returns
    log_rets = np.diff(np.log(hist_prices))
    mu_daily = np.mean(log_rets)
    sigma_daily = np.std(log_rets, ddof=1)
    mu_annual = mu_daily * trading_days_per_year
    sigma_annual = sigma_daily * np.sqrt(trading_days_per_year)
    T = days_to_expiry / trading_days_per_year
    disc = np.exp(-r * T)

    results: Dict[str, Any] = {}
    methods = [method] if method in ("gbm", "bootstrap") else ["gbm", "bootstrap"]

    for m in methods:
        if m == "gbm":
            Z = np.random.normal(size=n_sims)
            ln_ST = (
                np.log(S0)
                + (mu_annual - 0.5 * sigma_annual**2) * T
                + sigma_annual * np.sqrt(T) * Z
            )
            ST = np.exp(ln_ST)
        else:  # bootstrap
            inds = np.random.randint(0, len(log_rets), size=(n_sims, days_to_expiry))
            sampled_sum = log_rets[inds].sum(axis=1)
            ST = S0 * np.exp(sampled_sum)

        # convert to Polars Series
        ST_pl = pl.Series("ST", ST)

        if option_type == "call":
            payoffs_pl = pl.Series("payoff", np.maximum(ST, 0) - K)
            prob_ITM = float((ST > K).mean())
        else:
            payoffs_pl = pl.Series("payoff", np.maximum(K - ST, 0))
            prob_ITM = float((ST < K).mean())

        expected_payoff = float(payoffs_pl.mean())
        disc_expected_payoff = disc * expected_payoff

        # Black-Scholes price (assume function exists)
        bs_call = float(black_scholes_call_price(S0, K, T, r, sigma_annual, q=q))
        bs_price = (
            bs_call
            if option_type == "call"
            else float(bs_call - S0 * np.exp(-q * T) + K * np.exp(-r * T))
        )

        pl_metrics = None
        if premium is not None:
            pl_arr = disc * payoffs_pl.to_numpy() - premium
            pl_median = float(np.median(pl_arr))
            alpha = 0.95
            VaR_95 = float(-np.percentile(pl_arr, 100 * (1 - alpha)))
            tail_mask = pl_arr <= np.percentile(pl_arr, 100 * (1 - alpha))
            CVaR_95 = float(-pl_arr[tail_mask].mean()) if tail_mask.any() else 0.0
            pl_metrics = {"pl_median": pl_median, "VaR_95": VaR_95, "CVaR_95": CVaR_95}

        results[m] = {
            "contractSymbol": contractSymbol,
            "S0": S0,
            "K": K,
            "days_to_expiry": days_to_expiry,
            "T_years": T,
            "mu_annual": mu_annual,
            "sigma_annual": sigma_annual,
            "ST": ST_pl,
            "payoffs": payoffs_pl,
            "prob_ITM": prob_ITM,
            "expected_payoff": expected_payoff,
            "disc_expected_payoff": disc_expected_payoff,
            "bs_price": bs_price,
            "pl_metrics": pl_metrics,
            "option_type": option_type,
        }

    preferred = "bootstrap" if "bootstrap" in results else list(results.keys())[0]
    summary = {
        "preferred_method": preferred,
        "contractSymbol": results[preferred]["contractSymbol"],
        "strike": results[preferred]["K"],
        "prob_ITM": results[preferred]["prob_ITM"],
        "expected_payoff": results[preferred]["expected_payoff"],
        "disc_expected_payoff": results[preferred]["disc_expected_payoff"],
        "bs_price": results[preferred]["bs_price"],
        "sigma_annual": results[preferred]["sigma_annual"],
        "option_type": option_type,
    }

    return {"results": results, "summary": summary}
