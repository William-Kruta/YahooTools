import datetime as dt
from dataclasses import dataclass
from math import erf, exp, log, sqrt, pi

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None


SQRT_2 = sqrt(2.0)
SQRT_2PI = sqrt(2.0 * pi)
SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / SQRT_2))


@dataclass
class BSContext:
    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    sqrtT: float
    disc_r: float
    disc_q: float
    d1: float
    d2: float
    Nd1: float
    Nd2: float
    n_d1: float
    zero_T: bool
    zero_sigma: bool


def years_to_expiry(
    expiry: dt.date | dt.datetime | str,
    now: dt.datetime | None = None,
    *,
    expiry_clock: dt.time = dt.time(16, 0),  # 4 PM local market time
    market_tz: str = "America/New_York",
) -> float:
    if isinstance(expiry, str):
        if "T" in expiry or " " in expiry:
            expiry_dt = dt.datetime.fromisoformat(expiry)
        else:
            expiry_dt = dt.datetime.fromisoformat(expiry + "T00:00:00")
    elif isinstance(expiry, dt.datetime):
        expiry_dt = expiry
    elif isinstance(expiry, dt.date):
        expiry_dt = dt.datetime.combine(expiry, expiry_clock)
    else:
        raise TypeError("expiry must be date | datetime | str")

    if expiry_dt.tzinfo is None:
        expiry_dt = expiry_dt.replace(
            tzinfo=ZoneInfo(market_tz) if ZoneInfo else dt.timezone.utc
        )
    if now is None:
        now = dt.datetime.now(expiry_dt.tzinfo)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=expiry_dt.tzinfo)

    dt_seconds = max(0.0, (expiry_dt - now).total_seconds())
    return dt_seconds / SECONDS_PER_YEAR


def make_context(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> BSContext:
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    zero_T = T <= 0.0
    eps = 1e-12
    zero_sigma = sigma <= 0.0
    sig = max(sigma, eps)
    if zero_T:
        # Dummy values; Greeks functions will branch on zero_T
        return BSContext(
            S,
            K,
            0.0,
            r,
            q,
            sigma,
            0.0,
            exp(-r * 0.0),
            exp(-q * 0.0),
            float("inf"),
            float("inf"),
            1.0,
            1.0,
            0.0,
            True,
            zero_sigma,
        )

    sqrtT = sqrt(T)
    disc_r = exp(-r * T)
    disc_q = exp(-q * T)
    d1 = (log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    n_d1 = _norm_pdf(d1)
    return BSContext(
        S,
        K,
        T,
        r,
        q,
        sigma,
        sqrtT,
        disc_r,
        disc_q,
        d1,
        d2,
        Nd1,
        Nd2,
        n_d1,
        False,
        zero_sigma,
    )


def calc_delta(ctx: BSContext, option: str = "call") -> float:
    if ctx.zero_T:
        if option == "call":
            return 1.0 if ctx.S > ctx.K else (0.5 if ctx.S == ctx.K else 0.0)
        else:
            return -1.0 if ctx.S < ctx.K else (-0.5 if ctx.S == ctx.K else 0.0)
    option = option.lower()
    if option == "call":
        return ctx.disc_q * ctx.Nd1
    elif option == "put":
        return -ctx.disc_q * _norm_cdf(-ctx.d1)
    else:
        raise ValueError("option must be 'call' or 'put'.")


def calc_gamma(ctx: BSContext) -> float:
    if ctx.zero_T:
        return 0.0
    sig = max(ctx.sigma, 1e-12)
    return (ctx.disc_q * ctx.n_d1) / (ctx.S * sig * ctx.sqrtT)


def calc_vega(ctx: BSContext, *, per_vol_point: bool = True) -> float:
    if ctx.zero_T or ctx.zero_sigma:
        return 0.0
    v = ctx.S * ctx.disc_q * ctx.n_d1 * ctx.sqrtT
    return v / 100.0 if per_vol_point else v


def calc_theta(ctx: BSContext, option: str = "call", *, per_day: bool = True) -> float:
    if ctx.zero_T:
        return 0.0
    sig = max(ctx.sigma, 1e-12)
    # Common first term
    t1 = -(ctx.S * ctx.disc_q * ctx.n_d1 * sig) / (2.0 * ctx.sqrtT)
    option = option.lower()
    if option == "call":
        t = (
            t1
            - ctx.r * ctx.K * ctx.disc_r * ctx.Nd2
            + ctx.q * ctx.S * ctx.disc_q * ctx.Nd1
        )
    elif option == "put":
        t = (
            t1
            + ctx.r * ctx.K * ctx.disc_r * _norm_cdf(-ctx.d2)
            - ctx.q * ctx.S * ctx.disc_q * _norm_cdf(-ctx.d1)
        )
    else:
        raise ValueError("option must be 'call' or 'put'.")
    return t / 365.25 if per_day else t


def calc_rho(ctx: BSContext, option: str = "call") -> float:
    if ctx.zero_T:
        return 0.0
    if option.lower() == "call":
        return ctx.K * ctx.T * ctx.disc_r * ctx.Nd2
    elif option.lower() == "put":
        return -ctx.K * ctx.T * ctx.disc_r * _norm_cdf(-ctx.d2)
    else:
        raise ValueError("option must be 'call' or 'put'.")
