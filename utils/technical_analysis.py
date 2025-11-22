import numpy as np
import polars as pl
from typing import Optional, Sequence


def SMA(series: pl.Series, window: int) -> pl.Series:
    """
    Calculate Simple Moving Average (SMA)

    Parameters
    ----------
    series : pl.Series
        Series containing stock prices.
    window : int
        Period to use for calculation.

    Returns
    -------
    pl.Series
        Series containing SMA calculations.
    """
    return series.rolling_mean(window)


def EMA(series: pl.Series, window: int) -> pl.Series:
    """
    Calculate Exponential Moving Average (EMA)

    Parameters
    ----------
    series : pl.Series
        Series containing stock prices.
    window : int
        Window to use for calculation.

    Returns
    -------
    pl.Series
        Series containing EMA calculations.
    """
    alpha = 2 / (window + 1)
    return series.ewm_mean(alpha=alpha, adjust=False)


def RSI(series: pl.Series, period: int = 14) -> pl.Series:
    """
    Calculate the Relative Strength Index (RSI)

    Parameters
    ----------
    series : pl.Series
        Series containing stock prices, close prices recommended.
    period : int, optional
        Period to use for calculation, by default 14

    Returns
    -------
    pl.Series
        Series containing RSI calculations.
    """
    # Price changes
    delta = series.diff()

    # Gains and losses
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    # Wilder smoothing (EMA with alpha = 1/period)
    alpha = 1 / period
    avg_gain = gain.ewm_mean(alpha=alpha, adjust=False)
    avg_loss = loss.ewm_mean(alpha=alpha, adjust=False)

    RS = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + RS))

    return rsi


def MACD(series: pl.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Parameters
    ----------
    series : pl.Series
        Series of values. Recommend closing prices.
    fast : int, optional
        The first period used in the calculation of the MACD line, by default 12
    slow : int, optional
        The second period that determines how quickly and how far a signal must move before the MACD is above or below it, by default 26
    signal : int, optional
        The period used in the calculation of the MACD signal line, by default 9

    Returns
    -------
    tuple
        Tuple containing (macd_line, signal_line, histogram)
    """

    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def OBV(close: pl.Series, volume: pl.Series) -> pl.Series:
    """
    Calculate On Balance Volume (OBV)

    Parameters
    ----------
    close : pl.Series
        Historical closing prices.
    volume : pl.Series
        Historical volume data.

    Returns
    -------
    pl.Series
        Series containing OBV calculations.
    """
    # Price change direction: +1, -1, or 0
    direction = close.diff().sign().fill_null(0)

    # Volume * direction, then cumulative sum
    return (volume * direction).cumsum()


def VWAP(
    high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series
) -> pl.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)

    Parameters
    ----------
    high : pl.Series
        Historical high prices.
    low : pl.Series
        Historical low prices.
    close : pl.Series
        Historical close prices.
    volume : pl.Series
        Historical volume.

    Returns
    -------
    pl.Series
        Series containing VWAP calculations.
    """

    typical_price = (high + low + close) / 3

    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()

    return cum_pv / cum_vol


def ATR(
    df: pl.DataFrame, length: int = 14, ma: str = "wilder", new_col: str = "ATR_14"
) -> pl.DataFrame:
    # compute TR as a polars Series
    tr_series = df.select(_true_range_pl(df)).to_series().to_list()  # list of TR values

    if ma == "sma":
        # use Polars rolling mean (keeps everything in Polars)
        return df.with_columns(
            pl.Series("tr", tr_series),
            pl.Series(new_col, pl.Series(tr_series).rolling_mean(window_size=length)),
        )
    else:
        # Wilder smoothing implemented in Python (fast, linear scan)
        tr = np.asarray(tr_series, dtype=float)
        atr = np.full_like(tr, np.nan, dtype=float)
        if len(tr) >= length:
            # first ATR value = SMA of first `length` TRs (placed at index length-1)
            first = tr[:length].mean()
            atr[length - 1] = first
            # iterate (Wilder recursion)
            for i in range(length, len(tr)):
                atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
        # attach to DataFrame
        return df.with_columns(pl.Series("tr", tr), pl.Series(new_col, atr))


def _true_range_pl(df: pl.DataFrame) -> pl.Series:
    """
    Helper function for 'ATR'

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe containing OHLCV data.

    Returns
    -------
    pl.Series
        Series containing ATR calculations.
    """

    return df.max_horizontal(
        [
            (pl.col("high") - pl.col("low")),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]
    ).alias("tr")
