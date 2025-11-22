import pandas as pd
import polars as pl
import mplfinance as mpl


def plot_stock_price(
    df: pl.DataFrame, indicator_columns: list = [], num_candles: int = 252
):
    df = (
        df.with_columns(pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d"))
        .select(
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            + indicator_columns
        )
        .to_pandas()
    )
    if num_candles != -1:
        df = df.tail(num_candles)
    # df["date"] = pd.to_datetime(df["date"])

    df.set_index("date", inplace=True)
    if indicator_columns != []:
        addplots = []
        for i in indicator_columns:
            addplots.append(mpl.make_addplot(df[i]))

    mpl.plot(
        df,
        type="candle",
        style="yahoo",
        addplot=addplots,
        volume=True,
        figsize=(12, 8),
    )
