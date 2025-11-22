from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import pandas as pd
import polars as pl
import yfinance as yf
import time

import datetime as dt
import logging


from .utils.utils import is_stale, date_to_timestamp
from .database import Database


class Candles(Database):
    def __init__(self, db_path: Path | str = "data/candles.db", log: bool = True):
        TABLE = "candles"
        super().__init__(db_path, TABLE)
        self.log = log
        self._minute_intervals = ["1m", "5m", "10m", "30m"]
        self._date_formats = {"intraday": "%Y-%m-%d %H:%M:%S"}
        CREATE_TABLE_QUERY = """
            CREATE TABLE IF NOT EXISTS {} (
                ticker TEXT NOT NULL,
                interval TEXT NOT NULL,
                date TEXT,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (ticker, interval, timestamp)
            );
            """
        INDEX_QUERY = "CREATE INDEX IF NOT EXISTS idx_candles_ti_ts ON {} (ticker, interval, timestamp);"
        self._init_schema(CREATE_TABLE_QUERY, INDEX_QUERY)

    def get_candles(
        self,
        ticker: str,
        interval: str,
        period: str = "max",
        stale_threshold: int = 3,
        force_update: bool = False,
    ) -> pl.DataFrame:
        """
        Polars version of get_candles.
        - self.read_data(...) is expected to return either a pandas.DataFrame or a polars.DataFrame.
        - self._download_candles(...) may return pandas or polars; we normalize to Polars.
        - self.insert_data(...) must accept a Polars DataFrame (or handle conversion internally).
        - stale_threshold is interpreted in days for daily candles or in days for minute candles as well.
        """

        # choose period for intraday intervals
        if interval not in self._minute_intervals:
            period = "max"
        else:
            period = "7d"

        if force_update:
            stale_threshold = 0  # force update

        # SQL strings (same placeholders as original)
        read_query = """
            SELECT *
            FROM {}
            WHERE ticker = ? AND interval = ?
            ORDER BY ticker, timestamp;
        """
        read_params = (ticker, interval)

        insert_query = """INSERT OR IGNORE INTO {}
                    (ticker, interval, date, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?,  ?)"""

        # load local data (pandas or polars)
        local_df = self.read_data(read_query, read_params)
        # normalize to Polars for processing
        if isinstance(local_df, pd.DataFrame):
            local_pl = pl.from_pandas(local_df)
        elif isinstance(local_df, pl.DataFrame):
            local_pl = local_df
        else:
            # handle None or unexpected by creating empty polars frame with expected columns
            local_pl = pl.DataFrame(
                {
                    c: []
                    for c in [
                        "ticker",
                        "interval",
                        "date",
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                }
            )

        # If no local data: download, insert, return (as Polars DF)
        if local_pl.is_empty():
            if self.log:
                print(f"Downloading candles for {ticker}")
            web_df = self._download_candles(ticker, interval, period)
            web_pl = (
                pl.from_pandas(web_df) if isinstance(web_df, pd.DataFrame) else web_df
            )
            # insert into DB (insert_data should accept Polars or handle conversion)
            self.insert_data(web_pl, insert_query)
            return web_pl

        # Determine last_date from the last row (ORDER BY timestamp was used in SELECT)
        # Convert to a python date for staleness check (robust to string / datetime / date)
        # Grab last value from the 'date' column
        try:
            last_date_raw = local_pl.select("date").to_series().to_list()[-1]
        except Exception:
            # fallback: empty or missing column -> treat as stale
            last_date_raw = None

        # Parse last_date_raw into a date object (pandas handles many formats)
        if last_date_raw is None or (
            isinstance(last_date_raw, float) and pd.isna(last_date_raw)
        ):
            is_stale_flag = True
        else:
            # pandas parsing is robust for strings, pd.Timestamp, and date objects
            parsed_last = pd.to_datetime(last_date_raw, errors="coerce")
            if pd.isna(parsed_last):
                is_stale_flag = True
            else:
                last_date = parsed_last.date()
                now_date = dt.datetime.now().date()
                # staleness in days
                if stale_threshold == 0:
                    is_stale_flag = True
                else:
                    delta_days = (now_date - last_date).days
                    is_stale_flag = delta_days > stale_threshold

        if is_stale_flag:
            if self.log:
                print(f"Updating candles for {ticker}")
            # Download only the missing range (start=last_date, end=now_date) when possible
            # ensure we pass Python date/str; _download_candles should accept either
            start = last_date if "last_date" in locals() else None
            end = dt.datetime.now().date() + dt.timedelta(days=1)
            web_df = self._download_candles(
                ticker, interval, period, start=start, end=end
            )
            web_pl = (
                pl.from_pandas(web_df) if isinstance(web_df, pd.DataFrame) else web_df
            )
            # insert and merge
            self.insert_data(web_pl, insert_query)
            print(local_pl.dtypes)
            print(local_pl.tail())
            print(web_pl.dtypes)
            print(web_pl.tail())
            web_pl = web_pl.with_columns(pl.col("volume").cast(pl.Float64))
            combined = (
                pl.concat([local_pl, web_pl], how="vertical").unique().sort(by="date")
            )

            return combined
        else:
            if self.log:
                print(f"Locally fetched candles for {ticker}")
            return local_pl

    def batch_get_candles(
        self,
        tickers: list,
        interval: str,
        stale_threshold: int = 3,
        force_update: bool = False,
    ):
        data = []

        if isinstance(tickers, dict):
            tickers = list(tickers.keys())

        read_placeholders = ",".join(["?"] * len(tickers))
        read_query = f"""
            SELECT *
            FROM {self.TABLE}
            WHERE ticker IN ({read_placeholders}) AND interval = ?
            ORDER BY ticker, timestamp;
        """
        read_params = (*tickers, interval)

        local_df = self.read_data(read_query, read_params, require_format=False)
        if force_update:
            stale_tickers = tickers
            min_date = ""
        else:
            stale_tickers, min_date = self._batch_check_staleness(
                local_df, tickers, stale_threshold
            )

        if len(stale_tickers) > 0:
            now = dt.datetime.now().date()
            web_df = self._batch_download_candles(
                stale_tickers, interval=interval, start=min_date, end=now
            )
            web_df = web_df.with_columns(pl.col("volume").cast(pl.Float64))
            combined = (
                pl.concat([local_df, web_df], how="vertical").unique().sort(by="date")
            )
            return combined
        else:
            return local_df

        # for t in tickers:
        #     df = self.get_candles(
        #         t, interval, stale_threshold=stale_threshold, force_update=force_update
        #     )
        #     data.append(df)

        # data = pl.concat(data, how="vertical")
        # return data

    def _batch_download_candles(
        self,
        tickers: list,
        interval: str,
        period: str = "max",
        start: str = "",
        end: str = "",
    ):
        if isinstance(tickers, dict):
            tickers = list(tickers.keys())

        args = {
            "tickers": tickers,
            "period": period,
            "interval": interval,
            "group_by": "ticker",
            "auto_adjust": True,
            "progress": False,
        }
        if start != "":
            args["start"] = start
            args["end"] = end

        data_wide = yf.download(
            **args
            # tickers=tickers,
            # period=period,
            # interval=interval,
            # group_by="ticker",
            # auto_adjust=True,
            # progress=False,
        )
        if len(tickers) == 1:
            ticker = tickers[0]
            # Rename columns to match the multi-index structure for consistency
            data_wide.columns = pd.MultiIndex.from_product(
                [[ticker], data_wide.columns]
            )
        df_long = (
            data_wide.stack(level=0, future_stack=True)
            .rename_axis(["Date", "ticker"])
            .reset_index()
        )
        df_long = df_long.rename(
            columns={
                "Date": "date",
                "Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Volume": "volume",
            }
        )
        df_long["timestamp"] = df_long["date"].astype("int64") // 10**9
        df_long["interval"] = interval
        df_long = df_long[
            [
                "ticker",
                "interval",
                "date",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

        df = pl.from_pandas(df_long).with_columns(pl.col("date").cast(pl.String))
        return df

    def _download_candles(
        self, ticker: str, interval: str, period: str, start: str = "", end: str = ""
    ):
        ticker = ticker.upper()

        kwargs = {
            "tickers": ticker,
            "interval": interval,
            "period": period,
            "multi_level_index": False,
        }
        if start != "" and end != "":
            kwargs["start"] = start
            kwargs["end"] = end
        df = yf.download(**kwargs).reset_index()
        if interval not in self._minute_intervals:
            df["timestamp"] = (df["Date"].astype("int64") // 1_000_000_000).astype(
                "int64"
            )
            rename = {
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        else:
            df["timestamp"] = (df["Datetime"].astype("int64") // 1_000_000_000).astype(
                "int64"
            )
            rename = {
                "Datetime": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        df["interval"] = interval
        df["ticker"] = ticker
        df = df.rename(columns=rename)
        df = df[
            [
                "ticker",
                "interval",
                "date",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        df["date"] = df["date"].astype(str)
        df = pl.from_pandas(df)
        return df

    def _batch_check_staleness(
        self, df: pl.DataFrame, tickers: list, stale_threshold: int
    ):
        if isinstance(tickers, dict):
            tickers = list(tickers.keys())
        df = df.with_columns(pl.col("date").str.to_datetime())
        last_dates_pl = df.group_by("ticker").agg(
            pl.col("date").max().alias("max_date")
        )

        now = dt.datetime.now()
        # epoch microseconds for 'now'
        now_us = int(now.timestamp() * 1_000_000)
        MICROS_PER_DAY = 86_400_000_000  # 86400 * 1e6

        # compute age in days and is_stale flag (vectorized)
        last_dates_pl = last_dates_pl.with_columns(
            [
                (
                    (pl.lit(now_us) - pl.col("max_date").cast(pl.Int64))
                    / MICROS_PER_DAY
                ).alias("age_days"),
                (
                    (
                        (pl.lit(now_us) - pl.col("max_date").cast(pl.Int64))
                        / MICROS_PER_DAY
                    )
                    > stale_threshold
                ).alias("is_stale"),
            ]
        )
        min_date = last_dates_pl.select(pl.col("max_date").min()).item()
        stale_tickers = last_dates_pl.filter(pl.col("is_stale") == True)
        stale_tickers = stale_tickers["ticker"].to_list()
        return stale_tickers, min_date
        # last_dates_pl = (
        #     df.with_columns(
        #         pl.col("date").cast(pl.Datetime, strict=False).alias("date_dt")
        #     )
        #     .group_by("ticker")
        #     .agg(pl.col("date_dt").max().dt.date().alias("last_date_date"))
        #     .sort("ticker")
        # )
        print(f"Last: {last_dates_pl}")

    def _download_since(
        self, ticker: str, interval: str, start_ts: pd.Timestamp
    ) -> pd.DataFrame:
        # request data strictly after start_ts by starting 1 second later
        start = (start_ts + pd.Timedelta(seconds=1)).isoformat(sep=" ")
        # yfinance end param can be now UTC
        end = pd.Timestamp.utcnow().tz_localize("UTC").isoformat(sep=" ")
        return yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            threads=False,
            auto_adjust=False,
        )

    def get_risk_free_rate(self, ticker: str = "^TNX", return_decimal: bool = True):
        candles = self.get_candles(ticker, "1d")
        recent_price = candles["close"][-1]
        if return_decimal:
            return recent_price / 100
        return recent_price

    def delete_candles(self, date: str, ticker: str):
        query = """DELETE FROM {} WHERE date = ? AND ticker = ?"""
        self.delete_records(query, (date, ticker.upper()))


# Example usage
if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "MSFT", "NFLX", "TSLA", "NVDA", "AMD"]
    db = Candles("data/candles.db")
    db.batch_get_candles(tickers, "1d")
    # db.download_candles("AAPL", "1d", "max")
    # with SQLiteCandleStore("data/candles.db") as store:
    #     result = store.update_many(cfg)
    #     print(result)
    #     # fetch last 30 days for AAPL
    #     now_ms = int(time.time() * 1000)
    #     start_ms = now_ms - 30 * 24 * 3600 * 1000
    #     df = store.fetch_range("AAPL", "1d", start_ms, now_ms)
    #     print(df.tail(5))
