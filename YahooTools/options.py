import re
import sqlite3
import logging
import datetime as dt
import yfinance as yf
import pandas as pd
import polars as pl
import numpy as np


from typing import Any, Dict, Literal, Optional
from scipy.stats import norm


from enum import Enum

from .candles import Candles
from .database import Database
from .dividends import Dividends
from .utils.utils import calc_date_delta, is_stale
from .utils.greeks import (
    years_to_expiry,
    make_context,
    calc_delta,
    calc_gamma,
    calc_vega,
    calc_theta,
    calc_rho,
)
from .utils.risk import simulate_option_risk


class Options(Database):
    def __init__(self, db_path: str, log: bool = True):
        super().__init__(db_path, "options", log)
        self.candles = Candles(db_path, log)
        self.dividends = Dividends(db_path, log)
        self.candle_cache = {}
        self.dividend_yield_cache = {}
        self.RISK_FREE_TICKER = "^TNX"
        self.RISK_FREE_RATE = None
        self.columns = [
            "ticker",
            "contractSymbol",
            "lastTradeDate",
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "change",
            "percentChange",
            "volume",
            "OI",
            "IV",
            "ITM",
            "type",
            "stock_price",
            "dividend",
            "risk_free_rate",
            "time_collected",
            "expiration_date",
            "dte",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        ]
        CREATE_TABLE_QUERY = """ CREATE TABLE IF NOT EXISTS {} (
        ticker TEXT,
        contractSymbol TEXT, 
        lastTradeDate TEXT,
        strike REAL,
        lastPrice REAL,
        bid REAL,
        ask REAL,
        change REAL,
        percentChange REAL,
        volume REAL,
        OI REAL,
        IV REAL,
        ITM BOOLEAN,
        type REAL,
        stock_price REAL,
        dividend REAL,
        risk_free_rate REAL,
        time_collected TEXT,
        expiration_date TEXT,
        dte REAL,
        delta REAL,
        gamma REAL,
        vega REAL,
        theta REAL,
        rho REAL,
        PRIMARY KEY (contractSymbol, time_collected));
        """
        INDEX_QUERY = "CREATE INDEX IF NOT EXISTS idx_options_ti_ts ON {} (contractSymbol, time_collected);"
        self._init_schema(CREATE_TABLE_QUERY, INDEX_QUERY)

    def set_risk_free_rate(self):
        self.RISK_FREE_RATE = self.candles.get_risk_free_rate(self.RISK_FREE_TICKER)

    def get_risk_free_rate(self):
        if self.RISK_FREE_RATE is None:
            self.set_risk_free_rate()
        return self.RISK_FREE_RATE

    def _in_cache(self, ticker: str) -> bool:
        keys = list(self.candle_cache.keys())
        if ticker in keys:
            return True
        return False

    def _load_candle_cache(self, tickers: list):
        if isinstance(tickers, str):
            tickers = [tickers]
        for t in tickers:
            if not self._in_cache(t):
                self.candle_cache[t] = self.candles.get_candles(t, "1d")

    def get_candle_cache(self, ticker: str):
        if self.candle_cache == {} or not self._in_cache(ticker):
            self._load_candle_cache(ticker)
        return self.candle_cache

    def fetch_from_candle_cache(self, ticker: str):
        candle_cache = self.get_candle_cache(ticker)
        return candle_cache[ticker]

    def get_options(
        self,
        tickers: list,
        option_type: Literal["call", "put"],
        weekly: bool = False,
        custom_date: str = "",
        stale_threshold: int = 3,
        include_greeks: bool = False,
        force_update: bool = False,
    ):
        if isinstance(tickers, str):
            tickers = [tickers]

        # SQL strings (same as before)
        placeholders = ",".join("?" for _ in tickers)
        read_query = f"""
            SELECT * FROM {self.TABLE}
            WHERE ticker IN ({placeholders}) AND type = ?
            ORDER BY ticker, expiration_date;
        """
        read_params = [*tickers, option_type]

        column_string = ",".join(c for c in self.columns)
        column_placeholders = ",".join("?" for _ in self.columns)
        insert_query = f"""INSERT OR IGNORE INTO {self.TABLE}
                            ({column_string})
                            VALUES ({column_placeholders})"""

        if force_update:
            stale_threshold = 0

        # read_data is expected to return either a pandas.DataFrame or polars.DataFrame
        local_df = self.read_data(read_query, read_params, require_format=False)
        local_df = local_df.with_columns(pl.col("ITM").cast(pl.Boolean))

        # Normalize to Polars DataFrame for processing (but keep a pandas copy available for robust datetime parsing)
        if isinstance(local_df, pd.DataFrame):
            local_pl = pl.from_pandas(local_df)
        elif isinstance(local_df, pl.DataFrame):
            local_pl = local_df
        else:
            # empty / None -> create empty polars df with expected columns
            local_pl = pl.DataFrame({c: [] for c in self.columns})

        # Case: no local rows -> download all requested tickers
        if local_pl.is_empty():
            if self.log:
                print(f"Downloading options for {tickers}")
            web_df = self._download_options(
                tickers, weekly=weekly, custom_date=custom_date
            )
            # ensure web_df is Polars
            if isinstance(web_df, pd.DataFrame):
                web_pl = pl.from_pandas(web_df)
            else:
                web_pl = web_df
            # insert and return filtered result
            self.insert_data(web_pl, insert_query)
            return web_pl.filter(pl.col("type") == option_type)

        # Determine missing tickers (use Polars operations)
        local_tickers = local_pl.select(pl.col("ticker")).unique().to_series().to_list()
        new_tickers = list(set(tickers) - set(local_tickers))
        if new_tickers:
            if self.log:
                print(f"Downloading options for {new_tickers}")
            web_df = self._download_options(
                new_tickers,
                weekly=weekly,
                custom_date=custom_date,
                candle_force_update=force_update,
            )
            web_pl = (
                pl.from_pandas(web_df) if isinstance(web_df, pd.DataFrame) else web_df
            )
            self.insert_data(web_pl, insert_query)
            # merge into local (keep as Polars)
            combined = pl.concat([local_pl, web_pl], how="vertical").unique()
            return combined.filter(pl.col("type") == option_type)

        # --- Staleness check using most recent record per ticker ---
        # Polars' datetime parsing for mixed formats/timezones can be fragile.
        # Parse the 'time_collected' column with pandas (robust), then use that to
        # compute staleness and select stale tickers. We only use pandas for parsing/time logic;
        # main data remains in Polars.
        time_series = local_pl.select("time_collected").to_series().to_list()
        # create a pandas Series and parse to UTC-aware datetimes
        parsed = pd.to_datetime(pd.Series(time_series), utc=True, errors="coerce")

        # Build a small pandas DataFrame with ticker + parsed timestamp to find latest per ticker
        mini_pd = pd.DataFrame(
            {
                "ticker": local_pl.select("ticker").to_series().to_list(),
                "_tc_utc": parsed,
            }
        )

        # get most recent row per ticker
        mini_latest = (
            mini_pd.sort_values("_tc_utc")
            .drop_duplicates(subset=["ticker"], keep="last")
            .reset_index(drop=True)
        )

        # Determine which tickers are stale
        if stale_threshold == 0:
            stale_tickers = mini_latest["ticker"].tolist()
        else:
            now_utc = pd.Timestamp.now(tz="UTC")
            mini_latest["age_minutes"] = (
                now_utc - mini_latest["_tc_utc"]
            ).dt.total_seconds() / 60.0
            # treat parse failures as stale
            mini_latest.loc[mini_latest["_tc_utc"].isna(), "age_minutes"] = float("inf")
            stale_tickers = mini_latest.loc[
                mini_latest["age_minutes"] > stale_threshold * 86400, "ticker"
            ].tolist()

        # If any tickers are stale, download only those and insert
        if stale_tickers:
            if self.log:
                print(f"Updating options for {stale_tickers}")
            web_df = self._download_options(
                stale_tickers,
                weekly=weekly,
                custom_date=custom_date,
                include_greeks=include_greeks,
                candle_force_update=force_update,
            )
            web_pl = (
                pl.from_pandas(web_df) if isinstance(web_df, pd.DataFrame) else web_df
            )
            self.insert_data(web_pl, insert_query)
            # refresh local: concat and dedupe, then filter by option_type
            local_pl = pl.concat([local_pl, web_pl], how="vertical").unique()
            return local_pl.filter(pl.col("type") == option_type)

        # else: local data is fresh
        if self.log:
            print(f"Locally fetched options for {tickers}")
        return local_pl.filter(pl.col("type") == option_type)

    def _download_options(
        self,
        tickers: list,
        weekly: bool = False,
        custom_date: str = "",
        include_greeks: bool = False,
        candle_force_update: bool = False,
    ):
        if isinstance(tickers, str):
            tickers = [tickers]

        now = dt.datetime.now()
        op_data = []
        for t in tickers:
            print(f"TICKER: {t}")
            obj = yf.Ticker(t)
            dates = obj.options
            self.candle_cache[t] = self.candles.get_candles(
                t, "1d", force_update=candle_force_update
            )
            stock_price = self.candle_cache[t]["close"][-1]
            self.dividend_yield_cache[t] = self.dividends.calc_dividend_yield(
                t, stock_price
            )
            if weekly:
                dates = [dates[0]]
            elif not weekly and custom_date != "":
                dates = [custom_date]
            for d in dates:
                chain = obj.option_chain(d)
                calls = chain.calls
                puts = chain.puts
                calls["ticker"] = t
                puts["ticker"] = t
                calls["type"] = "call"
                puts["type"] = "put"
                calls["stock_price"] = stock_price
                puts["stock_price"] = stock_price
                calls["dividend"] = self.dividend_yield_cache[t]
                puts["dividend"] = self.dividend_yield_cache[t]
                op_data.append(calls)
                op_data.append(puts)

        option_data = pd.concat(op_data)
        option_data["risk_free_rate"] = self.get_risk_free_rate()
        option_data["time_collected"] = now
        # Format timestamps to string
        option_data["lastTradeDate"] = option_data["lastTradeDate"].astype(str)
        option_data["time_collected"] = option_data["time_collected"].astype(str)
        option_data["expiration_date"] = option_data["contractSymbol"].apply(
            parse_contract_symbol
        )
        option_data["dte"] = option_data["expiration_date"].apply(
            lambda x: calc_date_delta(now, x)
        )
        rename = {
            "impliedVolatility": "IV",
            "inTheMoney": "ITM",
            "openInterest": "OI",
        }
        drop = ["contractSize", "currency"]
        option_data = option_data.drop(drop, axis=1)
        option_data.rename(rename, axis=1, inplace=True)
        # exit()
        if include_greeks:
            option_data = option_data.apply(lambda x: self.row_greeks(x), axis=1)
        else:
            option_data["delta"] = np.nan
            option_data["gamma"] = np.nan
            option_data["vega"] = np.nan
            option_data["theta"] = np.nan
            option_data["rho"] = np.nan
        # option_data = option_data.apply(lambda x: self.row_simulate_risk(x), axis=1)
        option_data = option_data[self.columns]
        return pl.from_pandas(option_data)

    def row_greeks(self, s):
        S = s["stock_price"]
        K = s["strike"]
        T = years_to_expiry(s["expiration_date"], s["time_collected"])
        q = s["dividend"]
        r = s["risk_free_rate"]
        sigma = float(s["IV"]) / 100
        ctx = make_context(S, K, T, r, q, sigma)
        s["delta"] = calc_delta(ctx, s["type"])
        s["gamma"] = calc_gamma(ctx)
        s["vega"] = calc_vega(ctx)
        s["theta"] = calc_theta(ctx)
        s["rho"] = calc_rho(ctx)
        return s

    def row_simulate_risk(self, row: pd.Series):
        historical_price = self.candle_cache[row["ticker"]]["close"]
        data = simulate_option_risk(
            hist_prices=historical_price,
            S0=row["stock_price"],
            K=row["strike"],
            days_to_expiry=row["dte"],
            r=self.get_risk_free_rate(),
            premium=row["bid"],
        )
        print(data)
        exit()
        pass

    def _handle_options(
        self, df: pd.DataFrame, ticker: str, option_type: str
    ) -> pd.DataFrame:
        df["ticker"] = ticker
        df["type"] = option_type


def parse_contract_symbol(contract_symbol: str) -> str:
    match = re.search(r"\d{6}", contract_symbol)
    if match:
        date_str = match.group()
        formatted = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
        return formatted
    else:
        return None


if __name__ == "__main__":

    op = Options("data\\candles.db")
    op = Options("data\\candles.db")
    tickers = ["AAPL", "MSFT", "AMD"]
    data = op.get_options(tickers, weekly=True, option_type="call")
    print(data)
