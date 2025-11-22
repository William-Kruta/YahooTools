import sqlite3
import pandas as pd
import polars as pl
import yfinance as yf
import datetime as dt
import logging

try:
    from .utils.utils import is_stale, date_to_timestamp
    from .database import Database
except ImportError:
    from utils.utils import is_stale, date_to_timestamp
    from database import Database


class Dividends(Database):
    def __init__(self, db_path: str, log: bool):
        TABLE = "dividends"
        super().__init__(db_path, TABLE, log)
        # TABLE CREATION
        CREATE_TABLE_QUERY = """
            CREATE TABLE IF NOT EXISTS {} (
                ticker TEXT NOT NULL,
                date TEXT,
                dividend REAL,
                frequency REAL,
                PRIMARY KEY (ticker, date)
            );
            """
        INDEX_QUERY = "CREATE INDEX IF NOT EXISTS idx_dividends ON {} (ticker, date);"
        self._init_schema(
            create_table_query=CREATE_TABLE_QUERY, index_query=INDEX_QUERY
        )

    def _determine_frequency(self, history: pd.DataFrame) -> int:
        """
        Calculate the frequency of a dividend.


        history: pd.DataFrame
            Historical dividend data.
        Returns
        -------
        int
            Frequency of the dividend. If annual -> 1, if quarterly -> 4, if monthly -> 12
        """
        last_12m = history[
            history.index >= (history.index.max() - pd.DateOffset(years=1))
        ]
        count = len(last_12m)
        if count >= 11:
            return 12
        elif 3 <= count <= 5:
            return 4
        elif count == 1:
            return 1
        else:
            return 0

    def _download_dividends(self, ticker: str):
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.dividends
        if df.empty:
            date = dt.datetime.now().date().strftime("%Y-%m-%d")
            df = pl.DataFrame(
                {"ticker": [ticker], "date": [date], "dividend": 0, "frequency": 0}
            )
            df = df.with_columns(
                [
                    pl.col("dividend").cast(pl.Float64),
                    pl.col("frequency").cast(pl.Float64),
                ]
            )
        else:
            freq = self._determine_frequency(df)
            df = df.reset_index()
            df.rename({"Date": "date", "Dividends": "dividend"}, axis=1, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            df["ticker"] = ticker
            df["frequency"] = float(freq)
            df = df[["ticker", "date", "dividend", "frequency"]]
            df = pl.from_pandas(df)
        return df

    def get_dividends(self, ticker: str, stale_threshold: int = 120) -> pl.DataFrame:
        """
        Polars version of get_dividends.
        - stale_threshold is in minutes.
        - Returns a Polars DataFrame.
        """
        ticker = ticker.upper()

        # SQL
        read_query = "SELECT * FROM {} WHERE ticker = ?"
        read_params = (ticker,)

        insert_query = """INSERT OR IGNORE INTO {}
                    (ticker, date, dividend, frequency)
                    VALUES (?, ?, ?, ?)"""

        # load local data (self.read_data may return pandas or polars)
        local_df = self.read_data(read_query, params=read_params)

        # normalize to Polars
        if isinstance(local_df, pd.DataFrame):
            local_pl = pl.from_pandas(local_df)
        elif isinstance(local_df, pl.DataFrame):
            local_pl = local_df
        else:
            local_pl = pl.DataFrame(
                {"ticker": [], "date": [], "dividend": [], "frequency": []}
            )

        # if no local data, download, insert, return
        if local_pl.is_empty():
            if self.log:
                print(f"Downloading dividends for {ticker}")
            web_df = self._download_dividends(ticker)
            web_pl = (
                pl.from_pandas(web_df) if isinstance(web_df, pd.DataFrame) else web_df
            )
            self.insert_data(web_pl, insert_query)
            return web_pl

        # get last date value (robust to types)
        try:
            last_date_raw = local_pl.select("date").to_series().to_list()[-1]
        except Exception:
            last_date_raw = None

        # parse to timezone-aware timestamp (UTC) using pandas for robustness
        if last_date_raw is None:
            is_stale_flag = True
        else:
            parsed_last = pd.to_datetime(last_date_raw, utc=True, errors="coerce")
            if pd.isna(parsed_last):
                is_stale_flag = True
            else:
                now_utc = pd.Timestamp.now(tz="UTC")
                age_minutes = (now_utc - parsed_last).total_seconds() / 86400.0
                is_stale_flag = age_minutes > stale_threshold

        # if stale -> download, insert, merge, return
        if is_stale_flag:
            if self.log:
                print(f"Updating dividends for {ticker}")
            web_df = self._download_dividends(ticker)
            web_pl = (
                pl.from_pandas(web_df) if isinstance(web_df, pd.DataFrame) else web_df
            )
            self.insert_data(web_pl, insert_query)
            # concat and dedupe
            combined = pl.concat([local_pl, web_pl], how="vertical").unique()
            return combined
        else:
            if self.log:
                print(f"Locally fetched dividends for {ticker}")
            return local_pl

    def delete_dividends(self, ticker: str):
        ticker = ticker.upper()
        query = "DELETE FROM {} WHERE ticker = ?"
        self.delete_records(query, (ticker,))

    def calc_dividend_yield(
        self,
        ticker: str,
        stock_price: float,
        return_percent: bool = False,
    ) -> float:
        """
        Calculate the dividend yield for a stock.

        Parameters
        ----------
        ticker : str
            Companies ticker.
        stock_price : float
            Current stock price.
        return_percent : bool, optional
            Determines if to return value as percent or decimal Ex: 2% or 0.02, by default False

        Returns
        -------
        float
            Yield of the dividend.
        """
        ticker.upper()
        df = self.get_dividends(ticker)
        freq = df["frequency"][-1]
        if freq != 0:
            div = df["dividend"].tail(int(freq)).sum()
            div_yield = div / stock_price
            if not return_percent:
                div_yield *= 100
        else:
            div_yield = 0
        return div_yield


if __name__ == "__main__":
    ticker = "AAPL"
    div = Dividends("data\\candles.db")
    # data = div.get_dividends(ticker)
    # div.delete_dividends("RKLB")
    data = div.calc_dividend_yield(ticker, 272.83)
    # div.drop_table()
    print(data)
