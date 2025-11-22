import pandas as pd
import polars as pl
import yfinance as yf
import datetime as dt

# Custom
from .database import Database
from .candles import Candles
from .utils.utils import is_stale


class FinancialStatements(Database):
    def __init__(self, db_path: str, log: bool = True):
        super().__init__(db_path, "financial_statements", log)
        self.candles = Candles(db_path, log)
        self.candle_cache = {}

        CREATE_TABLE_QUERY = """CREATE TABLE IF NOT EXISTS {} (
            ticker TEXT, 
            date TEXT,
            statement_type TEXT,
            label TEXT,
            value REAL,
            period TEXT,
            PRIMARY KEY (ticker, date, label, period));"""
        INDEX_QUERY = "CREATE INDEX IF NOT EXISTS idx_statements_ti_ts ON {} (ticker, date, label, period);"
        self._init_schema(CREATE_TABLE_QUERY, INDEX_QUERY)

    def get_balance_sheet(
        self, ticker: str, quarterly: bool, stale_threshold: int = -1
    ):
        ticker = ticker.upper()
        df = self._get_statement(ticker, "balance_sheet", quarterly, stale_threshold)
        return df

    def get_cash_flow(self, ticker: str, quarterly: bool, stale_threshold: int = 100):
        ticker = ticker.upper()
        df = self._get_statement(ticker, "cash_flow", quarterly, stale_threshold)
        return df

    def get_income_statements(
        self, ticker: str, quarterly: bool, stale_threshold: int = 100
    ):
        ticker = ticker.upper()
        df = self._get_statement(ticker, "income_statement", quarterly, stale_threshold)
        return df

    def _get_statement(
        self, ticker: str, statement_type: str, quarterly: bool, stale_threshold: int
    ) -> pl.DataFrame:
        if quarterly:
            period = "Q"
        else:
            period = "A"

        if stale_threshold == -1:
            if quarterly:
                stale_threshold = 120
            else:
                stale_threshold = 400
        # Read db
        read_query = f"""
        SELECT * FROM {self.TABLE} 
        WHERE ticker = '{ticker}' AND statement_type = '{statement_type.lower()}' AND period = '{period}'
        ORDER BY date
        """
        print(f"READ: {read_query}")
        # Insert db
        insert_query = f"""INSERT OR IGNORE INTO {self.TABLE}
        (ticker, date, statement_type, label, value, period)
        VALUES (?, ?, ?, ?, ?, ?)"""

        # Logic to get or update statements.
        local_df = self.read_data(read_query)
        if local_df.is_empty():
            web_df = self._download_statement(ticker, statement_type, quarterly)
            self.insert_data(web_df, insert_query)
            return web_df

        try:
            last_date_raw = local_df.select("date").to_series().to_list()[-1]
        except Exception:
            # fallback: empty or missing column -> treat as stale
            last_date_raw = None

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
                print(f"Updating {statement_type} for {ticker}")
            web_df = self._download_statement(ticker, statement_type, quarterly)
            web_pl = (
                pl.from_pandas(web_df) if isinstance(web_df, pd.DataFrame) else web_df
            )
            # insert and merge
            self.insert_data(web_pl, insert_query)
            combined = pl.concat([local_df, web_pl], how="vertical").unique()
            return combined
        else:
            if self.log:
                print(f"Locally fetched {statement_type} for {ticker}")
            return local_df

    def _download_statement(
        self, ticker: str, statement_type: str, quarterly: bool
    ) -> pl.DataFrame:
        obj = yf.Ticker(ticker)
        if statement_type == "balance_sheet":
            if quarterly:
                statement = obj.quarterly_balance_sheet
                period = "Q"
            else:
                statement = obj.balance_sheet
                period = "A"
        elif statement_type == "cash_flow":
            if quarterly:
                statement = obj.cash_flow
                period = "Q"
            else:
                statement = obj.cash_flow
                period = "A"
        elif statement_type == "income_statement":
            if quarterly:
                statement = obj.quarterly_income_stmt
                period = "Q"
            else:
                statement = obj.income_stmt
                period = "A"
        df_flat = statement.reset_index().melt(
            id_vars="index", var_name="date", value_name="value"
        )
        df_flat.rename(columns={"index": "label"}, inplace=True)
        df_flat["ticker"] = ticker
        df_flat["date"] = df_flat["date"].astype(str)
        df_flat["period"] = period
        df_flat["statement_type"] = statement_type
        df_flat = df_flat[
            ["ticker", "date", "statement_type", "label", "value", "period"]
        ]
        return pl.from_pandas(df_flat)


if __name__ == "__main__":

    st = FinancialStatements("data\candles.db")
    st.drop_table()
    # df = st.get_balance_sheet("AAPL", False)
