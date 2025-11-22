import sqlite3
from pathlib import Path
from typing import Self
import pandas as pd
import polars as pl


class Database:
    def __init__(self, db_path: str, table: str, log: bool = True):
        self.db_path = Path(db_path)
        self.TABLE = table
        self.log = log
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = self._connect()

    def _connect(self, timeout: int = 30) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path), timeout=timeout, detect_types=sqlite3.PARSE_DECLTYPES
        )
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    def _init_schema(self, create_table_query: str, index_query: str) -> None:
        cur = self.conn.cursor()
        cur.execute(create_table_query.format(self.TABLE))
        cur.execute(index_query.format(self.TABLE))
        cur.close()

    def read_data(
        self,
        read_query: str,
        params: tuple = None,
        require_format: bool = True,
        format_data: tuple = (),
    ):
        if require_format:
            if format_data == ():
                read_query = read_query.format(self.TABLE)
            else:
                params = (self.TABLE) + format_data
                read_query = read_query.format(*params)

        with self.conn:
            cur = self.conn.cursor()
            if params is None:
                cur.execute(read_query)
            else:
                cur.execute(read_query, params)
            rows = cur.fetchall()
            columns = [col[0] for col in cur.description]
            if not rows:
                df = pl.DataFrame({c: [] for c in columns})
            else:
                df = pl.from_records(rows, schema=columns, orient="row")

            return df

    def insert_data(self, df: pl.DataFrame, insert_query: str) -> None:
        if df.is_empty():
            return

        if self.log:
            print(f"Inserting/updating {len(df)} records into the database...")
        records = df.to_numpy().tolist()
        with self.conn:
            self.conn.executemany(insert_query.format(self.TABLE), records)

    def delete_records(self, query: str, params: tuple = None):
        with self.conn:
            self.conn.execute(query.format(self.TABLE), params)

    def drop_table(self):
        query = "DROP TABLE IF EXISTS {}"
        with self.conn:
            self.conn.execute(query.format(self.TABLE))

    def close(self) -> None:
        try:
            self.conn.commit()
        except Exception:
            pass
        self.conn.close()

    # context manager support
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
