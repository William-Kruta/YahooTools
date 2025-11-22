import yfinance as yf

from .database import Database


class Futures(Database):
    def __init__(self, db_path, log: bool = True):
        super().__init__(db_path, "futures", log)

    def _download_futures(self, tickers: list):
        if isinstance(tickers, dict):
            tickers = list(tickers.keys())
        elif isinstance(tickers, str):
            tickers = [tickers]

        data = yf.download(tickers)
        print(data)


if __name__ == "__main__":

    futures_tickers = {
        "ES=F": "S&P 500 (E-Mini)",
        "NQ=F": "Nasdaq 100 (E-Mini)",
        "YM=F": "Dow Jones (E-Mini)",
        "CL=F": "Crude Oil (WTI)",
        "GC=F": "Gold",
        "SI=F": "Silver",
        "BTC=F": "Bitcoin Futures",
        "ZN=F": "10-Year T-Note",
    }

    futures = Futures("data\\candles.db")
    futures._download_futures(futures_tickers)
