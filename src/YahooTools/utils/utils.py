import datetime as dt
import polars as pl


def calc_date_delta(
    start_date: str, end_date: str, date_format: str = "%Y-%m-%d"
) -> int:
    """
    Calculate the number of days in a period.

    Parameters
    ----------
    start_date : str
        Start of the period.
    end_date : str
        End of the period.
    date_format : str, optional
        Format to use when converting to datetime objects, by default "%Y-%m-%d"

    Returns
    -------
    int
        Number of days elapsed during a period.
    """
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, date_format).date()
    elif isinstance(start_date, dt.datetime):
        start_date = start_date.date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, date_format).date()
    elif isinstance(end_date, dt.datetime):
        end_date = end_date.date()
    delta = end_date - start_date
    return delta.days


def clean_tickers(tickers: list):
    if isinstance(tickers, str):
        tickers = [tickers]
    mappings = {".": "-", "/": "-"}
    index = 0
    for t in tickers:
        for k, v in mappings.items():
            if k in t:
                tickers[index] = t.replace(k, v)
        index += 1
    return tickers


def date_to_timestamp(date: str, date_format: str = "%Y-%m-%d") -> int:
    """
    Converts date to a timestamp.

    Parameters
    ----------
    date : str
        Date to convert.
    date_format : str, optional
        Format to use when converting to datetime objects, by default "%Y-%m-%d"

    Returns
    -------
    int
       Integer of the timestamp.
    """
    if isinstance(date, str):
        date = dt.datetime.strptime(date, date_format)
    ts_ms = int(dt.datetime.timestamp(date))
    return ts_ms


def is_stale(
    start_date: str, end_date: str, stale_threshold: int, date_format: str = "%Y-%m-%d"
) -> bool:
    """
    Determine if period is stale based on "stale_threshold".

    Parameters
    ----------
    start_date : str
        Start of the period.
    end_date : str
        End of the period.
    stale_threshold : int
        Determines what is considered stale. In days.
    date_format : str, optional
        Format to use when converting to datetime objects, by default "%Y-%m-%d"

    Returns
    -------
    bool
        Boolean indicating if period is stale.
    """
    delta = calc_date_delta(start_date, end_date, date_format=date_format)
    if delta >= stale_threshold:
        return True
    return False


def load_tickers_from_csv(path_to_csv: str, ticker_column: str) -> list:

    df = pl.read_csv(path_to_csv)

    tickers = df[ticker_column].to_list()
    tickers = clean_tickers(tickers)
    print(tickers)
