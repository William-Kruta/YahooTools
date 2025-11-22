from enum import Enum


class FilterTypes(Enum):
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUAL_TO = "equal_to"


class StatementLabels(Enum):
    REVENUE = "revenue"
    LONG_TERM_DEBT = "Long Term Debt"
    NET_INCOME = "net_income"
    TOTAL_ASSETS = "Total Assets"
    TOTAL_DEBT = "Total Debt"
    TOTAL_LIABILITIES = "Total Liabilities"


class TechnicalIndicators(Enum):
    EMA = "ema"
    MACD = "macd"
    RSI = "rsi"
    SMA = "sma"
