import polars as pl
from typing import Literal

# Custom
from ..options import Options
from ..candles import Candles
from .risk import simulate_row


def screen_options(
    tickers: list,
    option_type: str,
    dte: int,
    long: bool,
    itm: bool,
    probability_filter: float,
    db_path: str,
    probability_column: Literal[
        "probability_of_profit", "expected_annual_return"
    ] = "probability_of_profit",
    sort_by: Literal[
        "annaul_yield", "prob_ITM", "probability_of_profit"
    ] = "annual_yield",
    verbose: bool = True,
    export: bool = False,
    export_path: str = "option_screener.csv",
):
    ROUND_TO = 2

    op = Options(db_path, log=False)
    risk_free_rate = op.get_risk_free_rate()

    options_df = op.get_options(
        tickers, option_type=option_type, weekly=False, force_update=False
    )
    options_df = options_df.filter((pl.col("dte") <= dte) & (pl.col("bid") > 0))
    if itm is not None:
        options_df = options_df.filter(pl.col("ITM") == itm)

    if options_df.is_empty():
        if export:
            pl.DataFrame().write_csv(export_path)
        else:
            print(pl.DataFrame())
        return

    unique_tickers = options_df.select("ticker").unique().to_series().to_list()
    candle_cache = {t: op.fetch_from_candle_cache(t)["close"] for t in unique_tickers}

    rows = list(options_df.iter_rows(named=True))
    results = [
        simulate_row(row, candle_cache[row["ticker"]], risk_free_rate, n_sims=100_000)
        for row in rows
    ]
    results_df = pl.DataFrame(results) if results else pl.DataFrame()

    df_merged = options_df.join(
        results_df.select(
            [
                "contractSymbol",
                "prob_ITM",
                "expected_payoff",
                "disc_expected_payoff",
                "bs_price",
            ]
        ),
        on="contractSymbol",
        how="left",
    )

    df_merged = df_merged.select(
        [
            "contractSymbol",
            "strike",
            "stock_price",
            "bid",
            "ask",
            "volume",
            "dte",
            "OI",
            "IV",
            "type",
            "prob_ITM",
            "expected_payoff",
            "bs_price",
        ]
    )

    # Build expression objects (compose them, don't reference alias names)
    premium_expr = (pl.col("bid") * 100).alias("premium")
    collateral_expr = (pl.col("strike") * 100).alias("collateral")

    # Convert prob_ITM to percent if it's a fraction (<=1) else keep as-is
    prob_itm_pct_expr = (
        pl.when(pl.col("prob_ITM").is_null())
        .then(pl.lit(None))
        .when(pl.col("prob_ITM") <= 1)
        .then(pl.col("prob_ITM") * 100)
        .otherwise(pl.col("prob_ITM"))
    )

    # probability_of_profit depends on long / short — compose from prob_itm_pct_expr
    if long:
        prob_profit_expr = prob_itm_pct_expr.alias("probability_of_profit")
        direction_expr = pl.lit("long").alias("direction")
    else:
        prob_profit_expr = (100 - prob_itm_pct_expr).alias("probability_of_profit")
        direction_expr = pl.lit("short").alias("direction")

    # yield / annual_yield (computed from premium and collateral expressions)
    yield_expr = ((pl.col("bid") * 100) / (pl.col("strike") * 100) * 100).alias("yield")
    annual_yield_expr = (
        ((pl.col("bid") * 100) / (pl.col("strike") * 100) * 100) * 52
    ).alias("annual_yield")

    # Apply all derived expressions in one call
    df_merged = df_merged.with_columns(
        [
            premium_expr,
            collateral_expr,
            prob_profit_expr,
            direction_expr,
            yield_expr,
            annual_yield_expr,
        ]
    )

    # Round numeric columns (only existing ones)
    cols_to_round = [
        "IV",
        "stock_price",
        "expected_payoff",
        "bs_price",
        "probability_of_profit",
        "yield",
        "prob_ITM",
        "annual_yield",
    ]
    round_exprs = [
        pl.col(c).round(ROUND_TO) for c in cols_to_round if c in df_merged.columns
    ]
    if round_exprs:
        df_merged = df_merged.with_columns(round_exprs)

    # Optionally rename prob_ITM_pct back to prob_ITM (keeps original prob_ITM if you prefer)
    print(df_merged.columns)
    if "prob_ITM_pct" in df_merged.columns:
        df_merged = df_merged.rename({"prob_ITM_pct": "prob_ITM"})

    # expected_annual_return = (annual_yield * probability_of_profit) / 100
    df_merged = df_merged.with_columns(
        (pl.col("annual_yield") * pl.col("probability_of_profit") / 100)
        .alias("expected_annual_return")
        .round(ROUND_TO)
    )

    # Ensure probability column exists, then filter
    if probability_column not in df_merged.columns:
        raise ValueError(f"{probability_column} not available in results")
    df_merged = df_merged.filter(pl.col(probability_column) > probability_filter)
    df_merged = df_merged.sort(sort_by, descending=True)

    if not verbose:
        df_merged = df_merged.select(
            [
                "contractSymbol",
                "strike",
                "stock_price",
                "dte",
                "premium",
                "bs_price",
                "annual_yield",
                "prob_ITM",
                "probability_of_profit",
            ]
        )
    print(df_merged.columns)
    if export:
        df_merged.write_csv(export_path, include_header=True)
    else:
        print(df_merged)
        return df_merged


if __name__ == "__main__":

    path = "data/candles.db"
    # tickers = ["AAPL", "MSFT", "NVDA", "AMD"]
    tickers = ["AAPL"]
    screen_options(
        tickers,
        option_type="put",
        dte=7,
        long=False,
        itm=False,
        probability_filter=0,
        db_path=path,
        verbose=False,
    )
