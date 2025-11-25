# data_prep.py
import pandas as pd


def load_sp500_data(csv_path: str):
    """
    Load S&P500 price data, filter out the index, and build the daily returns matrix.

    Returns
    -------
    stock_data_df : pd.DataFrame
        Full raw dataframe including ^GSPC.
    stock_data : pd.DataFrame
        DataFrame without the ^GSPC index.
    stock_prices : pd.DataFrame
        Tidy frame with Date, Ticker, Close, Returns.
    returns_matrix : pd.DataFrame
        Pivot: index=Date, columns=Ticker, values=Returns.
    """
    stock_data_df = pd.read_csv(csv_path)
    stock_data_df["Date"] = pd.to_datetime(stock_data_df["Date"])
    stock_data = stock_data_df[stock_data_df["Ticker"] != "^GSPC"]

    stock_prices = (
        stock_data[["Date", "Ticker", "Close"]]
        .sort_values(["Ticker", "Date"])
    )
    stock_prices["Returns"] = stock_prices.groupby("Ticker")["Close"].pct_change()
    stock_prices = stock_prices.dropna(subset=["Returns"])

    returns_matrix = pd.pivot_table(
        stock_prices,
        index="Date",
        columns="Ticker",
        values="Returns",
    )

    return stock_data_df, stock_data, stock_prices, returns_matrix


def report_missing_returns(returns_matrix: pd.DataFrame) -> pd.Series:
    """
    Compute and print fraction of missing returns per ticker,
    returning the non-zero entries sorted descending.
    """
    missing_fraction = returns_matrix.isna().mean()
    missing_fraction_sorted = missing_fraction.sort_values(ascending=False)
    non_zero_missing = missing_fraction_sorted[missing_fraction_sorted > 0]

    print("Tickers with missing returns (non-zero):")
    print(non_zero_missing)

    return non_zero_missing


def build_index_series(stock_data_df: pd.DataFrame, split_date):
    """
    Build ^GSPC daily returns and cumulative returns from the split_date onward.
    """
    snp = stock_data_df[stock_data_df["Ticker"] == "^GSPC"].copy()
    snp["Date"] = pd.to_datetime(snp["Date"])
    snp_train = snp.set_index("Date").loc[pd.to_datetime(split_date):]
    snp_returns = snp_train["Close"].pct_change().dropna()
    snp_cum_ret = (1 + snp_returns).cumprod()
    return snp_returns, snp_cum_ret


def index_stats_in_window(stock_data_df: pd.DataFrame, start, end):
    """
    Compute mean and std of ^GSPC daily returns within [start, end].
    Returns (mu, sigma).
    """
    snp = stock_data_df[stock_data_df["Ticker"] == "^GSPC"].copy()
    snp["Date"] = pd.to_datetime(snp["Date"])
    snp_window = snp.set_index("Date").loc[start:end]
    snp_returns = snp_window["Close"].pct_change().dropna()
    return float(snp_returns.mean()), float(snp_returns.std())
