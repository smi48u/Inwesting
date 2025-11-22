"""Hierarchical Risk Parity portfolio allocator.

Terminal tool that downloads price history from Yahoo Finance, computes HRP weights,
And prints allocations.
"""
import argparse
from datetime import datetime
from typing import List, Sequence

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download prices from Yahoo Finance and compute Hierarchical Risk Parity weights."
        )
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated list of ticker symbols (e.g., 'SPY,IVV,AGG').",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, the Yahoo Finance default period is used.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today if start is provided.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help=(
            "Yahoo Finance period (e.g., '1y', '6mo', '5y'). Ignored when start/end are provided."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="single",
        choices=["single", "average", "complete", "ward"],
        help="Linkage method for hierarchical clustering.",
    )
    return parser.parse_args()


def sanitize_dates(start: str | None, end: str | None) -> tuple[str | None, str | None]:
    if start is None:
        return None, None
    try:
        datetime.strptime(start, "%Y-%m-%d")
    except ValueError as err:
        raise SystemExit(f"Invalid start date: {start}. Use YYYY-MM-DD.") from err
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    else:
        try:
            datetime.strptime(end, "%Y-%m-%d")
        except ValueError as err:
            raise SystemExit(f"Invalid end date: {end}. Use YYYY-MM-DD.") from err
    return start, end


def download_prices(tickers: Sequence[str], start: str | None, end: str | None, period: str) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        period=None if start else period,
        progress=False,
        auto_adjust=True,
    )
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    if prices.empty:
        raise SystemExit("No price data returned. Check tickers or date range.")
    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna(axis=1, how="all")
    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"Warning: missing prices for {', '.join(sorted(missing))}. They will be ignored.")
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices).diff().dropna(how="all")
    if returns.empty:
        raise SystemExit("Not enough data to compute returns.")
    return returns


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    distance = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(distance.values, 0.0)
    return distance


def quasi_diagonal_order(linkage_matrix: np.ndarray, labels: List[str]) -> List[str]:
    leaf_order = sch.leaves_list(linkage_matrix)
    return [labels[i] for i in leaf_order]


def cluster_variance(cov: pd.DataFrame, assets: Sequence[str]) -> float:
    subset = cov.loc[assets, assets]
    weights = np.full(len(assets), 1.0 / len(assets))
    return float(weights @ subset.values @ weights)


def hrp_allocation(cov: pd.DataFrame, ordered_assets: Sequence[str]) -> pd.Series:
    weights = pd.Series(1.0, index=ordered_assets)

    def allocate(cluster_assets: List[str]):
        if len(cluster_assets) == 1:
            return
        mid = len(cluster_assets) // 2
        left = cluster_assets[:mid]
        right = cluster_assets[mid:]
        var_left = cluster_variance(cov, left)
        var_right = cluster_variance(cov, right)
        allocation_left = 1 - var_left / (var_left + var_right)
        allocation_right = 1 - allocation_left
        weights[left] *= allocation_left
        weights[right] *= allocation_right
        allocate(left)
        allocate(right)

    allocate(list(ordered_assets))
    return weights / weights.sum()


def main() -> None:
    args = parse_args()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if not tickers:
        raise SystemExit("Provide at least one ticker.")

    start, end = sanitize_dates(args.start, args.end)
    prices = download_prices(tickers, start, end, args.period)
    returns = log_returns(prices)

    cov = returns.cov()
    corr = returns.corr()
    distance = correlation_distance(corr)

    condensed_distance = squareform(distance.values, checks=False)
    linkage_matrix = sch.linkage(condensed_distance, method=args.method)
    ordered_assets = quasi_diagonal_order(linkage_matrix, list(distance.index))

    weights = hrp_allocation(cov, ordered_assets)

    print("\nHRP allocation:")
    for ticker, weight in weights.sort_values(ascending=False).items():
        print(f"  {ticker}: {weight:.2%}")


if __name__ == "__main__":
    main()
