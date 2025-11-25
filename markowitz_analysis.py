# markowitz_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import utils as ut


@dataclass
class RawResults:
    start_date: pd.Timestamp
    split_date: pd.Timestamp
    returns_train: pd.DataFrame
    returns_test: pd.DataFrame
    tickers: pd.Index
    N: int
    R_train: np.ndarray
    mu_train: np.ndarray
    sigma_train: np.ndarray
    min_vol_max_sharpe: list[float]
    weights_raw: list[pd.Series]
    efficient_vols: list[float]
    efficient_weights: list[np.ndarray]
    target_returns: np.ndarray
    rand_mus: np.ndarray
    rand_sigmas: np.ndarray
    indiv_mus: pd.Series
    indiv_sigmas: pd.Series


@dataclass
class PCAResults:
    sigma_pca: np.ndarray
    min_vol_max_sharpe_pca: list[float]
    weights_pca: list[pd.Series]
    efficient_vols_pca: list[float]
    efficient_weights_pca: list[np.ndarray]
    target_returns_pca: np.ndarray


@dataclass
class RMTResults:
    Corr: np.ndarray
    eigvals_rmt: np.ndarray
    eigvecs_rmt: np.ndarray
    lambda_plus: float
    eigvals_clean: np.ndarray
    Corr_clean: np.ndarray
    sigma_rmt: np.ndarray
    min_vol_max_sharpe_rmt: list[float]
    weights_rmt: list[pd.Series]
    efficient_vols_rmt: list[float]
    efficient_weights_rmt: list[np.ndarray]
    target_returns_rmt: np.ndarray


def fit_raw_markowitz(
    returns_matrix: pd.DataFrame,
    stock_data: pd.DataFrame,
    end_date: str = "2024-12-31",
    risk_free: float = 0.04 / 252,
    num_random: int = 1000,
) -> RawResults:
    """
    End-to-end raw Markowitz fit: split train/test, estimate mu/Sigma,
    efficient frontier, and random portfolios.
    """
    start_date = stock_data["Date"].min()
    split_date = pd.to_datetime(end_date)

    returns_train, returns_test, tickers, N = ut.split_dataset(
        returns_matrix, start_date, split_date
    )

    R_train, mu_train, sigma_train = ut.mu_and_sigma(returns_train)

    min_vol_max_sharpe, _, wt = ut.min_vol_max_sharpe(
        mu_train, sigma_train, N, returns_train, risk_free_rate=risk_free
    )

    efficient_vols, efficient_weights, target_returns = ut.efficient_frontier(
        mu_train, sigma_train, N
    )

    rand_mus = []
    rand_sigmas = []
    for _ in range(num_random):
        mu, sigma = ut.random_portfolio(returns_train)
        rand_mus.append(mu)
        rand_sigmas.append(sigma)

    indiv_mus = returns_train.mean(axis=0)
    indiv_sigmas = returns_train.std(axis=0)

    return RawResults(
        start_date=start_date,
        split_date=split_date,
        returns_train=returns_train,
        returns_test=returns_test,
        tickers=tickers,
        N=N,
        R_train=R_train,
        mu_train=mu_train,
        sigma_train=sigma_train,
        min_vol_max_sharpe=min_vol_max_sharpe,
        weights_raw=wt,
        efficient_vols=efficient_vols,
        efficient_weights=efficient_weights,
        target_returns=np.asarray(target_returns),
        rand_mus=np.asarray(rand_mus),
        rand_sigmas=np.asarray(rand_sigmas),
        indiv_mus=indiv_mus,
        indiv_sigmas=indiv_sigmas,
    )


def fit_pca_markowitz(
    raw: RawResults,
    risk_free: float = 0.04 / 252,
    var_explained: float = 0.90,
) -> PCAResults:
    """
    PCA-based covariance cleaning and Markowitz frontier.
    """
    R_train = raw.R_train
    mu_train = raw.mu_train
    sigma_train = raw.sigma_train
    N = raw.N
    returns_train = raw.returns_train

    # PCA on covariance
    eigvals, eigvecs = np.linalg.eigh(sigma_train)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    cum_var = np.cumsum(eigvals) / np.sum(eigvals)
    K = int(np.searchsorted(cum_var, var_explained) + 1)

    top_vals = eigvals[:K]
    top_vecs = eigvecs[:, :K]
    sigma_pca = top_vecs @ np.diag(top_vals) @ top_vecs.T

    min_vol_max_sharpe_pca, _, wt_pca = ut.min_vol_max_sharpe(
        mu_train, sigma_pca, N, returns_train, risk_free_rate=risk_free
    )

    efficient_vols_pca, efficient_weights_pca, target_returns_pca = ut.efficient_frontier(
        mu_train, sigma_pca, N
    )

    return PCAResults(
        sigma_pca=sigma_pca,
        min_vol_max_sharpe_pca=min_vol_max_sharpe_pca,
        weights_pca=wt_pca,
        efficient_vols_pca=efficient_vols_pca,
        efficient_weights_pca=efficient_weights_pca,
        target_returns_pca=np.asarray(target_returns_pca),
    )


def fit_rmt_markowitz(
    raw: RawResults,
    risk_free: float = 0.04 / 252,
) -> RMTResults:
    """
    RMT-based correlation cleaning and Markowitz frontier.
    """
    R_train = raw.R_train
    mu_train = raw.mu_train
    N = raw.N
    returns_train = raw.returns_train

    scaler = StandardScaler()
    R_std = scaler.fit_transform(R_train)

    Corr = np.corrcoef(R_std, rowvar=False)
    # print(Corr)

    eigvals_rmt, eigvecs_rmt = np.linalg.eigh(Corr)
    idx = np.argsort(eigvals_rmt)
    eigvals_rmt = eigvals_rmt[idx]
    eigvecs_rmt = eigvecs_rmt[:, idx]

    T, N_obs = R_std.shape
    assert N_obs == N

    q = N / T
    lambda_plus = (1 + np.sqrt(q)) ** 2

    signal_mask = eigvals_rmt > lambda_plus
    eigvals_clean = np.where(eigvals_rmt > lambda_plus, eigvals_rmt, lambda_plus)

    Corr_clean = eigvecs_rmt @ np.diag(eigvals_clean) @ eigvecs_rmt.T
    Corr_clean = (Corr_clean + Corr_clean.T) / 2  # symmetrize

    vols = R_train.std(axis=0)
    sigma_rmt = vols[:, None] * Corr_clean * vols[None, :]

    min_vol_max_sharpe_rmt, _, wt_rmt = ut.min_vol_max_sharpe(
        mu_train, sigma_rmt, N, returns_train, risk_free_rate=risk_free
    )

    efficient_vols_rmt, efficient_weights_rmt, target_returns_rmt = ut.efficient_frontier(
        mu_train, sigma_rmt, N
    )

    return RMTResults(
        Corr=Corr,
        eigvals_rmt=eigvals_rmt,
        eigvecs_rmt=eigvecs_rmt,
        lambda_plus=lambda_plus,
        eigvals_clean=eigvals_clean,
        Corr_clean=Corr_clean,
        sigma_rmt=sigma_rmt,
        min_vol_max_sharpe_rmt=min_vol_max_sharpe_rmt,
        weights_rmt=wt_rmt,
        efficient_vols_rmt=efficient_vols_rmt,
        efficient_weights_rmt=efficient_weights_rmt,
        target_returns_rmt=np.asarray(target_returns_rmt),
    )


def fit_quarter_frontiers(
    returns_matrix: pd.DataFrame,
    stock_data_df: pd.DataFrame,
    risk_free: float = 0.04 / 252,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute quarterly raw efficient frontiers and index stats for 3-, 6-, and 9-month shifts.
    """
    from pandas import DateOffset
    from pandas.tseries.offsets import MonthEnd

    start_base = stock_data_df["Date"].min()
    end_base = pd.to_datetime("2024-12-31")

    quarters = {}
    offsets = [3, 6, 9]
    labels = ["Q1", "Q2", "Q3"]

    for k, label in zip(offsets, labels):
        start_q = start_base + DateOffset(months=k) + MonthEnd(0)
        end_q = end_base + DateOffset(months=k) + MonthEnd(0)

        returns_train_q, returns_test_q, tickers_q, N_q = ut.split_dataset(
            returns_matrix, start_q, end_q
        )

        R_train_q, mu_train_q, sigma_train_q = ut.mu_and_sigma(returns_train_q)

        min_vol_max_sharpe_q, _, _ = ut.min_vol_max_sharpe(
            mu_train_q, sigma_train_q, N_q, returns_train_q, risk_free_rate=risk_free
        )

        efficient_vols_q, efficient_weights_q, target_returns_q = ut.efficient_frontier(
            mu_train_q, sigma_train_q, N_q
        )

        quarters[label] = {
            "start": start_q,
            "end": end_q,
            "returns_train": returns_train_q,
            "returns_test": returns_test_q,
            "tickers": tickers_q,
            "N": N_q,
            "R_train": R_train_q,
            "mu_train": mu_train_q,
            "sigma_train": sigma_train_q,
            "min_vol_max_sharpe": min_vol_max_sharpe_q,
            "efficient_vols": efficient_vols_q,
            "efficient_weights": efficient_weights_q,
            "target_returns": np.asarray(target_returns_q),
        }

    return quarters


def compute_cumrets_all(
    raw: RawResults, pca: PCAResults, rmt: RMTResults
) -> Dict[str, Dict[str, Any]]:
    """
    Compute cumulative returns and daily returns for all 6 portfolios on the common test set.
    """
    returns_test = raw.returns_test

    wt_raw_min, wt_raw_sharpe = raw.weights_raw
    wt_pca_min, wt_pca_sharpe = pca.weights_pca
    wt_rmt_min, wt_rmt_sharpe = rmt.weights_rmt

    cum_raw_minvol, ret_raw_minvol = ut.compute_portfolio_cumret(wt_raw_min, returns_test)
    cum_pca_minvol, ret_pca_minvol = ut.compute_portfolio_cumret(wt_pca_min, returns_test)
    cum_rmt_minvol, ret_rmt_minvol = ut.compute_portfolio_cumret(wt_rmt_min, returns_test)

    cum_raw_sharpe, ret_raw_sharpe = ut.compute_portfolio_cumret(wt_raw_sharpe, returns_test)
    cum_pca_sharpe, ret_pca_sharpe = ut.compute_portfolio_cumret(wt_pca_sharpe, returns_test)
    cum_rmt_sharpe, ret_rmt_sharpe = ut.compute_portfolio_cumret(wt_rmt_sharpe, returns_test)

    return {
        "minvol": {
            "raw": {"cum": cum_raw_minvol, "ret": ret_raw_minvol},
            "pca": {"cum": cum_pca_minvol, "ret": ret_pca_minvol},
            "rmt": {"cum": cum_rmt_minvol, "ret": ret_rmt_minvol},
        },
        "sharpe": {
            "raw": {"cum": cum_raw_sharpe, "ret": ret_raw_sharpe},
            "pca": {"cum": cum_pca_sharpe, "ret": ret_pca_sharpe},
            "rmt": {"cum": cum_rmt_sharpe, "ret": ret_rmt_sharpe},
        },
    }


def top_holdings(
    raw: RawResults, pca: PCAResults, rmt: RMTResults, top_n: int = 10
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Extract top-N weights for each of the six portfolios.
    """
    wt_raw_min, wt_raw_sharpe = raw.weights_raw
    wt_pca_min, wt_pca_sharpe = pca.weights_pca
    wt_rmt_min, wt_rmt_sharpe = rmt.weights_rmt

    return {
        "raw": {
            "minvol": wt_raw_min.sort_values(ascending=False).head(top_n),
            "minall": wt_raw_min.sort_values(ascending=False),
            "sharpe": wt_raw_sharpe.sort_values(ascending=False).head(top_n),
            "sharpeall": wt_raw_sharpe.sort_values(ascending=False)
        },
        "pca": {
            "minvol": wt_pca_min.sort_values(ascending=False).head(top_n),
            "minall": wt_raw_min.sort_values(ascending=False),
            "sharpe": wt_pca_sharpe.sort_values(ascending=False).head(top_n),
            "sharpeall": wt_raw_sharpe.sort_values(ascending=False)
        },
        "rmt": {
            "minvol": wt_rmt_min.sort_values(ascending=False).head(top_n),
            "minall": wt_raw_min.sort_values(ascending=False),
            "sharpe": wt_rmt_sharpe.sort_values(ascending=False).head(top_n),
            "sharpeall": wt_raw_sharpe.sort_values(ascending=False)
        },
    }
