import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from numpy.linalg import eigh
from sklearn.preprocessing import StandardScaler
import random

def split_dataset(returns_matrix, start, split):
    returns_train = returns_matrix.loc[start:split].dropna(axis=1)
    returns_test  = returns_matrix.loc[split:].dropna(axis=1)
    common = returns_train.columns.intersection(returns_test.columns)

    returns_train = returns_train[common]
    returns_test  = returns_test[common]
    tickers = common
    N = len(tickers)

    return returns_train, returns_test, tickers, N

def rand_weights(n):
    """Generate random portfolio weights that sum to 1."""
    k = np.random.dirichlet(np.ones(n) * 0.001)
    return k / k.sum()
    
def random_portfolio(returns_train):
    """
    Compute mean and volatility for a random portfolio using TRAIN data.
    """
    random_N = random.randint(2, len(returns_train.columns))
    random_tickers = random.sample(list(returns_train.columns), random_N)
    R = returns_train[random_tickers].values          # T x N
    N = R.shape[1]

    # Mean return per asset (train-only)
    p = R.mean(axis=0)                # shape (N,)

    # Covariance matrix (train-only)
    C = np.cov(R, rowvar=False)       # shape (N x N)

    # Random weights
    w = rand_weights(N)

    # Portfolio expected return and risk
    mu = w @ p
    sigma = np.sqrt(w.T @ C @ w)

    return mu, sigma

def mu_and_sigma(returns_train):
    # Convert the returns matrix to a numpy array
    R_train = returns_train.values

    # Compute for the expected returns or mean per asset
    mu_train = R_train.mean(axis=0)        # (N,)

    # Get the covariance matrix of asset returns
    # Set rowvar to False since np.cov, by default, treats rows as variables, columns as observations
    Sigma_train = np.cov(R_train, rowvar=False)   # (N × N)
    return R_train, mu_train, Sigma_train

# Portfolio volatility
def portfolio_vol(w, Sigma):
    return np.sqrt(w.T @ Sigma @ w)

# Sharpe
def neg_sharpe(w, mu, Sigma, rf=0):
    ret = w @ mu
    vol = np.sqrt(w.T @ Sigma @ w)
    return -(ret - rf) / vol

def min_vol_max_sharpe(mu_train, Sigma_train, N, returns_train, risk_free_rate=0.0):
    initial_guess = np.ones(N)/N
    bounds = [(0,1)] * N
    cons_sum1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    res_minvol = minimize(
    lambda w: portfolio_vol(w, Sigma_train),
    initial_guess,
    bounds=bounds,
    constraints=[cons_sum1]
    )
    w_minvol = pd.Series(res_minvol.x, index=returns_train.columns)

    # Portfolio volatility
    min_vol_std = portfolio_vol(w_minvol.values, Sigma_train)

    # Portfolio expected return
    min_vol_return = w_minvol.values @ mu_train

    res_sharpe = minimize(
    neg_sharpe,
    initial_guess,
    args=(mu_train, Sigma_train, risk_free_rate),
    bounds=bounds,
    constraints=[cons_sum1]
    )

    w_sharpe = pd.Series(res_sharpe.x, index=returns_train.columns)


    # Portfolio volatility
    max_sharpe_std = portfolio_vol(w_sharpe.values, Sigma_train)

    # Portfolio expected return
    max_sharpe_return = w_sharpe.values @ mu_train

    return [min_vol_return, min_vol_std, max_sharpe_return, max_sharpe_std], [mu_train, Sigma_train], [w_minvol, w_sharpe]

def efficient_frontier(mu_train, Sigma_train, N):
    target_returns = np.linspace(mu_train.min(), mu_train.max(), 10)

    efficient_vols = []
    efficient_weights = []


    initial_guess = np.ones(N)/N
    bounds = [(0,1)] * N
    cons_sum1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    for R_target in target_returns:

        cons_frontier = [
            cons_sum1,
            {'type': 'eq', 'fun': lambda w, Rt=R_target: w @ mu_train - Rt}
        ]

        res = minimize(
            lambda w: portfolio_vol(w, Sigma_train),
            initial_guess,
            bounds=bounds,
            constraints=cons_frontier
        )

        efficient_vols.append(res.fun)   # <-- correct list
        efficient_weights.append(res.x)  # <-- correct list

    return efficient_vols, efficient_weights, target_returns

def compute_portfolio_cumret(weights, returns_test):
    """
    Compute cumulative returns of a portfolio using TEST returns.
    weights: pd.Series indexed by tickers
    returns_test: DataFrame of test-period returns
    """

    # reorder test matrix to match weight index
    R = returns_test[weights.index]

    # portfolio daily returns
    port_ret = (R * weights).sum(axis=1)

    # cumulative returns (growth of $1)
    cumret = (1 + port_ret).cumprod()

    return cumret, port_ret