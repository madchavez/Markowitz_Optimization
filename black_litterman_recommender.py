from __future__ import annotations

from ast import If
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import utils as ut
from markowitz_analysis import PCAResults, RMTResults, RawResults


@dataclass
class MFImplicitResults:
    aligned_matrix: pd.DataFrame
    user_factors: np.ndarray
    item_factors: np.ndarray
    scores: pd.DataFrame


@dataclass
class BLResults:
    model_name: str
    investor_id: Any
    tickers: pd.Index
    prior_mu: np.ndarray
    sigma_prior: np.ndarray
    tau: float
    P: np.ndarray
    Q: np.ndarray
    Omega: np.ndarray
    posterior_mu: np.ndarray
    posterior_sigma: np.ndarray
    recommender_scores: pd.Series
    view_table: pd.DataFrame
    min_vol_max_sharpe_bl: list[float]
    weights_bl: list[pd.Series]
    efficient_vols_bl: list[float]
    efficient_weights_bl: list[np.ndarray]
    target_returns_bl: np.ndarray

def stocktwits_to_sentiment_matrix(
    stocktwits_df: pd.DataFrame,
    user_col: str = "user_id",
    asset_col: str = "ticker",
    sentiment_col: str = "sentiment",
    bullish_values: tuple = (1, "bullish", "Bullish", "BULLISH"),
    bearish_values: tuple = (-1, "bearish", "Bearish", "BEARISH"),
) -> pd.DataFrame:
    df = stocktwits_df[[user_col, asset_col, sentiment_col]].copy()

    def map_sentiment(x):
        if x in bullish_values:
            return 1.0
        if x in bearish_values:
            return 0.0
        return np.nan

    df["sent_score"] = df[sentiment_col].map(map_sentiment)

    utility = df.pivot(
        index=user_col,
        columns=asset_col,
        values="sent_score",
    )

    return utility.sort_index(axis=0).sort_index(axis=1)

def als_explicit(M, d, lam=0.1, max_iter=50, tol=1e-4, seed=42):
    """
    Explicit-feedback ALS for matrix with:
        1   = positive
        0   = negative
        NaN = missing

    Parameters
    ----------
    M : array-like, shape (n_users, n_items)
        Ratings matrix with NaN for missing.
    d : int
        Number of latent factors.
    lam : float
        L2 regularization strength.
    max_iter : int
        Maximum ALS iterations.
    tol : float
        Stop when RMSE improvement is below this threshold.
    seed : int
        Random seed.

    Returns
    -------
    U : ndarray, shape (n_users, d)
    V : ndarray, shape (n_items, d)
    rmse_hist : list of float
    """
    M = np.asarray(M, dtype=float)
    n_users, n_items = M.shape
    rng = np.random.default_rng(seed)

    U = 0.1 * rng.standard_normal((n_users, d))
    V = 0.1 * rng.standard_normal((n_items, d))

    I = np.eye(d)
    rmse_hist = []

    for _ in range(max_iter):
        # update item factors
        for j in range(n_items):
            obs = np.isfinite(M[:, j])
            if not np.any(obs):
                continue

            Uj = U[obs, :]
            yj = M[obs, j]

            A = Uj.T @ Uj + lam * I
            b = Uj.T @ yj
            V[j, :] = np.linalg.solve(A, b)

        # update user factors
        for i in range(n_users):
            obs = np.isfinite(M[i, :])
            if not np.any(obs):
                continue

            Vi = V[obs, :]
            yi = M[i, obs]

            A = Vi.T @ Vi + lam * I
            b = Vi.T @ yi
            U[i, :] = np.linalg.solve(A, b)

        pred = U @ V.T
        rmse = np.sqrt(np.nanmean((M - pred) ** 2))
        rmse_hist.append(rmse)

        if len(rmse_hist) > 1 and abs(rmse_hist[-2] - rmse_hist[-1]) < tol:
            break

    return U, V, rmse_hist

def recommend_from_als(M, U, V, user_idx, top_n=10):
    M = np.asarray(M, dtype=float)
    scores = U[user_idx] @ V.T
    unrated = ~np.isfinite(M[user_idx])

    candidates = np.where(unrated)[0]
    ranked = candidates[np.argsort(scores[candidates])[::-1]]

    return ranked[:top_n], scores[ranked[:top_n]]

def _align_utility_to_returns(
    utility_matrix: pd.DataFrame,
    tickers: pd.Index,
) -> pd.DataFrame:
    aligned = utility_matrix.copy()
    aligned.columns = aligned.columns.astype(str)
    tickers = pd.Index(tickers.astype(str))

    aligned = aligned.reindex(columns=tickers, fill_value=0)
    aligned = aligned.fillna(0)
    aligned = (aligned > 0).astype(float)
    return aligned


def fit_implicit_mf(
    utility_matrix: pd.DataFrame,
    n_factors: int = 12,
    alpha: float = 20.0,
    reg: float = 0.10,
    n_iter: int = 15,
    seed: int = 42,
) -> MFImplicitResults:
    """
    Weighted-ALS matrix factorization for implicit/binary feedback.

    Parameters
    ----------
    utility_matrix
        User x asset binary interaction matrix.
    alpha
        Confidence amplification for positive interactions.
    reg
        L2 regularization.
    n_iter
        ALS iterations.
    """
    X_obs = utility_matrix.fillna(0).astype(float)
    pref = (X_obs.values > 0).astype(float)
    conf = 1.0 + alpha * X_obs.values

    n_users, n_items = pref.shape
    rng = np.random.default_rng(seed)
    user_factors = 0.01 * rng.standard_normal((n_users, n_factors))
    item_factors = 0.01 * rng.standard_normal((n_items, n_factors))
    eye = np.eye(n_factors)

    for _ in range(n_iter):
        item_gram = item_factors.T @ item_factors
        for u in range(n_users):
            pos_mask = pref[u] > 0
            A = item_gram + reg * eye
            b = np.zeros(n_factors)

            if np.any(pos_mask):
                Y_pos = item_factors[pos_mask]
                conf_pos = conf[u, pos_mask]
                A += (Y_pos.T * (conf_pos - 1.0)) @ Y_pos
                b += Y_pos.T @ conf_pos

            user_factors[u] = np.linalg.solve(A + 1e-8 * eye, b)

        user_gram = user_factors.T @ user_factors
        for i in range(n_items):
            pos_mask = pref[:, i] > 0
            A = user_gram + reg * eye
            b = np.zeros(n_factors)

            if np.any(pos_mask):
                X_pos = user_factors[pos_mask]
                conf_pos = conf[pos_mask, i]
                A += (X_pos.T * (conf_pos - 1.0)) @ X_pos
                b += X_pos.T @ conf_pos

            item_factors[i] = np.linalg.solve(A + 1e-8 * eye, b)

    scores = pd.DataFrame(
        user_factors @ item_factors.T,
        index=utility_matrix.index,
        columns=utility_matrix.columns,
    )

    return MFImplicitResults(
        aligned_matrix=X_obs,
        user_factors=user_factors,
        item_factors=item_factors,
        scores=scores,
    )


def investor_recommender_scores(
    mf,
    investor_id,
    assets: pd.Index | None = None,
    exclude_observed: bool = True,
) -> pd.Series:
    if investor_id not in mf.scores.index:
        raise KeyError(f"Investor '{investor_id}' not found in utility matrix index.")

    scores = mf.scores.loc[investor_id].copy()

    if assets is not None:
        scores = scores.reindex(pd.Index(assets.astype(str)))

    if exclude_observed:
        observed = np.isfinite(
            mf.aligned_matrix.loc[investor_id].reindex(scores.index)
        )
        scores.loc[observed] = np.nan

    return scores.sort_values(ascending=False)


def investor_recommender_scores_als(
    M,
    U,
    V,
    investor_id,
    user_index=None,
    item_index=None,
    assets: pd.Index | None = None,
    exclude_observed: bool = True,
) -> pd.Series:
    """
    Get recommender scores for one investor from explicit-feedback ALS output.

    Parameters
    ----------
    M : array-like or pd.DataFrame, shape (n_users, n_items)
        Original ratings matrix used in ALS, with:
            1   = positive
            0   = negative
            NaN = missing
    U : ndarray, shape (n_users, d)
        User latent factors from als_explicit.
    V : ndarray, shape (n_items, d)
        Item latent factors from als_explicit.
    investor_id : int or label
        Investor row identifier. If M is a DataFrame, this may be a row label.
        If M is an ndarray, this should usually be the integer row position unless
        user_index is supplied.
    user_index : sequence, optional
        User labels corresponding to rows of M/U.
    item_index : sequence, optional
        Item labels corresponding to columns of M/V.
    assets : pd.Index, optional
        Subset of assets/items to return, in the desired order.
    exclude_observed : bool, default True
        If True, set already observed entries in M for this investor to NaN.

    Returns
    -------
    pd.Series
        Scores sorted descending.
    """
    M_arr = np.asarray(M, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    if U.shape[0] != M_arr.shape[0]:
        raise ValueError("U row count must match number of users in M.")
    if V.shape[0] != M_arr.shape[1]:
        raise ValueError("V row count must match number of items in M.")

    # Resolve user labels
    if isinstance(M, pd.DataFrame):
        resolved_user_index = pd.Index(M.index)
        resolved_item_index = pd.Index(M.columns.astype(str))
    else:
        resolved_user_index = (
            pd.Index(user_index)
            if user_index is not None
            else pd.RangeIndex(M_arr.shape[0])
        )
        resolved_item_index = (
            pd.Index(pd.Index(item_index).astype(str))
            if item_index is not None
            else pd.Index(pd.RangeIndex(M_arr.shape[1]).astype(str))
        )

    if investor_id not in resolved_user_index:
        raise KeyError(f"Investor '{investor_id}' not found.")

    user_pos = resolved_user_index.get_loc(investor_id)

    # Predicted scores for this investor
    scores = pd.Series(
        U[user_pos] @ V.T,
        index=resolved_item_index,
        dtype=float,
    )

    # Restrict to requested assets if provided
    if assets is not None:
        scores = scores.reindex(pd.Index(assets.astype(str)))

    # Mask already observed entries
    if exclude_observed:
        observed = np.isfinite(
            pd.Series(M_arr[user_pos], index=resolved_item_index).reindex(scores.index)
        )
        scores.loc[observed] = np.nan

    return scores.sort_values(ascending=False)

def build_relative_views_from_scores(
    scores: pd.Series,
    tickers: pd.Index,
    sigma_prior: np.ndarray,
    tau: float = 0.05,
    n_views: int = 5,
    view_return_scale: float = 0.03 / 252,
    min_confidence: float = 0.35,
    max_confidence: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert latent-factor recommender scores into relative Black-Litterman views.

    Q is expressed in the same periodicity as the return inputs.
    If returns are daily, keep view_return_scale daily as well.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    n_assets = len(tickers)

    k = min(n_views, len(ranked) // 2)
    if k < 1:
        raise ValueError("Not enough scored assets to build Black-Litterman views.")

    top_assets = ranked.head(k).index.tolist()
    bottom_assets = ranked.tail(k).index.tolist()

    score_std = float(ranked.std(ddof=0))
    if not np.isfinite(score_std) or score_std == 0:
        score_std = 1.0

    P = np.zeros((k, n_assets), dtype=float)
    Q = np.zeros((k, 1), dtype=float)
    omega_diag = []
    rows = []

    tickers_list = list(pd.Index(tickers.astype(str)))

    for r, (long_asset, short_asset) in enumerate(zip(top_assets, bottom_assets)):
        i = tickers_list.index(str(long_asset))
        j = tickers_list.index(str(short_asset))

        P[r, i] = 1.0
        P[r, j] = -1.0

        gap = float(ranked.loc[long_asset] - ranked.loc[short_asset])
        gap_unit = float(np.tanh(gap / score_std))
        confidence = min_confidence + (max_confidence - min_confidence) * abs(gap_unit)
        q_value = view_return_scale * gap_unit

        view_var = float(P[r] @ sigma_prior @ P[r])
        omega_value = max(1e-10, tau * view_var * (1.0 - confidence) / confidence)

        Q[r, 0] = q_value
        omega_diag.append(omega_value)
        rows.append(
            {
                "long_asset": long_asset,
                "short_asset": short_asset,
                "score_long": ranked.loc[long_asset],
                "score_short": ranked.loc[short_asset],
                "score_gap": gap,
                "q_view": q_value,
                "confidence": confidence,
                "omega_diag": omega_value,
            }
        )

    Omega = np.diag(omega_diag)
    view_table = pd.DataFrame(rows)
    return P, Q, Omega, view_table


def black_litterman_posterior(
    prior_mu: np.ndarray,
    sigma_prior: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form Black-Litterman posterior.

    Returns
    -------
    posterior_mu : np.ndarray
        Posterior expected returns.
    posterior_sigma : np.ndarray
        Posterior covariance used for downstream optimization.
    """
    prior_mu = np.asarray(prior_mu, dtype=float).reshape(-1, 1)
    sigma_prior = np.asarray(sigma_prior, dtype=float)
    sigma_prior = (sigma_prior + sigma_prior.T) / 2.0

    n = sigma_prior.shape[0]
    k = P.shape[0]

    tau_sigma = tau * sigma_prior + ridge * np.eye(n)
    omega_reg = Omega + ridge * np.eye(k)

    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(omega_reg)

    middle = inv_tau_sigma + P.T @ inv_omega @ P
    rhs = inv_tau_sigma @ prior_mu + P.T @ inv_omega @ Q

    posterior_mu = np.linalg.solve(middle, rhs).ravel()

    M = np.linalg.solve(middle, np.eye(n))
    posterior_sigma = sigma_prior + M
    posterior_sigma = (posterior_sigma + posterior_sigma.T) / 2.0

    return posterior_mu, posterior_sigma


def fit_black_litterman_generic(
    returns_train: pd.DataFrame,
    prior_mu: np.ndarray,
    sigma_prior: np.ndarray,
    utility_matrix: pd.DataFrame,
    investor_id: Any,
    model_name: str = "black_litterman",
    risk_free: float = 0.04 / 252,
    n_factors: int = 12,
    alpha: float = 20.0,
    reg: float = 0.10,
    n_iter: int = 15,
    tau: float = 0.05,
    n_views: int = 5,
    view_return_scale: float = 0.03 / 252,
    exclude_observed: bool = True,
    seed: int = 42,
) -> BLResults:
    tickers = pd.Index(returns_train.columns.astype(str))
    utility_aligned = _align_utility_to_returns(utility_matrix, tickers)

    mf = fit_implicit_mf(
        utility_aligned,
        n_factors=n_factors,
        alpha=alpha,
        reg=reg,
        n_iter=n_iter,
        seed=seed,
    )

    scores = investor_recommender_scores(
        mf,
        investor_id=investor_id,
        assets=tickers,
        exclude_observed=exclude_observed,
    )

    P, Q, Omega, view_table = build_relative_views_from_scores(
        scores=scores,
        tickers=tickers,
        sigma_prior=sigma_prior,
        tau=tau,
        n_views=n_views,
        view_return_scale=view_return_scale,
    )

    posterior_mu, posterior_sigma = black_litterman_posterior(
        prior_mu=prior_mu,
        sigma_prior=sigma_prior,
        P=P,
        Q=Q,
        Omega=Omega,
        tau=tau,
    )

    N = len(tickers)
    min_vol_max_sharpe_bl, _, weights_bl = ut.min_vol_max_sharpe(
        posterior_mu,
        posterior_sigma,
        N,
        returns_train,
        risk_free_rate=risk_free,
    )
    efficient_vols_bl, efficient_weights_bl, target_returns_bl = ut.efficient_frontier(
        posterior_mu,
        posterior_sigma,
        N,
    )

    return BLResults(
        model_name=model_name,
        investor_id=investor_id,
        tickers=tickers,
        prior_mu=np.asarray(prior_mu, dtype=float),
        sigma_prior=np.asarray(sigma_prior, dtype=float),
        tau=tau,
        P=P,
        Q=Q,
        Omega=Omega,
        posterior_mu=posterior_mu,
        posterior_sigma=posterior_sigma,
        recommender_scores=scores,
        view_table=view_table,
        min_vol_max_sharpe_bl=min_vol_max_sharpe_bl,
        weights_bl=weights_bl,
        efficient_vols_bl=efficient_vols_bl,
        efficient_weights_bl=efficient_weights_bl,
        target_returns_bl=np.asarray(target_returns_bl),
    )


def fit_black_litterman_raw(
    raw: RawResults,
    utility_matrix: pd.DataFrame,
    investor_id: Any,
    **kwargs,
) -> BLResults:
    return fit_black_litterman_generic(
        returns_train=raw.returns_train,
        prior_mu=raw.mu_train,
        sigma_prior=raw.sigma_train,
        utility_matrix=utility_matrix,
        investor_id=investor_id,
        model_name="BL (Raw)",
        **kwargs,
    )


def fit_black_litterman_pca(
    raw: RawResults,
    pca: PCAResults,
    utility_matrix: pd.DataFrame,
    investor_id: Any,
    **kwargs,
) -> BLResults:
    return fit_black_litterman_generic(
        returns_train=raw.returns_train,
        prior_mu=raw.mu_train,
        sigma_prior=pca.sigma_pca,
        utility_matrix=utility_matrix,
        investor_id=investor_id,
        model_name="BL (PCA)",
        **kwargs,
    )


def fit_black_litterman_rmt(
    raw: RawResults,
    rmt: RMTResults,
    utility_matrix: pd.DataFrame,
    investor_id: Any,
    **kwargs,
) -> BLResults:
    return fit_black_litterman_generic(
        returns_train=raw.returns_train,
        prior_mu=raw.mu_train,
        sigma_prior=rmt.sigma_rmt,
        utility_matrix=utility_matrix,
        investor_id=investor_id,
        model_name="BL (RMT)",
        **kwargs,
    )


def compute_cumrets_bl(
    raw: RawResults,
    bl: BLResults,
) -> dict[str, dict[str, pd.Series]]:
    returns_test = raw.returns_test.reindex(columns=bl.tickers)
    wt_min, wt_sharpe = bl.weights_bl

    cum_min, ret_min = ut.compute_portfolio_cumret(wt_min, returns_test)
    cum_sharpe, ret_sharpe = ut.compute_portfolio_cumret(wt_sharpe, returns_test)

    return {
        "minvol": {"cum": cum_min, "ret": ret_min},
        "sharpe": {"cum": cum_sharpe, "ret": ret_sharpe},
    }


def top_holdings_bl(bl: BLResults, top_n: int = 10) -> dict[str, pd.Series]:
    wt_min, wt_sharpe = bl.weights_bl
    return {
        "minvol": wt_min.sort_values(ascending=False).head(top_n),
        "minvolall": wt_min.sort_values(ascending=False),
        "sharpe": wt_sharpe.sort_values(ascending=False).head(top_n),
        "sharpeall": wt_sharpe.sort_values(ascending=False),
    }

def dcg_at_k(relevances, k):
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return np.sum(rel * discounts)

def recall_at_k(relevances, k):
    rel = np.asarray(relevances, dtype=float)
    total_relevant = np.sum(rel > 0)
    if total_relevant == 0:
        return np.nan
    hits_at_k = np.sum(rel[:k] > 0)
    return float(hits_at_k / total_relevant)

def coverage_at_k(ranked_items, n_items, k=10):
    """
    User-level coverage@k as fraction of total catalog appearing in top-k.
    """
    if n_items <= 0:
        return np.nan
    top_items = list(ranked_items[:k])
    if len(top_items) == 0:
        return 0.0
    return float(len(set(top_items)) / n_items)

def diversity_at_k(ranked_items, item_sim, k=10):
    """
    User-level intra-list diversity@k as 1 minus mean pairwise similarity.
    """
    top_items = list(ranked_items[:k])
    if len(top_items) < 2:
        return 0.0

    if isinstance(item_sim, pd.DataFrame):
        sim_block = (
            item_sim.reindex(index=top_items, columns=top_items)
            .fillna(0.0)
            .astype(float)
            .to_numpy(copy=True)
        )
    else:
        sim_arr = np.asarray(item_sim, dtype=float)
        if sim_arr.ndim != 2 or sim_arr.shape[0] != sim_arr.shape[1]:
            raise ValueError("item_sim must be a square 2D matrix.")
        idx = np.asarray(top_items, dtype=int)
        sim_block = sim_arr[np.ix_(idx, idx)]

    sim_block = np.clip(sim_block, 0.0, 1.0)
    upper = sim_block[np.triu_indices(len(top_items), k=1)]
    if upper.size == 0:
        return 0.0
    return float(1.0 - upper.mean())

def ndcg_from_als(M_train, M_test, U, V, user_idx, k=10):
    """
    NDCG@k for one user.

    M_train : training matrix used to fit ALS
    M_test  : holdout matrix with hidden positives/negatives for evaluation
    """
    M_train = np.asarray(M_train, dtype=float)
    M_test = np.asarray(M_test, dtype=float)

    # predicted scores for all items
    scores = U[user_idx] @ V.T

    # recommend only items not observed in train
    candidate_mask = ~np.isfinite(M_train[user_idx])
    candidates = np.where(candidate_mask)[0]

    # ranked candidate items
    ranked = candidates[np.argsort(scores[candidates])[::-1]]

    # binary relevance from test set: only hidden positives count
    rel = np.nan_to_num(M_test[user_idx, ranked], nan=0.0)

    dcg = dcg_at_k(rel, k)

    # ideal ranking = all relevant hidden items first
    ideal_rel = np.sort(np.nan_to_num(M_test[user_idx, candidates], nan=0.0))[::-1]
    idcg = dcg_at_k(ideal_rel, k)

    if idcg == 0:
        return np.nan  # or 0.0, depending on convention

    return dcg / idcg

def mean_ndcg_from_als(M_train, M_test, U, V, k=10):
    vals = []
    n_users = np.asarray(M_train).shape[0]

    for user_idx in range(n_users):
        val = ndcg_from_als(M_train, M_test, U, V, user_idx, k=k)
        if not np.isnan(val):
            vals.append(val)

    return float(np.mean(vals)) if vals else np.nan

def recall_from_als(M_train, M_test, U, V, user_idx, k=10):
    """
    Recall@k for one user using ALS latent factors.
    """
    M_train = np.asarray(M_train, dtype=float)
    M_test = np.asarray(M_test, dtype=float)

    scores = U[user_idx] @ V.T
    candidate_mask = ~np.isfinite(M_train[user_idx])
    candidates = np.where(candidate_mask)[0]
    ranked = candidates[np.argsort(scores[candidates])[::-1]]

    rel = np.nan_to_num(M_test[user_idx, ranked], nan=0.0)
    return recall_at_k(rel, k)

def mean_recall_from_als(M_train, M_test, U, V, k=10):
    vals = []
    n_users = np.asarray(M_train).shape[0]

    for user_idx in range(n_users):
        val = recall_from_als(M_train, M_test, U, V, user_idx, k=k)
        if not np.isnan(val):
            vals.append(val)

    return float(np.mean(vals)) if vals else np.nan

def coverage_from_als(M_train, U, V, user_idx, k=10):
    M_arr = np.asarray(M_train, dtype=float)
    n_items = M_arr.shape[1]
    ranked_idx, _ = recommend_from_als(
        M=M_arr,
        U=U,
        V=V,
        user_idx=user_idx,
        top_n=k,
    )
    return coverage_at_k(ranked_idx.tolist(), n_items=n_items, k=k)

def mean_coverage_from_als(M_train, U, V, k=10):
    vals = []
    n_users = np.asarray(M_train).shape[0]
    for user_idx in range(n_users):
        v = coverage_from_als(M_train=M_train, U=U, V=V, user_idx=user_idx, k=k)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def diversity_from_als(M_train, U, V, user_idx, item_sim, k=10):
    M_arr = np.asarray(M_train, dtype=float)
    ranked_idx, _ = recommend_from_als(
        M=M_arr,
        U=U,
        V=V,
        user_idx=user_idx,
        top_n=k,
    )
    return diversity_at_k(ranked_idx.tolist(), item_sim=item_sim, k=k)

def mean_diversity_from_als(M_train, U, V, k=10):
    M_arr = np.asarray(M_train, dtype=float)
    item_sim = _cosine_similarity_matrix(np.nan_to_num(M_arr, nan=0.0).T)

    vals = []
    n_users = M_arr.shape[0]
    for user_idx in range(n_users):
        v = diversity_from_als(
            M_train=M_arr,
            U=U,
            V=V,
            user_idx=user_idx,
            item_sim=item_sim,
            k=k,
        )
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def make_holdout(util, test_frac=0.2, min_pos=2, seed=42):
    """
    Hold out some observed bullish entries (1.0) per user for evaluation.
    Train keeps the rest; test stores only held-out positives.
    """
    rng = np.random.default_rng(seed)

    train = util.copy()
    test = pd.DataFrame(0.0, index=util.index, columns=util.columns)

    for investor_id in util.index:
        pos_items = util.columns[util.loc[investor_id].eq(1.0)]
        if len(pos_items) < min_pos:
            continue

        n_test = max(1, int(np.floor(test_frac * len(pos_items))))
        held_out = rng.choice(pos_items.to_numpy(), size=n_test, replace=False)

        train.loc[investor_id, held_out] = np.nan
        test.loc[investor_id, held_out] = 1.0

    return train, test


def ndcg_from_recommend_fn(M_train, M_test, U, V, investor_id, k=10):
    user_idx = M_train.index.get_loc(investor_id)

    ranked_idx, ranked_pred = recommend_from_als(
        M=M_train,
        U=U,
        V=V,
        user_idx=user_idx,
        top_n=k,
    )

    ranked_items = M_train.columns[ranked_idx]
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)
    dcg = dcg_at_k(rel, k)

    observed_train = np.isfinite(M_train.loc[investor_id])
    candidate_items = M_train.columns[~observed_train]
    ideal_rel = np.sort(
        np.nan_to_num(M_test.loc[investor_id, candidate_items].values, nan=0.0)
    )[::-1]
    idcg = dcg_at_k(ideal_rel, k)

    return np.nan if idcg == 0 else dcg / idcg

def ndcg_from_scores(M_train, M_test, U, V, investor_id, k=10):
    """
    NDCG@k for one investor, using investor_recommender_scores_als.
    Assumes M_test contains held-out positives as 1.0 and everything else as 0/NaN.
    """
    ranked_scores = investor_recommender_scores_als(
        M=M_train,
        U=U,
        V=V,
        investor_id=investor_id,
        exclude_observed=True,
    )

    ranked_items = ranked_scores.dropna().index.tolist()
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)

    dcg = dcg_at_k(rel, k)

    # Ideal ranking among the same candidate set = all held-out positives first
    observed_train = np.isfinite(M_train.loc[investor_id])
    candidate_items = M_train.columns[~observed_train]
    ideal_rel = np.sort(
        np.nan_to_num(M_test.loc[investor_id, candidate_items].values, nan=0.0)
    )[::-1]

    idcg = dcg_at_k(ideal_rel, k)
    return np.nan if idcg == 0 else dcg / idcg

def mean_ndcg_from_scores(M_train, M_test, U, V, k=10):
    vals = []
    for investor_id in M_train.index:
        v = ndcg_from_scores(M_train, M_test, U, V, investor_id=investor_id, k=k)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def recall_from_scores(M_train, M_test, U, V, investor_id, k=10):
    ranked_scores = investor_recommender_scores_als(
        M=M_train,
        U=U,
        V=V,
        investor_id=investor_id,
        exclude_observed=True,
    )

    ranked_items = ranked_scores.dropna().index.tolist()
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)
    return recall_at_k(rel, k)

def mean_recall_from_scores(M_train, M_test, U, V, k=10):
    vals = []
    for investor_id in M_train.index:
        v = recall_from_scores(M_train, M_test, U, V, investor_id=investor_id, k=k)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def coverage_from_scores(M_train, U, V, investor_id, k=10):
    ranked_scores = investor_recommender_scores_als(
        M=M_train,
        U=U,
        V=V,
        investor_id=investor_id,
        exclude_observed=True,
    )
    ranked_items = ranked_scores.dropna().index.tolist()
    return coverage_at_k(ranked_items, n_items=len(M_train.columns), k=k)

def mean_coverage_from_scores(M_train, U, V, k=10):
    vals = []
    for investor_id in M_train.index:
        v = coverage_from_scores(
            M_train=M_train,
            U=U,
            V=V,
            investor_id=investor_id,
            k=k,
        )
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def diversity_from_scores(M_train, U, V, investor_id, item_sim, k=10):
    ranked_scores = investor_recommender_scores_als(
        M=M_train,
        U=U,
        V=V,
        investor_id=investor_id,
        exclude_observed=True,
    )
    ranked_items = ranked_scores.dropna().index.tolist()
    return diversity_at_k(ranked_items, item_sim=item_sim, k=k)

def mean_diversity_from_scores(M_train, U, V, k=10):
    item_sim = item_item_similarity(M_train)
    vals = []
    for investor_id in M_train.index:
        v = diversity_from_scores(
            M_train=M_train,
            U=U,
            V=V,
            investor_id=investor_id,
            item_sim=item_sim,
            k=k,
        )
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def popularity_scores(M_train):
    """
    Global item popularity from training data only.
    Here: mean of observed entries per asset.
    """
    return M_train.mean(axis=0, skipna=True).fillna(0.0).sort_values(ascending=False)


def recommend_from_popularity(M_train, investor_id, pop_scores, top_n=10):
    """
    Recommend most popular unseen assets for one investor.
    """
    scores = pop_scores.copy()

    observed = np.isfinite(M_train.loc[investor_id])
    scores.loc[observed] = np.nan

    scores = scores.sort_values(ascending=False)
    return scores.head(top_n)


def ndcg_from_popularity(M_train, M_test, investor_id, pop_scores, k=10):
    ranked_scores = recommend_from_popularity(
        M_train=M_train,
        investor_id=investor_id,
        pop_scores=pop_scores,
        top_n=len(M_train.columns),
    )

    ranked_items = ranked_scores.dropna().index.tolist()
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)

    dcg = dcg_at_k(rel, k)

    observed_train = np.isfinite(M_train.loc[investor_id])
    candidate_items = M_train.columns[~observed_train]
    ideal_rel = np.sort(
        np.nan_to_num(M_test.loc[investor_id, candidate_items].values, nan=0.0)
    )[::-1]

    idcg = dcg_at_k(ideal_rel, k)
    return np.nan if idcg == 0 else dcg / idcg


def mean_ndcg_from_popularity(M_train, M_test, k=10):
    pop_scores = popularity_scores(M_train)
    vals = []

    for investor_id in M_train.index:
        v = ndcg_from_popularity(
            M_train=M_train,
            M_test=M_test,
            investor_id=investor_id,
            pop_scores=pop_scores,
            k=k,
        )
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan

def recall_from_popularity(M_train, M_test, investor_id, pop_scores, k=10):
    ranked_scores = recommend_from_popularity(
        M_train=M_train,
        investor_id=investor_id,
        pop_scores=pop_scores,
        top_n=len(M_train.columns),
    )

    ranked_items = ranked_scores.dropna().index.tolist()
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)
    return recall_at_k(rel, k)

def mean_recall_from_popularity(M_train, M_test, k=10):
    pop_scores = popularity_scores(M_train)
    vals = []

    for investor_id in M_train.index:
        v = recall_from_popularity(
            M_train=M_train,
            M_test=M_test,
            investor_id=investor_id,
            pop_scores=pop_scores,
            k=k,
        )
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan

def coverage_from_popularity(M_train, investor_id, pop_scores, k=10):
    ranked_scores = recommend_from_popularity(
        M_train=M_train,
        investor_id=investor_id,
        pop_scores=pop_scores,
        top_n=len(M_train.columns),
    )
    ranked_items = ranked_scores.dropna().index.tolist()
    return coverage_at_k(ranked_items, n_items=len(M_train.columns), k=k)

def mean_coverage_from_popularity(M_train, k=10):
    pop_scores = popularity_scores(M_train)
    vals = []
    for investor_id in M_train.index:
        v = coverage_from_popularity(
            M_train=M_train,
            investor_id=investor_id,
            pop_scores=pop_scores,
            k=k,
        )
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan

def diversity_from_popularity(M_train, investor_id, pop_scores, item_sim, k=10):
    ranked_scores = recommend_from_popularity(
        M_train=M_train,
        investor_id=investor_id,
        pop_scores=pop_scores,
        top_n=len(M_train.columns),
    )
    ranked_items = ranked_scores.dropna().index.tolist()
    return diversity_at_k(ranked_items, item_sim=item_sim, k=k)

def mean_diversity_from_popularity(M_train, k=10):
    pop_scores = popularity_scores(M_train)
    item_sim = item_item_similarity(M_train)
    vals = []
    for investor_id in M_train.index:
        v = diversity_from_popularity(
            M_train=M_train,
            investor_id=investor_id,
            pop_scores=pop_scores,
            item_sim=item_sim,
            k=k,
        )
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _cosine_similarity_matrix(X):
    X = np.asarray(X, dtype=float)
    gram = X @ X.T
    norms = np.linalg.norm(X, axis=1)
    denom = np.outer(norms, norms)
    sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    np.fill_diagonal(sim, 0.0)
    return sim


def _prune_similarity_topk(sim, top_k_neighbors, axis=1):
    sim = np.asarray(sim, dtype=float)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("sim must be a square 2D matrix.")
    n = sim.shape[0]

    pruned = sim.copy()
    np.fill_diagonal(pruned, 0.0)
    pruned[pruned < 0] = 0.0

    if top_k_neighbors is None or top_k_neighbors <= 0 or top_k_neighbors >= n:
        return pruned

    keep = np.zeros_like(pruned, dtype=bool)

    if axis == 1:
        for i in range(n):
            row = pruned[i]
            idx = np.argpartition(row, -top_k_neighbors)[-top_k_neighbors:]
            keep[i, idx] = True
    elif axis == 0:
        for j in range(n):
            col = pruned[:, j]
            idx = np.argpartition(col, -top_k_neighbors)[-top_k_neighbors:]
            keep[idx, j] = True
    else:
        raise ValueError("axis must be 0 (column-wise) or 1 (row-wise).")

    return np.where(keep, pruned, 0.0)


def user_user_similarity(M_train):
    """
    User-user cosine similarity from train matrix (NaN treated as missing).
    """
    n_users = M_train.shape[0]
    if n_users > 5000:
        raise MemoryError(
            "Dense user-user similarity is too large for this dataset. "
            "Use user_user_topk_neighbors(...) or mean_ndcg_from_user_user(...)."
        )

    X = M_train.fillna(0.0).astype(float).values
    sim = _cosine_similarity_matrix(X)
    return pd.DataFrame(sim, index=M_train.index, columns=M_train.index)


def user_user_topk_neighbors(M_train, top_k_neighbors=25):
    """
    Build memory-safe top-k user neighborhoods using cosine KNN.

    Returns
    -------
    neighbor_idx : np.ndarray, shape (n_users, k)
        Neighbor user-row indices for each user. Missing slots are -1.
    neighbor_sim : np.ndarray, shape (n_users, k)
        Corresponding non-negative cosine similarities.
    """
    if top_k_neighbors is None or top_k_neighbors <= 0:
        raise ValueError("top_k_neighbors must be a positive integer.")

    X = M_train.fillna(0.0).astype(np.float32).to_numpy(copy=True)
    n_users = X.shape[0]
    if n_users == 0:
        return np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms

    k = min(int(top_k_neighbors), max(0, n_users - 1))
    if k == 0:
        return np.empty((n_users, 0), dtype=np.int32), np.empty((n_users, 0), dtype=np.float32)

    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    knn.fit(X)
    distances, indices = knn.kneighbors(X, return_distance=True)
    similarities = np.clip(1.0 - distances, 0.0, None).astype(np.float32, copy=False)

    neighbor_idx = np.full((n_users, k), -1, dtype=np.int32)
    neighbor_sim = np.zeros((n_users, k), dtype=np.float32)

    for user_idx in range(n_users):
        row_idx = indices[user_idx]
        row_sim = similarities[user_idx]
        keep = row_idx != user_idx

        row_idx = row_idx[keep]
        row_sim = row_sim[keep]

        take = min(k, row_idx.size)
        if take > 0:
            neighbor_idx[user_idx, :take] = row_idx[:take]
            neighbor_sim[user_idx, :take] = row_sim[:take]

    return neighbor_idx, neighbor_sim


def item_item_similarity(M_train):
    """
    Item-item cosine similarity from train matrix (NaN treated as missing).
    """
    X = M_train.fillna(0.0).astype(float).values.T
    sim = _cosine_similarity_matrix(X)
    return pd.DataFrame(sim, index=M_train.columns, columns=M_train.columns)


def recommend_from_user_user(
    M_train,
    investor_id,
    user_sim,
    top_n=10,
    top_k_neighbors=25,
):
    """
    Recommend assets for one investor using user-user neighborhood CF.
    """
    if investor_id not in M_train.index:
        raise KeyError(f"Investor '{investor_id}' not found.")

    user_idx = M_train.index.get_loc(investor_id)
    X = M_train.astype(float).to_numpy()
    X_filled = np.nan_to_num(X, nan=0.0)
    observed_mask = np.isfinite(X).astype(float)

    if isinstance(user_sim, tuple) and len(user_sim) == 2:
        neighbor_idx, neighbor_sim = user_sim
        neighbor_idx = np.asarray(neighbor_idx)
        neighbor_sim = np.asarray(neighbor_sim, dtype=float)

        if user_idx >= neighbor_idx.shape[0] or user_idx >= neighbor_sim.shape[0]:
            raise ValueError("Neighbor arrays do not align with M_train users.")

        row_idx = neighbor_idx[user_idx]
        row_sim = np.clip(neighbor_sim[user_idx], 0.0, None)
        keep = row_idx >= 0

        row_idx = row_idx[keep].astype(int, copy=False)
        row_sim = row_sim[keep]

        if (
            top_k_neighbors is not None
            and top_k_neighbors > 0
            and top_k_neighbors < row_idx.size
        ):
            row_idx = row_idx[:top_k_neighbors]
            row_sim = row_sim[:top_k_neighbors]

        if row_idx.size:
            weighted_sum = row_sim @ X_filled[row_idx]
            normalizer = row_sim @ observed_mask[row_idx]
        else:
            weighted_sum = np.zeros(X.shape[1], dtype=float)
            normalizer = np.zeros(X.shape[1], dtype=float)
    else:
        if isinstance(user_sim, pd.DataFrame):
            sim_row = (
                user_sim.loc[investor_id]
                .reindex(M_train.index)
                .fillna(0.0)
                .astype(float)
                .values
            )
        else:
            sim_row = np.asarray(user_sim, dtype=float)[user_idx].copy()

        sim_row[user_idx] = 0.0
        sim_row = np.clip(sim_row, 0.0, None)

        if (
            top_k_neighbors is not None
            and top_k_neighbors > 0
            and top_k_neighbors < len(sim_row)
        ):
            keep = np.argpartition(sim_row, -top_k_neighbors)[-top_k_neighbors:]
            mask = np.zeros_like(sim_row, dtype=bool)
            mask[keep] = True
            sim_row = np.where(mask, sim_row, 0.0)

        weighted_sum = sim_row @ X_filled
        normalizer = sim_row @ observed_mask

    scores = np.divide(
        weighted_sum,
        normalizer,
        out=np.zeros_like(weighted_sum, dtype=float),
        where=normalizer > 0,
    )

    scores = pd.Series(scores, index=M_train.columns, dtype=float)
    observed_user = np.isfinite(M_train.loc[investor_id])
    scores.loc[observed_user] = np.nan

    return scores.sort_values(ascending=False).head(top_n)


def recommend_from_item_item(
    M_train,
    investor_id,
    item_sim,
    top_n=10,
    top_k_neighbors=25,
):
    """
    Recommend assets for one investor using item-item neighborhood CF.
    """
    if investor_id not in M_train.index:
        raise KeyError(f"Investor '{investor_id}' not found.")

    if isinstance(item_sim, pd.DataFrame):
        sim = (
            item_sim.reindex(index=M_train.columns, columns=M_train.columns)
            .fillna(0.0)
            .astype(float)
            .to_numpy(copy=True)
        )
    else:
        sim = np.asarray(item_sim, dtype=float).copy()

    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, None)

    if top_k_neighbors is not None and top_k_neighbors > 0 and top_k_neighbors < sim.shape[0]:
        sim = _prune_similarity_topk(sim, top_k_neighbors=top_k_neighbors, axis=0)

    user_ratings = M_train.loc[investor_id].astype(float).values
    observed_user = np.isfinite(user_ratings)
    user_values = np.nan_to_num(user_ratings, nan=0.0)

    weighted_sum = user_values @ sim
    normalizer = observed_user.astype(float) @ sim

    scores = np.divide(
        weighted_sum,
        normalizer,
        out=np.zeros_like(weighted_sum, dtype=float),
        where=normalizer > 0,
    )

    scores = pd.Series(scores, index=M_train.columns, dtype=float)
    scores.loc[observed_user] = np.nan
    return scores.sort_values(ascending=False).head(top_n)


def ndcg_from_user_user(M_train, M_test, investor_id, user_sim, k=10, top_k_neighbors=25):
    ranked_scores = recommend_from_user_user(
        M_train=M_train,
        investor_id=investor_id,
        user_sim=user_sim,
        top_n=len(M_train.columns),
        top_k_neighbors=top_k_neighbors,
    )

    ranked_items = ranked_scores.dropna().index.tolist()
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)
    dcg = dcg_at_k(rel, k)

    observed_train = np.isfinite(M_train.loc[investor_id])
    candidate_items = M_train.columns[~observed_train]
    ideal_rel = np.sort(
        np.nan_to_num(M_test.loc[investor_id, candidate_items].values, nan=0.0)
    )[::-1]
    idcg = dcg_at_k(ideal_rel, k)

    return np.nan if idcg == 0 else dcg / idcg

def recall_from_user_user(M_train, M_test, investor_id, user_sim, k=10, top_k_neighbors=25):
    ranked_scores = recommend_from_user_user(
        M_train=M_train,
        investor_id=investor_id,
        user_sim=user_sim,
        top_n=len(M_train.columns),
        top_k_neighbors=top_k_neighbors,
    )

    ranked_items = ranked_scores.dropna().index.tolist()
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)
    return recall_at_k(rel, k)

def coverage_from_user_user(M_train, investor_id, user_sim, k=10, top_k_neighbors=25):
    ranked_scores = recommend_from_user_user(
        M_train=M_train,
        investor_id=investor_id,
        user_sim=user_sim,
        top_n=len(M_train.columns),
        top_k_neighbors=top_k_neighbors,
    )
    ranked_items = ranked_scores.dropna().index.tolist()
    return coverage_at_k(ranked_items, n_items=len(M_train.columns), k=k)

def diversity_from_user_user(
    M_train,
    investor_id,
    user_sim,
    item_sim,
    k=10,
    top_k_neighbors=25,
):
    ranked_scores = recommend_from_user_user(
        M_train=M_train,
        investor_id=investor_id,
        user_sim=user_sim,
        top_n=len(M_train.columns),
        top_k_neighbors=top_k_neighbors,
    )
    ranked_items = ranked_scores.dropna().index.tolist()
    return diversity_at_k(ranked_items, item_sim=item_sim, k=k)


def ndcg_from_item_item(M_train, M_test, investor_id, item_sim, k=10, top_k_neighbors=25):
    ranked_scores = recommend_from_item_item(
        M_train=M_train,
        investor_id=investor_id,
        item_sim=item_sim,
        top_n=len(M_train.columns),
        top_k_neighbors=top_k_neighbors,
    )

    ranked_items = ranked_scores.dropna().index.tolist()
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)
    dcg = dcg_at_k(rel, k)

    observed_train = np.isfinite(M_train.loc[investor_id])
    candidate_items = M_train.columns[~observed_train]
    ideal_rel = np.sort(
        np.nan_to_num(M_test.loc[investor_id, candidate_items].values, nan=0.0)
    )[::-1]
    idcg = dcg_at_k(ideal_rel, k)

    return np.nan if idcg == 0 else dcg / idcg

def recall_from_item_item(M_train, M_test, investor_id, item_sim, k=10, top_k_neighbors=25):
    ranked_scores = recommend_from_item_item(
        M_train=M_train,
        investor_id=investor_id,
        item_sim=item_sim,
        top_n=len(M_train.columns),
        top_k_neighbors=top_k_neighbors,
    )

    ranked_items = ranked_scores.dropna().index.tolist()
    rel = np.nan_to_num(M_test.loc[investor_id, ranked_items].values, nan=0.0)
    return recall_at_k(rel, k)

def coverage_from_item_item(M_train, investor_id, item_sim, k=10, top_k_neighbors=25):
    ranked_scores = recommend_from_item_item(
        M_train=M_train,
        investor_id=investor_id,
        item_sim=item_sim,
        top_n=len(M_train.columns),
        top_k_neighbors=top_k_neighbors,
    )
    ranked_items = ranked_scores.dropna().index.tolist()
    return coverage_at_k(ranked_items, n_items=len(M_train.columns), k=k)

def diversity_from_item_item(
    M_train,
    investor_id,
    item_sim,
    k=10,
    top_k_neighbors=25,
):
    ranked_scores = recommend_from_item_item(
        M_train=M_train,
        investor_id=investor_id,
        item_sim=item_sim,
        top_n=len(M_train.columns),
        top_k_neighbors=top_k_neighbors,
    )
    ranked_items = ranked_scores.dropna().index.tolist()
    return diversity_at_k(ranked_items, item_sim=item_sim, k=k)


def mean_ndcg_from_user_user(M_train, M_test, k=10, top_k_neighbors=25):
    neighbor_idx, neighbor_sim = user_user_topk_neighbors(
        M_train,
        top_k_neighbors=top_k_neighbors,
    )

    X = M_train.astype(float).to_numpy()
    X_filled = np.nan_to_num(X, nan=0.0)
    observed_bool = np.isfinite(X)
    observed_float = observed_bool.astype(float)
    test_vals = (
        M_test.reindex(index=M_train.index, columns=M_train.columns)
        .astype(float)
        .to_numpy()
    )

    vals = []
    n_items = X.shape[1]

    for user_idx in range(X.shape[0]):
        row_idx = neighbor_idx[user_idx]
        row_sim = neighbor_sim[user_idx]
        keep = row_idx >= 0

        row_idx = row_idx[keep].astype(int, copy=False)
        row_sim = row_sim[keep].astype(float, copy=False)

        if row_idx.size == 0:
            continue

        weighted_sum = row_sim @ X_filled[row_idx]
        normalizer = row_sim @ observed_float[row_idx]
        scores = np.divide(
            weighted_sum,
            normalizer,
            out=np.zeros(n_items, dtype=float),
            where=normalizer > 0,
        )

        observed_user = observed_bool[user_idx]
        scores[observed_user] = -np.inf

        ranked_idx = np.argsort(scores)[::-1]
        ranked_idx = ranked_idx[np.isfinite(scores[ranked_idx])]

        rel = np.nan_to_num(test_vals[user_idx, ranked_idx], nan=0.0)
        dcg = dcg_at_k(rel, k)

        ideal_rel = np.sort(
            np.nan_to_num(test_vals[user_idx, ~observed_user], nan=0.0)
        )[::-1]
        idcg = dcg_at_k(ideal_rel, k)

        if idcg > 0:
            vals.append(dcg / idcg)

    return float(np.mean(vals)) if vals else np.nan

def mean_recall_from_user_user(M_train, M_test, k=10, top_k_neighbors=25):
    neighbor_idx, neighbor_sim = user_user_topk_neighbors(
        M_train,
        top_k_neighbors=top_k_neighbors,
    )

    X = M_train.astype(float).to_numpy()
    X_filled = np.nan_to_num(X, nan=0.0)
    observed_bool = np.isfinite(X)
    observed_float = observed_bool.astype(float)
    test_vals = (
        M_test.reindex(index=M_train.index, columns=M_train.columns)
        .astype(float)
        .to_numpy()
    )

    vals = []

    for user_idx in range(X.shape[0]):
        row_idx = neighbor_idx[user_idx]
        row_sim = neighbor_sim[user_idx]
        keep = row_idx >= 0

        row_idx = row_idx[keep].astype(int, copy=False)
        row_sim = row_sim[keep].astype(float, copy=False)

        if row_idx.size == 0:
            continue

        weighted_sum = row_sim @ X_filled[row_idx]
        normalizer = row_sim @ observed_float[row_idx]
        scores = np.divide(
            weighted_sum,
            normalizer,
            out=np.zeros(X.shape[1], dtype=float),
            where=normalizer > 0,
        )

        observed_user = observed_bool[user_idx]
        scores[observed_user] = -np.inf

        ranked_idx = np.argsort(scores)[::-1]
        ranked_idx = ranked_idx[np.isfinite(scores[ranked_idx])]

        rel = np.nan_to_num(test_vals[user_idx, ranked_idx], nan=0.0)
        v = recall_at_k(rel, k)
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan

def mean_coverage_from_user_user(M_train, k=10, top_k_neighbors=25):
    neighbor_idx, neighbor_sim = user_user_topk_neighbors(
        M_train,
        top_k_neighbors=top_k_neighbors,
    )
    X = M_train.astype(float).to_numpy()
    X_filled = np.nan_to_num(X, nan=0.0)
    observed_bool = np.isfinite(X)
    observed_float = observed_bool.astype(float)
    n_items = X.shape[1]

    vals = []

    for user_idx in range(X.shape[0]):
        row_idx = neighbor_idx[user_idx]
        row_sim = neighbor_sim[user_idx]
        keep = row_idx >= 0

        row_idx = row_idx[keep].astype(int, copy=False)
        row_sim = row_sim[keep].astype(float, copy=False)

        if row_idx.size:
            weighted_sum = row_sim @ X_filled[row_idx]
            normalizer = row_sim @ observed_float[row_idx]
        else:
            weighted_sum = np.zeros(n_items, dtype=float)
            normalizer = np.zeros(n_items, dtype=float)

        scores = np.divide(
            weighted_sum,
            normalizer,
            out=np.zeros(n_items, dtype=float),
            where=normalizer > 0,
        )

        observed_user = observed_bool[user_idx]
        scores[observed_user] = -np.inf

        ranked_idx = np.argsort(scores)[::-1]
        ranked_idx = ranked_idx[np.isfinite(scores[ranked_idx])]

        v = coverage_at_k(ranked_idx.tolist(), n_items=n_items, k=k)
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan

def mean_diversity_from_user_user(M_train, k=10, top_k_neighbors=25):
    neighbor_idx, neighbor_sim = user_user_topk_neighbors(
        M_train,
        top_k_neighbors=top_k_neighbors,
    )
    X = M_train.astype(float).to_numpy()
    X_filled = np.nan_to_num(X, nan=0.0)
    observed_bool = np.isfinite(X)
    observed_float = observed_bool.astype(float)
    item_sim = _cosine_similarity_matrix(X_filled.T)
    n_items = X.shape[1]

    vals = []

    for user_idx in range(X.shape[0]):
        row_idx = neighbor_idx[user_idx]
        row_sim = neighbor_sim[user_idx]
        keep = row_idx >= 0

        row_idx = row_idx[keep].astype(int, copy=False)
        row_sim = row_sim[keep].astype(float, copy=False)

        if row_idx.size:
            weighted_sum = row_sim @ X_filled[row_idx]
            normalizer = row_sim @ observed_float[row_idx]
        else:
            weighted_sum = np.zeros(n_items, dtype=float)
            normalizer = np.zeros(n_items, dtype=float)

        scores = np.divide(
            weighted_sum,
            normalizer,
            out=np.zeros(n_items, dtype=float),
            where=normalizer > 0,
        )

        observed_user = observed_bool[user_idx]
        scores[observed_user] = -np.inf

        ranked_idx = np.argsort(scores)[::-1]
        ranked_idx = ranked_idx[np.isfinite(scores[ranked_idx])]

        v = diversity_at_k(ranked_idx.tolist(), item_sim=item_sim, k=k)
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan


def mean_ndcg_from_item_item(M_train, M_test, k=10, top_k_neighbors=25):
    sim = item_item_similarity(M_train)
    if top_k_neighbors is not None and top_k_neighbors > 0 and top_k_neighbors < sim.shape[0]:
        sim = pd.DataFrame(
            _prune_similarity_topk(sim.values, top_k_neighbors=top_k_neighbors, axis=0),
            index=sim.index,
            columns=sim.columns,
        )
        top_k_neighbors = None

    vals = []
    for investor_id in M_train.index:
        v = ndcg_from_item_item(
            M_train=M_train,
            M_test=M_test,
            investor_id=investor_id,
            item_sim=sim,
            k=k,
            top_k_neighbors=top_k_neighbors,
        )
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan

def mean_recall_from_item_item(M_train, M_test, k=10, top_k_neighbors=25):
    sim = item_item_similarity(M_train)
    if top_k_neighbors is not None and top_k_neighbors > 0 and top_k_neighbors < sim.shape[0]:
        sim = pd.DataFrame(
            _prune_similarity_topk(sim.values, top_k_neighbors=top_k_neighbors, axis=0),
            index=sim.index,
            columns=sim.columns,
        )
        top_k_neighbors = None

    vals = []
    for investor_id in M_train.index:
        v = recall_from_item_item(
            M_train=M_train,
            M_test=M_test,
            investor_id=investor_id,
            item_sim=sim,
            k=k,
            top_k_neighbors=top_k_neighbors,
        )
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan

def mean_coverage_from_item_item(M_train, k=10, top_k_neighbors=25):
    sim = item_item_similarity(M_train).to_numpy(copy=True)
    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, None)
    if top_k_neighbors is not None and top_k_neighbors > 0 and top_k_neighbors < sim.shape[0]:
        sim = _prune_similarity_topk(sim, top_k_neighbors=top_k_neighbors, axis=0)

    X = M_train.astype(float).to_numpy()
    user_values = np.nan_to_num(X, nan=0.0)
    observed_bool = np.isfinite(X)
    observed_float = observed_bool.astype(float)
    n_items = X.shape[1]

    vals = []
    for user_idx in range(X.shape[0]):
        weighted_sum = user_values[user_idx] @ sim
        normalizer = observed_float[user_idx] @ sim
        scores = np.divide(
            weighted_sum,
            normalizer,
            out=np.zeros(n_items, dtype=float),
            where=normalizer > 0,
        )

        observed_user = observed_bool[user_idx]
        scores[observed_user] = -np.inf

        ranked_idx = np.argsort(scores)[::-1]
        ranked_idx = ranked_idx[np.isfinite(scores[ranked_idx])]

        v = coverage_at_k(ranked_idx.tolist(), n_items=n_items, k=k)
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan

def mean_diversity_from_item_item(M_train, k=10, top_k_neighbors=25):
    item_sim = item_item_similarity(M_train).to_numpy(copy=True)
    sim = item_sim.copy()
    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, None)
    if top_k_neighbors is not None and top_k_neighbors > 0 and top_k_neighbors < sim.shape[0]:
        sim = _prune_similarity_topk(sim, top_k_neighbors=top_k_neighbors, axis=0)

    X = M_train.astype(float).to_numpy()
    user_values = np.nan_to_num(X, nan=0.0)
    observed_bool = np.isfinite(X)
    observed_float = observed_bool.astype(float)
    n_items = X.shape[1]

    vals = []
    for user_idx in range(X.shape[0]):
        weighted_sum = user_values[user_idx] @ sim
        normalizer = observed_float[user_idx] @ sim
        scores = np.divide(
            weighted_sum,
            normalizer,
            out=np.zeros(n_items, dtype=float),
            where=normalizer > 0,
        )

        observed_user = observed_bool[user_idx]
        scores[observed_user] = -np.inf

        ranked_idx = np.argsort(scores)[::-1]
        ranked_idx = ranked_idx[np.isfinite(scores[ranked_idx])]

        v = diversity_at_k(ranked_idx.tolist(), item_sim=item_sim, k=k)
        if not np.isnan(v):
            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def _rank_unseen(scores, observed_mask):
    scores = np.asarray(scores, dtype=float).copy()
    scores[observed_mask] = -np.inf
    ranked_idx = np.argsort(scores)[::-1]
    return ranked_idx[np.isfinite(scores[ranked_idx])]


def _metrics_from_ranked_idx(ranked_idx, test_row, ideal_rel, item_sim, k=10):
    ranked_idx = np.asarray(ranked_idx, dtype=int)

    rel = (
        np.nan_to_num(test_row[ranked_idx], nan=0.0)
        if ranked_idx.size
        else np.array([], dtype=float)
    )
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(ideal_rel, k)
    ndcg = np.nan if idcg == 0 else float(dcg / idcg)

    recall = recall_at_k(rel, k)
    diversity = diversity_at_k(ranked_idx.tolist(), item_sim=item_sim, k=k)

    return {
        "ndcg": ndcg,
        "recall": recall,
        "diversity": diversity,
    }


def _mean_or_nan(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def compute_seed_metrics(
    util,
    U,
    V,
    seed,
    k=10,
    top_k_neighbors=25,
    test_frac=0.2,
    min_pos=2,
):
    """
    Compute NDCG, Recall, and Diversity for one holdout seed in one pass.

    Returns
    -------
    dict
        {
            "ndcg": {...},
            "recall": {...},
            "diversity": {...},
        }
    """
    util_train, util_test = make_holdout(
        util,
        test_frac=test_frac,
        min_pos=min_pos,
        seed=seed,
    )

    X = util_train.astype(float).to_numpy(copy=True)
    X_filled = np.nan_to_num(X, nan=0.0)
    observed_bool = np.isfinite(X)
    observed_float = observed_bool.astype(float)
    user_values = X_filled
    test_vals = (
        util_test.reindex(index=util_train.index, columns=util_train.columns)
        .astype(float)
        .to_numpy(copy=True)
    )

    n_users, n_items = X.shape

    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)

    if U_arr.shape[0] != n_users:
        raise ValueError("U row count must match util row count.")
    if V_arr.shape[0] != n_items:
        raise ValueError("V row count must match util column count.")

    # shared structures
    als_scores_all = U_arr @ V_arr.T

    pop_scores = (
        util_train.mean(axis=0, skipna=True)
        .reindex(util_train.columns)
        .fillna(0.0)
        .to_numpy(copy=True)
    )

    neighbor_idx, neighbor_sim = user_user_topk_neighbors(
        util_train,
        top_k_neighbors=top_k_neighbors,
    )

    item_sim_full = _cosine_similarity_matrix(X_filled.T)

    sim_pruned = item_sim_full.copy()
    np.fill_diagonal(sim_pruned, 0.0)
    sim_pruned = np.clip(sim_pruned, 0.0, None)
    if (
        top_k_neighbors is not None
        and top_k_neighbors > 0
        and top_k_neighbors < sim_pruned.shape[0]
    ):
        sim_pruned = _prune_similarity_topk(
            sim_pruned,
            top_k_neighbors=top_k_neighbors,
            axis=0,
        )

    ii_weighted_sum_all = user_values @ sim_pruned
    ii_normalizer_all = observed_float @ sim_pruned
    ii_scores_all = np.divide(
        ii_weighted_sum_all,
        ii_normalizer_all,
        out=np.zeros_like(ii_weighted_sum_all, dtype=float),
        where=ii_normalizer_all > 0,
    )

    collectors = {
        "als": {"ndcg": [], "recall": [], "diversity": []},
        "popularity": {"ndcg": [], "recall": [], "diversity": []},
        "user_user": {"ndcg": [], "recall": [], "diversity": []},
        "item_item": {"ndcg": [], "recall": [], "diversity": []},
    }

    for user_idx in range(n_users):
        observed_user = observed_bool[user_idx]
        candidate_mask = ~observed_user
        ideal_rel = np.sort(
            np.nan_to_num(test_vals[user_idx, candidate_mask], nan=0.0)
        )[::-1]

        # ALS
        als_ranked = _rank_unseen(als_scores_all[user_idx], observed_user)
        als_metrics = _metrics_from_ranked_idx(
            als_ranked,
            test_vals[user_idx],
            ideal_rel,
            item_sim=item_sim_full,
            k=k,
        )

        # Popularity
        pop_ranked = _rank_unseen(pop_scores, observed_user)
        pop_metrics = _metrics_from_ranked_idx(
            pop_ranked,
            test_vals[user_idx],
            ideal_rel,
            item_sim=item_sim_full,
            k=k,
        )

        # User-user
        row_idx = neighbor_idx[user_idx]
        row_sim = neighbor_sim[user_idx]
        keep = row_idx >= 0

        row_idx = row_idx[keep].astype(int, copy=False)
        row_sim = row_sim[keep].astype(float, copy=False)

        if row_idx.size:
            uu_weighted_sum = row_sim @ X_filled[row_idx]
            uu_normalizer = row_sim @ observed_float[row_idx]
        else:
            uu_weighted_sum = np.zeros(n_items, dtype=float)
            uu_normalizer = np.zeros(n_items, dtype=float)

        uu_scores = np.divide(
            uu_weighted_sum,
            uu_normalizer,
            out=np.zeros(n_items, dtype=float),
            where=uu_normalizer > 0,
        )

        uu_ranked = _rank_unseen(uu_scores, observed_user)
        uu_metrics = _metrics_from_ranked_idx(
            uu_ranked,
            test_vals[user_idx],
            ideal_rel,
            item_sim=item_sim_full,
            k=k,
        )

        # Item-item
        ii_ranked = _rank_unseen(ii_scores_all[user_idx], observed_user)
        ii_metrics = _metrics_from_ranked_idx(
            ii_ranked,
            test_vals[user_idx],
            ideal_rel,
            item_sim=item_sim_full,
            k=k,
        )

        for metric_name, metric_val in als_metrics.items():
            if not np.isnan(metric_val):
                collectors["als"][metric_name].append(metric_val)

        for metric_name, metric_val in pop_metrics.items():
            if not np.isnan(metric_val):
                collectors["popularity"][metric_name].append(metric_val)

        for metric_name, metric_val in uu_metrics.items():
            if not np.isnan(metric_val):
                collectors["user_user"][metric_name].append(metric_val)

        for metric_name, metric_val in ii_metrics.items():
            if not np.isnan(metric_val):
                collectors["item_item"][metric_name].append(metric_val)

    return {
        "ndcg": {
            model: _mean_or_nan(vals["ndcg"])
            for model, vals in collectors.items()
        },
        "recall": {
            model: _mean_or_nan(vals["recall"])
            for model, vals in collectors.items()
        },
        "diversity": {
            model: _mean_or_nan(vals["diversity"])
            for model, vals in collectors.items()
        },
    }


def compute_mean_ndcgs(util, U, V, seed, k=10, top_k_neighbors=25):
    return compute_seed_metrics(
        util=util,
        U=U,
        V=V,
        seed=seed,
        k=k,
        top_k_neighbors=top_k_neighbors,
    )["ndcg"]


def compute_mean_recalls(util, U, V, seed, k=10, top_k_neighbors=25):
    return compute_seed_metrics(
        util=util,
        U=U,
        V=V,
        seed=seed,
        k=k,
        top_k_neighbors=top_k_neighbors,
    )["recall"]


def compute_mean_diversities(util, U, V, seed, k=10, top_k_neighbors=25):
    return compute_seed_metrics(
        util=util,
        U=U,
        V=V,
        seed=seed,
        k=k,
        top_k_neighbors=top_k_neighbors,
    )["diversity"]


def aggregate_seed_metric_results(results_by_seed, seeds=None):
    """
    Convert nested per-seed results into summary tables.

    Parameters
    ----------
    results_by_seed : dict
        {seed: {"ndcg": {...}, "recall": {...}, "diversity": {...}}}
    seeds : sequence, optional
        Preserve row order.

    Returns
    -------
    dict
        {
            "tables": {
                "ndcg": DataFrame,
                "recall": DataFrame,
                "diversity": DataFrame,
            },
            "summary": {
                "ndcg": DataFrame,
                "recall": DataFrame,
                "diversity": DataFrame,
                "combined": DataFrame,
            },
        }
    """
    metric_names = ["ndcg", "recall", "diversity"]
    model_order = ["popularity", "user_user", "item_item", "als"]

    if seeds is None:
        seeds = sorted(results_by_seed)

    tables = {}
    summary = {}

    pretty_metric = {
        "ndcg": "NDCG@10",
        "recall": "Recall@10",
        "diversity": "Diversity@10",
    }

    pretty_model = {
        "popularity": "Popularity",
        "user_user": "User-User",
        "item_item": "Item-Item",
        "als": "ALS",
    }

    for metric in metric_names:
        df = pd.DataFrame(
            [
                [results_by_seed[s][metric][m] for m in model_order]
                for s in seeds
            ],
            index=seeds,
            columns=[pretty_model[m] for m in model_order],
        )
        df.index.name = "Seed"
        tables[metric] = df

        metric_label = pretty_metric[metric]
        summary[metric] = pd.DataFrame(
            {
                f"Mean {metric_label}": df.mean(),
                f"Std {metric_label}": df.std(ddof=1),
            }
        )

    summary["combined"] = pd.concat(
        [summary["ndcg"], summary["recall"], summary["diversity"]],
        axis=1,
    )

    return {"tables": tables, "summary": summary}


def evaluate_recommenders_parallel(
    util,
    U,
    V,
    seeds,
    k=10,
    top_k_neighbors=25,
    test_frac=0.2,
    min_pos=2,
    max_workers=None,
    backend="process",
):
    """
    Parallel evaluation across seeds.

    backend='process' is the default because this workload is largely CPU-bound.
    Switch to backend='thread' only if process serialization overhead dominates.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must contain at least one value.")

    if max_workers is None:
        max_workers = min(len(seeds), max(1, os.cpu_count() or 1))

    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

    results_by_seed = {}
    with executor_cls(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                compute_seed_metrics,
                util,
                U,
                V,
                seed,
                k,
                top_k_neighbors,
                test_frac,
                min_pos,
            ): seed
            for seed in seeds
        }

        for fut in as_completed(futures):
            seed = futures[fut]
            results_by_seed[seed] = fut.result()

    aggregated = aggregate_seed_metric_results(results_by_seed, seeds=seeds)

    return {
        "by_seed": results_by_seed,
        "tables": aggregated["tables"],
        "summary": aggregated["summary"],
    }