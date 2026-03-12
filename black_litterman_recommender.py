from __future__ import annotations

from ast import If
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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