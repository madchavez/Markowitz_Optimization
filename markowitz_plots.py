# markowitz_plots.py
import matplotlib.pyplot as plt

from markowitz_analysis import RawResults, PCAResults, RMTResults


def plot_frontier_raw(raw: RawResults, title_suffix: str = "Raw; Jan 1, 2020 to Dec 31, 2024"):
    plt.figure(figsize=(12, 8))

    plt.scatter(
        raw.efficient_vols,
        raw.target_returns,
        color="red",
        linewidth=2.5,
        label="Efficient Frontier (Raw)",
    )

    mv = raw.min_vol_max_sharpe
    plt.scatter(mv[1], mv[0], color="green", marker="o", s=180, label="Min Vol (Raw)")
    plt.scatter(mv[3], mv[2], color="green", marker="*", s=250, label="Max Sharpe (Raw)")

    plt.scatter(
        raw.rand_sigmas,
        raw.rand_mus,
        color="blue",
        marker=".",
        s=250,
        label="Random Portfolios",
    )

    plt.scatter(
        raw.indiv_sigmas,
        raw.indiv_mus,
        color="pink",
        alpha=0.8,
        marker=".",
        s=250,
        label="Individual Stocks",
    )

    plt.title(f"Markowitz Efficient Frontier ({title_suffix})", fontsize=16)
    plt.xlabel("Portfolio Risk (Std Dev)", fontsize=13)
    plt.ylabel("Portfolio Expected Return", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_frontier_pca(raw: RawResults, pca: PCAResults, title_suffix: str = "PCA; Jan 1, 2020 to Dec 31, 2024"):
    plt.figure(figsize=(12, 8))

    plt.scatter(
        pca.efficient_vols_pca,
        pca.target_returns_pca,
        color="red",
        linewidth=2.5,
        label="Efficient Frontier (PCA)",
    )

    mv = pca.min_vol_max_sharpe_pca
    plt.scatter(mv[1], mv[0], color="green", marker="o", s=180, label="Min Vol (PCA)")
    plt.scatter(mv[3], mv[2], color="green", marker="*", s=250, label="Max Sharpe (PCA)")

    plt.scatter(
        raw.rand_sigmas,
        raw.rand_mus,
        color="blue",
        marker=".",
        s=250,
        label="Random Portfolios",
    )

    plt.scatter(
        raw.indiv_sigmas,
        raw.indiv_mus,
        color="pink",
        alpha=0.8,
        marker=".",
        s=250,
        label="Individual Stocks",
    )

    plt.title(f"Markowitz Efficient Frontier ({title_suffix})", fontsize=16)
    plt.xlabel("Portfolio Risk (Std Dev)", fontsize=13)
    plt.ylabel("Portfolio Expected Return", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_frontier_rmt(raw: RawResults, rmt: RMTResults, title_suffix: str = "RMT; Jan 1, 2020 to Dec 31, 2024"):
    plt.figure(figsize=(12, 8))

    plt.scatter(
        rmt.efficient_vols_rmt,
        rmt.target_returns_rmt,
        color="red",
        linewidth=2.5,
        label="Efficient Frontier (RMT)",
    )

    mv = rmt.min_vol_max_sharpe_rmt
    plt.scatter(mv[1], mv[0], color="green", marker="o", s=180, label="Min Vol (RMT)")
    plt.scatter(mv[3], mv[2], color="green", marker="*", s=250, label="Max Sharpe (RMT)")

    plt.scatter(
        raw.rand_sigmas,
        raw.rand_mus,
        color="blue",
        marker=".",
        s=250,
        label="Random Portfolios",
    )

    plt.scatter(
        raw.indiv_sigmas,
        raw.indiv_mus,
        color="pink",
        alpha=0.8,
        marker=".",
        s=250,
        label="Individual Stocks",
    )

    plt.title(f"Markowitz Efficient Frontier ({title_suffix})", fontsize=16)
    plt.xlabel("Portfolio Risk (Std Dev)", fontsize=13)
    plt.ylabel("Portfolio Expected Return", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_rmt_spectrum(rmt: RMTResults):
    eigvals = rmt.eigvals_rmt
    lambda_plus = rmt.lambda_plus

    plt.figure(figsize=(14, 6))
    plt.hist(eigvals, bins=80, alpha=0.75, edgecolor="black")
    plt.axvline(
        lambda_plus,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=r"$\lambda_{+}$ (MP)",
    )
    plt.title("Eigenvalue Spectrum of Full-Universe Correlation vs MP Bounds", fontsize=16)
    plt.xlabel("Eigenvalue", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.legend()
    plt.show()


def plot_quarter_frontiers(
    quarter_results: dict,
    rmt: RMTResults,
    pca: PCAResults,
    snp_stats: dict,
):
    """
    Plot 3 subplots comparing quarterly raw efficient frontiers vs. PCA/RMT minvol/maxSharpe and ^GSPC point.
    snp_stats[label] = (mu, sigma)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    quarters = [
        ("Raw; Q1", "Q1"),
        ("Raw; Q2", "Q2"),
        ("Raw; Q3", "Q3"),
    ]

    for ax, (label, key) in zip(axes, quarters):
        q = quarter_results[key]
        vols = q["efficient_vols"]
        rets = q["target_returns"]
        mm = q["min_vol_max_sharpe"]
        sm, ss = snp_stats[key]  # mu, sigma

        # Efficient Frontier
        ax.scatter(vols, rets, color="red", linewidth=2.5, label="Efficient Frontier")

        # Min Vol / Max Sharpe (raw, per quarter)
        ax.scatter(mm[1], mm[0], color="green", marker="o", s=180, label="Min Vol")
        ax.scatter(mm[3], mm[2], color="green", marker="*", s=250, label="Max Sharpe")

        # Global RMT
        mv_rmt = rmt.min_vol_max_sharpe_rmt
        ax.scatter(
            mv_rmt[1],
            mv_rmt[0],
            color="blue",
            marker=".",
            s=180,
            label="Min Vol (RMT)",
        )
        ax.scatter(
            mv_rmt[3],
            mv_rmt[2],
            color="blue",
            marker="*",
            s=250,
            label="Max Sharpe (RMT)",
        )

        # Global PCA
        mv_pca = pca.min_vol_max_sharpe_pca
        ax.scatter(
            mv_pca[1],
            mv_pca[0],
            color="orange",
            marker=".",
            s=180,
            label="Min Vol (PCA)",
        )
        ax.scatter(
            mv_pca[3],
            mv_pca[2],
            color="orange",
            marker="*",
            s=250,
            label="Max Sharpe (PCA)",
        )

        # ^GSPC point
        ax.scatter(
            ss,
            sm,
            color="black",
            marker="*",
            s=250,
            label="^GSPC",
        )

        ax.set_title(f"Efficient Frontier ({label})", fontsize=14)
        ax.set_xlabel("Risk (Std Dev)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Expected Return")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_cumrets_minvol(cumrets: dict, snp_cum_ret):
    plt.figure(figsize=(14, 8))

    plt.plot(cumrets["minvol"]["raw"]["cum"], label="Raw MinVol", linewidth=2)
    plt.plot(cumrets["minvol"]["pca"]["cum"], label="PCA MinVol", linewidth=2)
    plt.plot(cumrets["minvol"]["rmt"]["cum"], label="RMT MinVol", linewidth=2)

    plt.plot(snp_cum_ret, label="^GSPC", linewidth=2, color="black")
    plt.title(
        "Out-of-Sample Cumulative Returns (Minimum Volatility Portfolios vs ^GSPC)",
        fontsize=16,
    )
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_cumrets_maxsharpe(cumrets: dict, snp_cum_ret):
    plt.figure(figsize=(14, 8))

    plt.plot(cumrets["sharpe"]["raw"]["cum"], label="Raw MaxSharpe", linewidth=2)
    plt.plot(cumrets["sharpe"]["pca"]["cum"], label="PCA MaxSharpe", linewidth=2)
    plt.plot(cumrets["sharpe"]["rmt"]["cum"], label="RMT MaxSharpe", linewidth=2)

    plt.plot(snp_cum_ret, label="^GSPC", linewidth=2, color="black")
    plt.title(
        "Out-of-Sample Cumulative Returns (Maximum Sharpe Portfolios vs ^GSPC)",
        fontsize=16,
    )
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_top10_holdings(top: dict):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Top 10 Holdings per Portfolio", fontsize=16, weight="bold")

    def plot_bar(ax, series, title):
        series.sort_values().plot(
            kind="barh",
            ax=ax,
            edgecolor="black",
        )
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Portfolio Weight")
        ax.set_ylabel("Ticker")
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        ax.set_xlim(0, series.max() * 1.3)

    # Row 1: MinVar; Row 2: MaxSharpe
    plot_bar(axes[0, 0], top["raw"]["minvol"], f"MinVar (Raw)")
    plot_bar(axes[1, 0], top["raw"]["sharpe"], "MaxSharpe (Raw)")

    plot_bar(axes[0, 1], top["pca"]["minvol"], "MinVar (PCA)")
    plot_bar(axes[1, 1], top["pca"]["sharpe"], "MaxSharpe (PCA)")

    plot_bar(axes[0, 2], top["rmt"]["minvol"], "MinVar (RMT)")
    plot_bar(axes[1, 2], top["rmt"]["sharpe"], "MaxSharpe (RMT)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
