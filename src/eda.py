import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PLOTS_DIR
from src.preprocessing import load_data, clean_data, feature_engineering
from src import logger as log


def run():
    """Run complete Exploratory Data Analysis and save all visualisations."""
    log.section("Loading & preparing data", "\U0001F4C2")
    df = load_data()
    raw_shape = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_dup = int(df.duplicated().sum())
    df = clean_data(df)
    df = feature_engineering(df)
    log.data_summary(raw_shape, n_missing, n_dup, n_features_eng=4)
    log.info(f"After feature engineering: {df.shape[0]} rows x {df.shape[1]} cols")

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    log.section("Generating 12 EDA visualisations", "\U0001F3A8")

    with log.progress_bar(12, "EDA Plots") as progress:
        task = progress.add_task("Generating plots...", total=12)

        # 1. Distribution of Attack Types
        fig, ax = plt.subplots(figsize=(10, 6))
        counts = df["Attack Type"].value_counts()
        sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
        ax.set_title("Distribution of Attack Types", fontsize=14, fontweight="bold")
        ax.set_xlabel("Count")
        ax.set_ylabel("Attack Type")
        for i, v in enumerate(counts.values):
            ax.text(v + 0.5, i, str(v), va="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "01_attack_type_distribution.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[1/12] Attack Type distribution")

        # 2. Financial Loss distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df["Financial Loss (in Million $)"], bins=30, edgecolor="black",
                     color="steelblue", alpha=0.7)
        axes[0].set_title("Distribution of Financial Loss", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Financial Loss (Million $)")
        axes[0].set_ylabel("Frequency")
        sns.boxplot(x=df["Financial Loss (in Million $)"], ax=axes[1], color="steelblue")
        axes[1].set_title("Box Plot of Financial Loss", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Financial Loss (Million $)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "02_financial_loss_distribution.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[2/12] Financial Loss distribution")

        # 3. Attacks by Country
        fig, ax = plt.subplots(figsize=(12, 6))
        cc = df["Country"].value_counts()
        sns.barplot(x=cc.index, y=cc.values, ax=ax, palette="coolwarm")
        ax.set_title("Number of Cyber Attacks by Country", fontsize=14, fontweight="bold")
        ax.set_xlabel("Country"); ax.set_ylabel("Number of Attacks")
        plt.xticks(rotation=45, ha="right")
        for i, v in enumerate(cc.values):
            ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "03_attacks_by_country.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[3/12] Attacks by Country")

        # 4. Financial Loss by Attack Type
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="Attack Type", y="Financial Loss (in Million $)",
                    ax=ax, palette="Set2")
        ax.set_title("Financial Loss by Attack Type", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "04_loss_by_attack_type.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[4/12] Loss by Attack Type")

        # 5. Financial Loss by Target Industry
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="Target Industry", y="Financial Loss (in Million $)",
                    ax=ax, palette="Set3")
        ax.set_title("Financial Loss by Target Industry", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "05_loss_by_industry.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[5/12] Loss by Target Industry")

        # 6. Temporal trends
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        yearly_cnt = df.groupby("Year").size()
        axes[0].plot(yearly_cnt.index, yearly_cnt.values, marker="o", linewidth=2,
                     color="steelblue")
        axes[0].fill_between(yearly_cnt.index, yearly_cnt.values, alpha=0.3,
                             color="steelblue")
        axes[0].set_title("Number of Attacks per Year", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Year"); axes[0].set_ylabel("Count")
        axes[0].set_xticks(range(2015, 2025))
        yearly_loss = df.groupby("Year")["Financial Loss (in Million $)"].mean()
        axes[1].plot(yearly_loss.index, yearly_loss.values, marker="s", linewidth=2,
                     color="tomato")
        axes[1].fill_between(yearly_loss.index, yearly_loss.values, alpha=0.3,
                             color="tomato")
        axes[1].set_title("Average Financial Loss per Year", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Year"); axes[1].set_ylabel("Avg Loss (Million $)")
        axes[1].set_xticks(range(2015, 2025))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "06_attacks_over_time.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[6/12] Temporal trends")

        # 7. Heatmap: Attack Type vs Target Industry
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot = pd.crosstab(df["Attack Type"], df["Target Industry"])
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title("Attack Type vs Target Industry", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "07_attack_vs_industry_heatmap.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[7/12] Attack vs Industry heatmap")

        # 8. Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        num_cols = ["Year", "Financial Loss (in Million $)", "Number of Affected Users",
                    "Incident Resolution Time (in Hours)", "Loss_per_User",
                    "Users_per_Hour", "Loss_per_Hour", "Log_Financial_Loss"]
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    ax=ax, square=True, linewidths=0.5)
        ax.set_title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "08_correlation_heatmap.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[8/12] Correlation heatmap")

        # 9. Attack Source pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        src = df["Attack Source"].value_counts()
        colors = sns.color_palette("pastel")[: len(src)]
        ax.pie(src.values, labels=src.index, autopct="%1.1f%%", colors=colors,
               startangle=90)
        ax.set_title("Distribution of Attack Sources", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "09_attack_source_distribution.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[9/12] Attack Source distribution")

        # 10. Resolution Time by Defense Mechanism
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="Defense Mechanism Used",
                    y="Incident Resolution Time (in Hours)", ax=ax, palette="Set2")
        ax.set_title("Resolution Time by Defense Mechanism", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "10_resolution_by_defense.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[10/12] Resolution by Defense")

        # 11. Security Vulnerability Type analysis
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        vc = df["Security Vulnerability Type"].value_counts()
        sns.barplot(x=vc.values, y=vc.index, ax=axes[0], palette="viridis")
        axes[0].set_title("Vulnerability Type Distribution", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Count")
        vl = (df.groupby("Security Vulnerability Type")["Financial Loss (in Million $)"]
                .mean().sort_values(ascending=False))
        sns.barplot(x=vl.values, y=vl.index, ax=axes[1], palette="magma")
        axes[1].set_title("Avg Financial Loss by Vulnerability", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Avg Loss (Million $)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "11_vulnerability_analysis.png"), dpi=150)
        plt.close()
        progress.update(task, advance=1, description="[11/12] Vulnerability analysis")

        # 12. Scatter matrix coloured by Attack Type
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        pairs = [
            ("Financial Loss (in Million $)", "Number of Affected Users"),
            ("Financial Loss (in Million $)", "Incident Resolution Time (in Hours)"),
            ("Number of Affected Users", "Incident Resolution Time (in Hours)"),
            ("Financial Loss (in Million $)", "Loss_per_User"),
            ("Year", "Financial Loss (in Million $)"),
            ("Year", "Number of Affected Users"),
        ]
        attack_types = df["Attack Type"].unique()
        cmap = dict(zip(attack_types, sns.color_palette("husl", len(attack_types))))
        for idx, (xc, yc) in enumerate(pairs):
            ax = axes[idx // 3][idx % 3]
            for at in attack_types:
                m = df["Attack Type"] == at
                ax.scatter(df.loc[m, xc], df.loc[m, yc], alpha=0.6, label=at,
                           color=cmap[at], s=30)
            ax.set_xlabel(xc, fontsize=9)
            ax.set_ylabel(yc, fontsize=9)
            ax.set_title(f"{xc} vs {yc}", fontsize=10)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(attack_types),
                   bbox_to_anchor=(0.5, 1.02), fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(PLOTS_DIR, "12_scatter_plots.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        progress.update(task, advance=1, description="[12/12] Scatter plots done")

    log.success(f"All 12 EDA plots saved to {PLOTS_DIR}/")
    return df


if __name__ == "__main__":
    run()
