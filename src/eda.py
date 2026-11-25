"""Exploratory data analysis: every function saves one figure to reports/figures/."""
import matplotlib
matplotlib.use("Agg")  # works without a display; remove this line inside Jupyter if you want inline plots
import matplotlib.pyplot as plt
import seaborn as sns
from .config import FIG_DIR, TARGET

sns.set_theme(style="whitegrid")


def _save(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()


def plot_target_distribution(df):                      # Pie / bar  (Day 1)
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax[0].set_title("Booking status (share)")
    sns.countplot(data=df, x=TARGET, ax=ax[1])
    ax[1].set_title("Booking status (count)")
    _save("01_target_distribution.png")


def plot_lead_time(df):                                 # Box plot + histogram
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x=TARGET, y="lead_time", ax=ax[0])
    ax[0].set_title("Lead time vs booking status")
    sns.histplot(data=df, x="lead_time", hue=TARGET, bins=40, ax=ax[1])
    ax[1].set_title("Lead time distribution")
    _save("02_lead_time.png")


def plot_price(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x=TARGET, y="avg_price_per_room", ax=ax[0])
    ax[0].set_title("Average room price vs status")
    sns.histplot(df["avg_price_per_room"], bins=40, ax=ax[1])
    ax[1].set_title("Average room price distribution")
    _save("03_price.png")


def plot_cancel_rate_by(df, col, filename, title):      # Bar plot of cancellation rate
    rate = (df.assign(cancelled=(df[TARGET] == "Canceled").astype(int))
              .groupby(col)["cancelled"].mean().sort_values(ascending=False))
    plt.figure(figsize=(8, 4))
    sns.barplot(x=rate.index.astype(str), y=rate.values)
    plt.ylabel("Cancellation rate")
    plt.xlabel(col)
    plt.title(title)
    plt.xticks(rotation=30)
    _save(filename)


def plot_monthly_trend(df):                             # Line plot
    rate = (df.assign(cancelled=(df[TARGET] == "Canceled").astype(int))
              .groupby("arrival_month")["cancelled"].mean())
    plt.figure(figsize=(8, 4))
    plt.plot(rate.index, rate.values, marker="o")
    plt.xticks(range(1, 13))
    plt.xlabel("Arrival month")
    plt.ylabel("Cancellation rate")
    plt.title("Cancellation rate by arrival month")
    _save("07_monthly_trend.png")


def plot_correlation(df):
    num = df.select_dtypes("number").copy()
    num["cancelled"] = (df[TARGET] == "Canceled").astype(int)
    plt.figure(figsize=(11, 8))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap (numeric features)")
    _save("08_correlation_heatmap.png")


def run_all_eda(df):
    plot_target_distribution(df)
    plot_lead_time(df)
    plot_price(df)
    plot_cancel_rate_by(df, "market_segment_type", "04_cancel_by_segment.png", "Cancellation rate by market segment")
    plot_cancel_rate_by(df, "no_of_special_requests", "05_cancel_by_requests.png", "Cancellation rate by special requests")
    plot_cancel_rate_by(df, "repeated_guest", "06_cancel_by_repeated_guest.png", "Cancellation rate: repeated guest")
    plot_monthly_trend(df)
    plot_correlation(df)
