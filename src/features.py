import pandas as pd

CATEGORICAL = ["category"]
NUMERIC = [
    "budget", "bid_amount", "avg_competitor_bid", "delivery_days", "client_expected_days",
    "num_bids", "client_rating", "client_hire_rate", "payment_verified", "profile_rating",
    "jobs_completed", "proposal_words", "hours_to_bid", "keyword_match", "personalized",
    "bid_to_budget", "bid_vs_avg", "delivery_ratio",
]
TARGET = "won"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bid_to_budget"] = df["bid_amount"] / df["budget"]
    df["bid_vs_avg"] = df["bid_amount"] / df["avg_competitor_bid"]
    df["delivery_ratio"] = df["delivery_days"] / df["client_expected_days"]
    return df