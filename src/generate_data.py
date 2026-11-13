import numpy as np
import pandas as pd
from pathlib import Path

CATEGORIES = ["Web Development", "Data Science", "Mobile App", "Design", "Writing", "Marketing"]
CAT_SCALE = dict(zip(CATEGORIES, [400, 600, 700, 250, 120, 200]))


def generate(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    category = rng.choice(CATEGORIES, n)
    scale = np.array([CAT_SCALE[c] for c in category])
    budget = np.round(scale * rng.lognormal(0, 0.5, n), 0).clip(20)
    avg_comp = budget * rng.normal(0.85, 0.12, n).clip(0.4, 1.2)
    bid_amount = np.round(avg_comp * rng.normal(1.0, 0.22, n).clip(0.4, 1.8), 0).clip(10)
    expected_days = rng.integers(3, 31, n)
    delivery_days = np.round(expected_days * rng.normal(1.0, 0.3, n).clip(0.4, 2.0)).clip(1).astype(int)
    num_bids = rng.poisson(15, n) + 1
    client_rating = rng.uniform(3.0, 5.0, n).round(1)
    client_hire_rate = rng.uniform(0.1, 0.95, n).round(2)
    payment_verified = rng.binomial(1, 0.8, n)
    profile_rating = rng.uniform(4.0, 5.0, n).round(2)
    jobs_completed = rng.integers(0, 150, n)
    proposal_words = rng.normal(180, 90, n).clip(30, 600).round().astype(int)
    hours_to_bid = rng.exponential(8, n).round(1)
    keyword_match = rng.beta(4, 3, n).round(2)
    personalized = rng.binomial(1, 0.5, n)

    bid_vs_avg = bid_amount / avg_comp
    delivery_ratio = delivery_days / expected_days
    logit = (
        -2.2
        - 3.0 * (bid_vs_avg - 1)
        - 0.07 * (num_bids - 15)
        + 1.0 * (profile_rating - 4.5)
        + 0.25 * np.log1p(jobs_completed) - 0.9
        + 1.6 * (keyword_match - 0.57)
        + 0.5 * personalized
        - 0.45 * np.log1p(hours_to_bid) + 0.9
        - 0.6 * ((proposal_words - 180) / 150) ** 2
        - 1.0 * np.maximum(delivery_ratio - 1, 0)
        + 0.3 * payment_verified
        + 0.6 * (client_hire_rate - 0.5)
        + rng.normal(0, 0.4, n)
    )
    won = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame(dict(
        category=category, budget=budget, bid_amount=bid_amount,
        avg_competitor_bid=avg_comp.round(0), delivery_days=delivery_days,
        client_expected_days=expected_days, num_bids=num_bids,
        client_rating=client_rating, client_hire_rate=client_hire_rate,
        payment_verified=payment_verified, profile_rating=profile_rating,
        jobs_completed=jobs_completed, proposal_words=proposal_words,
        hours_to_bid=hours_to_bid, keyword_match=keyword_match,
        personalized=personalized, won=won,
    ))


if __name__ == "__main__":
    df = generate()
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/bids.csv", index=False)
    print(f"Saved {len(df)} rows | win rate: {df.won.mean():.1%}")