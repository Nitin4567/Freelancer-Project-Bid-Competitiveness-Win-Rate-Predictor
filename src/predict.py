import joblib
import numpy as np
import pandas as pd

from .features import CATEGORICAL, NUMERIC, add_features

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load("models/model.joblib")
    return _model


def win_probability(bid: dict) -> float:
    return float(win_probabilities([bid])[0])


def win_probabilities(bids: list) -> np.ndarray:
    df = add_features(pd.DataFrame(bids))
    return _get_model().predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]


def suggestions(bid: dict) -> list:
    base = win_probability(bid)
    options = []
    for pct in (5, 10, 15):
        options.append((f"Lower your price by {pct}%", {"bid_amount": round(bid["bid_amount"] * (1 - pct / 100))}))
    if bid["delivery_days"] > 1:
        options.append(("Deliver 20% faster", {"delivery_days": max(1, round(bid["delivery_days"] * 0.8))}))
    if bid["hours_to_bid"] > 2:
        options.append(("Bid within 2 hours of posting", {"hours_to_bid": 2.0}))
    if not bid["personalized"]:
        options.append(("Personalize the proposal to the client", {"personalized": 1}))
    if abs(bid["proposal_words"] - 180) > 40:
        options.append(("Aim for a ~180-word proposal", {"proposal_words": 180}))
    if bid["keyword_match"] < 0.8:
        options.append(("Mirror more keywords from the job post", {"keyword_match": min(1.0, bid["keyword_match"] + 0.2)}))

    probs = win_probabilities([{**bid, **change} for _, change in options])
    out = [{"tip": t, "new_prob": float(p), "gain": float(p - base)} for (t, _), p in zip(options, probs)]
    return sorted(out, key=lambda r: r["gain"], reverse=True)


def price_curve(bid: dict, lo=0.5, hi=1.5, steps=25) -> pd.DataFrame:
    prices = np.linspace(bid["bid_amount"] * lo, bid["bid_amount"] * hi, steps).round()
    probs = win_probabilities([{**bid, "bid_amount": p} for p in prices])
    return pd.DataFrame({"bid_amount": prices, "win_probability": probs})