import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.generate_data import CATEGORIES, generate

st.set_page_config(page_title="Bid Win-Rate Predictor", page_icon="🎯", layout="wide")
st.title("🎯 Bid Competitiveness & Win-Rate Predictor")

# Train automatically on first launch
if not Path("models/model.joblib").exists():
    from src.train import main as train
    with st.spinner("First launch: training the model..."):
        Path("data").mkdir(exist_ok=True)
        if not Path("data/bids.csv").exists():
            generate().to_csv("data/bids.csv", index=False)
        train()

from src.predict import price_curve, suggestions, win_probability

with st.sidebar:
    st.header("Project")
    category = st.selectbox("Category", CATEGORIES)
    budget = st.number_input("Client budget ($)", 20, 50000, 400, step=10)
    avg_comp = st.number_input("Average competitor bid ($)", 10, 50000, 340, step=10)
    expected_days = st.number_input("Client's expected days", 1, 120, 14)
    num_bids = st.number_input("Bids so far", 0, 200, 15)
    st.header("Client")
    client_rating = st.slider("Client rating", 3.0, 5.0, 4.5, 0.1)
    hire_rate = st.slider("Client hire rate", 0.0, 1.0, 0.5, 0.05)
    verified = st.checkbox("Payment verified", True)
    st.header("Your bid")
    bid_amount = st.number_input("Your bid ($)", 10, 50000, 340, step=10)
    delivery_days = st.number_input("Delivery days", 1, 120, 14)
    proposal_words = st.slider("Proposal length (words)", 30, 600, 180)
    hours_to_bid = st.slider("Hours after posting", 0.0, 72.0, 4.0, 0.5)
    keyword_match = st.slider("Keyword match with job post", 0.0, 1.0, 0.6, 0.05)
    personalized = st.checkbox("Personalized proposal", True)
    st.header("Your profile")
    profile_rating = st.slider("Profile rating", 4.0, 5.0, 4.8, 0.05)
    jobs_completed = st.number_input("Jobs completed", 0, 1000, 25)

bid = dict(
    category=category, budget=budget, bid_amount=bid_amount, avg_competitor_bid=avg_comp,
    delivery_days=delivery_days, client_expected_days=expected_days, num_bids=num_bids,
    client_rating=client_rating, client_hire_rate=hire_rate, payment_verified=int(verified),
    profile_rating=profile_rating, jobs_completed=jobs_completed, proposal_words=proposal_words,
    hours_to_bid=hours_to_bid, keyword_match=keyword_match, personalized=int(personalized),
)

prob = win_probability(bid)
base_rate = json.load(open("models/metrics.json"))["base_rate"]

c1, c2, c3 = st.columns(3)
c1.metric("Win probability", f"{prob:.0%}", f"{(prob - base_rate) * 100:+.0f} pts vs. average bid")
c2.metric("Your bid vs. competitors", f"{bid_amount / avg_comp:.0%}")
c3.metric("Your bid vs. budget", f"{bid_amount / budget:.0%}")

left, right = st.columns(2)
with left:
    st.subheader("💡 How to improve")
    tips = suggestions(bid)
    shown = [s for s in tips[:5] if s["gain"] > 0.002]
    for s in shown:
        st.write(f"**{s['tip']}** → {s['new_prob']:.0%} (+{s['gain'] * 100:.1f} pts)")
    if not shown:
        st.success("This bid is already well optimized.")
with right:
    st.subheader("📈 What-if: price vs. win probability")
    st.line_chart(price_curve(bid).set_index("bid_amount"))

with st.expander("Model performance & key drivers"):
    st.json(json.load(open("models/metrics.json"))["results"])
    st.bar_chart(pd.read_csv("models/importance.csv").head(10).set_index("feature"))