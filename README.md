# 🎯 Bid Competitiveness & Win-Rate Predictor

A machine learning web app that predicts the probability of a freelance bid winning, and suggests how to make it more competitive.

**🔗 Live app:** https://freelancer-project-bid-competitiveness-win-rate-predictor-akel.streamlit.app/


## Features
- **Win probability** for a bid, based on price, delivery time, proposal quality, timing, client and profile details
- **Improvement tips**: tests changes like a lower price or a faster bid and shows how much each would raise the win chance
- **What-if chart**: win probability across different price points
- **Model insights**: performance metrics and the most influential factors

## How it works
1. **Data:** a synthetic dataset of 6,000 bids with won/lost outcomes
2. **Features:** bid-to-budget ratio, bid vs. average competitor bid, delivery vs. client's expected days, plus client and profile fields
3. **Models:** Logistic Regression and a calibrated Gradient Boosting model, with the best by ROC-AUC kept
4. **Calibration:** probabilities are calibrated, so "30%" means roughly a 30% chance of winning
5. **Insights:** permutation importance and a one-change-at-a-time what-if engine

## Tech stack
Python · scikit-learn · pandas · NumPy · Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The model trains automatically on first launch.

## Project structure
```
├── app.py              # Streamlit app
├── requirements.txt
└── src/
    ├── features.py     # feature engineering
    ├── generate_data.py# synthetic data generator
    ├── train.py        # model training and evaluation
    └── predict.py      # predictions and tips
```

## Note on data
The model is currently trained on **synthetic data** to demonstrate the full pipeline. The tips show how the system works, not real market behaviour.

## Next steps
- Retrain on real freelance bid data
- Add a "log my bid" form
- Add SHAP explanations and XGBoost/LightGBM models
