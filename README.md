# GAU → HYD Flight ML v3 — daily expanding learning system

This version is designed for the actual question: **when should I book 21/22 Nov 2026?**

## Every day

1. Track the exact target flights **6E565 and 6E187** for **21 Nov and 22 Nov**.
2. Search a **broad panel of departure dates** from tomorrow through 30 days after 22 Nov. This covers September/October/November and surrounding dates while they are still searchable.
3. Save every observation permanently in `data/`. Older observations are never discarded.
4. Retrain the model from all eligible historical examples.
5. Generate a fresh **10-day prediction path** for every target flight/date.
6. Estimate the probability of a **≥7% drop** and **≥7% rise** over each forecast horizon.
7. Produce a booking-risk signal: possible large drop, high increase risk, mixed/volatile, or no large move detected.
8. Compare old predictions with later real prices without changing the old prediction. This is walk-forward, leakage-safe evaluation.

## Why the extra departure dates matter

The exact 21/22 Nov flights start with very little history. The broader daily panel creates many learning examples across different days-to-departure values. For example, an October departure that has already finished supplies a much more mature booking curve for learning how a price behaves 20, 15, 10, or 5 days before departure.

The model predicts **future percentage/log-price movement**, not an absolute rupee value copied from another flight. That lets broader route-level data teach movement patterns while the exact target flight's current price remains the anchor.

## Daily training

The model is retrained every day. It uses only observations that existed by that run's date. For each 1–10 day horizon it uses several tree-based regressors and weights them using a chronological validation split based only on older observations. Separate classifiers estimate large-drop and large-rise probabilities.

As soon as a real future price arrives, the corresponding old prediction becomes eligible for later training, and the model learns from it on subsequent runs.

## Accuracy

The dashboard reports current realized MAE/sMAPE from predictions that have already encountered their true future prices. It does **not** claim 100% accuracy.

## Limitations

Google Flights does not provide a documented public historical-price API for this purpose. This project builds its own history from repeated searches through the `fli` client. It does not invent missing historical prices and does not bypass CAPTCHA/access controls. The unofficial client may stop working if Google changes its interface.

## Free deployment

A public GitHub repository + GitHub Actions + GitHub Pages can run/host the project at no monetary cost under GitHub's applicable limits.
