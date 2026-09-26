"""Pair classifier + threshold -> final matches. Owner: Person 3.

Plan: LightGBM binary classifier on features.py output, labels from train ground
truth. Threshold is tuned for macro F0.5 on a held-out S1 split (evaluate.py),
not for pair-level accuracy.
"""
import pandas as pd


def train(features: pd.DataFrame, labels: pd.Series):
    raise NotImplementedError


def predict_proba(model, features: pd.DataFrame):
    raise NotImplementedError


def select_matches(pairs: pd.DataFrame, proba, threshold: float) -> dict:
    """Keep pairs with proba >= threshold -> {s1_id: [match ids]}."""
    raise NotImplementedError
