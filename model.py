"""Shared linear regression logic used by both the Gradio and Flask apps."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "Cafe Cold-Drink Sales": {
        "file": os.path.join(BASE, "data", "cafe_sales.xlsx"),
        "target": "Cold_Drink_Sales_PKR",
    },
    "Laptop Prices": {
        "file": os.path.join(BASE, "data", "laptop_prices.xlsx"),
        "target": "Price_USD",
    },
}


class RegressionModel:
    def __init__(self, name):
        cfg = DATASETS[name]
        self.name = name
        self.df = pd.read_excel(cfg["file"])
        self.target = cfg["target"]
        self.features = [c for c in self.df.columns if c != self.target]
        self._train()

    def _train(self):
        X, y = self.df[self.features], self.df[self.target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LinearRegression().fit(X_tr, y_tr)
        pred = self.model.predict(X_te)
        self.metrics = {
            "R2": round(r2_score(y_te, pred), 4),
            "MAE": round(mean_absolute_error(y_te, pred), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2),
            "Train rows": len(X_tr),
            "Test rows": len(X_te),
        }
        self.coefs = {f: round(float(c), 4) for f, c in zip(self.features, self.model.coef_)}
        self.intercept = round(float(self.model.intercept_), 4)

    def predict(self, values):
        row = pd.DataFrame([values], columns=self.features)
        return float(self.model.predict(row)[0])

    def feature_ranges(self):
        d = self.df[self.features]
        return {f: (float(d[f].min()), float(d[f].max()), float(d[f].median())) for f in self.features}


_cache = {}

def get_model(name):
    if name not in _cache:
        _cache[name] = RegressionModel(name)
    return _cache[name]
