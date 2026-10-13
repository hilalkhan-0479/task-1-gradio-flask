# Linear Regression with Gradio and Flask

Linear regression on two different datasets (Excel files), each served through a **Gradio** UI and a **Flask** UI.

## Datasets (in `data/`)
| File | Target | Features |
|---|---|---|
| `cafe_sales.xlsx` | Cold_Drink_Sales_PKR | Temperature_C, Foot_Traffic, Promo_Running |
| `laptop_prices.xlsx` | Price_USD | RAM_GB, Storage_GB, Battery_Hours, Screen_Inches |

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python gradio_app.py   # http://127.0.0.1:7860
python flask_app.py    # http://127.0.0.1:5000
```

## Files
- `model.py` – loads data, trains `LinearRegression`, computes R2/MAE/RMSE, predicts
- `gradio_app.py` – Gradio interface
- `flask_app.py` + `templates/index.html` – Flask interface
