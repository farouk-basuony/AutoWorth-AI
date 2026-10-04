# AutoWorth AI — Used Car Price & Deal Advisor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://autoworth-ml.streamlit.app/)
> 🔗 **Live Demo:** [Click here to launch the AutoWorth AI App](https://autoworth-ml.streamlit.app/)

Individual final project for the **Artificial Intelligence** course at **Telecom Egypt (WE)**.
Student: Mohamed Ahmed Farouk · Supervised by: Eng. Fayrouz Ahmed

AutoWorth AI estimates the fair market price of a used car from the 100,000 UK Used Car Dataset, then compares that estimate with the seller's asking price and returns an actionable deal rating.

---

## What the system answers

> **"What is the expected market price of this car, and is the seller offering a good deal?"**

**Example:**
* **Estimated market price:** £10,697
* **Seller price:** £9,200
* **Difference:** £1,497 cheaper
* **Deal rating:** Great Deal

---

## Dataset

* **Source:** Kaggle [100,000 UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)
* **Manufacturer files used for modelling:** Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, Volkswagen (99,187 initial listings, 97,050 after cleaning).
* *Note:* `cclass.csv`, `focus.csv`, and unclean files are excluded as they lack tax/mpg data or contain unstandardized duplicates.

---

## Project Pipeline

1. **Data Integration:** Load individual brand CSVs and assign the `make` feature.
2. **Cleaning & Sanity Checks:** Remove duplicates, anomalous years (e.g., 2060, pre-1995), engine size 0.0 entries, and cap unrealistic MPG (>120).
3. **Exploratory Data Analysis:** Price distribution, mileage decay, brand tiering, and boxplots across transmission types.
4. **Leakage Prevention:** 70% / 15% / 15% Train/Val/Test strict split before any transformations.
5. **Feature Engineering:**
   * `car_age`: 2020 − year
   * `mileage_per_year`: mileage ÷ max(car_age, 1)
   * `is_premium`: German premium brand flag (Audi, BMW, Mercedes)
   * `engine_efficiency`: mpg ÷ engineSize
6. **Model Benchmarking:** Train Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
7. **Hyperparameter Tuning:** RandomizedSearchCV on XGBoost parameters (`learning_rate`, `max_depth`, `n_estimators`, `colsample_bytree`).
8. **Final Evaluation:** Evaluate the tuned XGBoost on the unbiased held-out test set.
9. **Explainability & Diagnostics:** Feature importances and residual/error diagnostics.
10. **Productization:** Interactive Streamlit web interface with real-time deal evaluation.

---

## Deal Ratings

| Rating | Rule |
| :--- | :--- |
| 🟢 **Great Deal** | Seller is more than 10% cheaper than the predicted market price |
| 🟩 **Good Deal** | 5% to 10% cheaper |
| 🟡 **Fair Price** | Within ±5% |
| 🟠 **Slightly Overpriced** | 5% to 10% more expensive |
| 🔴 **Overpriced** | More than 10% above the predicted market price |

---

## Results from the Trained Pipeline

Final **Tuned XGBoost** model:

| Split | R² | MAE (£) | RMSE (£) |
| :--- | :---: | :---: | :---: |
| **Validation (Tuned)** | 0.967 | 1,083.46 | 1,702.57 |
| **Test (held-out)** | **0.968** | **1,070.86** | **1,654.05** |

Tuning improved XGBoost validation R² from 0.954 to 0.967, MAE from £1,345 to £1,083 and RMSE from £2,009 to £1,703. The test-set result confirms strong generalization with no data leakage.

**Best parameters (RandomizedSearchCV):** `subsample=0.9`, `n_estimators=300`, `min_child_weight=5`, `max_depth=10`, `learning_rate=0.1`, `colsample_bytree=0.8`.

---

## Limitations

* The model is trained on the **UK market** (prices in GBP), so it is not meant for direct use on other markets.
* Largest errors occur on rare high-performance trims (e.g. BMW M4, Audi A8), where optional extras are not captured.
* Car age is computed relative to a reference year of 2020.

---

## Run Locally

Requires **Python 3.11 or 3.12** (the pinned `scikit-learn` has no prebuilt package for newer versions).

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

---

## Repository Layout

```text
├── AutoWorth_AI.ipynb               # Full exploratory and training walkthrough
├── app.py                           # Streamlit web application interface
├── requirements.txt                 # Python production dependencies
├── artifacts/                       # Pretrained model and preprocessing pipeline
└── AutoWorth_AI_Presentation.pptx   # Final presentation deck
```
