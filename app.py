"""AutoWorth AI — Streamlit Smart Deal Advisor."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="AutoWorth AI", page_icon="🚗", layout="wide")

# ---------- Style ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2.2rem; padding-bottom: 2rem;}
    h1 {margin-bottom: 0.2rem;}
    .verdict {padding: 1rem; border-radius: 12px; text-align: center;
              font-weight: 800; font-size: 1.25rem; letter-spacing: 0.04em;
              margin: 0.6rem 0 0.4rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Banner colours per verdict: (background, text)
VERDICT_STYLE = {
    "GREAT DEAL": ("#15803d", "#ffffff"),
    "GOOD DEAL": ("#65a30d", "#ffffff"),
    "FAIR PRICE": ("#eab308", "#111111"),
    "SLIGHTLY OVERPRICED": ("#f97316", "#111111"),
    "OVERPRICED": ("#dc2626", "#ffffff"),
    "Unknown": ("#6b7280", "#ffffff"),
}


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(ARTIFACTS / "autoworth_bundle.joblib")
    return bundle


def deal_rating(predicted: float, seller: float) -> tuple[str, str, str]:
    if predicted <= 0:
        return "Unknown", "gray", "Predicted price is not valid."
    pct = (seller - predicted) / predicted * 100
    if pct < -10:
        return "GREAT DEAL", "green", "More than 10% cheaper than the predicted market price."
    if pct < -5:
        return "GOOD DEAL", "green", "5–10% cheaper than the predicted market price."
    if pct <= 5:
        return "FAIR PRICE", "gold", "Within about ±5% of the predicted market price."
    if pct <= 10:
        return "SLIGHTLY OVERPRICED", "orange", "5–10% more expensive than the predicted market price."
    return "OVERPRICED", "red", "More than 10% above the predicted market price."


def main():
    if not (ARTIFACTS / "autoworth_bundle.joblib").exists():
        st.error(
            "Model artifacts were not found. Run `python train_and_export.py` first, "
            "or execute the AutoWorth_AI notebook through the export cell."
        )
        return

    bundle = load_artifacts()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_order = bundle["feature_order"]
    choices = bundle["choices"]
    reference_year = bundle["reference_year"]

    st.title("🚗 AutoWorth AI")
    st.caption("Used car fair-market price estimator and smart deal advisor")

    left, right = st.columns([1.15, 1], gap="large")

    # ---------- Inputs ----------
    with left:
        with st.container(border=True):
            st.subheader("Car details")

            c1, c2 = st.columns(2)
            make = c1.selectbox("Make", choices["makes"])
            models = choices["models_by_make"].get(make, choices["models"])
            model_name = c2.selectbox("Model", models)

            year = st.slider(
                "Year", int(choices["year_min"]), int(choices["year_max"]), 2017
            )

            c1, c2 = st.columns(2)
            mileage = c1.number_input("Mileage", min_value=0, max_value=250000, value=25000, step=500)
            seller_price = c2.number_input(
                "Seller Price (£)", min_value=500, max_value=150000, value=17000, step=100
            )

            c1, c2 = st.columns(2)
            transmission = c1.selectbox("Transmission", choices["transmissions"])
            fuel_type = c2.selectbox("Fuel Type", choices["fuel_types"])

            c1, c2, c3 = st.columns(3)
            engine_size = c1.number_input(
                "Engine (L)", min_value=0.5, max_value=6.6, value=2.0, step=0.1
            )
            tax = c2.number_input("Tax (£)", min_value=0, max_value=600, value=145, step=5)
            mpg = c3.number_input("MPG", min_value=1.0, max_value=300.0, value=50.0, step=0.5)

            submit = st.button("Estimate market price", type="primary", use_container_width=True)

    # ---------- Result ----------
    with right:
        with st.container(border=True):
            st.subheader("Deal assessment")

            if not submit:
                st.info("Enter the listing details and click **Estimate market price**.")
            else:
                car_age = max(reference_year - int(year), 0)
                mileage_per_year = mileage / max(car_age, 1)
                premium_makes = {"Audi", "BMW", "Mercedes"}
                is_premium = int(make in premium_makes)
                engine_efficiency = mpg / max(engine_size, 0.1)

                row = pd.DataFrame(
                    [
                        {
                            "make": make,
                            "model": model_name,
                            "year": year,
                            "mileage": mileage,
                            "transmission": transmission,
                            "fuelType": fuel_type,
                            "tax": tax,
                            "mpg": mpg,
                            "engineSize": engine_size,
                            "car_age": car_age,
                            "mileage_per_year": mileage_per_year,
                            "is_premium": is_premium,
                            "engine_efficiency": engine_efficiency,
                        }
                    ]
                )[feature_order]

                X = preprocessor.transform(row)
                predicted = float(model.predict(X)[0])
                difference = seller_price - predicted
                rating, color, meaning = deal_rating(predicted, seller_price)
                bg, fg = VERDICT_STYLE.get(rating, VERDICT_STYLE["Unknown"])

                m1, m2 = st.columns(2)
                m1.metric("Estimated market price", f"£{predicted:,.0f}")
                m2.metric("Seller price", f"£{seller_price:,.0f}")

                cheaper = difference < 0
                st.metric(
                    "Difference",
                    f"£{abs(difference):,.0f} {'cheaper' if cheaper else 'more expensive'}",
                )

                st.markdown(
                    f"<div class='verdict' style='background:{bg};color:{fg}'>{rating}</div>",
                    unsafe_allow_html=True,
                )
                st.write(meaning)
                st.caption(
                    f"Reference year for car age is {reference_year}. "
                    "The rating compares the seller asking price with the model's predicted market value."
                )


if __name__ == "__main__":
    main()