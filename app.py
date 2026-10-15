
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Assistant",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "house_price_model.joblib"
RECOMMENDATION_PATH = BASE_DIR / "house_recommendation_data.csv"


# --------------------------------------------------
# 2. LOAD SAVED MODEL AND HISTORICAL DATA
# --------------------------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_houses(path):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error("Model file not found: house_price_model.joblib")
    st.stop()

if not RECOMMENDATION_PATH.exists():
    st.error(
        "Recommendation data not found: "
        "house_recommendation_data.csv"
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    houses = load_houses(str(RECOMMENDATION_PATH))
except Exception as exc:
    st.error(f"Could not load the application files: {exc}")
    st.stop()


required_columns = [
    "Id", "Neighborhood", "SalePrice", "GrLivArea",
    "BedroomAbvGr", "TotalBathrooms", "OverallQual",
    "YearBuilt", "GarageCars", "Fireplaces", "KitchenQual"
]

missing_columns = [
    col for col in required_columns
    if col not in houses.columns
]

if missing_columns:
    st.error(f"Missing required data columns: {missing_columns}")
    st.stop()


# Flag unusual size-price combinations for review.
# This rule identifies candidates for review, not confirmed errors.
houses["Review_Status"] = np.where(
    (houses["GrLivArea"] > 4000) &
    (houses["SalePrice"] < 300000),
    "Review required",
    "No flag under this rule"
)


neighborhood_options = sorted(
    houses["Neighborhood"].dropna().astype(str).unique().tolist()
)

kitchen_order = ["Po", "Fa", "TA", "Gd", "Ex"]

available_kitchens = set(
    houses["KitchenQual"].dropna().astype(str).unique()
)

kitchen_options = [
    quality for quality in kitchen_order
    if quality in available_kitchens
]

year_min = int(houses["YearBuilt"].min())
year_max = int(houses["YearBuilt"].max())


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🏠 House Price Assistant")

st.write(
    "Estimate a house price or discover historical properties "
    "that match your preferences."
)

st.info(
    "This prototype uses historical house sales from Ames, Iowa. "
    "Recommendations are not current listings, and predictions "
    "are estimates rather than professional appraisals."
)


# --------------------------------------------------
# 4. TWO APPLICATION TABS
# --------------------------------------------------

predict_tab, recommend_tab = st.tabs(
    ["💰 Predict House Price", "🔎 Find Matching Houses"]
)


# --------------------------------------------------
# 5. PRICE PREDICTION FORM
# --------------------------------------------------

with predict_tab:

    st.subheader("Enter the property characteristics")

    st.write(
        "Choose the available values that best describe the house. "
        "The trained model uses these 11 features."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            overall_qual = st.slider(
                "Overall quality",
                min_value=1,
                max_value=10,
                value=6
            )

            overall_cond = st.slider(
                "Overall condition",
                min_value=1,
                max_value=10,
                value=5
            )

            gr_liv_area = st.number_input(
                "Above-ground living area (sq. ft.)",
                min_value=300,
                max_value=6000,
                value=1500,
                step=50
            )

            total_bsmt_sf = st.number_input(
                "Basement area (sq. ft.)",
                min_value=0,
                max_value=10000,
                value=800,
                step=50
            )

            garage_cars = st.selectbox(
                "Garage capacity (cars)",
                options=[0, 1, 2, 3, 4],
                index=2
            )

        with col2:
            full_bath = st.selectbox(
                "Full bathrooms above ground",
                options=[0, 1, 2, 3, 4],
                index=2
            )

            bedroom_abvgr = st.selectbox(
                "Bedrooms above ground",
                options=list(range(1, 9)),
                index=2
            )

            year_built = st.slider(
                "Year built",
                min_value=year_min,
                max_value=year_max,
                value=min(max(2000, year_min), year_max)
            )

            lot_area = st.number_input(
                "Lot area (sq. ft.)",
                min_value=1000,
                max_value=250000,
                value=8500,
                step=500
            )

            neighborhood = st.selectbox(
                "Neighborhood",
                options=neighborhood_options
            )

        kitchen_qual = st.selectbox(
            "Kitchen quality",
            options=kitchen_options,
            format_func=lambda x: {
                "Ex": "Excellent",
                "Gd": "Good",
                "TA": "Typical / Average",
                "Fa": "Fair",
                "Po": "Poor"
            }.get(x, x)
        )

        predict_clicked = st.form_submit_button(
            "Predict House Price",
            use_container_width=True
        )

    if predict_clicked:

        input_data = pd.DataFrame([{
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "GarageCars": garage_cars,
            "FullBath": full_bath,
            "BedroomAbvGr": bedroom_abvgr,
            "YearBuilt": year_built,
            "LotArea": lot_area,
            "Neighborhood": neighborhood,
            "KitchenQual": kitchen_qual
        }])

        try:
            predicted_price = float(model.predict(input_data)[0])

            if not np.isfinite(predicted_price):
                st.error("The model returned an invalid price.")
            else:
                st.success("Price prediction completed.")

                st.metric(
                    "Estimated historical-market sale price",
                    f"${predicted_price:,.0f}"
                )

                st.caption(
                    "The estimate is based on historical training data. "
                    "It is not a guarantee of a property's current market value."
                )

                with st.expander("Review the inputs used"):
                    st.dataframe(
                        input_data,
                        use_container_width=True
                    )

        except Exception as exc:
            st.error(f"Could not predict the house price: {exc}")


# --------------------------------------------------
# 6. BUYER RECOMMENDATION FORM
# --------------------------------------------------

with recommend_tab:

    st.subheader("Find properties matching your requirements")

    st.write(
        "Filter historical sales by budget, property features, "
        "neighborhood, and optional preferences."
    )

    with st.form("recommendation_form"):

        budget_range = st.slider(
            "Sale-price budget range ($)",
            min_value=0,
            max_value=800000,
            value=(100000, 200000),
            step=5000
        )

        preferred_neighborhoods = st.multiselect(
            "Preferred neighborhoods (leave empty for all)",
            options=neighborhood_options,
            default=[]
        )

        col1, col2 = st.columns(2)

        with col1:
            min_area = st.number_input(
                "Minimum living area (sq. ft.)",
                min_value=0,
                max_value=6000,
                value=1500,
                step=100
            )

            min_bedrooms = st.selectbox(
                "Minimum bedrooms",
                options=list(range(0, 9)),
                index=3
            )

            min_bathrooms = st.selectbox(
                "Minimum total bathroom equivalents",
                options=[0.0, 0.5, 1.0, 1.5, 2.0,
                         2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                index=4
            )

        with col2:
            min_quality = st.slider(
                "Minimum overall quality",
                min_value=1,
                max_value=10,
                value=6
            )

            require_garage = st.checkbox(
                "Require a garage"
            )

            require_fireplace = st.checkbox(
                "Require a fireplace"
            )

            exclude_review = st.checkbox(
                "Hide records flagged for review",
                value=True
            )

        sort_preference = st.selectbox(
            "Rank matching houses by",
            options=[
                "Lowest historical price",
                "Highest quality",
                "Largest living area"
            ]
        )

        find_clicked = st.form_submit_button(
            "Find Matching Houses",
            use_container_width=True
        )

    if find_clicked:

        matches = houses[
            (houses["SalePrice"] >= budget_range[0]) &
            (houses["SalePrice"] <= budget_range[1]) &
            (houses["GrLivArea"] >= min_area) &
            (houses["BedroomAbvGr"] >= min_bedrooms) &
            (houses["TotalBathrooms"] >= min_bathrooms) &
            (houses["OverallQual"] >= min_quality)
        ].copy()

        if preferred_neighborhoods:
            matches = matches[
                matches["Neighborhood"].isin(
                    preferred_neighborhoods
                )
            ]

        if require_garage:
            matches = matches[matches["GarageCars"] >= 1]

        if require_fireplace:
            matches = matches[matches["Fireplaces"] >= 1]

        if exclude_review:
            matches = matches[
                matches["Review_Status"] == "No flag under this rule"
            ]

        if sort_preference == "Lowest historical price":
            matches = matches.sort_values("SalePrice")

        elif sort_preference == "Highest quality":
            matches = matches.sort_values(
                ["OverallQual", "SalePrice"],
                ascending=[False, True]
            )

        else:
            matches = matches.sort_values(
                ["GrLivArea", "SalePrice"],
                ascending=[False, True]
            )

        st.subheader("Recommendation results")

        st.write(f"Matching historical houses: **{len(matches)}**")

        result_columns = [
            "Id",
            "Neighborhood",
            "SalePrice",
            "GrLivArea",
            "BedroomAbvGr",
            "TotalBathrooms",
            "OverallQual",
            "YearBuilt",
            "Review_Status"
        ]

        if matches.empty:
            st.warning(
                "No houses matched every selected requirement. "
                "Try expanding the budget or relaxing one of the filters."
            )
        else:
            st.dataframe(
                matches[result_columns].head(50),
                use_container_width=True,
                hide_index=True
            )

            if len(matches) > 50:
                st.info(
                    "Showing the first 50 matching houses. "
                    "Adjust your filters to narrow the results."
                )

            st.caption(
                "These are historical sale records, not current "
                "property listings. A record without a review flag "
                "is not guaranteed to be error-free."
            )

            csv_data = matches[result_columns].to_csv(index=False)

            st.download_button(
                "Download matching houses as CSV",
                data=csv_data,
                file_name="matching_houses.csv",
                mime="text/csv"
            )
