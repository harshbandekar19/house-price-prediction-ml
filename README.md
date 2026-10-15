# House Price Prediction and Recommendation System

A machine learning project that analyzes historical house sales in Ames, Iowa, predicts a property's historical sale price, and recommends matching historical properties based on buyer preferences.

> **Important:** Predictions are estimates based on historical data. Recommendations are matches from the dataset, not current real-estate listings or professional appraisals.

## Project objectives

- Explore and understand housing data using exploratory data analysis (EDA).
- Clean missing values while distinguishing absent features (such as no garage) from unknown measurements.
- Engineer property features such as house age, time since remodeling, total house area, outdoor area, and total bathrooms.
- Compare regression models and evaluate their prediction errors.
- Explain model behavior using feature importance, permutation importance, SHAP, and residual analysis.
- Provide an interactive interface for price prediction and historical-house filtering.

## Dataset

- Original file: `Data/data.csv`
- Shape: 1,460 rows and 81 original columns
- Target: `SalePrice`
- `Id` is a record identifier and is excluded from model features.
- `Data/house prices.docx` contains the supplied project brief/documentation.

The data represents historical house sales in Ames, Iowa, USA.

## Project workflow

1. Load the data and inspect shape, data types, descriptive statistics, and missing values.
2. Perform univariate and bivariate EDA, including distributions, correlation, and category-level price comparisons.
3. Handle missing values according to their meaning and inspect unusual observations.
4. Create engineered features from existing columns.
5. Split the data into training and testing sets for model evaluation.
6. Build preprocessing pipelines using imputation, scaling for numerical features where appropriate, and one-hot encoding for categorical features.
7. Compare Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting models, including a log-target experiment.
8. Use five-fold cross-validation and `RandomizedSearchCV` to tune Gradient Boosting.
9. Evaluate with MAE, RMSE, and R²; review residuals and errors across price ranges.
10. Build a Streamlit interface for prediction and rule-based filtering/ranking of historical houses.

## Model results

The notebook's **full-feature tuned Gradient Boosting model** achieved the following exploratory results:

| Evaluation | Result |
|---|---:|
| Mean 5-fold CV MAE | $14,882.55 |
| Test MAE | $15,730.70 |
| Test RMSE | $24,830.93 |
| Test R² | 0.9196 |

The Streamlit application uses a **separate model with 11 user-friendly input features**, to make the form easier to complete. Its reported five-fold cross-validation results were:

| Evaluation | Result |
|---|---:|
| CV MAE | $17,450.12 |
| CV RMSE | $31,636.52 |
| CV R² | 0.8079 |

These results belong to different models and should not be compared as though they were the same fitted model. The test set was inspected during model development, so the notebook's test metrics are exploratory rather than a final untouched-test estimate.

## Streamlit application

The app has two main functions:

- **Predict House Price:** Estimates a historical sale price from 11 property characteristics.
- **Find Matching Houses:** Filters and ranks historical records using preferences such as budget, living area, bedrooms, bathrooms, quality, and neighborhood.

The recommendation feature is rule-based filtering and sorting; it is not a separate machine learning model.

## Repository files

- `PRCP_1020_HousePricePred.ipynb` — analysis, feature engineering, model comparison, tuning, evaluation, explainability, and recommendations.
- `app.py` — Streamlit user interface.
- `house_price_model.joblib` — saved model used by the app.
- `house_recommendation_data.csv` — records used for historical-house recommendations.
- `Data/data.csv` — original dataset.
- `Data/house prices.docx` — supplied project brief/documentation.
- `requirements.txt` — Python package dependencies.

Keep `app.py`, `house_price_model.joblib`, and `house_recommendation_data.csv` in the same project folder.

## Run the application locally

Use a Python environment compatible with the package versions below. The serialized model was created with scikit-learn 1.6.1, so use that version to avoid model-compatibility issues.

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Launch Streamlit:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed in the terminal, usually `http://localhost:8501`.

The notebook was developed in Google Colab and may contain Google Drive paths; update its data path if running it outside Colab.

## Limitations and future improvements

- The dataset reflects a specific location and historical period; results may not generalize to other areas or current market conditions.
- Expensive properties had larger prediction errors in the test analysis.
- An untouched final test set and newer external validation data would strengthen future evaluation.
- The application is an educational prototype, not a valuation service or property marketplace.
