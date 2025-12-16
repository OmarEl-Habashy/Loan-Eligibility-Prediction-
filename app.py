import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

# ----------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "new_all_cleaned_train.csv"
IMAGES_DIR = BASE_DIR / "data" / "images"

# The exact feature order used for training (Person B)
FEATURE_COLUMNS = [
    "Loan_Amount_Term",
    "Credit_History",
    "Total_Income",
    "LoanAmount_Log",
    "Total_Income_Log",
    "Loan_to_Income_Ratio",
    "Gender_Male",
    "Married_Yes",
    "Dependents_1",
    "Dependents_2",
    "Dependents_3+",
    "Education_Not Graduate",
    "Self_Employed_Yes",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
]


# ----------------------------------------------------------------------------------
# Load model, scaler, and data
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_processed_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception:
        return None


@st.cache_data
def load_eda_image(name: str):
    path = IMAGES_DIR / name
    if path.exists():
        return Image.open(path)
    return None


# ----------------------------------------------------------------------------------
# Feature engineering (Person A's logic reproduced)
# ----------------------------------------------------------------------------------
def build_feature_row(
    gender: str,
    married: str,
    dependents: str,
    education: str,
    self_employed: str,
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_amount_term: float,
    credit_history: float,
    property_area: str,
) -> pd.DataFrame:
    """
    Recreate EXACTLY the engineered features that were used for training:

    - Total_Income = ApplicantIncome + CoapplicantIncome
    - LoanAmount_Log = log(LoanAmount)
    - Total_Income_Log = log(Total_Income)
    - Loan_to_Income_Ratio = LoanAmount / Total_Income
    - One-hot encoding for:
        Gender, Married, Dependents, Education, Self_Employed, Property_Area
      matching new_all_cleaned_train.csv
    """

    # 1. Basic numeric features
    total_income = applicant_income + coapplicant_income

    # Avoid log(0)
    safe_loan_amount = max(loan_amount, 1.0)
    safe_total_income = max(total_income, 1.0)

    loan_amount_log = float(np.log(safe_loan_amount))
    total_income_log = float(np.log(safe_total_income))

    loan_to_income_ratio = loan_amount / safe_total_income

    # 2. One-hot encodings
    gender_male = 1 if gender == "Male" else 0
    married_yes = 1 if married == "Yes" else 0

    dependents_str = str(dependents)
    dep_1 = 1 if dependents_str == "1" else 0
    dep_2 = 1 if dependents_str == "2" else 0
    dep_3plus = 1 if dependents_str in ["3", "3+", "4", "4+"] else 0

    edu_not_grad = 1 if education == "Not Graduate" else 0
    self_emp_yes = 1 if self_employed == "Yes" else 0

    prop_semiurban = 1 if property_area == "Semiurban" else 0
    prop_urban = 1 if property_area == "Urban" else 0

    row_dict = {
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Total_Income": total_income,
        "LoanAmount_Log": loan_amount_log,
        "Total_Income_Log": total_income_log,
        "Loan_to_Income_Ratio": loan_to_income_ratio,
        "Gender_Male": gender_male,
        "Married_Yes": married_yes,
        "Dependents_1": dep_1,
        "Dependents_2": dep_2,
        "Dependents_3+": dep_3plus,
        "Education_Not Graduate": edu_not_grad,
        "Self_Employed_Yes": self_emp_yes,
        "Property_Area_Semiurban": prop_semiurban,
        "Property_Area_Urban": prop_urban,
    }

    df_features = pd.DataFrame([row_dict], columns=FEATURE_COLUMNS)
    return df_features


# ----------------------------------------------------------------------------------
# Prediction helper (Person B's model + scaler)
# ----------------------------------------------------------------------------------
def predict_loan_eligibility(input_df: pd.DataFrame):
    model, scaler = load_model_and_scaler()

    X_scaled = scaler.transform(input_df)

    # Predicted class (0 = Rejected, 1 = Approved)
    y_pred = model.predict(X_scaled)[0]

    # Probability of approval, if available
    proba = None
    try:
        proba = model.predict_proba(X_scaled)[0][1]
    except Exception:
        pass

    return int(y_pred), proba


# ----------------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Loan Eligibility Prediction Dashboard",
        page_icon="💳",
        layout="wide",
    )

    st.title("💳 Loan Eligibility Prediction Dashboard")
    st.caption(
        "Person A → Data Cleaning & Feature Engineering · "
        "Person B → Model Training · "
        "Person C → This Interactive App"
    )

    df = load_processed_data()

    # Top-level metrics row (uses Person A+B work)
    col_m1, col_m2, col_m3 = st.columns(3)

    if df is not None and "Loan_Status" in df.columns:
        total_rows = len(df)
        approve_rate = df["Loan_Status"].mean() * 100  # assuming 1 = approved
        avg_income = df["Total_Income"].mean()

        col_m1.metric("Total Historical Applicants", f"{total_rows}")
        col_m2.metric("Historical Approval Rate", f"{approve_rate:.1f}%")
        col_m3.metric("Average Total Income", f"{avg_income:,.0f}")
    else:
        col_m1.metric("Total Historical Applicants", "N/A")
        col_m2.metric("Historical Approval Rate", "N/A")
        col_m3.metric("Average Total Income", "N/A")

    st.markdown("---")

    # Tabs: Prediction | Dataset Insights | About
    tab_pred, tab_insights, tab_about = st.tabs(
        ["🔮 Prediction", "📊 Dataset Insights", "ℹ️ About the Project"]
    )

    # ----------------------------------------------------------------------
    # TAB 1 – Prediction
    # ----------------------------------------------------------------------
    with tab_pred:
        st.subheader("🔮 Predict a New Applicant's Loan Eligibility")

        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            st.markdown("### Applicant Information")

            with st.form("prediction_form"):
                c1, c2 = st.columns(2)
                with c1:
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    married = st.selectbox("Married", ["Yes", "No"])
                    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
                    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
                with c2:
                    applicant_income = st.number_input(
                        "Applicant Income",
                        min_value=0.0,
                        value=5000.0,
                        step=500.0,
                        help="Monthly income of the main applicant.",
                    )
                    coapplicant_income = st.number_input(
                        "Co-Applicant Income",
                        min_value=0.0,
                        value=0.0,
                        step=500.0,
                        help="Monthly income of the co-applicant (if any).",
                    )
                    loan_amount = st.number_input(
                        "Loan Amount",
                        min_value=1.0,
                        value=128.0,
                        step=1.0,
                        help="Loan amount in thousands (as in the dataset).",
                    )
                    loan_amount_term = st.number_input(
                        "Loan Term (in days)",
                        min_value=1.0,
                        value=360.0,
                        step=12.0,
                    )
                    credit_history = st.selectbox(
                        "Credit History",
                        options=[1.0, 0.0],
                        format_func=lambda x: "Meets guidelines (1.0)"
                        if x == 1.0
                        else "Does NOT meet guidelines (0.0)",
                    )

                property_area = st.selectbox(
                    "Property Area",
                    ["Urban", "Semiurban", "Rural"],
                )

                submitted = st.form_submit_button("Predict Eligibility")

            if submitted:
                input_df = build_feature_row(
                    gender=gender,
                    married=married,
                    dependents=dependents,
                    education=education,
                    self_employed=self_employed,
                    applicant_income=applicant_income,
                    coapplicant_income=coapplicant_income,
                    loan_amount=loan_amount,
                    loan_amount_term=loan_amount_term,
                    credit_history=credit_history,
                    property_area=property_area,
                )

                pred_class, proba = predict_loan_eligibility(input_df)

                with right_col:
                    st.markdown("### Result")

                    if pred_class == 1:
                        st.success("✅ **Predicted Loan Status: APPROVED**")
                    else:
                        st.error("❌ **Predicted Loan Status: REJECTED**")

                    if proba is not None:
                        st.metric(
                            "Model Confidence (Approval Probability)",
                            f"{proba:.1%}",
                        )

                    st.markdown("#### Model Input Features")
                    st.caption("These are the engineered features sent to the model (Person A's work).")
                    st.dataframe(input_df, use_container_width=True)

        st.info(
            "This tab combines Person A's engineered features and Person B's trained model "
            "into a single user-friendly interface built by Person C."
        )

    # ----------------------------------------------------------------------
    # TAB 2 – Dataset Insights (uses Person A & B EDA work: images)
    # ----------------------------------------------------------------------
    with tab_insights:
        st.subheader("📊 Dataset Insights (from EDA)")

        st.write(
            "These visualizations summarize the historical loan dataset used by "
            "Person A (feature engineering) and Person B (model training)."
        )

        img1 = load_eda_image("income_distribution.png")
        img2 = load_eda_image("credit_history_impact.png")
        img3 = load_eda_image("correlation_heatmap.png")

        if img1 or img2:
            c1, c2 = st.columns(2)
            with c1:
                if img1:
                    st.image(img1, caption="Income Distribution", use_container_width=True)
            with c2:
                if img2:
                    st.image(img2, caption="Credit History Impact on Approval", use_container_width=True)

        if img3:
            st.image(img3, caption="Feature Correlation Heatmap", use_container_width=True)

        if df is not None:
            st.markdown("#### Sample of Processed Dataset")
            st.dataframe(df.head(), use_container_width=True)

    # ----------------------------------------------------------------------
    # TAB 3 – About the Project (good for viva)
    # ----------------------------------------------------------------------
    with tab_about:
        st.subheader("ℹ️ About This Loan Eligibility Project")

        st.markdown(
            """
            **Person A – Data & Features**
            - Collected the raw Kaggle loan dataset.
            - Cleaned missing values and handled categorical variables.
            - Engineered key features such as:
              - `Total_Income = ApplicantIncome + CoapplicantIncome`
              - `LoanAmount_Log`, `Total_Income_Log`
              - `Loan_to_Income_Ratio`
              - Dummy variables: `Gender_Male`, `Married_Yes`, `Dependents_1`, `Dependents_2`,
                `Dependents_3+`, `Education_Not Graduate`, `Self_Employed_Yes`,
                `Property_Area_Semiurban`, `Property_Area_Urban`.
            - Saved the final processed dataset as `data/raw/new_all_cleaned_train.csv`.

            **Person B – Modeling**
            - Used the processed dataset from Person A.
            - Split the data into features (X) and target (`Loan_Status_NUM`).
            - Applied `StandardScaler` to the numerical feature set.
            - Trained and evaluated multiple models, selected **GaussianNB** as the best.
            - Saved:
              - `loan_model.pkl` → trained Gaussian Naive Bayes model.
              - `scaler.pkl` → fitted StandardScaler used in training.

            **Person C – Deployment (this app)**
            - Implemented this Streamlit dashboard (`app.py`).
            - Reproduces Person A's feature engineering for any new user input.
            - Uses Person B's `scaler.pkl` and `loan_model.pkl` to make predictions.
            - Presents results as:
              - Loan Approved / Rejected.
              - Model confidence (approval probability).
              - The exact engineered features vector sent to the model.
            """
        )

        st.success(
            "In the viva you can clearly explain the pipeline: "
            "Person A (data), Person B (model), Person C (app + inference)."
        )


if __name__ == "__main__":
    main()
