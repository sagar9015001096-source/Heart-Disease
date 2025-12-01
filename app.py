import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================
# 1. MODEL LOADING / TRAINING
# =========================

@st.cache_resource
def load_and_train_model(csv_file: str):
    df = pd.read_csv(csv_file)

    # Drop unused columns & create binary target
    df_clean = df.copy().drop(['id', 'dataset'], axis=1)
    df_clean['target'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)
    df_clean = df_clean.drop('num', axis=1)

    # Map numeric codes -> human-readable labels
    sex_map = {0: "Female", 1: "Male"}
    cp_map = {1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal', 4: 'asymptomatic'}
    restecg_map = {0: 'normal', 1: 'st-t abnormality', 2: 'lv hypertrophy'}
    slope_map = {1: 'upsloping', 2: 'flat', 3: 'downsloping'}
    thal_map = {3: 'normal', 6: 'fixed defect', 7: 'reversable defect'}
    fbs_map = {0: 'False', 1: 'True'}
    exang_map = {0: 'False', 1: 'True'}

    df_clean['sex'] = df_clean['sex'].map(sex_map).fillna(df_clean['sex'])
    df_clean['cp'] = df_clean['cp'].map(cp_map).fillna(df_clean['cp'])
    df_clean['restecg'] = df_clean['restecg'].map(restecg_map).fillna(df_clean['restecg'])
    df_clean['slope'] = df_clean['slope'].map(slope_map).fillna(df_clean['slope'])
    df_clean['thal'] = df_clean['thal'].map(thal_map).fillna(df_clean['thal'])
    df_clean['fbs'] = df_clean['fbs'].map(fbs_map).astype(str)
    df_clean['exang'] = df_clean['exang'].map(exang_map).astype(str)

    # Features / target
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']

    numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    # Pipelines
    num_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, X.columns.tolist(), accuracy, df_clean


# =========================
# 2. BASIC CONFIG & STYLING
# =========================

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Global CSS
st.markdown(
    """
<style>
/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* App background */
.stApp {
    background: radial-gradient(circle at top left, #edf2ff 0, #ffffff 55%, #f9fbff 100%);
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Big titles */
.big-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #0f172a;
}
.sub-title {
    font-size: 1rem;
    color: #4b5563;
    margin-bottom: 1.2rem;
}

/* Cards */
.card, .form-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.09);
    border: 1px solid rgba(148, 163, 184, 0.35);
}
.form-card {
    margin-top: 0.8rem;
}

/* Inputs – force nice light theme */
.stNumberInput > div, 
.stTextInput > div,
.stSelectbox > div,
.stSlider > div {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
}

/* Inner text of select */
.stSelectbox div[data-baseweb="select"] > div {
    color: #111827 !important;
}

/* Slider bar color */
.stSlider > div > div > div {
    background: #2563eb !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    background-color: #e5e7eb;
}
.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
}

/* Labels */
.stSlider label, .stNumberInput label, .stSelectbox label {
    font-weight: 500 !important;
    color: #111827 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Load model
try:
    model, feature_cols, model_accuracy, data = load_and_train_model("heart_disease_uci.csv")
except Exception as e:
    st.error(f"❌ Error loading model or CSV file: {e}")
    st.stop()


# =========================
# 3. HELPERS
# =========================

def get_risk_label(prob_disease: float) -> str:
    if prob_disease < 0.25:
        return "Low"
    elif prob_disease < 0.50:
        return "Moderate"
    elif prob_disease < 0.75:
        return "High"
    return "Very High"


# =========================
# 4. PAGE CONTENT
# =========================

def home_page():
    left, right = st.columns([2, 1])

    with left:
        st.markdown(
            '<div class="big-title">💖 Heart Disease Prediction System</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-title">Machine learning–based clinical decision support.</div>',
            unsafe_allow_html=True,
        )
        st.write(
            """
This web application estimates the **probability of heart disease** using 
standard clinical parameters.

You can:
- Enter patient data and get a risk estimation  
- Explore dataset statistics on the dashboard  
- Use this as a polished **final-year project** demonstration  
            """
        )

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📌 Model Overview")
        st.write(f"**Validation Accuracy:** `{model_accuracy * 100:.1f}%`")
        total = len(data)
        disease_cases = int((data["target"] == 1).sum())
        st.write(f"**Total Records:** `{total}`")
        st.write(f"**Heart Disease Cases:** `{disease_cases}`")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔍 Sample of Dataset")
    st.dataframe(data.head(), use_container_width=True)


def prediction_page():
    st.markdown(
        '<div class="big-title">🩺 Predict Heart Disease</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Enter patient clinical details below.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age (years)", 18, 100, 50)
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 240)
            thalch = st.number_input("Max Heart Rate Achieved", 70, 220, 150)

        with col2:
            sex = st.selectbox("Sex", ("Male", "Female"))
            cp = st.selectbox(
                "Chest Pain Type",
                ("typical angina", "atypical angina", "non-anginal", "asymptomatic"),
            )
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ("True", "False"))
            exang = st.selectbox("Exercise Induced Angina", ("True", "False"))

        col3, col4, col5 = st.columns(3)
        with col3:
            restecg = st.selectbox(
                "Resting ECG",
                ("lv hypertrophy", "normal", "st-t abnormality"),
            )
        with col4:
            slope = st.selectbox(
                "Slope of ST Segment",
                ("upsloping", "flat", "downsloping"),
            )
        with col5:
            ca = st.slider("Number of Major Vessels (0–3)", 0, 3, 0)
            thal = st.selectbox(
                "Thallium Test Result",
                ("normal", "fixed defect", "reversable defect"),
            )

        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            0.0,
            6.2,
            1.0,
            step=0.1,
        )

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        # Prepare input in same order as training
        input_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "thalch": thalch,
            "oldpeak": oldpeak,
            "fbs": fbs,
            "restecg": restecg,
            "exang": exang,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        input_df = pd.DataFrame([input_data], columns=feature_cols)

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        prob_disease = float(proba[1])
        prob_no = float(proba[0])
        risk_label = get_risk_label(prob_disease)

        st.markdown("### 🧾 Result")
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if pred == 1:
                st.error(f"High Risk of Heart Disease ({prob_disease:.2%})")
            else:
                st.success(f"Low Risk of Heart Disease ({prob_no:.2%})")

            st.write(f"**Risk Category:** {risk_label}")
            st.caption(
                "⚠️ This is a model prediction and **not** a medical diagnosis. "
                "Always consult a healthcare professional."
            )

        with res_col2:
            st.write("**Risk Meter**")
            st.progress(int(prob_disease * 100))


def dashboard_page():
    st.markdown(
        '<div class="big-title">📊 Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Visual summary of the dataset.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Target Distribution (0 = No Disease, 1 = Disease)**")
        st.bar_chart(data["target"].value_counts().sort_index())
    with col2:
        st.markdown("**Average Cholesterol by Heart Disease Status**")
        st.bar_chart(data.groupby("target")["chol"].mean())

    st.markdown("**Age Distribution**")
    st.line_chart(data["age"].value_counts().sort_index())


def about_page():
    st.markdown(
        '<div class="big-title">ℹ️ About</div>',
        unsafe_allow_html=True,
    )
    st.write(
        """
### Project Description
This **Heart Disease Prediction System** is built using:

- Python, Pandas, Scikit-learn  
- Random Forest Classifier  
- Streamlit for the web interface  

It is suitable as a **final-year academic project**, demonstrating:
- Data preprocessing and feature engineering  
- Model training and evaluation  
- Interactive web-based prediction UI  
- Basic data visualization

> **Disclaimer:** This tool is for educational purposes only and must not be used as a
substitute for professional medical advice, diagnosis, or treatment.
        """
    )


# =========================
# 5. TOP NAVIGATION VIA TABS
# =========================

tabs = st.tabs(["🏠 Home", "🩺 Prediction", "📊 Dashboard", "ℹ️ About"])

with tabs[0]:
    home_page()

with tabs[1]:
    prediction_page()

with tabs[2]:
    dashboard_page()

with tabs[3]:
    about_page()

