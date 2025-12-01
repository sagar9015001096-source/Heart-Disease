import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -------------------- 1) MODEL LOADING & TRAINING --------------------

@st.cache_resource
def load_and_train_model(csv_file: str):
    df = pd.read_csv(csv_file)

    # Clean + target
    df_clean = df.copy().drop(['id', 'dataset'], axis=1)
    df_clean['target'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)
    df_clean = df_clean.drop('num', axis=1)

    # Map codes -> labels (so they match the form options)
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

    X = df_clean.drop('target', axis=1)
    y = df_clean['target']

    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

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
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
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


# -------------------- 2) BASIC CONFIG & THEME --------------------

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Custom CSS for nicer UI
custom_css = """
<style>
/* Remove Streamlit default padding and header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
    background: radial-gradient(circle at top left, #f0f4ff 0, #ffffff 50%, #f9f9ff 100%);
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Title */
.big-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-size: 1rem;
    color: #4B5563;
    margin-bottom: 1.2rem;
}

/* Card style */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.25);
}

/* Metrics row */
.metric-label {
    font-size: 0.85rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #111827;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.4rem 1rem;
    border-radius: 999px;
    background-color: #E5E7EB;
}
.stTabs [aria-selected="true"] {
    background-color: #1D4ED8 !important;
    color: white !important;
}

/* Form labels */
.stSlider label, .stNumberInput label, .stSelectbox label {
    font-weight: 500 !important;
    color: #111827 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# load model
try:
