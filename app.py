import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ================= MODEL TRAINING =================

@st.cache_resource
def load_and_train_model(csv_file):
    df = pd.read_csv(csv_file)
    df_clean = df.copy().drop(["id", "dataset"], axis=1)
    df_clean["target"] = df_clean["num"].apply(lambda x: 0 if x == 0 else 1)
    df_clean = df_clean.drop("num", axis=1)

    sex_map = {0: "Female", 1: "Male"}
    cp_map = {1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal', 4: 'asymptomatic'}
    restecg_map = {0: 'normal', 1: 'st-t abnormality', 2: 'lv hypertrophy'}
    slope_map = {1: 'upsloping', 2: 'flat', 3: 'downsloping'}
    thal_map = {3: 'normal', 6: 'fixed defect', 7: 'reversable defect'}
    fbs_map = {0: 'False', 1: 'True'}
    exang_map = {0: 'False', 1: 'True'}

    df_clean["sex"] = df_clean["sex"].map(sex_map).fillna(df_clean["sex"])
    df_clean["cp"] = df_clean["cp"].map(cp_map).fillna(df_clean["cp"])
    df_clean["restecg"] = df_clean["restecg"].map(restecg_map).fillna(df_clean["restecg"])
    df_clean["slope"] = df_clean["slope"].map(slope_map).fillna(df_clean["slope"])
    df_clean["thal"] = df_clean["thal"].map(thal_map).fillna(df_clean["thal"])
    df_clean["fbs"] = df_clean["fbs"].map(fbs_map).astype(str)
    df_clean["exang"] = df_clean["exang"].map(exang_map).astype(str)

    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_transformer, num_cols), ("cat", cat_transformer, cat_cols)]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))]
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, X.columns.tolist(), accuracy, df_clean


# ================ PAGE CONFIG =================
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ================ DARK MODE CSS =================
st.markdown("""
<style>
body, .stApp {
    background-color: #0A0A0A !important;
    color: white !important;
}

/* Titles */
.big-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
}
.sub-title {
    font-size: 1.1rem;
    color: #b9b9b9;
    margin-bottom: 1rem;
}

/* Cards */
.card, .form-card {
    background: #141414;
    border-radius: 12px;
    padding: 22px;
    border: 1px solid #2a2a2a;
    box-shadow: 0 0 25px rgba(0,0,0,0.5);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: #1b1b1b;
    border-radius: 25px;
    color: #bfbfbf;
    padding: 6px 18px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #007bff !important;
    color: #ffffff !important;
    border: 1px solid #00aaff;
    box-shadow: 0 0 12px #0099ff;
}

/* Inputs */
.stNumberInput > div, .stSelectbox > div, .stTextInput > div, .stSlider > div {
    background-color: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    color: white !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    color: white !important;
}
.stSlider > div > div > div {
    background: #00aaff !important;
}
</style>
""", unsafe_allow_html=True)


# load ML model
try:
    model, feature_cols, model_accuracy, data = load_and_train_model("heart_disease_uci.csv")
except Exception as e:
    st.error(f" CSV/MODEL ERROR: {e}")
    st.stop()


# ================ UTIL =================
def get_risk_label(prob):
    if prob < 0.25: return "Low"
    if prob < 0.5: return "Moderate"
    if prob < 0.75: return "High"
    return "Very High"


# ================ PAGES =================
def home():
    c1, c2 = st.columns([2,1])

    with c1:
        st.markdown('<div class="big-title"> Heart Disease Prediction System</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Machine learning–based clinical decision support.</div>', unsafe_allow_html=True)

        st.write("""
This tool predicts the chance of heart disease based on clinical parameters.

✔ AI-powered prediction  
✔ Interactive dashboard  
✔ Final-year project ready  
        """)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(" ML Model Overall Stats")
        st.write(f"**Accuracy:** `{model_accuracy * 100:.2f}%`")
        total = len(data)
        st.write(f"**Records:** `{total}`")
        st.write(f"**Heart Disease Cases:** `{(data['target']==1).sum()}`")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔍 Sample Dataset")
    st.dataframe(data.head(), use_container_width=True)


def prediction():
    st.markdown('<div class="big-title"> Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter patient clinical values.</div>', unsafe_allow_html=True)

    with st.form("form"):
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 100, 50)
            trestbps = st.number_input("Resting BP", 80, 200, 120)
            chol = st.number_input("Cholesterol", 100, 600, 240)
            thalch = st.number_input("Max Heart Rate", 70, 220, 150)
        with c2:
            sex = st.selectbox("Sex", ("Male", "Female"))
            cp = st.selectbox("Chest Pain", ('typical angina','atypical angina','non-anginal','asymptomatic'))
            fbs = st.selectbox("Fasting Blood Sugar > 120", ('True','False'))
            exang = st.selectbox("Exercise Induced Angina", ('True','False'))

        slope = st.selectbox("Slope", ('upsloping','flat','downsloping'))
        restecg = st.selectbox("Resting ECG", ('lv hypertrophy','normal','st-t abnormality'))
        ca = st.slider("Major Vessels (0–3)", 0, 3, 0)
        thal = st.selectbox("Thallium Test Result", ('normal','fixed defect','reversable defect'))
        oldpeak = st.number_input("ST Depression", 0.0, 6.2, 1.0)

        submit = st.form_submit_button("Predict 🔍")
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        input_df = pd.DataFrame([{
            "age": age,"sex": sex,"cp": cp,"trestbps": trestbps,"chol": chol,
            "thalch": thalch,"oldpeak": oldpeak,"fbs": fbs,"restecg": restecg,
            "exang": exang,"slope": slope,"ca": ca,"thal": thal
        }], columns=feature_cols)

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        label = get_risk_label(proba)

        if pred == 1:
            st.error(f"⚠ HIGH RISK ({proba:.2%}) — {label}")
        else:
            st.success(f"✔ LOW RISK ({1-proba:.2%}) — {label}")

        st.progress(int(proba * 100))


def dashboard():
    st.markdown('<div class="big-title"> Dashboard</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Disease vs No Disease**")
        st.bar_chart(data["target"].value_counts().sort_index())

    with col2:
        st.write("**Avg Cholesterol by Status**")
        st.bar_chart(data.groupby("target")["chol"].mean())

    st.write("**Age Distribution**")
    st.line_chart(data["age"].value_counts().sort_index())


def about():
    st.markdown('<div class="big-title"> About</div>', unsafe_allow_html=True)
    st.write("""
Heart Disease Prediction using:
- Python, Pandas, NumPy
- Scikit-Learn (Random Forest)
- Streamlit Web UI

✔ Ideal for final-year academic project  
✔ Includes ML + dashboard + prediction UI  

⚠ For **educational use only** — not a medical device.
""")


# ================= TABS NAVIGATION =================
tabs = st.tabs([" Home", " Prediction", " Dashboard", " About"])
with tabs[0]: home()
with tabs[1]: prediction()
with tabs[2]: dashboard()
with tabs[3]: about()


