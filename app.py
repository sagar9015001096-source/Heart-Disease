import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_and_train_model(csv_file: str):
    df = pd.read_csv(csv_file)

    df_clean = df.copy().drop(['id', 'dataset'], axis=1)
    df_clean['target'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)
    df_clean = df_clean.drop('num', axis=1)

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

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    model = Pipeline([('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns.tolist(), accuracy, df_clean


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at top left, #e7effc 0, #ffffff 50%, #fdfdff 100%);
    font-family: "Segoe UI", system-ui;
}

/* Title */
.big-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #0F172A;
}
.sub-title {
    font-size: 1.05rem;
    color: #374151;
    margin-bottom: 1rem;
}

/* Card */
.card {
    background: white;
    padding: 1.3rem;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(50,50,93,.1);
    border: 1px solid rgba(0,0,0,.05);
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD TRAINED MODEL --------------------
try:
    model, feature_cols, model_accuracy, data = load_and_train_model("heart_disease_uci.csv")
except Exception as e:
    st.error(f"Error loading model or CSV file: {e}")
    st.stop()


# -------------------- RISK LABEL --------------------
def get_risk_label(prob_disease):
    if prob_disease < 0.25: return "Low"
    elif prob_disease < 0.50: return "Moderate"
    elif prob_disease < 0.75: return "High"
    return "Very High"


# -------------------- HOME PAGE --------------------
def home_page():
    st.markdown('<div class="big-title">💖 Heart Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Artificial Intelligence for clinical decision support</div>', unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        st.write("""
This interactive ML-powered platform estimates the **probability of heart disease** 
based on clinical patient details.

You can:
- Perform predictions  
- Explore dataset visualizations  
- Export results  
- Use the app for **final-year project & seminars**
""")
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 📌 Model Summary")
        st.write(f"**Accuracy:** `{model_accuracy*100:.1f}%`")
        st.write(f"**Dataset Samples:** `{len(data)}`")
        st.write(f"**Disease Cases:** `{(data['target']==1).sum()}`")
        st.write('</div>', unsafe_allow_html=True)

    st.markdown("### 🔍 Sample Dataset")
    st.dataframe(data.head(), use_container_width=True)


# -------------------- PREDICTION PAGE --------------------
def prediction_page():
    st.markdown('<div class="big-title">🩺 Predict Heart Disease</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter patient clinical details</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 100, 50)
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)
            thalch = st.number_input("Max Heart Rate", 70, 220, 150)
        with c2:
            sex = st.selectbox("Sex", ("Male", "Female"))
            cp = st.selectbox("Chest Pain Type", ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'))
            fbs = st.selectbox("Fasting Blood Sugar > 120", ('True', 'False'))
            exang = st.selectbox("Exercise Induced Angina", ('True', 'False'))

        slope = st.selectbox("ST Slope", ('upsloping', 'flat', 'downsloping'))
        restecg = st.selectbox("Resting ECG", ('lv hypertrophy', 'normal', 'st-t abnormality'))
        ca = st.slider("Major Vessels (0–3)", 0, 3, 0)
        thal = st.selectbox("Thallium Test Result", ('normal', 'fixed defect', 'reversable defect'))
        oldpeak = st.number_input("Oldpeak", 0.0, 6.2, 1.0, step=0.1)

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        input_data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                      'thalch': thalch, 'oldpeak': oldpeak, 'fbs': fbs, 'restecg': restecg,
                      'exang': exang, 'slope': slope, 'ca': ca, 'thal': thal}

        df_input = pd.DataFrame([input_data], columns=feature_cols)
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]
        prob = float(proba[1])
        risk = get_risk_label(prob)

        st.markdown("## 🧾 Result")
        if pred == 1:
            st.error(f"High Risk ({prob:.2%}) — {risk}")
        else:
            st.success(f"Low Risk ({1-prob:.2%}) — {risk}")
        st.progress(int(prob * 100))


# -------------------- DASHBOARD PAGE --------------------
def dashboard_page():
    st.markdown('<div class="big-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Dataset visual summary</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Target Distribution**")
        st.bar_chart(data["target"].value_counts().sort_index())
    with c2:
        st.write("**Average Cholesterol by Status**")
        st.bar_chart(data.groupby("target")["chol"].mean())

    st.write("**Age Distribution**")
    st.line_chart(data["age"].value_counts().sort_index())


# -------------------- ABOUT PAGE --------------------
def about_page():
    st.markdown('<div class="big-title">ℹ️ About</div>', unsafe_allow_html=True)
    st.write("""
This project demonstrates **Machine Learning + Streamlit integration** for heart
disease detection using **clinical attributes**.

✔ Final-year project ready  
✔ Clean dashboard and prediction UI  
✔ Exportable + reusable  
""")


# -------------------- TOP NAVIGATION --------------------
tabs = st.tabs(["🏠 Home", "🩺 Prediction", "📊 Dashboard", "ℹ️ About"])

with tabs[0]: home_page()
with tabs[1]: prediction_page()
with tabs[2]: dashboard_page()
with tabs[3]: about_page()
