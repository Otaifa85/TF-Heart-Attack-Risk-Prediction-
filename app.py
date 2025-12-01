import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO

# Optional extras
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# --------------------------------------------------------
# 1. LOAD DATASET (embedded heart attack dataset)
# --------------------------------------------------------

DATA_PATH = "Heart Attack Data Set.csv"

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load dataset from '{DATA_PATH}': {e}")
    st.stop()

X = df.drop(columns=["target"])
y = df["target"]

# --------------------------------------------------------
# 2. TRAIN/VAL/TEST SPLIT (60/20/20)
# --------------------------------------------------------

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,   # 0.25 of remaining 0.8 ⇒ 0.2
    random_state=42,
    stratify=y_temp
)


# --------------------------------------------------------
# 3. PREPROCESSOR (MUST MATCH MODEL)
# --------------------------------------------------------

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

preprocessor.fit(X_train)  # only on train


def to_dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else arr


X_train_proc = to_dense(preprocessor.transform(X_train))
X_val_proc = to_dense(preprocessor.transform(X_val))
X_test_proc = to_dense(preprocessor.transform(X_test))


# --------------------------------------------------------
# 4. ACTIVE MODEL MANAGEMENT (no globals)
# --------------------------------------------------------

MODEL_DEFAULT_PATH = "final_model.h5"


def load_model_from_path(path: str):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model from {path}: {e}")
        return None


def get_active_model():
    # Cached model in session
    if "active_model" in st.session_state:
        return st.session_state["active_model"]

    # Try to load default trained model
    try:
        model = tf.keras.models.load_model(MODEL_DEFAULT_PATH)
        st.session_state["active_model"] = model
        st.session_state["active_model_path"] = MODEL_DEFAULT_PATH
        return model
    except Exception:
        st.warning("⚠ No default model found. Train a model in '🧪 Train Model' tab.")
        return None


def set_active_model(model, path: str | None = None):
    st.session_state["active_model"] = model
    if path is not None:
        st.session_state["active_model_path"] = path


# --------------------------------------------------------
# 5. HELPER FUNCTIONS: GAUGE, SHAP, PDF, RECOMMENDATION
# --------------------------------------------------------

def plot_gauge(prob: float):
    value_pct = prob * 100.0

    if prob < 0.33:
        gauge_color = "green"
    elif prob < 0.66:
        gauge_color = "orange"
    else:
        gauge_color = "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_pct,
            number={"suffix": "%"},
            title={"text": "Predicted Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 33], "color": "#14532d"},
                    {"range": [33, 66], "color": "#854d0e"},
                    {"range": [66, 100], "color": "#7f1d1d"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=0))
    return fig


def wrapped_model(X_df: pd.DataFrame):
    """
    Wrapper for SHAP: takes raw features, preprocess, then uses active model.
    """
    model = get_active_model()
    if model is None:
        return np.zeros(len(X_df))
    X_proc = preprocessor.transform(X_df)
    X_proc = to_dense(X_proc)
    return model.predict(X_proc).flatten()


def compute_shap_for_instance(input_df: pd.DataFrame):
    if not SHAP_AVAILABLE:
        return None, "SHAP not installed. Run: pip install shap"

    try:
        bg = X_train.sample(min(100, len(X_train)), random_state=0)
        explainer = shap.Explainer(wrapped_model, bg, feature_names=ALL_FEATURES)
        shap_values = explainer(input_df)
        vals = shap_values.values[0]
        feature_names = list(shap_values.feature_names) if shap_values.feature_names is not None else ALL_FEATURES

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": vals
        }).sort_values("shap_value", key=np.abs, ascending=False)

        return shap_df, None
    except Exception as e:
        return None, f"Could not compute SHAP values: {e}"


def get_doctor_recommendation(prob: float):
    if prob < 0.30:
        risk_level = "Low Risk"
        recommendation = (
            "Patient appears within a normal cardiovascular risk range. "
            "Recommend maintaining a healthy lifestyle: balanced diet, "
            "regular physical activity, non-smoking, and routine annual check-ups."
        )
    elif prob < 0.60:
        risk_level = "Moderate Risk"
        recommendation = (
            "Patient shows moderate risk. Recommend lifestyle modification: "
            "reduce sodium and saturated fat intake, achieve/maintain healthy weight, "
            "engage in 3–4 sessions of moderate exercise per week, "
            "and schedule follow-up evaluation with a family physician or cardiologist."
        )
    else:
        risk_level = "High Risk"
        recommendation = (
            "Patient is at high estimated risk. Recommend urgent clinical assessment, "
            "including ECG, laboratory investigations (lipid profile, blood glucose), "
            "and possible cardiology referral. Immediate lifestyle intervention is strongly advised."
        )

    return risk_level, recommendation


def generate_pdf_report(patient_data, prob: float, pred_label: str,
                        risk_level: str, recommendation: str,
                        shap_df: pd.DataFrame | None):
    if not REPORTLAB_AVAILABLE:
        return None, "reportlab not installed. Run: pip install reportlab"

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Heart Attack Risk Assessment Report")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(50, y, "Facility: Fanshawe College – AI Department")
    y -= 15
    c.drawString(50, y, "Department: Applied Machine Learning & Predictive Analytics Unit")
    y -= 25

    # Risk summary
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "1. Risk Summary")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(60, y, f"Predicted Risk Category (Binary Model): {pred_label}")
    y -= 15
    c.drawString(60, y, f"Estimated Probability of Heart Attack: {prob:.2%}")
    y -= 15
    c.drawString(60, y, f"Clinical Risk Level: {risk_level}")
    y -= 25

    # Doctor recommendation
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "2. Doctor Recommendation (Model-Based)")
    y -= 18
    c.setFont("Helvetica", 11)

    for line in recommendation.split(". "):
        if line.strip():
            c.drawString(60, y, "- " + line.strip().rstrip(".") + ".")
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)

    y -= 10

    # Patient info
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "3. Patient Clinical Profile (Input Features)")
    y -= 18
    c.setFont("Helvetica", 11)
    for key, val in patient_data.items():
        c.drawString(60, y, f"{key}: {val}")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica-Bold", 13)
            c.drawString(50, y, "3. Patient Clinical Profile (cont.)")
            y -= 18
            c.setFont("Helvetica", 11)

    # SHAP explanation
    if shap_df is not None and not shap_df.empty:
        if y < 140:
            c.showPage()
            y = height - 50
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "4. Key Features Influencing the Model Prediction")
        y -= 18
        c.setFont("Helvetica", 11)
        top_n = shap_df.head(5)
        for _, row in top_n.iterrows():
            direction = "increasing" if row["shap_value"] > 0 else "decreasing"
            c.drawString(
                60, y,
                f"- {row['feature']}: {direction} estimated risk (SHAP = {row['shap_value']:.3f})"
            )
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)

    # Footer
    if y < 120:
        c.showPage()
        y = height - 50

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "5. Report Generation")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(60, y, "Generated by: Khaled Otaifa (AI Student – Fanshawe College)")
    y -= 14
    c.drawString(60, y, "Powered by: AI Model (Heart Attack Risk Predictor v1.0)")
    y -= 25

    c.setFont("Helvetica", 9)
    c.drawString(
        50, 40,
        "This report is produced by a machine learning decision-support tool and "
        "is not a substitute for professional medical judgment."
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue(), None


# --------------------------------------------------------
# 6. ADVANCED TRAINING TAB (Option B2)
# --------------------------------------------------------

def render_training_tab_b2():
    st.header("🧠 Train a New Model (Advanced B2 Mode)")

    st.write(
        "This training mode uses the same embedded heart attack dataset that powers the app.\n"
        "You can customize architecture, regularization, optimizer, learning rate scheduler, and more."
    )

    # ------------------------
    # MODEL ARCHITECTURE
    # ------------------------
    with st.expander("📐 Neural Network Architecture", expanded=True):
        num_layers = st.selectbox("Number of hidden layers", [1, 2, 3, 4], index=1)

        activation_choice = st.selectbox(
            "Activation Function (for all layers)",
            ["ReLU", "LeakyReLU", "Tanh", "Sigmoid"],
            index=0
        )

        layer_configs = []
        for i in range(num_layers):
            st.markdown(f"**Layer {i+1} Configuration**")
            c1, c2 = st.columns(2)
            with c1:
                units = st.slider(
                    f"Units in Layer {i+1}",
                    8, 256,
                    32 if i == 0 else 16,
                    step=8,
                    key=f"units_{i}"
                )
            with c2:
                dropout = st.slider(
                    f"Dropout for Layer {i+1}",
                    0.0, 0.6,
                    0.2,
                    0.05,
                    key=f"dropout_{i}"
                )
            layer_configs.append((units, dropout))
            st.markdown("---")

    # ------------------------
    # REGULARIZATION
    # ------------------------
    with st.expander("🧬 Regularization", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            l1_value = st.selectbox("L1 Regularization", [0.0, 0.0001, 0.001, 0.01], index=0)
        with col_b:
            l2_value = st.selectbox("L2 Regularization", [0.0, 0.0001, 0.001, 0.01], index=2)

    # ------------------------
    # TRAINING SETTINGS
    # ------------------------
    with st.expander("⚙️ Training Settings", expanded=True):
        col_c, col_d = st.columns(2)
        with col_c:
            epochs = st.slider("Epochs", 10, 300, 50, step=10)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        with col_d:
            optimizer_name = st.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"], index=0)
            learning_rate = st.selectbox("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], index=2)

    # ------------------------
    # LR SCHEDULER
    # ------------------------
    with st.expander("📉 Learning Rate Scheduler (ReduceLROnPlateau)", expanded=False):
        use_scheduler = st.checkbox("Enable ReduceLROnPlateau", value=False)
        if use_scheduler:
            scheduler_factor = st.slider("LR Reduction Factor", 0.1, 0.9, 0.5)
            scheduler_patience = st.slider("Scheduler Patience (epochs)", 1, 10, 3)
        else:
            scheduler_factor = None
            scheduler_patience = None

    # ------------------------
    # ADVANCED OPTIONS
    # ------------------------
    with st.expander("🔧 Advanced Options", expanded=False):
        val_split = st.slider("Validation Split (from training set)", 0.1, 0.4, 0.2, step=0.05)
        early_patience = st.slider("Early Stopping Patience", 3, 20, 5)

    # ------------------------
    # BUILD MODEL
    # ------------------------
    input_dim = X_train_proc.shape[1]

    def build_advanced_model():
        # regularizer
        reg = None
        if l1_value > 0 and l2_value > 0:
            reg = tf.keras.regularizers.l1_l2(l1=l1_value, l2=l2_value)
        elif l1_value > 0:
            reg = tf.keras.regularizers.l1(l1_value)
        elif l2_value > 0:
            reg = tf.keras.regularizers.l2(l2_value)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        act = activation_choice.lower()
        for units, dr in layer_configs:
            if act == "leakyrelu":
                model.add(tf.keras.layers.Dense(units, kernel_regularizer=reg))
                model.add(tf.keras.layers.LeakyReLU())
            elif act == "tanh":
                model.add(tf.keras.layers.Dense(units, activation="tanh", kernel_regularizer=reg))
            elif act == "sigmoid":
                model.add(tf.keras.layers.Dense(units, activation="sigmoid", kernel_regularizer=reg))
            else:  # relu
                model.add(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg))

            if dr > 0:
                model.add(tf.keras.layers.Dropout(dr))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        lr = float(learning_rate)
        opt_name = optimizer_name.lower()
        if opt_name == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif opt_name == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

        model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------
    # TRAIN MODEL
    # ------------------------
    if st.button("🚀 Start Training", type="primary"):

        model = build_advanced_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_patience,
                restore_best_weights=True
            )
        ]

        if use_scheduler:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                    verbose=1
                )
            )

        st.info("⏳ Training in progress...")

        history = model.fit(
            X_train_proc,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=0
        )

        st.success("🎉 Training complete!")
        hist = history.history

        # Loss plot
        st.subheader("📈 Training & Validation Loss")
        fig1, ax1 = plt.subplots()
        ax1.plot(hist["loss"], label="Train Loss", linestyle="--")
        ax1.plot(hist["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # Accuracy plot
        if "accuracy" in hist:
            st.subheader("📈 Training & Validation Accuracy")
            fig2, ax2 = plt.subplots()
            ax2.plot(hist["accuracy"], label="Train Acc", linestyle="--")
            ax2.plot(hist["val_accuracy"], label="Val Acc")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

        # Evaluate on test set
        y_test_prob = model.predict(X_test_proc).flatten()
        y_test_pred = (y_test_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)

        st.subheader("🧪 Test Set Performance")
        st.write(
            f"**Accuracy:** {acc:.3f} | "
            f"**Precision:** {prec:.3f} | "
            f"**Recall:** {rec:.3f} | "
            f"**F1-score:** {f1:.3f}"
        )

        # Save model
        model.save("advanced_trained_model_b2.h5")
        st.success("💾 Model saved as `advanced_trained_model_b2.h5`.")

        # Set as active model
        set_active_model(model, "advanced_trained_model_b2.h5")

        # Store history and metrics for Model Visuals tab
        st.session_state["train_history"] = hist
        st.session_state["train_metrics"] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        st.info("✅ This model is now the active prediction model used in all tabs.")


# --------------------------------------------------------
# 7. STREAMLIT PAGE CONFIG + THEME
# --------------------------------------------------------

st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

theme_choice = st.sidebar.radio(
    "Theme",
    ["Dark", "Light"],
    index=0 if st.session_state["theme"] == "Dark" else 1
)
st.session_state["theme"] = theme_choice

if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .stApp { background-color: #020617; color: #e5e7eb; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background-color: #f3f4f6; color: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("❤️ Heart Attack Risk Prediction App")
st.write("Predict and analyze heart attack risk using a deep learning model trained on clinical data.")


# --------------------------------------------------------
# 8. TABS
# --------------------------------------------------------

tab_single, tab_batch, tab_visuals, tab_train = st.tabs(
    ["👤 Single Prediction", "📂 Batch Prediction", "📊 Model Visuals", "🧪 Train Model"]
)


# --------------------------------------------------------
# 9. SINGLE-PATIENT PREDICTION
# --------------------------------------------------------

with tab_single:
    st.header("Single Patient Prediction")
    st.subheader("Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

    with col2:
        sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (1 = Yes, 0 = No)", [0, 1])
        restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
        exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1])
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed, 2 = Reversible)", [0, 1, 2])

    shap_df = None

    if st.button("Predict Heart Attack Risk"):
        model = get_active_model()
        if model is None:
            st.error("❌ No active model available. Train one in '🧪 Train Model' tab.")
        else:
            input_df = pd.DataFrame([[
                age,
                trestbps,
                chol,
                thalach,
                oldpeak,
                sex,
                cp,
                fbs,
                restecg,
                exang,
                slope,
                ca,
                thal,
            ]], columns=ALL_FEATURES)

            X_proc = preprocessor.transform(input_df)
            X_proc = to_dense(X_proc)
            prob = float(model.predict(X_proc)[0][0])
            pred = 1 if prob >= 0.5 else 0
            pred_label = "HIGH RISK" if pred == 1 else "LOW RISK"

            st.subheader("🩺 Prediction Result")
            if pred == 1:
                st.error(f"⚠ {pred_label} of Heart Attack (Probability: {prob:.2f})")
            else:
                st.success(f"💚 {pred_label} of Heart Attack (Probability: {prob:.2f})")

            st.write("Model Confidence Score:", round(prob, 3))

            risk_level, rec_text = get_doctor_recommendation(prob)
            st.markdown("### Doctor Recommendation (Model-Based)")
            if prob < 0.30:
                st.success(f"**{risk_level}:** {rec_text}")
            elif prob < 0.60:
                st.warning(f"**{risk_level}:** {rec_text}")
            else:
                st.error(f"**{risk_level}:** {rec_text}")

            st.markdown("### Risk Gauge")
            st.plotly_chart(plot_gauge(prob), use_container_width=True)

            st.markdown("### Feature Contributions (SHAP Bar Chart)")
            if SHAP_AVAILABLE:
                shap_df, shap_err = compute_shap_for_instance(input_df)
                if shap_df is not None:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    top = shap_df.head(10)
                    ax.barh(top["feature"], top["shap_value"])
                    ax.set_xlabel("SHAP value (impact on risk)")
                    ax.invert_yaxis()
                    st.pyplot(fig)
                else:
                    st.info(shap_err)
            else:
                st.info("SHAP not installed. Run `pip install shap` to view feature contributions.")

            st.markdown("### Download Medical Report (PDF)")
            patient_data = {
                "Age": age,
                "Sex (0=F,1=M)": sex,
                "Chest Pain Type": cp,
                "Resting Blood Pressure": trestbps,
                "Cholesterol": chol,
                "Fasting Blood Sugar >120": fbs,
                "Resting ECG": restecg,
                "Max Heart Rate": thalach,
                "Exercise-Induced Angina": exang,
                "ST Depression": oldpeak,
                "Slope of ST Segment": slope,
                "Number of Vessels (ca)": ca,
                "Thalassemia": thal,
            }

            if REPORTLAB_AVAILABLE:
                pdf_bytes, pdf_err = generate_pdf_report(
                    patient_data,
                    prob,
                    pred_label,
                    risk_level,
                    rec_text,
                    shap_df
                )
                if pdf_bytes is not None:
                    st.download_button(
                        label="📄 Download Patient Report (PDF)",
                        data=pdf_bytes,
                        file_name="heart_attack_report.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.info(pdf_err)
            else:
                st.info("PDF generation requires 'reportlab'. Run: pip install reportlab")


# --------------------------------------------------------
# 10. BATCH PREDICTION
# --------------------------------------------------------

with tab_batch:
    st.header("Batch Prediction from CSV")
    st.write("""
    Upload a CSV file with patient medical data.  
    The file must include **exactly these columns**:

    ```
    age, trestbps, chol, thalach, oldpeak,
    sex, cp, fbs, restecg, exang, slope, ca, thal
    ```
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if "target" in batch_df.columns:
                batch_df = batch_df.drop(columns=["target"])

            missing_cols = [c for c in ALL_FEATURES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                batch_df = batch_df[ALL_FEATURES]

                model = get_active_model()
                if model is None:
                    st.error("❌ No active model available. Train one in '🧪 Train Model' tab.")
                else:
                    X_batch = preprocessor.transform(batch_df)
                    X_batch = to_dense(X_batch)
                    probs = model.predict(X_batch).reshape(-1)
                    preds = (probs >= 0.5).astype(int)
                    labels = np.where(preds == 1, "High Risk", "Low Risk")

                    results_df = batch_df.copy()
                    results_df["risk_probability"] = probs
                    results_df["prediction"] = labels

                    st.success(f"Processed {len(results_df)} patients successfully.")
                    st.dataframe(results_df)

                    csv_out = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions as CSV",
                        csv_out,
                        "heart_attack_predictions.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# --------------------------------------------------------
# 11. MODEL VISUALS
# --------------------------------------------------------

with tab_visuals:
    st.header("Model Visuals")

    # Training curves from last trained model (this session)
    if "train_history" in st.session_state:
        hist = st.session_state["train_history"]

        st.subheader("Training & Validation Loss")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(hist["loss"], label="Train Loss", linestyle="--")
        ax1.plot(hist["val_loss"], label="Val Loss", linestyle="-")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        st.pyplot(fig1)

        if "accuracy" in hist and "val_accuracy" in hist:
            st.subheader("Training & Validation Accuracy")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(hist["accuracy"], label="Train Acc", linestyle="--")
            ax2.plot(hist["val_accuracy"], label="Val Acc", linestyle="-")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            st.pyplot(fig2)

        if "train_metrics" in st.session_state:
            st.subheader("Test Metrics (Last Trained Model)")
            m = st.session_state["train_metrics"]
            st.write(
                f"**Accuracy:** {m['accuracy']:.3f}  |  "
                f"**Precision:** {m['precision']:.3f}  |  "
                f"**Recall:** {m['recall']:.3f}  |  "
                f"**F1-score:** {m['f1']:.3f}"
            )
    else:
        st.info("Train a model in the '🧪 Train Model' tab to see training curves here.")

    # Confusion Matrix for current active model on test set
    st.subheader("Confusion Matrix on Test Set")
    model = get_active_model()
    if model is not None:
        y_test_prob = model.predict(X_test_proc).flatten()
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_test_pred)

        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["No Attack", "Attack"])
        ax_cm.set_yticklabels(["No Attack", "Attack"])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")

        fig_cm.colorbar(im, ax=ax_cm)
        st.pyplot(fig_cm)
    else:
        st.info("No active model available to evaluate confusion matrix.")


# --------------------------------------------------------
# 12. TRAIN MODEL TAB (B2)
# --------------------------------------------------------

with tab_train:
    render_training_tab_b2()
