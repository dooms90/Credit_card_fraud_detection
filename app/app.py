import streamlit as st
import pandas as pd
import numpy as np
import joblib
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

model = joblib.load('app_model.pkl')
scaler = joblib.load('app_scaler.pkl')
samples = pd.read_csv('sample_transactions.csv')

feature_cols = [col for col in samples.columns if col != 'actual_label']

if 'inputs' not in st.session_state:
    st.session_state.inputs = {col: 0.0 for col in feature_cols}
    st.session_state.inputs['scaled_amount'] = 0.0
    st.session_state.inputs['scaled_time'] = 0.0

st.title("💳 Credit Card Fraud Detection")
st.write("A machine learning demo that predicts whether a transaction is fraudulent, trained on real anonymized transaction data (Kaggle Credit Card Fraud dataset).")

st.subheader("🔄 Try a Real Example")
col1, col2, col3 = st.columns(3)

def load_sample(label):
    subset = samples[samples['actual_label'] == label]
    row = subset.sample(1).iloc[0]
    for col in feature_cols:
        st.session_state.inputs[col] = float(row[col])

with col1:
    if st.button("✅ Load Normal Transaction"):
        load_sample(0)
with col2:
    if st.button("⚠️ Load Fraud Transaction"):
        load_sample(1)
with col3:
    if st.button("🔁 Reset All to 0"):
        for col in feature_cols:
            st.session_state.inputs[col] = 0.0

st.divider()

st.subheader("Transaction Details")

c1, c2 = st.columns(2)
with c1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0,
                               value=float(scaler.inverse_transform([[st.session_state.inputs.get('scaled_amount', 0.0)]])[0][0]) if 'scaled_amount' in feature_cols else 100.0)
with c2:
    time = st.number_input("Time (seconds since first transaction)", min_value=0.0,
                             value=50000.0)

with st.expander("Advanced: anonymized features (V1–V28)"):
    st.caption("These are PCA-transformed values from the original dataset — not human-interpretable individually. Use the sample loader above instead of guessing.")
    v_values = []
    cols = st.columns(4)
    for i, col_name in enumerate([c for c in feature_cols if c not in ('scaled_amount', 'scaled_time')]):
        col = cols[i % 4]
        val = col.number_input(col_name, value=float(st.session_state.inputs.get(col_name, 0.0)), format="%.4f", key=col_name)
        v_values.append((col_name, val))

st.divider()
if st.button("🔍 Predict", type="primary"):
    scaled_amount = scaler.transform([[amount]])[0][0]
    scaled_time = scaler.transform([[time]])[0][0]

    v_dict = dict(v_values)
    row = [scaled_amount, scaled_time] + [v_dict[c] for c in feature_cols if c not in ('scaled_amount', 'scaled_time')]
    input_data = np.array([row])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        if prediction == 1:
            st.error(f"⚠️ FRAUD\n\n{probability:.1%} confidence")
        else:
            st.success(f"✅ NORMAL\n\n{(1-probability):.1%} confidence")
    with r2:
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Fraud probability: {probability:.2%}")

    if hasattr(model, 'coef_'):
        st.subheader("🔎 What influenced this prediction")
        coefs = model.coef_[0]
        col_names = ['scaled_amount', 'scaled_time'] + [c for c in feature_cols if c not in ('scaled_amount', 'scaled_time')]
        contributions = np.array(row) * coefs
        top_idx = np.argsort(np.abs(contributions))[::-1][:5]

        explain_df = pd.DataFrame({
            'Feature': [col_names[i] for i in top_idx],
            'Contribution': [contributions[i] for i in top_idx]
        })
        explain_df['Direction'] = explain_df['Contribution'].apply(lambda x: 'Pushes toward Fraud' if x > 0 else 'Pushes toward Normal')
        st.dataframe(explain_df, hide_index=True, use_container_width=True)

st.divider()
st.caption("⚠️ Educational demo only — trained on a public research dataset, not production financial data.")
