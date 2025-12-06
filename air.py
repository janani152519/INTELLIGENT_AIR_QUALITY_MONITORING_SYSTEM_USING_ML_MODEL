import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="🌫️ Air Quality Dashboard — Clean UI", layout="wide")

# CSS
st.markdown("""
<style>
:root { --card-bg: rgba(255,255,255,0.06); --card-border: rgba(255,255,255,0.08); }
.stApp { padding: 18px 24px; }
.aq-badge { padding:10px 18px; border-radius:12px; font-weight:700; display:inline-block; border:2px solid transparent; }
.aq-good { background:#d1fae5; border-color:#16a34a; color:#064e3b; }
.aq-moderate { background:#facc1533; border-color:#b45309; color:#92400e; }
.aq-poor { background:#fed7aa; border-color:#ea580c; color:#7c2d12; }
.aq-hazardous { background:#ef444433; border-color:#991b1b; color:#6b0b0b; }
.aq-unknown { background:#ddd; color:#222; border-color:#ccc; }
.info-card { background: rgba(255,255,255,0.02); border:1px solid var(--card-border); border-radius:14px; padding:14px; min-width:260px; margin-bottom:12px; }
.info-title { font-size:15px; margin:0 0 6px 0; font-weight:600; color:inherit; }
.info-value { font-size:17px; margin:0; font-weight:700; color:inherit; }
.scroll-row { display:flex; gap:16px; overflow-x:auto; padding-bottom:8px; }
.scroll-row::-webkit-scrollbar { height:8px; }
.scroll-row::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("updated_pollution_dataset (2).csv")
df.columns = df.columns.str.strip()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df = df.dropna(subset=['Air Quality'])
df['Air Quality'] = df['Air Quality'].astype(str).str.strip().str.title()

# Features & model
features = ["Temperature","Humidity","PM2.5","PM10","NO2","SO2","CO",
            "Proximity_to_Industrial_Areas","Population_Density"]
X = df[features]
y = df['Air Quality']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.18, random_state=42, stratify=y if len(y.unique())>1 else None
)
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Safety & small helper functions (unchanged)
def mask_recommendation(aqi):
    if aqi == "Good": return "No mask needed"
    elif aqi == "Moderate": return "Surgical mask recommended"
    elif aqi == "Poor": return "N95 mask required"
    elif aqi == "Hazardous": return "Avoid going outside"
    return "Unknown"

def pregnant_safety(aqi, so2, no2):
    if aqi == "Good" and so2 < 20 and no2 < 40: return "Safe for short outdoor exposure"
    elif aqi == "Moderate": return "Limit outdoor time"
    elif aqi == "Poor": return "Unsafe – avoid outdoors"
    elif aqi == "Hazardous": return "Highly dangerous – stay indoors"
    return "Unknown"

def kids_safety(aqi):
    if aqi == "Good": return "Safe for school/outdoor play"
    elif aqi == "Moderate": return "Caution for long play"
    elif aqi == "Poor": return "Avoid heavy outdoor activity"
    elif aqi == "Hazardous": return "Not safe for kids at all"
    return "Unknown"

def senior_safety(aqi):
    if aqi in ["Good","Moderate"]: return "Safe"
    elif aqi == "Poor": return "Caution"
    elif aqi == "Hazardous": return "Dangerous – stay indoors"
    return "Unknown"

def asthma_risk(aqi, no2, o3):
    if aqi == "Good": return "Low risk"
    elif aqi == "Moderate": return "Moderate risk"
    else: return "High risk"

def exercise_safety(aqi):
    if aqi == "Good": return "Perfect for exercise"
    elif aqi == "Moderate": return "Moderate – avoid long cardio"
    elif aqi == "Poor": return "Risky – avoid running"
    else: return "Unsafe – stay indoors"

def indoor_aqi(aqi, humidity, wind_speed):
    if wind_speed > 2: return "Indoor AQI lower"
    elif humidity > 70: return "Indoor AQI higher"
    else: return aqi

def irritation_risk(so2, pm25):
    score = so2*0.6 + pm25*0.4
    if score <= 50: return "No irritation expected"
    elif score <= 100: return "Mild irritation"
    else: return "High irritation risk"

def voc_toxicity(pm25, pm10, no2, so2, co):
    score = 0.3*pm25 + 0.2*pm10 + 0.2*no2 + 0.2*so2 + 0.1*co
    if score <= 20:
        return "Low VOC exposure"
    elif score <= 50:
        return "Moderate VOC exposure"
    else:
        return "High VOC toxicity"

def best_time(aqi_current, aqi_forecast):
    if aqi_forecast == aqi_current: return "Wait 1–2 hours, air will improve"
    else: return "Go out now — air may worsen later"

def resp_score(pm25, no2, o3):
    return 0.4*pm25 + 0.4*no2 + 0.2*o3

def industrial_impact(so2, nox):
    total = so2 + nox
    if total <= 80: return "Low industrial impact"
    elif total <= 150: return "Moderate industrial impact"
    else: return "High industrial emission detected"

# Sidebar Inputs
st.sidebar.header("Input Environmental Values")
input_values = {}
for col in features:
    col_min = float(np.nanmin(X[col]))
    col_max = float(np.nanmax(X[col]))
    col_mid = float(np.nanmedian(X[col]))
    if col_min == col_max: col_max = col_min + 1.0
    step = max((col_max - col_min)/200, 0.01)
    input_values[col] = st.sidebar.slider(label=col, min_value=col_min, max_value=col_max, value=col_mid, step=step)

# Show Input Table BEFORE Prediction
st.subheader("📝 Current Input Values")
st.table(pd.DataFrame([input_values]))

# Prediction Button
if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([input_values])
    predicted_label = clf.predict(input_df)[0]

    # AQI Badge
    badge_class = {
        "Good":"aq-good",
        "Moderate":"aq-moderate",
        "Poor":"aq-poor",
        "Hazardous":"aq-hazardous"
    }.get(predicted_label, "aq-unknown")
    st.markdown(f"<h1>🌫️ Air Quality Prediction Dashboard</h1><span class='aq-badge {badge_class}'>{predicted_label}</span>", unsafe_allow_html=True)

    # Safety Cards with colored border according to AQI
    st.subheader("🛡️ Safety Insights")
    st.markdown("<div class='scroll-row'>", unsafe_allow_html=True)
    border_color_map = {
        "Good": "#16a34a",
        "Moderate": "#b45309",
        "Poor": "#c2410c",
        "Hazardous": "#991b1b"
    }
    border_color = border_color_map.get(predicted_label, "#ccc")
    cards = [
        ("😷 Mask Recommendation", mask_recommendation(predicted_label)),
        ("🤰 Pregnant Women", pregnant_safety(predicted_label, input_values.get('SO2',0), input_values.get('NO2',0))),
        ("🧒 Kids Safety", kids_safety(predicted_label)),
        ("👵 Senior Citizen", senior_safety(predicted_label)),
        ("💨 Asthma Risk", asthma_risk(predicted_label, input_values.get('NO2',0), input_values.get('CO',0))),
        ("🏃 Exercise Safety", exercise_safety(predicted_label)),
        ("🏠 Indoor AQI", indoor_aqi(predicted_label, input_values.get('Humidity',0), 1.5)),
        ("👁️ Irritation Risk", irritation_risk(input_values.get('SO2',0), input_values.get('PM2.5',0))),
        ("☣️ VOC Toxicity",voc_toxicity(input_values.get('PM2.5', 0),input_values.get('PM10', 0),input_values.get('NO2', 0),input_values.get('SO2', 0),input_values.get('CO', 0) )),
        ("⏰ Best Time Outside", best_time(predicted_label, predicted_label)),
        ("🌬️ Resp. Score", resp_score(input_values.get('PM2.5',0), input_values.get('NO2',0), input_values.get('CO',0))),
        ("🏭 Industrial Impact", industrial_impact(input_values.get('SO2',0), input_values.get('NO2',0))),
    ]
    for t, v in cards:
        st.markdown(f"""
        <div class='info-card' style='border:2px solid {border_color}'>
            <div class='info-title'>{t}</div>
            <div class='info-value'>{v}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Table with Reasoning
    st.subheader("📝 Input Values & Prediction Reasoning")
    # Reasoning based on predicted label
    if predicted_label == "Good":
        reasoning = "All pollutant levels are within safe limits — air quality is excellent."
    elif predicted_label == "Moderate":
        reasoning = "Pollutant levels are moderate — acceptable for most activities."
    elif predicted_label == "Poor":
        reasoning = "Pollutant levels are high — may affect sensitive individuals."
    elif predicted_label == "Hazardous":
        reasoning = "Pollutant levels are dangerously high — avoid outdoor exposure."
    else:
        reasoning = "Prediction based on input values."
    result_table = input_df.copy()
    result_table['Predicted AQI'] = predicted_label
    result_table['Reasoning'] = reasoning
    st.table(result_table)

    # ------------------ FIXED GRAPHS ------------------

    # 1) Interactive Pie chart using Plotly
    st.subheader("🥧 AQI Distribution in Dataset")
    # enforce a sensible fixed label order for display
    label_order = ["Good", "Moderate", "Poor", "Hazardous"]
    counts_ordered = []
    labels_present = []
    for lbl in label_order:
        if lbl in df['Air Quality'].values:
            counts_ordered.append(df['Air Quality'].value_counts().get(lbl, 0))
            labels_present.append(lbl)
    # color map aligned to our badge styles
    color_map = {"Good": "#22c55e", "Moderate": "#facc15", "Poor": "#fb923c", "Hazardous": "#ef4444"}
    colors = [color_map.get(lbl, "#999999") for lbl in labels_present]

    if sum(counts_ordered) == 0:
        st.write("No AQI data")
    else:
        # pull out the predicted slice
        pull = [0.1 if lbl == predicted_label else 0.0 for lbl in labels_present]
        fig3 = go.Figure(data=[go.Pie(
            labels=labels_present,
            values=counts_ordered,
            marker_colors=colors,
            pull=pull,
            textfont=dict(color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        fig3.update_layout(
            title="AQI Distribution in Dataset",
            showlegend=True
        )
        st.plotly_chart(fig3, width='stretch')

    # 2) Interactive Feature Importance using Plotly
    st.subheader("📊 Feature Importance")
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    feat_sorted = [features[i] for i in order]
    imp_sorted = importances[order]

    fig1 = go.Figure(data=[go.Bar(
        x=imp_sorted,
        y=feat_sorted,
        orientation='h',
        marker_color='rgba(31, 119, 180, 0.8)',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    )])
    fig1.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Features",
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig1, width='stretch')

    # 3) AQI vs PM2.5 scatter — your input values only
    st.subheader("📈 AQI vs PM2.5 (your input)")
    input_pm25 = input_values.get('PM2.5', 0)
    label_order = ["Good", "Moderate", "Poor", "Hazardous"]
    label_map = {lbl: idx for idx, lbl in enumerate(label_order)}
    input_y = label_map.get(predicted_label, 0)
    fig2 = go.Figure(data=[go.Scatter(
        x=[input_pm25],
        y=[input_y],
        mode='markers',
        marker=dict(size=20, color='red', symbol='circle'),
        name='Your Input'
    )])
    fig2.update_layout(
        title="AQI vs PM2.5 (Your Input)",
        xaxis_title="PM2.5",
        yaxis_title="Air Quality",
        yaxis=dict(tickvals=list(label_map.values()), ticktext=list(label_map.keys()))
    )
    st.plotly_chart(fig2, width='stretch')

    # 4) Your Input Dataset Graph (Moderate)
    st.subheader("📊 Your Input Dataset (Moderate)")
    fig4 = go.Figure(data=[go.Bar(
        x=list(input_values.keys()),
        y=list(input_values.values()),
        marker_color='orange'
    )])
    fig4.update_layout(
        title="Your Input Values (Predicted: Moderate)",
        xaxis_title="Features",
        yaxis_title="Values"
    )
    st.plotly_chart(fig4, width='stretch')

    # Confusion Matrix
    st.subheader("📋 Confusion Matrix")
    cm = confusion_matrix(y_test, clf.predict(X_test))
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
    # Create a new DataFrame for display with error information
    display_df = cm_df.copy().astype(str)
    for i in range(len(display_df)):
        for j in range(len(display_df.columns)):
            if i != j:
                display_df.iloc[i, j] = f"{cm_df.iloc[i, j]} (error)"
            else:
                display_df.iloc[i, j] = f"{cm_df.iloc[i, j]} (correct)"
    st.table(display_df)

    # Model summary
    st.subheader("🔎 Model Summary")
    acc = clf.score(X_test, y_test)
    st.info(f"Accuracy on test split: {acc:.3f}")

else:
    st.markdown("<h3>Ready — set inputs in the sidebar and press Predict</h3>", unsafe_allow_html=True)
