import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Smart Grid Analytics - Household Power", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_dashboard.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

@st.cache_resource
def load_models():
    reg = joblib.load("src/rf_regressor.pkl")
    clf = joblib.load("src/rf_classifier.pkl")
    scaler = joblib.load("src/scaler.pkl")
    return reg, clf, scaler

df = load_data()
reg_model, clf_model, scaler = load_models()

st.title("⚡ Gestión Inteligente de Consumo Eléctrico Residencial")
st.markdown("**Proyecto Integrador - Maestría en Inteligencia Artificial | Universidad de La Salle**")

# --- MÓDULO 1: FILTROS DINÁMICOS ---
st.sidebar.header("🎛️ Filtros Dinámicos")
niveles = st.sidebar.multiselect(
    "Nivel de Demanda:",
    options=[0, 1, 2],
    default=[0, 1, 2],
    format_func=lambda x: {0: "Baja (<1 kW)", 1: "Media (1-3 kW)", 2: "Alta (>=3 kW)"}[x]
)
horas = st.sidebar.slider("Rango Horario (Hora del día):", 0, 23, (0, 23))
potencia_min, potencia_max = st.sidebar.slider(
    "Rango Potencia Activa (kW):",
    float(df['Global_active_power'].min()),
    float(df['Global_active_power'].max()),
    (0.0, 10.6)
)

df_filtrado = df[
    (df['Demand_Level'].isin(niveles)) &
    (df['Hour'].between(horas[0], horas[1])) &
    (df['Global_active_power'].between(potencia_min, potencia_max))
]

# --- MÓDULO 2: TARJETAS KPI EJECUTIVAS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros Filtrados", f"{len(df_filtrado):,}")
col2.metric("Potencia Activa Promedio", f"{df_filtrado['Global_active_power'].mean():.2f} kW")
col3.metric("Voltaje Promedio Red", f"{df_filtrado['Voltage'].mean():.1f} V")
col4.metric("Factor de Potencia (cos θ)", f"{df_filtrado['Power_Factor'].mean():.3f}")

st.divider()

# --- MÓDULO 3: SIMULADOR DE PREDICCIÓN EN VIVO ---
st.subheader("🔮 Módulo de Predicción en Vivo (Random Forest)")
c1, c2, c3, c4 = st.columns(4)
v_in = c1.number_input("Voltaje (V)", 220.0, 255.0, 240.0)
i_in = c2.number_input("Intensidad Global (A)", 0.1, 46.0, 4.5)
q_in = c3.number_input("Potencia Reactiva (kVAR)", 0.0, 1.5, 0.12)
h_in = c4.slider("Hora del día", 0, 23, 19)

s1, s2, s3, s4 = st.columns(4)
sub1 = s1.number_input("Sub_metering 1 (Cocina Wh)", 0.0, 80.0, 0.0)
sub2 = s2.number_input("Sub_metering 2 (Lavandería Wh)", 0.0, 80.0, 1.0)
sub3 = s3.number_input("Sub_metering 3 (Clima/Agua Wh)", 0.0, 35.0, 17.0)
dow_in = s4.selectbox(
    "Día de la semana",
    options=list(range(7)),
    format_func=lambda x: ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][x]
)

apparent_p = (v_in * i_in) / 1000.0
pf = 0.98

if st.button("Predecir Consumo y Régimen", type="primary"):
    input_data = np.array([[q_in, v_in, i_in, sub1, sub2, sub3, apparent_p, pf, h_in, dow_in, 6]])
    input_scaled = scaler.transform(input_data)
    pred_kw = reg_model.predict(input_scaled)[0]
    pred_clase = clf_model.predict(input_scaled)[0]
    etiquetas = {
        0: "🟢 Baja Demanda (< 1.0 kW)",
        1: "🟡 Demanda Media (1.0 - 3.0 kW)",
        2: "🔴 Alta Demanda (≥ 3.0 kW)"
    }
    
    r1, r2 = st.columns(2)
    r1.success(f"**Potencia Activa Estimada:** {pred_kw:.3f} kW")
    r2.info(f"**Clasificación de Carga:** {etiquetas[pred_clase]}")

st.divider()

# --- MÓDULO 4: SEGMENTACIÓN NO SUPERVISADA (CLUSTERING) ---
st.subheader("📊 Módulo de Segmentación Operativa (K-Means K=3)")
g1, g2 = st.columns([2, 1])

with g1:
    muestra_graf = df_filtrado.sample(min(3000, len(df_filtrado)), random_state=42)
    fig = px.scatter(
        muestra_graf,
        x="Global_intensity",
        y="Global_active_power",
        color=muestra_graf['Cluster_KMeans'].astype(str),
        title="Distribución de Clústeres: Corriente vs Potencia Activa",
        labels={
            "Global_intensity": "Intensidad (A)",
            "Global_active_power": "Potencia Activa (kW)",
            "color": "Clúster"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.markdown("**Perfiles Promedio por Clúster**")
    perfil = df.groupby('Cluster_KMeans')[['Global_active_power', 'Global_intensity', 'Voltage']].mean().round(2)
    st.dataframe(perfil, use_container_width=True)
