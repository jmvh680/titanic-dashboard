#correrlo con "streamlit run titanic_dashboard.py"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Configuración de página
st.set_page_config(page_title="Dashboard Titanic", layout="wide")

# Cargar datos
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Sex"] = df["Sex"].str.lower()
    df["IsChild"] = df["Age"] < 12
    return df

df = cargar_datos()

st.title("🚢 Dashboard Interactivo - Titanic")

# Filtros
st.sidebar.header("Filtros")
sexo = st.sidebar.multiselect("Sexo", options=df["Sex"].unique(), default=df["Sex"].unique())
clase = st.sidebar.multiselect("Clase", options=sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique()))

# Aplicar filtros
df_filtrado = df[(df["Sex"].isin(sexo)) & (df["Pclass"].isin(clase))]

# Layout de columnas
col1, col2 = st.columns(2)

# Gráfico 1: Supervivencia por sexo
with col1:
    st.subheader("Supervivencia por Sexo")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Sex", hue="Survived", data=df_filtrado, ax=ax1)
    ax1.set_title("Supervivencia por sexo")
    st.pyplot(fig1)

# Gráfico 2: Distribución de edades
with col2:
    st.subheader("Distribución de Edades")
    fig2, ax2 = plt.subplots()
    sns.histplot(df_filtrado["Age"], bins=30, kde=True, ax=ax2)
    ax2.set_title("Distribución de edades")
    st.pyplot(fig2)

# Segunda fila
col3, col4 = st.columns(2)

# Gráfico 3: Supervivencia de niños
with col3:
    st.subheader("Supervivencia: Niños vs Adultos")
    fig3, ax3 = plt.subplots()
    sns.barplot(x="IsChild", y="Survived", data=df_filtrado, ax=ax3)
    ax3.set_xlabel("¿Es niño?")
    ax3.set_title("Supervivencia de niños vs adultos")
    st.pyplot(fig3)

# Gráfico 4: Edad por clase y supervivencia
with col4:
    st.subheader("Edad por Clase y Supervivencia")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x="Pclass", y="Age", hue="Survived", data=df_filtrado, ax=ax4)
    ax4.set_title("Edad por clase y supervivencia")
    st.pyplot(fig4)

# Mostrar tabla
st.subheader("Vista previa de los datos filtrados")
st.dataframe(df_filtrado.head(20))

from sklearn.ensemble import RandomForestClassifier

# Preparar datos para entrenamiento
modelo_df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
modelo_df["Sex"] = modelo_df["Sex"].map({"male": 0, "female": 1})
y = df["Survived"]

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(modelo_df, y)

st.markdown("---")
st.header("🔮 Predicción de Supervivencia")

with st.form("form_prediccion"):
    col1, col2, col3 = st.columns(3)

    with col1:
        sexo_input = st.selectbox("Sexo", ["male", "female"])
        pclass_input = st.selectbox("Clase", [1, 2, 3])
    
    with col2:
        edad_input = st.slider("Edad", 0, 80, 30)
        sibsp_input = st.number_input("Hermanos/Pareja a bordo", min_value=0, max_value=10, value=0)
    
    with col3:
        parch_input = st.number_input("Padres/Hijos a bordo", min_value=0, max_value=10, value=0)
        fare_input = st.slider("Tarifa pagada (Fare)", 0.0, 600.0, 50.0)

    submitted = st.form_submit_button("Predecir")

    if submitted:
        datos = [[
            pclass_input,
            0 if sexo_input == "male" else 1,
            edad_input,
            sibsp_input,
            parch_input,
            fare_input
        ]]
        prediccion = modelo.predict(datos)[0]
        proba = modelo.predict_proba(datos)[0][1]

        st.success(f"🌊 {'Sobrevive' if prediccion == 1 else 'No sobrevive'} con una probabilidad de {proba:.2%}")
