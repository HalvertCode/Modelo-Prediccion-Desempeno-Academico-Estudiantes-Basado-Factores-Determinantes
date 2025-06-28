import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import altair as alt

# Crear carpeta de modelos
os.makedirs("models", exist_ok=True)

# Función para cargar modelo guardado
def load_saved_model(path="models/random_forest.pkl"):
    return joblib.load(path) if os.path.exists(path) else None

# Título y navegación
st.title("Predicción de Calificaciones de Estudiantes 📊📈")
st.sidebar.header("Navegación")
page = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Explorar Datos", "Entrenar Modelo", "Ajuste de Hiperparámetros", "Evaluar Modelo", "Hacer Predicción"]
)

# Sesión de datos
def get_data():
    return st.session_state.get('data')

def set_data(df):
    st.session_state['data'] = df

# ============================
# Explorar Datos
if page == "Explorar Datos":
    st.header("Exploración de Datos 📊")
    uploaded = st.file_uploader("Sube tu dataset (CSV)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        set_data(df)
        st.subheader("Vista previa del dataset (scrollable)")
        st.dataframe(df, height=400)
        st.subheader("Estadísticas descriptivas")
        st.write(df.describe())
        st.subheader("Tipos de datos y valores nulos")
        st.text(df.info())

# ============================
# Entrenar Modelo
elif page == "Entrenar Modelo":
    st.header("Entrenar Modelo 🤖")
    df = get_data()
    if df is None:
        upload = st.file_uploader("Sube tu dataset para entrenamiento (CSV)", type=["csv"], key="train_upload")
        if upload:
            df = pd.read_csv(upload)
            set_data(df)
    if df is not None:
        st.subheader("Esquema de datos")
        st.write(df.dtypes)
        target = 'Exam_Score'
        st.info("La variable objetivo (target) siempre es Exam_Score para entrenamiento")

        col1, col2 = st.columns([1, 2])
        train_clicked = col1.button("Iniciar entrenamiento", key="train_btn")
        status = col2.empty()

        if train_clicked:
            status.text("Entrenando modelo. Por favor, espera...")
            # Preprocesamiento
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)
            df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            X = df_encoded.drop(columns=[target])
            y = df_encoded[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            # Guardar en sesión
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = X.columns.tolist()
            st.session_state['orig_cat_cols'] = cat_cols
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols.remove(target)
            st.session_state['orig_num_cols'] = num_cols
            # Métricas
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            status.success(f"Entrenamiento completado: MAE={mae:.3f}, RMSE={rmse:.3f}")

        if 'model' in st.session_state:
            if st.button("Guardar modelo en disco", key="save_btn"):
                joblib.dump(st.session_state['model'], "models/random_forest.pkl")
                st.success("Modelo guardado en models/random_forest.pkl 📦")
            if st.button("Cargar modelo guardado", key="load_btn"):
                mdl = load_saved_model()
                if mdl:
                    st.session_state['model'] = mdl
                    st.success("Modelo cargado desde disco")
                else:
                    st.error("No se encontró ningún modelo guardado.")

# ============================
# Ajuste de Hiperparámetros
elif page == "Ajuste de Hiperparámetros":
    st.header("Ajuste de Hiperparámetros 🔧")
    df = get_data()
    if df is None:
        st.warning("Sube y explora tu dataset primero.")
    else:
        target = 'Exam_Score'
        # Parámetros de búsqueda
        st.subheader("Define rangos para RandomizedSearchCV")
        n_estimators = st.slider("n_estimators", 50, 500, (100, 300), step=50)
        max_depth = st.slider("max_depth (None=0)", 0, 50, (0, 20), step=5)
        min_samples_split = st.slider("min_samples_split", 2, 20, (2, 10), step=2)
        min_samples_leaf = st.slider("min_samples_leaf", 1, 10, (1, 4), step=1)
        n_iter = st.number_input("Número de iteraciones (n_iter)", min_value=1, max_value=50, value=10)
        cv = st.number_input("Número de folds (cv)", min_value=2, max_value=10, value=3)

        if st.button("Ejecutar RandomizedSearchCV"):
            with st.spinner("Buscando mejores hiperparámetros..."):
                # Preprocesamiento igual al entrenamiento
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                for col in cat_cols:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                X = df_encoded.drop(columns=[target])
                y = df_encoded[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # Espacio de búsqueda
                param_dist = {
                    'n_estimators': list(range(n_estimators[0], n_estimators[1]+1, 50)),
                    'max_depth': [None] + list(range(max_depth[0] or 1, max_depth[1]+1, 5)),
                    'min_samples_split': list(range(min_samples_split[0], min_samples_split[1]+1, 2)),
                    'min_samples_leaf': list(range(min_samples_leaf[0], min_samples_leaf[1]+1, 1))
                }
                rand_search = RandomizedSearchCV(
                    estimator=RandomForestRegressor(random_state=42),
                    param_distributions=param_dist,
                    n_iter=n_iter, cv=cv, random_state=42, n_jobs=-1
                )
                rand_search.fit(X_train, y_train)
                best = rand_search.best_estimator_
                # Guardar mejor modelo
                st.session_state['model'] = best
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = X.columns.tolist()
                # Resultados
                st.subheader("Mejores hiperparámetros")
                st.write(rand_search.best_params_)
                y_pred = best.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.subheader("Métricas con mejor modelo")
                st.write(f"- MAE: {mae:.3f}")
                st.write(f"- RMSE: {rmse:.3f}")

# ============================
# Evaluar Modelo
elif page == "Evaluar Modelo":
    st.header("Evaluar Modelo 📈")
    model = st.session_state.get('model') or load_saved_model()
    if model is None or 'X_test' not in st.session_state:
        st.warning("Primero entrena o ajusta hiperparámetros.")
    else:
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.subheader("Métricas de evaluación")
        st.write(f"- MAE: {mae:.3f}")
        st.write(f"- RMSE: {rmse:.3f}")
        imp_df = pd.DataFrame({
            'feature': st.session_state['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        imp_df['top10'] = imp_df['importance'] >= imp_df['importance'].nlargest(10).min()
        chart = alt.Chart(imp_df).mark_bar().encode(
            x=alt.X('feature', sort='-y'),
            y='importance',
            color=alt.condition(
                alt.datum.top10,
                alt.value('orange'),
                alt.value('steelblue')
            )
        ).properties(width=700)
        st.subheader("Importancia de características (Top 10 resaltadas)")
        st.altair_chart(chart, use_container_width=True)

# ============================
# Hacer Predicción
else:
    st.header("Hacer Predicción 🎯")
    model = st.session_state.get('model') or load_saved_model()
    if model is None or 'feature_names' not in st.session_state:
        st.warning("No hay modelo. Entrena primero.")
    else:
        orig_num = st.session_state['orig_num_cols']
        orig_cat = st.session_state['orig_cat_cols']
        inputs = {}
        st.subheader("Variables numéricas")
        for feat in orig_num:
            val = st.text_input(f"{feat}", placeholder="Ej: 10.5", key=f"num_{feat}")
            if val == "" or not val.replace('.','',1).isdigit() or float(val) < 0:
                st.error(f"Ingresa un número válido no negativo para {feat}")
            inputs[feat] = val
        st.subheader("Variables categóricas")
        for feat in orig_cat:
            levels = get_data()[feat].dropna().unique().tolist()
            inputs[feat] = st.selectbox(feat, levels, index=0)
        can_pred = all(val != "" and val.replace('.','',1).isdigit() and float(val) >= 0 for val in [inputs[f] for f in orig_num])
        if st.button("Predecir", key="pred_btn"):
            if not can_pred:
                st.error("Completa correctamente todas las variables numéricas antes de predecir")
            else:
                for feat in orig_num:
                    inputs[feat] = float(inputs[feat])
                new_df = pd.DataFrame([inputs])
                for col in orig_cat:
                    new_df[col].fillna(get_data()[col].mode()[0], inplace=True)
                new_enc = pd.get_dummies(new_df, columns=orig_cat, drop_first=True)
                X_new = new_enc.reindex(columns=st.session_state['feature_names'], fill_value=0)
                pred = model.predict(X_new)[0]
                st.success(f"Predicción de calificación: {pred:.2f}")