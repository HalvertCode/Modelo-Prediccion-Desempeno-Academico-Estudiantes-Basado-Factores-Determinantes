## 📘 Modelo de Predicción del Desempeño Académico de Estudiantes

### ELABORADO POR ESTUDIANTES DE LA UNIVERSIDAD PRIVADA ANTENOR ORREGO

### 😸 GRUPO C INTEGRANTES

* **Guerrero Puicón, Halvert**
* **Lopez Ortega, Alvaro**
* **Ortiz Barboza, Anghelo**
* **Rodriguez Lara, Franklin**
* **Vasquez Saenz, Juan**

### 🎯 Descripción

Esta aplicación permite predecir el desempeño académico de estudiantes universitarios utilizando un modelo de Random Forest. Ofrece una interfaz interactiva para la carga de datos, ajuste de hiperparámetros y visualización de resultados.

### 🚀 Acceso a la Aplicación

👉 [Haz clic aquí para acceder a la aplicación desplegada en Streamlit Cloud](https://grupoc-modelo-prediccion-desempeno-academico-estudiantes.streamlit.app/)

### 🛠️ Características

* **Carga de Datos:** Permite subir archivos CSV con los datos de los estudiantes.
* **Exploración de Datos:** Visualiza estadísticas descriptivas y distribuciones de variables.
* **Ajuste de Hiperparámetros:** Interfaz para modificar parámetros del modelo y observar su impacto.
* **Entrenamiento del Modelo:** Entrena un modelo de Random Forest con los datos proporcionados.
* **Evaluación del Modelo:** Muestra métricas como MAE y RMSE para evaluar el rendimiento.

### 📂 Estructura del Proyecto

```
Modelo-Prediccion-Desempeno-Academico-Estudiantes-Basado-Factores-Determinantes/
├── data_notebook/
│   └── kaggle.json
│   └── PROYECTO_FINAL_GRUPO_C.ipynb
├── models/
│   └── random_forest.pkl
├── app.py
├── requirements.txt
├── Procfile
├── EJEMPLO-StudentPerformanceFactors.csv
└── README.md
```

### 📦 Requisitos

* Python 3.10
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Altair

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

### 📂 Dataset de ejemplo

Puedes usar el archivo [`EJEMPLO-StudentPerformanceFactors.csv`](./EJEMPLO-StudentPerformanceFactors.csv) incluido en este repositorio para probar la app web.


### 🧠 Modelo Utilizado

Se emplea un modelo de Random Forest Regressor para predecir el rendimiento académico basado en factores determinantes como hábitos de estudio, asistencia y participación en clase.