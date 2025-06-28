## 📘 Modelo de Predicción del Desempeño Académico de Estudiantes

### 🎯 Descripción

Esta aplicación permite predecir el desempeño académico de estudiantes universitarios utilizando un modelo de Random Forest. Ofrece una interfaz interactiva para la carga de datos, ajuste de hiperparámetros y visualización de resultados.

### 🚀 Acceso a la Aplicación

👉 [Haz clic aquí para acceder a la aplicación desplegada en Streamlit Cloud](https://grupoc-modelo-prediccion-desempeno-academico-estudiantes.streamlit.app/)

### 🛠️ Características

* **Carga de Datos:** Permite subir archivos CSV con los datos de los estudiantes.
* **Exploración de Datos:** Visualiza estadísticas descriptivas y distribuciones de variables.
* **Ajuste de Hiperparámetros:** Interfaz para modificar parámetros del modelo y observar su impacto.
* **Entrenamiento del Modelo:** Entrena un modelo de Random Forest con los datos proporcionados.
* **Evaluación del Modelo:** Muestra métricas como MAE, RMSE y R² para evaluar el rendimiento.

### 📂 Estructura del Proyecto

```
modelo-prediccion-estudiantes/
├── app.py
├── requirements.txt
├── Procfile
├── models/
│   └── random_forest.pkl
└── README.md
```

### 📦 Requisitos

* Python 3.10 o superior
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

### 🧠 Modelo Utilizado

Se emplea un modelo de Random Forest Regressor para predecir el rendimiento académico basado en factores determinantes como hábitos de estudio, asistencia y participación en clase.

### 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar esta aplicación, por favor realiza un fork del repositorio y envía un pull request.