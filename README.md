# 🩺 Proyecto 3: Clasificación de Cáncer de Mama (ML Supervisado)

## Introducción

Este proyecto implementa y evalúa tres modelos de Aprendizaje Automático (ML) Supervisado para la clasificación binaria de tumores de mama como **malignos** o **benignos**, utilizando el popular dataset de diagnóstico de Wisconsin. El objetivo es identificar el clasificador más robusto y preciso para una tarea crítica de diagnóstico médico.

El código está organizado de manera **modular**, separando la carga de datos, el entrenamiento, la evaluación y la visualización en scripts independientes para mejorar la claridad y la reproducibilidad.

---

## ⚙️ Flujo y Estructura del Proyecto

La ejecución se orquesta mediante el script principal (`00_main_executor.py`) que llama a los módulos en el siguiente orden:

1.  **Carga de Datos (`01_data_loader.py`):** Carga el Wisconsin Breast Cancer Dataset y muestra el conteo de clases.
2.  **Análisis Exploratorio (`02_eda_visualizer.py`):** Calcula y muestra la Matriz de Correlación de las 30 características. 
3.  **Setup de Entrenamiento (`03_trainer_setup.py`):** Define los Pipelines (incluyendo `StandardScaler` cuando es necesario) y los diccionarios de hiperparámetros (`param_grids`).
4.  **Entrenamiento y Evaluación (`04_model_trainer.py`):** Ejecuta `GridSearchCV` para los modelos Logistic Regression, Random Forest, y SVC. Muestra los reportes de clasificación detallados en la consola.
5.  **Visualización (`06_plot_results.py`):** Muestra la Matriz de Confusión del mejor modelo y las Curvas ROC comparativas.

## 🚀 Requisitos de Instalación

Instala todas las dependencias necesarias usando el archivo `requirements.txt`:

1.  *Instalar Dependencias:*
    ```bash
    pip install -r requirements.txt
    ```

### Contenido de `requirements.txt`:

Dependencias principales para el proyecto de Clasificación ML: pandas, numpy, scikit-learn, matplotlib, tabulate.

## 🛠️ Guía de Ejecución

Ejecuta el script principal para iniciar el flujo completo. Los resultados de texto y las tablas aparecerán en la consola, y los gráficos se abrirán en ventanas separadas.

```bash
python 00_main_executor.py
