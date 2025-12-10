import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Importar funciones de los módulos
from _01_data_loader import load_and_describe_data
from _02_eda_visualizer import perform_eda_and_plot
from _03_trainer_setup import setup_pipelines_and_grids
from _04_model_trainer import train_models_grid_search
from _06_plot_results import plot_best_model_cm, plot_roc_curves, plot_feature_importance

RANDOM_STATE = 42

print("=========================================================")
print("  PROYECTO 3: CLASIFICACIÓN CÁNCER DE MAMA (MODULAR) ")
print("=========================================================")

# --- 1) Carga y Descripción del Dataset ---
print("\n[PASO 1] Cargando y describiendo el dataset...")
X, y, data = load_and_describe_data()

# --- 2) Train/test split ---
print("\n[PASO 2] Realizando el split de datos (Train/Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Tamaño Train: {len(X_train)} | Tamaño Test: {len(X_test)}")

# --- 3) EDA y Visualización de Correlación ---
print("\n[PASO 3] Análisis Exploratorio de Datos (EDA) y Matriz de Correlación.")
# Este script abrirá la ventana gráfica de correlación.
perform_eda_and_plot(X)

# --- 4) Setup de Pipelines y Grids ---
print("\n[PASO 4] Configurando Pipelines y Grids de Hiperparámetros...")
pipelines, param_grids = setup_pipelines_and_grids(RANDOM_STATE)
print(f"  Modelos a entrenar: {list(pipelines.keys())}")

# --- 5) Entrenamiento de Modelos (GridSearchCV) y Evaluación de Resultados ---
print("\n[PASO 5] Iniciando Entrenamiento con GridSearchCV (scoring: ROC-AUC)...")
best_estimators, results_df = train_models_grid_search(
    pipelines, param_grids, X_train, y_train, X_test, y_test, data.target_names
)
# Mostrar tabla de resultados en la consola
print("\n--- RESUMEN DE RESULTADOS DE EVALUACIÓN (TEST) ---")
print(results_df.to_markdown(index=False))

# --- 6) Visualización de Resultados ---
best_name = results_df.iloc[0]['model']
best_model = best_estimators[best_name]

print(f"\n[PASO 6] Visualización de Resultados para el mejor modelo: **{best_name}**")

# Matriz de Confusión
plot_best_model_cm(best_model, X_test, y_test, best_name, data.target_names) # Abre ventana gráfica

# Curvas ROC
plot_roc_curves(best_estimators, X_test, y_test) # Abre ventana gráfica

# Importancia de Características (solo si Random Forest fue entrenado)
if 'RandomForestClassifier' in best_estimators:
    plot_feature_importance(best_estimators['RandomForestClassifier'], X.columns) # Abre ventana gráfica

print("\n==========================")
print("      EJECUCIÓN COMPLETA.     ")
print("==========================")