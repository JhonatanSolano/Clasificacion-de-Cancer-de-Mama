from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

def load_and_describe_data():
    """Carga el dataset de Cáncer de Mama y muestra una descripción básica."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Mostrar descripción en terminal
    print(f"  Dataset: {data.DESCR.splitlines()[0]}")
    print(f"  Número de muestras (filas): {X.shape[0]}")
    print(f"  Número de características (columnas): {X.shape[1]}")
    print(f"  Clases objetivo: {list(data.target_names)}")
    print("  Conteo de clases (0: Maligno, 1: Benigno):")
    print(y.value_counts())
    
    return X, y, data