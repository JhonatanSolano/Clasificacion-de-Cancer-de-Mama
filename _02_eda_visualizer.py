import matplotlib.pyplot as plt
import pandas as pd

def perform_eda_and_plot(X):
    """Calcula y muestra la matriz de correlación de características."""
    
    corr = X.corr()
    
    plt.figure(figsize=(12, 10))
    # Usamos cmap='coolwarm' para mejorar la visualización de la correlación positiva/negativa
    plt.imshow(corr, interpolation='nearest', cmap='coolwarm') 
    plt.title("Matriz de Correlación de Características")
    plt.colorbar(shrink=0.7)
    
    # Configurar los labels del eje para que sean legibles
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    
    plt.tight_layout()
    plt.show() # Muestra la gráfica