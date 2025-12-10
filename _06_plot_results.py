import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def plot_best_model_cm(model, X_test, y_test, model_name, target_names):
    """Muestra la Matriz de Confusión para el mejor modelo."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    
    # Añadir números al centro de cada casilla
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.show() # Muestra la gráfica

def plot_roc_curves(best_estimators, X_test, y_test):
    """Muestra las Curvas ROC para todos los modelos comparados."""
    plt.figure(figsize=(8, 6))
    
    for name, est in best_estimators.items():
        # Obtener probabilidades o decision function
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X_test)[:,1]
        else:
            p = est.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        
    plt.plot([0,1],[0,1],'k--', label='Chance (AUC=0.500)')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR) / Recall")
    plt.title("Curvas ROC para Comparación de Modelos")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show() # Muestra la gráfica
    
def plot_feature_importance(rf_estimator, feature_names):
    """Muestra la importancia de características para Random Forest."""
    
    # Acceder al clasificador dentro del pipeline
    rf = rf_estimator.named_steps['clf']
    
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    importances.head(10).plot(kind='bar', color='darkgreen')
    plt.title("Top 10 Importancia de Características (Random Forest)")
    plt.ylabel("Puntuación de Importancia")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() # Muestra la gráfica