import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report)

def train_models_grid_search(pipelines, param_grids, X_train, y_train, X_test, y_test, target_names):
    """Entrena modelos usando GridSearchCV y evalúa resultados iniciales."""
    
    best_estimators = {}
    results = []

    for name, pipeline in pipelines.items():
        print(f"  > Iniciando entrenamiento para: {name}...")
        
        # GridSearchCV para optimización de hiperparámetros (cv=5, scoring='roc_auc')
        grid = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        
        best = grid.best_estimator_
        best_estimators[name] = best

        # Evaluación en conjunto de Test
        y_pred = best.predict(X_test)
        
        # Obtener probabilidades/decision function para ROC-AUC
        if hasattr(best, "predict_proba"):
            y_proba = best.predict_proba(X_test)[:,1]
        else:
            y_proba = best.decision_function(X_test)

        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        
        # Mostrar reporte de clasificación detallado en la terminal
        print(f"\n  Resultados de Test para {name}: AUC={roc:.3f}, Acc={acc:.3f}")
        print(classification_report(y_test, y_pred, target_names=target_names))

        results.append({
            "model": name,
            "best_params": grid.best_params_,
            "accuracy": f"{acc:.4f}",
            "precision": f"{prec:.4f}",
            "recall": f"{rec:.4f}",
            "f1": f"{f1:.4f}",
            "roc_auc": f"{roc:.4f}"
        })
        
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    
    return best_estimators, results_df