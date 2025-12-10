from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def setup_pipelines_and_grids(random_state):
    """Define los pipelines de modelos y sus grillas de hiperparámetros."""
    
    # 1. Pipelines (preprocesamiento + clasificador)
    pipelines = {
        "LogisticRegression": Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', LogisticRegression(max_iter=5000, solver='liblinear', random_state=random_state))
        ]),
        "RandomForestClassifier": Pipeline([
            ('clf', RandomForestClassifier(random_state=random_state))
        ]),
        "SVC": Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', SVC(probability=True, random_state=random_state))
        ])
    }

    # 2. Grillas de parámetros
    param_grids = {
        "LogisticRegression": {'clf__C': [0.01, 0.1, 1, 10]},
        "RandomForestClassifier": {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 5, 10]},
        "SVC": {'clf__C': [0.1, 1, 10], 'clf__kernel': ['rbf', 'linear']}
    }
    
    return pipelines, param_grids