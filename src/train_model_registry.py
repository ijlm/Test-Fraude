import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

import mlflow


#directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

#metricas para evaluar los resultados del modelo 
def evaluate_models(y_test, y_pred):    
    accuracy=accuracy_score(y_test, y_pred)
    precision =precision_score(y_test, y_pred)
    recall =recall_score(y_test, y_pred)
    f1 =f1_score(y_test, y_pred)
    roc_auc =roc_auc_score(y_test, y_pred)
    matrix=confusion_matrix(y_test, y_pred)
    return (accuracy,precision,recall,f1,roc_auc,matrix)

#metrica que nos dara el corte optimo del modelo 
def definir_corte_optimo(y_test, y_prob, costo_fp, costo_fn, beneficio_tp, beneficio_tn):
    corte = np.arange(0, 1, 0.01)     
    punto_optimo = None
    max_rentabilidad = -float('inf')    
    for threshold in corte:
        y_pred = (y_prob >= threshold).astype(int)        
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fn = np.sum((y_test == 1) & (y_pred == 0))        
        rentabilidad = tp * beneficio_tp + tn * beneficio_tn - fp * costo_fp - fn * costo_fn        
        if rentabilidad > max_rentabilidad:
            max_rentabilidad = rentabilidad
            punto_optimo = threshold    
    return punto_optimo

#se entrenael modelo y se carga la información
def process(data):        
    #cargamos el dataset procesado para el modelo 
    df=pd.read_csv(data)
    df['fecha']=pd.to_datetime(df.fecha,format='%Y-%m-%d %H:%M:%S')
    df.drop(columns=['g','j','fecha'],inplace=True)

    #realizamos el sobre muestreo que nos ayudara con un mejor modelo
    X=df.drop(columns='fraude')
    y=df.fraude
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    #separamos los datos para la validación del modelo 
    X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled , test_size=0.3, random_state=42)

    # Entrenar el modelo LightGBM, con los hiperparámetros del modelo
    train_data = lgb.Dataset(X_train, label=y_train)

    params = {
        "objective": "binary","metric": "binary_logloss","boosting_type": "gbdt",
        "num_leaves": 31,"subsample_for_bin":200000,"max_depth":-1,
        "min_child_samples":20, "min_child_weight":0.001,"learning_rate": 0.1,
        "feature_fraction": 0.9,"n_jobs":-1,"boosting_type":'gbdt'
    }

    # Entrenar el modelo LightGBM
    num_round = 1000
    bst = lgb.train(params, train_data, num_round)
    y_pred = bst.predict(X_test)

    costo_fp = 1.0  # Costo de un falso positivo FP es todo ya que se pierde el ingreso por error del modelo
    costo_fn = 1.0  # Costo de un falso negativo FN es todo ya que el modelo no lo pudo detectar
    beneficio_tp = 0.0  # Detectar el fraude es bueno no genera costos, pero no genera ganacia adicional al ingreso
    beneficio_tn = 0.25  # Detectar los no fraudes correctamente ayuda a generar ese 25% adicional de ganacia
    threshold=definir_corte_optimo(y_test, y_pred, costo_fp, costo_fn, beneficio_tp, beneficio_tn)


    y_pred_binario = (y_pred > threshold).astype(int)
    accuracy,precision,recall,f1_score,roc_auc,matrix=evaluate_models(y_test, y_pred_binario)

    #registrarmos en mlflow para hacer seguimiento a las ejecuciones de los modelos y ver si mantiene su desempeño
    mlflow.set_tracking_uri('http://localhost:5000/')
    experiment_name = "meli_fraude"    
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name) 
    experiment = mlflow.get_experiment_by_name(experiment_name)        
    with mlflow.start_run(experiment_id = experiment.experiment_id):
        metrics = {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'threshold': threshold
        }
        mlflow.log_metrics(metrics)

    
    #guardamos el modelo compilado
    model = os.path.join(current_dir, '../model/modelo_lightgbm_fraude.txt')    
    bst.save_model(model)


if __name__ == "__main__":
    input_path = os.path.join(current_dir, '../processed_data/data_processed.csv')
    process(input_path)


