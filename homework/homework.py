# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#

# Carga de librerias
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Carga de datos
train_data_zip = 'files/input/train_data.csv.zip'
test_data_zip = 'files/input/test_data.csv.zip'

# Extraccion de los datos de los archivos zip
with ZipFile(train_data_zip, 'r') as zip_ref:
    with zip_ref.open('train_default_of_credit_card_clients.csv') as f:
        train_data=pd.read_csv(f)

with ZipFile(test_data_zip, 'r') as zip_ref:
    with zip_ref.open('test_default_of_credit_card_clients.csv') as f:
        test_data=pd.read_csv(f)

# Renombrar la columna "default payment next month" a "default"
train_data.rename(columns={'default payment next month': 'default'}, inplace=True)
test_data.rename(columns={'default payment next month': 'default'}, inplace=True)

# Remover la columna "ID"
train_data.drop(columns='ID', inplace=True)
test_data.drop(columns='ID', inplace=True)

# Recodificar la variable EDUCATION: 0 es "NaN"
train_data['EDUCATION'] = train_data['EDUCATION'].replace(0, np.nan)
test_data['EDUCATION'] = test_data['EDUCATION'].replace(0, np.nan)

# Recodificar la variable MARRIAGE: 0 es "NaN"

train_data['MARRIAGE'] = train_data['MARRIAGE'].replace(0, np.nan)
test_data['MARRIAGE'] = test_data['MARRIAGE'].replace(0, np.nan)

# Eliminar los registros con informacion no disponible (es decir, con al menos una columna con valor nulo)
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Agrupar los valores de EDUCATION > 4 en la categoria "others"
train_data.loc[train_data['EDUCATION'] > 4, 'EDUCATION'] = 4
test_data.loc[test_data['EDUCATION'] > 4, 'EDUCATION'] = 4

# Reclasificar las variables PAY_i, EDUCATION, SEX y MARRIAGE a categoricas
cat_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'EDUCATION', 'SEX', 'MARRIAGE']
for col in cat_columns:
    train_data[col] = train_data[col].astype('category')
    test_data[col] = test_data[col].astype('category')

# Guardar los datasets limpios
train_data.to_csv('train_data_clean.csv', index=False)


# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
x_train = train_data.drop(columns='default')
y_train = train_data['default']
x_test = test_data.drop(columns='default')
y_test = test_data['default']
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding. (SEX, EDUCATION, MARRIAGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
# - Ajusta un modelo de bosques aleatorios (rando forest).
#

# Crear el pipeline
pipeline = Pipeline([
    ('encoder', OneHotEncoder(drop='first')),
    ('model', RandomForestClassifier())
])

#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo como "files/models/model.pkl".
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
