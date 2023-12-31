
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

#si se tiene la data en una BD, esto es pensando en la producción donde los datos son etiquetdos y dejandos para el re-entrenamiento 
def extract_data_db(connection_string, query):
    engine = create_engine(connection_string)
    df = pd.read_sql(query, engine)
    return df

#para usar data local
def extract_data_local(input_path):    
    df = pd.read_csv(input_path,sep=',')
    return df

# Definir una función para aplicar las transformaciones
def transform_data(df):
    # Transformaciones que se aprendieron en la exploración
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d %H:%M:%S')
    df.drop(columns='o', inplace=True)
    df["p"] = np.where(df["p"] == "Y", 1, 0)
    df['b'].fillna(df['b'].median(), inplace=True)
    df['c'].fillna(df['c'].median(), inplace=True)
    df['d'].fillna(df['d'].median(), inplace=True)
    df['f'].fillna(df['f'].median(), inplace=True)
    df['l'].fillna(df['l'].median(), inplace=True)
    df['m'].fillna(df['m'].median(), inplace=True)
    
    var=pd.DataFrame(df[['g']].value_counts().head(4)).reset_index()['g'].values
    
    df['g'] = np.where(df['g'].isin(var), df['g'], 'otros')
    df = pd.concat([df, pd.get_dummies(df['g'], prefix='Country').astype(int)], axis=1)
    df = df[df['e'] < df['e'].quantile(0.99)]
    df = df[df['f'] < df['f'].quantile(0.99)]

    df['hour_early']=np.where((df.fecha.dt.hour>=0 ) & (df.fecha.dt.hour<5),1,0)
    
    return df

# Definir una función para cargar datos en un archivo
def save_data_to_file(df, output_path):
    df.to_csv(output_path, index=False)

# Ejecutar el proceso ETL
def etl_process():

    #En caso de conectar a BD ejecutar esta funcion 
    #connection_string = 'Driver={SQL Server Native Client 11.0};Server=tu_servidor;Database=tu_base_de_datos;Uid=tu_usuario;Pwd=tu_contraseña;'
    #query = 'SELECT * FROM tu_tabla'  # Reemplaza con tu consulta
    #extracted_data = extract_data_db(connection_string, query)

    input_path = os.path.join(current_dir, '../raw_data/MercadoLibre Data Scientist Technical Challenge - Dataset.csv')
    extracted_data = extract_data_local(input_path)

    
    # Aplicar transformaciones
    transformed_data = transform_data(extracted_data)
    
    # Escribir el DataFrame transformado en un archivo CSV
    output_path = os.path.join(current_dir, '../processed_data/data_processed.csv')
    save_data_to_file(transformed_data, output_path)


if __name__ == "__main__":
    etl_process()
