import fastapi
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Carga el modelo LightGBM, en este caso local, pero lo mismo se podria hacer desde el repositorio de modelos
model_filename = 'model/modelo_lightgbm_fraude.txt'
loaded_model = lgb.Booster(model_file=model_filename)

# Define la estructura de los datos de entrada
class InputData(BaseModel):
    a: int
    b: float
    c: float
    d: float
    e: float
    f: float
    h: int
    k: float
    l: float
    m: float
    n: int
    p: int
    monto: float
    score: int
    Country_AR: int
    Country_BR: int
    Country_US: int
    Country_UY: int
    Country_otros: int
    hour_early: int

#metodo post para llamado de la pai
@app.post("/predict/")
async def predict(data: InputData):    
    input_data = {'a': [data.a],'b': [data.b],'c': [data.c],'d': [data.d],'e': [data.e],'f': [data.f],
        'h': [data.h],'k': [data.k],'l': [data.l],'m': [data.m],'n': [data.n],'p': [data.p],'monto': [data.monto],
        'score': [data.score],'Country_AR': [data.Country_AR],'Country_BR': [data.Country_BR],'Country_US': [data.Country_US],
        'Country_UY': [data.Country_UY],'Country_otros': [data.Country_otros],'hour_early': [data.hour_early]
    }

    # Realiza predicciones con el modelo 
    X_test = pd.DataFrame(input_data)
    y_pred = loaded_model.predict(X_test)

    # Devuelve la predicción como respuesta
    return {"prediction": y_pred[0]}

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="127.0.0.1", port=80)