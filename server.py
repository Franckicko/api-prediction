
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# Chargement des modèles TROT
with open("model_trot.pkl", "rb") as f:
    model_a1 = pickle.load(f)

with open("model_trot_a2.pkl", "rb") as f:
    model_a2 = pickle.load(f)

# Chargement des modèles GALOP (XGBoost)
with open("model_galop_xgb.pkl", "rb") as f:
    model_galop_a1 = pickle.load(f)

with open("model_galop_a2_xgb.pkl", "rb") as f:
    model_galop_a2 = pickle.load(f)

# Préparation des hippodromes encodés - TROT
conn = sqlite3.connect("courses.db")
df_trot = pd.read_sql_query("SELECT DISTINCT HIPPODROME FROM courses WHERE lower(DISCIPLINE) = 'trot'", conn)
df_galop = pd.read_sql_query("SELECT DISTINCT HIPPODROME FROM courses WHERE lower(DISCIPLINE) IN ('plat', 'obs')", conn)
conn.close()

hippodromes_list_trot = df_trot['HIPPODROME'].astype(str).str.strip().str.upper().tolist()
hippodromes_list_galop = df_galop['HIPPODROME'].astype(str).str.strip().str.upper().tolist()

hippodrome_to_index = {name: idx for idx, name in enumerate(hippodromes_list_trot)}
hippodrome_to_index_galop = {name: idx for idx, name in enumerate(hippodromes_list_galop)}

# Modèle de requête
class PredictionRequest(BaseModel):
    hippodrome: str
    nbre_partants: int

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de Prédiction Hippique 🏇"}

@app.get("/courses")
def get_courses():
    conn = sqlite3.connect("courses.db")
    df = pd.read_sql_query("SELECT * FROM courses", conn)
    conn.close()
    return df.to_dict(orient="records")

# Route : prédiction du 1er cheval TROT (TOP 3)
@app.post("/predict")
def predict_course(data: PredictionRequest):
    hippo = data.hippodrome.strip().upper()
    if hippo not in hippodrome_to_index:
        return {"error": f"Hippodrome inconnu (trot) : {hippo}"}
    
    features = np.array([[hippodrome_to_index[hippo], data.nbre_partants]])
    proba = model_a1.predict_proba(features)[0]
    top3 = proba.argsort()[-3:][::-1]
    return {"predictions": top3.tolist()}

# Route : prédiction du 2ème cheval TROT (TOP 4)
@app.post("/predict_a2")
def predict_course_second(data: PredictionRequest):
    hippo = data.hippodrome.strip().upper()
    if hippo not in hippodrome_to_index:
        return {"error": f"Hippodrome inconnu (trot) : {hippo}"}
    
    features = np.array([[hippodrome_to_index[hippo], data.nbre_partants]])
    proba = model_a2.predict_proba(features)[0]
    top4 = proba.argsort()[-4:][::-1]
    return {"predictions_2eme": top4.tolist()}

# Route : prédiction du 1er cheval GALOP (TOP 3)
@app.post("/predict_galop")
def predict_course_galop(data: PredictionRequest):
    hippo = data.hippodrome.strip().upper()
    if hippo not in hippodrome_to_index_galop:
        return {"error": f"Hippodrome inconnu (galop) : {hippo}"}
    
    features = np.array([[hippodrome_to_index_galop[hippo], data.nbre_partants]])
    proba = model_galop_a1.predict_proba(features)[0]
    top3 = proba.argsort()[-3:][::-1]
    return {"predictions_galop": top3.tolist()}

# Route : prédiction du 2ème cheval GALOP (TOP 4)
@app.post("/predict_galop_a2")
def predict_course_galop_second(data: PredictionRequest):
    hippo = data.hippodrome.strip().upper()
    if hippo not in hippodrome_to_index_galop:
        return {"error": f"Hippodrome inconnu (galop) : {hippo}"}
    
    features = np.array([[hippodrome_to_index_galop[hippo], data.nbre_partants]])
    proba = model_galop_a2.predict_proba(features)[0]
    top4 = proba.argsort()[-4:][::-1]
    return {"predictions_galop_2eme": top4.tolist()}
