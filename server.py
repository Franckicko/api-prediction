from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# Chargement des modèles
with open("model_trot.pkl", "rb") as f:
    model_a1 = pickle.load(f)

with open("model_trot_a2.pkl", "rb") as f:
    model_a2 = pickle.load(f)

# Préparation des hippodromes encodés
conn = sqlite3.connect("courses.db")
df_trot = pd.read_sql_query("SELECT DISTINCT HIPPODROME FROM courses WHERE lower(DISCIPLINE) = 'trot'", conn)
conn.close()

hippodromes_list = df_trot['HIPPODROME'].astype(str).str.strip().str.upper().tolist()
hippodrome_to_index = {name: idx for idx, name in enumerate(hippodromes_list)}

# Modèle de requête
class PredictionRequest(BaseModel):
    hippodrome: str
    nbre_partants: int

@app.get("/courses")
def get_courses():
    conn = sqlite3.connect("courses.db")
    df = pd.read_sql_query("SELECT * FROM courses", conn)
    conn.close()
    return df.to_dict(orient="records")

# Route : prédiction du 1er cheval (TOP 3)
@app.post("/predict")
def predict_course(data: PredictionRequest):
    hippo = data.hippodrome.strip().upper()
    if hippo not in hippodrome_to_index:
        return {"error": f"Hippodrome inconnu : {hippo}"}
    
    features = np.array([[hippodrome_to_index[hippo], data.nbre_partants]])
    proba = model_a1.predict_proba(features)[0]
    top3 = proba.argsort()[-3:][::-1]
    return {"predictions": top3.tolist()}

# Route : prédiction du 2ème cheval (TOP 4)
@app.post("/predict_a2")
def predict_course_second(data: PredictionRequest):
    hippo = data.hippodrome.strip().upper()
    if hippo not in hippodrome_to_index:
        return {"error": f"Hippodrome inconnu : {hippo}"}
    
    features = np.array([[hippodrome_to_index[hippo], data.nbre_partants]])
    proba = model_a2.predict_proba(features)[0]
    top4 = proba.argsort()[-4:][::-1]
    return {"predictions_2eme": top4.tolist()}

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de Prédiction Hippique 🏇"}
# Route : récupération de toutes les courses