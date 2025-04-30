from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import pandas as pd
# Imports
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import pandas as pd
import numpy as np
import pickle

# App initialization
app = FastAPI()

# Charger le modèle au démarrage
with open(r'C:\Users\Francesco\Documents\FlutterProjects\PREDICTION_Q\assets\model_trot.pkl', 'rb') as f:
    model = pickle.load(f)

# Récupérer tous les hippodromes connus pour l'encodage
conn = sqlite3.connect(r'C:\Users\Francesco\Documents\FlutterProjects\PREDICTION_Q\assets\courses.db')
df_trot = pd.read_sql_query("SELECT DISTINCT HIPPODROME FROM courses WHERE lower(DISCIPLINE) = 'trot'", conn)
conn.close()
hippodromes_list = df_trot['HIPPODROME'].astype(str).apply(lambda x: x.strip().upper()).tolist()
hippodrome_to_index = {name: idx for idx, name in enumerate(hippodromes_list)}

# Classe pour recevoir les données de prédiction
class PredictionRequest(BaseModel):
    hippodrome: str
    nbre_partants: int

# Routes
@app.get("/courses")
def get_courses():
    # Connexion à la base SQLite
    conn = sqlite3.connect(r'C:\Users\Francesco\Documents\FlutterProjects\PREDICTION_Q\assets\courses.db')
    df = pd.read_sql_query("SELECT * FROM courses", conn)
    conn.close()

    # Convertir le DataFrame en liste de dictionnaires
    return df.to_dict(orient="records")

@app.post("/predict")
def predict_course(data: PredictionRequest):
    hippo = data.hippodrome.strip().upper()

    if hippo not in hippodrome_to_index:
        return {"error": f"Hippodrome inconnu : {hippo}"}

    encoded_hippo = hippodrome_to_index[hippo]
    input_data = np.array([[encoded_hippo, data.nbre_partants]])

    proba = model.predict_proba(input_data)[0]
    top3 = proba.argsort()[-3:][::-1]  # Les 3 meilleurs candidats

    return {"predictions": top3.tolist()}
