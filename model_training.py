import pandas as pd
import sqlite3
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Connexion à la base SQLite
conn = sqlite3.connect(r'C:\Users\Francesco\Documents\FlutterProjects\PREDICTION_Q\assets\courses.db')

# 2. Charger toutes les courses de TROT
df = pd.read_sql_query("SELECT * FROM courses WHERE lower(DISCIPLINE) = 'trot'", conn)
conn.close()

# 3. Préparer les données
features = ['HIPPODROME', 'NBRE_PARTANTS']
df = df.dropna(subset=features + ['A1'])  # Supprimer les lignes incomplètes

# Encodage du champ HIPPODROME en numérique
df['HIPPODROME'] = df['HIPPODROME'].astype(str)
df['hippodrome_encoded'] = pd.factorize(df['HIPPODROME'])[0]

# Définir X (entrées) et y (sortie/gagnant)
X = df[['hippodrome_encoded', 'NBRE_PARTANTS']]
y = df['A1']

# 4. Séparer entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entraîner le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy sur jeu de test : {accuracy*100:.2f}%")

# 7. Sauvegarder le modèle dans un fichier .pkl
model_path = r'C:\Users\Francesco\Documents\FlutterProjects\PREDICTION_Q\assets\model_trot.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Modèle sauvegardé sous : {model_path}")
