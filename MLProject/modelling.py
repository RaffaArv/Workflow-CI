import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def run_modelling():
    print("Membaca dataset...")
    df = pd.read_csv('Telcochurn_preprocessing/cleaned_telco_auto.csv')
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="CI_RandomForest"):
        print("Training model...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan artefak model secara eksplisit biar gampang ditarik Docker
        if os.path.exists("saved_model"):
            shutil.rmtree("saved_model")
        mlflow.sklearn.save_model(model, "saved_model")
        
        print("Training selesai!")

if __name__ == "__main__":
    run_modelling()