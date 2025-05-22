import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow

def train():
    
    df = pd.read_csv('../data/creditcard_sample.csv')
    
  
    X = df.drop('Class', axis=1)
    y = df['Class']
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    
  
    with mlflow.start_run():
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
    
  
    joblib.dump(model, '../models/model.joblib')

if __name__ == "__main__":
    train()
