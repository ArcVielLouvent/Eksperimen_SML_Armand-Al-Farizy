import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../preprocessing/dataset_preprocessing/data_clean.csv')

numeric_cols = df.select_dtypes(include='number').columns.tolist()

if 'Survived' in numeric_cols:
    target = 'Survived'
    numeric_cols.remove('Survived')
    features = numeric_cols
else:
    target = numeric_cols[-1]
    features = numeric_cols[:-1]
    df[target] = (df[target] > df[target].mean()).astype(int) 

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Basic_Autolog_Experiment")
mlflow.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    print("Pelatihan Dasar Selesai!")
