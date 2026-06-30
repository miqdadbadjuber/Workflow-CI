import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ambil parameter data_path jika ada
data_path = sys.argv[1] if len(sys.argv) > 1 else "student_performance_preprocessing"

# load data hasil preprocessing
train_df = pd.read_csv(f'{data_path}/train.csv')
test_df = pd.read_csv(f'{data_path}/test.csv')

X_train = train_df.drop(columns=['GradeClass'])
y_train = train_df['GradeClass']
X_test = test_df.drop(columns=['GradeClass'])
y_test = test_df['GradeClass']

# enable autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="random_forest_ci"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"akurasi model: {acc:.4f}")

print("training selesai!")