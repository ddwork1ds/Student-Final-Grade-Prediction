import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from pathlib import Path
import sys
BASE_DIR = Path.cwd()
sys.path.append(str(BASE_DIR))
from config import PROCESSED_DATA, MODEL_DIR
df = pd.read_csv(PROCESSED_DATA)

df.head()

RANDOM_STATE = 42

# Thiết lập Cross-Validation cố định
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

features = ['sex_encoded', 'age', 'failures', 'higher_encoded', 'absences', 'G_Avg']
target = "G3_10"

# Chuẩn bị dữ liệu 
X = df[features].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

models = {
    "Baseline(mean)": {
        "model": DummyRegressor(strategy="mean"),
        "params": {}
    },
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "params": {
            "max_depth": [3, 5, 7], 
            "min_samples_split": [5, 10], 
            "min_samples_leaf": [2, 5] 
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE),
        "params": {
            "max_depth": [3, 7], 
            "min_samples_leaf": [5, 10, 20], 
            "n_estimators": [50, 100]
        }
    }
}

best_models=[]

models_metrics = {}

for name, config in models.items():
    print(f"Training {name}")
    
    grid = GridSearchCV(config["model"], config["params"], cv=cv, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    best_models.append({
    "model": name,
    "best_params": grid.best_params_,
    "rmse": rmse,
    "r2": r2
})
    models_metrics[name] = {
        "best_params": grid.best_params_,
        "rmse": rmse,
        "r2": r2,
        "best_estimator": grid.best_estimator_
    }

best_models

results_df = pd.DataFrame(best_models)
results_df 

results_df.sort_values(by="rmse")

best_row = results_df.sort_values(by="rmse").iloc[0]

best_row

best_model_name = best_row["model"]

best_model_name

best_params = best_row['best_params']
best_params

best_model = models[best_model_name]["model"].set_params(**best_params)

best_model.fit(X_train, y_train)

joblib.dump(best_model, MODEL_DIR/"best_model.pkl")

joblib.dump(models_metrics, MODEL_DIR/"models_metrics.pkl")
print("✅ Đã lưu metrics của baseline và candidates: models_metrics.pkl")
print("📊 Metrics đã lưu:")
for name, metrics in models_metrics.items():
    print(f"  {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")