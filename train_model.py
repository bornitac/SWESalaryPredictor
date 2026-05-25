import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ── Load & Prepare Data ──────────────────────────────────────────────────────
df = pd.read_csv('ds_salaries.csv')

exp_score_map  = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
size_score_map = {'S': 1, 'M': 2, 'L': 3}

df['experience_score'] = df['experience_level'].map(exp_score_map)
df['is_remote']        = (df['remote_ratio'] == 100).astype(int)
df['size_score']       = df['company_size'].map(size_score_map)
df['is_us']            = (df['company_location'] == 'US').astype(int)

# ── Features ─────────────────────────────────────────────────────────────────
selected_columns = [
    'experience_level', 'employment_type', 'job_title',
    'company_location', 'company_size', 'remote_ratio',
    'work_year', 'experience_score', 'is_remote',
    'size_score', 'is_us'
]

categorical_cols = ['experience_level', 'employment_type', 'job_title',
                    'company_location', 'company_size']
numerical_cols   = ['remote_ratio', 'work_year', 'experience_score',
                    'is_remote', 'size_score', 'is_us']

df_model = df[selected_columns + ['salary_in_usd']].dropna()
X = df_model[selected_columns]
y = df_model['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessor ─────────────────────────────────────────────────────────────
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# ── Train Models ──────────────────────────────────────────────────────────────
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
lr_pipeline.fit(X_train, y_train)
lr_r2  = r2_score(y_test, lr_pipeline.predict(X_test))
lr_mae = mean_absolute_error(y_test, lr_pipeline.predict(X_test))
lr_mse = mean_squared_error(y_test, lr_pipeline.predict(X_test))

dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=42))])
dt_pipeline.fit(X_train, y_train)
dt_r2  = r2_score(y_test, dt_pipeline.predict(X_test))
dt_mae = mean_absolute_error(y_test, dt_pipeline.predict(X_test))
dt_mse = mean_squared_error(y_test, dt_pipeline.predict(X_test))

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
rf_pipeline.fit(X_train, y_train)
rf_r2  = r2_score(y_test, rf_pipeline.predict(X_test))
rf_mae = mean_absolute_error(y_test, rf_pipeline.predict(X_test))
rf_mse = mean_squared_error(y_test, rf_pipeline.predict(X_test))

print(f"Linear Regression  R²: {lr_r2:.4f}")
print(f"Decision Tree      R²: {dt_r2:.4f}")
print(f"Random Forest      R²: {rf_r2:.4f}")

# ── Save Models & Metrics ────────────────────────────────────────────────────
with open('swe_salary_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_pipeline, f)
with open('swe_salary_lr_model.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)
with open('swe_salary_dt_model.pkl', 'wb') as f:
    pickle.dump(dt_pipeline, f)

metrics = {
    'lr': {'r2': lr_r2, 'mae': lr_mae, 'rmse': float(np.sqrt(lr_mse))},
    'dt': {'r2': dt_r2, 'mae': dt_mae, 'rmse': float(np.sqrt(dt_mse))},
    'rf': {'r2': rf_r2, 'mae': rf_mae, 'rmse': float(np.sqrt(rf_mse))},
}
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Models and metrics saved successfully!")
