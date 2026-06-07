# Tech & Data Science Salary Predictor

An interactive web dashboard that predicts tech and data science salaries based on job title, experience level, employment type, company size, work setting, and location. Built using Python and Dash, trained on 3,755 records, and deployed on Render.

**Live Demo:** https://swe-salary-predictor.onrender.com

> Note: The app is hosted on Render's free tier and may take 1-2 minutes to load if it hasn't been accessed recently.

---

## What It Does

- Predicts estimated annual salary based on user inputs using a trained Random Forest model
- Compares prediction against dataset medians by experience level
- Visualizes salary trends by year, experience level, job title, work setting, and company size
- Shows model performance metrics for all three trained models

---

## How It Works

### Data
The dataset contains 3,755 records of AI, ML, and data science salaries from 2020–2023 sourced from Kaggle, covering 93 unique job titles across 72 countries.

### Feature Engineering
Several new features were derived from the raw data to improve model performance:
- **Experience Score** – numeric mapping of experience level (Entry=1, Mid=2, Senior=3, Executive=4)
- **Is Remote** – binary flag for fully remote positions
- **Size Score** – numeric mapping of company size (Small=1, Medium=2, Large=3)
- **Is US** – binary flag for US-based companies

### Models Trained
Three models were trained and evaluated using R², MAE, and RMSE:
- Linear Regression (baseline)
- Decision Tree Regressor
- Random Forest Regressor (best performing)

### Preprocessing Pipeline
Categorical columns (job title, experience level, employment type, company location, company size) were encoded using OneHotEncoder. Numerical columns were scaled using StandardScaler. Both transformations were combined using a ColumnTransformer pipeline from scikit-learn.

---

## Dashboard Tabs

- **Salary Predictor** – input your profile and get an estimated salary
- **Visualizations** – explore salary trends across experience, role, year, work setting, and company size
- **Model Performance** – compare R², MAE, and RMSE across all three models

---

## Built With

- **Python** – core scripting and ML
- **Pandas** – data loading and feature engineering
- **Scikit-learn** – preprocessing pipelines and ML models
- **Dash / Plotly** – interactive web dashboard
- **Render** – cloud deployment

---

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/bornitac/SWESalaryPredictor
   ```
2. Train the models:
   ```bash
   python train_model.py
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:10000` in your browser

---

## Data Source

**AI/ML/Data Science Salary Dataset** — Kaggle  
https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023
