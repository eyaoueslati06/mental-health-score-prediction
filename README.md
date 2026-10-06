# 🧠 Mental Health Score Prediction

A machine learning regression project that predicts a student's **Mental Health Score** based on social media usage, study habits, sleep, physical activity, stress level, and demographic information.

The project covers the complete machine learning workflow, including:

- Data exploration
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Categorical encoding
- Feature scaling
- Train-test splitting
- Machine learning pipelines
- Linear Regression
- Random Forest Regression
- Hyperparameter tuning
- Model evaluation
- Model serialization

---

## 📌 Project Objective

The goal of this project is to predict:

```text
Mental_Health_Score
```

using information about students' social media behavior and lifestyle.

Since `Mental_Health_Score` is a continuous numerical variable, this is a **regression problem**.

---

## 📊 Dataset

The dataset contains approximately **5,000 student records** and includes demographic, lifestyle, academic, and social media usage information.

### Features

| Feature | Description |
|---|---|
| `Age` | Student age |
| `Gender` | Student gender |
| `Country` | Student country |
| `Academic_Level` | Current academic level |
| `Most_Used_Platform` | Most frequently used social media platform |
| `Purpose_Of_Use` | Main reason for using social media |
| `Avg_Daily_Usage_Hours` | Average daily social media usage |
| `Daily_Unlocks` | Number of daily device unlocks |
| `Study_Hours` | Average study hours |
| `Physical_Activity_Hours` | Average physical activity hours |
| `Sleep_Hours_Per_Night` | Average nightly sleep |
| `Stress_Level` | Student stress level |
| `Mental_Health_Score` | Target variable |

---

## 🔎 Exploratory Data Analysis

Several visualizations were used to understand the relationships between the features and the target.

The analysis included:

- Distribution of Mental Health Score
- Correlation heatmap
- Stress Level vs Mental Health Score
- Social Media Usage vs Mental Health Score
- Sleep Hours vs Mental Health Score
- Outlier analysis
- Skewness analysis

### Key Observations

The correlation analysis showed several important relationships.

- `Avg_Daily_Usage_Hours` has a strong negative relationship with Mental Health Score.
- `Daily_Unlocks` also has a strong negative relationship with Mental Health Score.
- `Study_Hours` has a strong positive relationship with Mental Health Score.
- `Sleep_Hours_Per_Night` has a strong positive relationship with Mental Health Score.
- `Physical_Activity_Hours` has a moderate positive relationship with Mental Health Score.
- `Age` has only a weak relationship with Mental Health Score.

The analysis suggests that higher social media usage is associated with lower mental health scores, while more sleep, studying, and physical activity tend to be associated with higher scores.

> These relationships represent associations in the dataset and should not automatically be interpreted as causation.

---

## 🧹 Data Cleaning

The dataset was inspected for:

- Missing values
- Duplicate rows
- Invalid numerical values
- Outliers
- Data types

Duplicate records were removed using:

```python
df = df.drop_duplicates()
```

An invalid negative value was detected in `Physical_Activity_Hours`.

Because physical activity hours cannot realistically be negative, values below zero were clipped to zero:

```python
df["Physical_Activity_Hours"] = (
    df["Physical_Activity_Hours"].clip(lower=0)
)
```

---

## 📈 Skewness Analysis

Skewness was examined for numerical features.

Most numerical features were approximately symmetric, while `Study_Hours` showed the highest positive skewness:

```text
Study_Hours ≈ 0.436
```

Although this represents only mild skewness, `Study_Hours` was handled separately in the preprocessing pipeline using a logarithmic transformation.

---

## 🛠️ Feature Engineering

### Country Grouping

The original `Country` column contained a large number of unique categories.

Directly one-hot encoding all countries would create many additional sparse features.

To reduce this high cardinality:

1. The 10 most frequent countries were retained.
2. All remaining countries were grouped into:

```text
Other
```

A new feature was created:

```text
Grouped_country
```

This reduces dimensionality while retaining useful country information.

---

## 🔤 Encoding Strategy

Different categorical features require different encoding strategies.

### Ordinal Encoding

`Stress_Level` has a meaningful order:

```text
Low < Medium < High < Very High
```

Therefore, it was encoded using `OrdinalEncoder`.

```python
OrdinalEncoder(
    categories=[["Low", "Medium", "High", "Very High"]]
)
```

### One-Hot Encoding

Nominal categorical features do not have a natural order.

The following features were one-hot encoded:

```text
Gender
Academic_Level
Most_Used_Platform
Purpose_Of_Use
Grouped_country
```

The encoder uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

This allows the pipeline to safely handle unseen categories during prediction.

---

## ✂️ Train-Test Split

The features and target were separated:

```python
X = df[feature_col]
y = df["Mental_Health_Score"]
```

The dataset was then divided into:

- **70% training data**
- **30% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
```

The split is performed **before fitting the preprocessing transformations** to reduce the risk of data leakage.

---

## ⚙️ Preprocessing Pipeline

Scikit-learn's `Pipeline` and `ColumnTransformer` were used to apply different preprocessing techniques to different feature groups.

### 1. Study Hours

```text
Study_Hours
      ↓
log1p transformation
      ↓
StandardScaler
```

The logarithmic transformation helps reduce positive skewness.

---

### 2. Numerical Features

The following numerical variables were standardized:

```text
Age
Avg_Daily_Usage_Hours
Sleep_Hours_Per_Night
Physical_Activity_Hours
Daily_Unlocks
```

using:

```python
StandardScaler()
```

---

### 3. Ordinal Feature

```text
Stress_Level
      ↓
OrdinalEncoder
```

---

### 4. Nominal Features

```text
Gender
Academic_Level
Most_Used_Platform
Purpose_Of_Use
Grouped_country
      ↓
OneHotEncoder
```

---

## 🧩 ColumnTransformer

A `ColumnTransformer` combines all preprocessing operations into one reusable preprocessing object.

Conceptually:

```text
                         Raw Features
                              │
               ┌──────────────┼──────────────┐
               │              │              │
         Study Hours       Numerical     Categorical
               │              │              │
          log1p + scale       scale       encoding
               │              │              │
               └──────────────┼──────────────┘
                              ↓
                     Processed Features
```

This ensures that each type of feature receives the correct preprocessing automatically.

---

## 🤖 Machine Learning Models

Two regression algorithms were compared.

### 1. Linear Regression

Linear Regression was used as the baseline model.

```python
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
```

### 2. Random Forest Regressor

Random Forest was used to capture more complex and nonlinear relationships.

```python
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("random forest", RandomForestRegressor(random_state=42))
])
```

---

## 🎯 Model Evaluation

The models were evaluated using:

### R² Score

Measures how much variation in Mental Health Score is explained by the model.

Higher is better.

### MAE — Mean Absolute Error

Measures the average absolute difference between the actual and predicted Mental Health Scores.

Lower is better.

### RMSE — Root Mean Squared Error

Measures prediction error while penalizing large mistakes more heavily.

Lower is better.

---

## 📊 Model Results

| Model | Training R² | Test R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.724 | 0.740 | 0.536 | 0.676 |
| Random Forest | **0.981** | **0.878** | **0.346** | **0.463** |
| Tuned Random Forest | 0.955 | 0.865 | 0.369 | 0.487 |

The default **Random Forest Regressor achieved the highest test performance**, with an R² score of approximately:

```text
0.878
```

This means that the model explains approximately **87.8% of the variation in Mental Health Score** on the test data.

It also achieved the lowest MAE and RMSE among the evaluated models.

The difference between its training and testing R² also indicates that the default Random Forest fits the training data substantially more closely than the test data, which is why hyperparameter tuning was explored.

---

## 🔧 Random Forest Hyperparameter Tuning

`RandomizedSearchCV` was used to search for better Random Forest hyperparameters.

The following parameters were explored:

```python
param_grid = {
    "random forest__n_estimators": [100, 200, 300],
    "random forest__max_depth": [5, 10, 15],
    "random forest__min_samples_split": [2, 5, 10],
    "random forest__min_samples_leaf": [1, 2, 4]
}
```

The search used:

```text
15 random combinations
5-fold cross-validation
R² as the scoring metric
```

### Best Hyperparameters

```text
n_estimators      = 200
max_depth         = 15
min_samples_split = 5
min_samples_leaf  = 2
```

The tuned model achieved:

```text
Test R² ≈ 0.865
MAE     ≈ 0.369
RMSE    ≈ 0.487
```

Although tuning reduced some of the training-set fit, the default Random Forest still achieved the highest test-set performance in this experiment.

---

## 💾 Model Saving

The complete Random Forest pipeline is saved using `joblib`:

```python
import joblib

joblib.dump(
    rf_pipeline,
    "Mental_Health_Model.pkl"
)
```

Saving the complete pipeline rather than only the Random Forest model ensures that preprocessing and prediction are performed consistently on future data.

---

## 🧰 Tech Stack

### Programming

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Data Analysis

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Data Visualization

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)

### Machine Learning

![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

### Development

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/eyaoueslati06/mental-health-score-prediction.git
```

Move into the project directory:

```bash
cd mental-health-score-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Running the Project

After installing the dependencies, open the Jupyter notebook using VS Code or Jupyter Notebook and run the cells sequentially.

The notebook covers the complete workflow:

```text
Dataset
   ↓
Data Inspection
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Skewness Analysis
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
ColumnTransformer
   ↓
Machine Learning Pipeline
   ↓
Linear Regression
   ↓
Random Forest
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Saving
```

---

## 🌐 Part 2 — Web Application

In the second part of the project, the trained machine learning model was connected to a simple web application so users can enter student information and receive a predicted Mental Health Score.

The application contains both a **frontend** and a **FastAPI backend**.

### Frontend

The frontend is built using:

```text
index.html
style.css
script.js
```

- `index.html` contains the structure of the prediction form.
- `style.css` handles the visual design and layout.
- `script.js` collects the user input, sends it to the API, and displays the predicted Mental Health Score returned by the backend.

The frontend communicates with the FastAPI `/predict` endpoint using an HTTP request.

---

### FastAPI Backend

The backend is implemented in:

```text
main.py
```

The API loads the trained machine learning pipeline:

```python
model = joblib.load("Mental_Health_Model.pkl")
```

A `StudentData` Pydantic model is used to validate incoming user data such as:

- Age
- Gender
- Country
- Academic level
- Social media platform
- Purpose of use
- Daily usage hours
- Daily unlocks
- Study hours
- Physical activity
- Sleep hours
- Stress level

Validation rules are added using `Field` and `Literal` to ensure that the API receives valid values.

The backend also recreates the `Grouped_country` feature used during model training. Countries outside the selected frequent-country list are grouped into:

```text
Other
```

---

### API Endpoints

The application provides two main endpoints.

#### `GET /`

Returns a simple welcome message confirming that the API is running.

#### `POST /predict`

Receives student information, converts it into the same DataFrame structure used during model training, and sends it directly to the saved machine learning pipeline:

```python
prediction = model.predict(input_row)[0]
```

The API then returns the predicted Mental Health Score as JSON.

Example response:

```json
{
  "predicted_mental_health_score": 6.78
}
```

---

### CORS

`CORSMiddleware` is enabled so that the frontend can communicate with the FastAPI backend from the browser.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
```

---

### Application Flow

```text
User enters student information
            ↓
        index.html
            ↓
         script.js
            ↓
      POST /predict
            ↓
        FastAPI API
            ↓
   Pydantic validation
            ↓
Create model input DataFrame
            ↓
Mental_Health_Model.pkl
            ↓
      Model prediction
            ↓
JSON prediction response
            ↓
Displayed on the frontend
```

This second part turns the machine learning notebook into a simple end-to-end prediction application where the trained model can be used through a browser interface.

GitHub: [eyaoueslati06](https://github.com/eyaoueslati06)

Repository: [mental-health-score-prediction](https://github.com/eyaoueslati06/mental-health-score-prediction)
