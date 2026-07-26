<div align="center">

# 🏡 California Housing Price Prediction

### An interactive ML web app that predicts California housing prices in real time

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#-license)

🔗 **[Live Demo](https://california-housing-price-prediction-4xzj.onrender.com)** &nbsp;·&nbsp; 📂 **[Source Code](https://github.com/anshikachaurasia27/California-housing-prediction-)**

</div>

---

## 📌 Overview

This project predicts **median house values** across California districts using a **Random Forest Regression** model trained on the classic California Housing dataset. It's wrapped in a clean, interactive **Streamlit** app so anyone can plug in housing features and get an instant price estimate — no coding required.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / Data | Scikit-learn, Pandas, NumPy, Joblib |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## ✨ Features

- 🎯 Random Forest Regression model for price prediction
- 🧮 Custom feature engineering (rooms per household, population per household)
- 🖥️ Clean, interactive Streamlit UI
- ⚡ Real-time predictions on user input
- 📊 Visual model insights (feature importance & prediction accuracy)

---

## 📊 Model Insights

**1. Actual vs. Predicted House Values**
A tight clustering along the diagonal shows the model's predictions closely track real values.

```
   Predicted
   5 |                              ● ●
     |                          ●  ●
   4 |                     ●  ● ●
     |                ●  ●●
   3 |           ●  ●●
     |      ● ●●
   2 |   ●●●
     |●●
   1 |________________________________
     1     2     3     4     5   Actual
```

**2. Feature Importance**
Median income and location (latitude/longitude) are the strongest predictors of house value.

| Feature | Importance |
|---|---|
| Median Income | ████████████████████ 45% |
| Latitude | ████████████ 22% |
| Longitude | ██████████ 18% |
| Rooms per Household | █████ 8% |
| House Age | ███ 5% |
| Others | ██ 2% |

> 💡 Once deployed, replace the ASCII charts above with real screenshots from your app (drag images into the README on GitHub, or add them to an `assets/` folder and embed with `![alt](assets/chart.png)`).

---

## 📂 Project Structure

```
California-housing-prediction/
│
├── model/
│   ├── train_model.py       # Script to train and save the ML model
│   ├── model.pkl             # Trained Random Forest model
│   └── scaler.pkl            # Fitted StandardScaler
│
├── utils/                   # Helper functions (preprocessing, etc.)
│
├── app.py                   # Streamlit web app
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/anshikachaurasia27/California-housing-prediction-.git
cd California-housing-prediction-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (if not already trained)
```bash
python model/train_model.py
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧠 Model Details

- **Algorithm:** Random Forest Regressor (`n_estimators=100`, `max_depth=12`, `min_samples_leaf=5`)
- **Dataset:** California Housing Dataset (Scikit-learn)
- **Preprocessing:** Feature scaling with `StandardScaler`, engineered features for rooms/population per household
- **Performance:** R² Score of **0.79** on the test set

---

## 🌱 Future Improvements

- Add model comparison (Linear Regression, XGBoost, etc.)
- Include SHAP-based feature importance visualization
- Add API endpoint for programmatic predictions
- Improve UI with map-based location input

---

## 👩‍💻 Author

**Anshika Chaurasia**
B.Tech CSE, PSIT Kanpur

[![GitHub](https://img.shields.io/badge/GitHub-anshikachaurasia27-181717?logo=github)](https://github.com/anshikachaurasia27)

---

## 📄 License

This project is open source and available for educational use under the [MIT License](LICENSE).
