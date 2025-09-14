# ☕ Coffee Shop Daily Revenue Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Model-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat\&logo=github)

---

## 🌟 Project Overview

A **Machine Learning powered web app** that predicts the **daily revenue of a coffee shop** based on sales, weather, promotions, and customer data.
The app is built with **Streamlit** and has a **modern interactive UI**.

✨ This project is designed to impress HRs, recruiters, and business stakeholders.

---

## 🚀 Features

* 🎨 **Modern UI** with cards & blue-white theme
* 📊 **Accurate Revenue Predictions**
* ⏳ **Time-based Insights** (Day, Week, Quarter)
* 🧑‍💻 **Feature Engineering** for business data
* 💾 **Reusable Model** (saved with Joblib)
* 🌐 Ready for **deployment** on Streamlit Cloud / Netlify

---

## 📂 Project Structure

```bash
coffee-revenue-predictor/
│── coffee_sales_model.pkl        # Trained ML model
│── scaler.pkl                    # StandardScaler
│── selector.pkl                  # Feature Selector
│── Streamlit.py                  # Streamlit App
│── coffee_shop_sales_dataset.xlsx # Dataset
│── requirements.txt              # Dependencies
│── README.md                     # Documentation
```

---

## ⚙️ Tech Stack

| Layer             | Tools Used                       |
| ----------------- | -------------------------------- |
| **Language**      | Python 🐍                        |
| **Data Handling** | Pandas, NumPy                    |
| **Visualization** | Matplotlib, Seaborn              |
| **ML Model**      | Scikit-learn (Linear Regression) |
| **Frontend**      | Streamlit                        |
| **Model Saving**  | Joblib                           |

---

## 📊 Workflow

1️⃣ **Data Preprocessing**
✔ Convert `Date` → Day, Week, Quarter
✔ Encode categorical features (Day, Season)
✔ Scale numerical values

2️⃣ **Model Training**
✔ Feature selection with `SelectKBest`
✔ Trained **Linear Regression**
✔ Evaluated using **R², RMSE, MAE, MAPE**

3️⃣ **Deployment**
✔ Model, Scaler, Selector saved
✔ Streamlit App for **real-time prediction**

---

## 🔮 Example Prediction

**Input (HR Demo):**

* Month: `12 (Dec)`
* Weekend: ✅ Yes
* Temp: `18°C`
* Coffee Sales: `120`
* Staff: `7`

**Output:**
💰 Predicted Daily Revenue:

```diff
+ $2,350.75
```

---

## 📦 Installation

Clone the repo 👇

```bash
git clone https://github.com/your-username/coffee-revenue-predictor.git
cd coffee-revenue-predictor
```

Install dependencies 👇

```bash
pip install -r requirements.txt
```

Run the app 👇

```bash
streamlit run Streamlit.py
```

---

## 🌐 Deployment Options

* 🚀 [Streamlit Cloud]([https://streamlit.io/cloud](https://rohithragavender-coffee-revenue-streamlit-tfb1kk.streamlit.app/)) (1-click deploy)
* ⚡ Render / Heroku
* 💻 React Frontend + FastAPI Backend (for production apps)
* 🌍 Netlify + API

---

## 📈 Results

✔ Strong correlation between **Coffee Sales, Customer Count** and Revenue
✔ Time-based features improved seasonal forecasting
✔ Model shows **low overfitting** and good generalization

---

## ✨ Future Improvements

* 🔮 Add Deep Learning (LSTM for time series)
* ☁️ Connect to Weather API for real-time predictions
* 🎯 Deploy with **React UI** for HR demo

---

## 👨‍💻 Author

**Rohith Ragavender**
💼 Data Science & AI Enthusiast
📧 Email: rohithragavender@gmail.com
🌐 [LinkedIn](www.linkedin.com/in/rohith0410)


