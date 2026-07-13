# Walmart Sales Forecasting Web Application

An end-to-end Supervised Machine Learning and Web Application built from scratch to forecast weekly store sales. This project utilizes a **Random Forest Regressor** to predict sales based on environmental and economic indicators, served through an interactive web interface built with **Flask**.

---

##  Features
* **Predictive AI Brain:** Uses supervised machine learning to predict continuous numerical values (`Weekly_Sales`).
* **Real-time Forecasts:** An interactive HTML/CSS front-end form allowing users to submit data parameters instantly.
* **Automated Data Preprocessing:** Automatically handles string-based date parsing into numeric features (`Day`, `Month`, `Year`) for the ML pipeline.

---

## Project Architecture & Workflow

The application operates in a two-stage lifecycle:

1. **The Machine Learning Pipeline (`train_model.py`):** Parses historical datasets, engineers temporal features, splits data into train/test subsets (80/20), trains a Random Forest ensemble, and serializes the state into a portable binary format.
2. **The Server Bridge (`app.py`):** A lightweight web application route handling both `GET` (delivering the form) and `POST` (passing user variables safely to the model and outputting results).

```text
[ Historical Walmart Data ] ➔ [ Feature Engineering & Training ] ➔ [ Saved Model (.pkl) ]
                                                                           │
[ User Form Input (UI) ]    ➔ [ Flask Backend Route (POST) ]   ◄───────────┘
         │                             │
         └─── [ Predicted Sales ] ◄────┘


Project Directory Structure

walmart-sales-forecaster/
│
├── app.py                 # Flask web backend application script
├── train_model.py         # Data preprocessing and model training script
├── explore.py             # Simple data analysis diagnostic script
├── sales_model.pkl        # Serialized Random Forest model artifact (Binary)
├── Walmart.csv            # Historical training data source (Git ignored)
│
├── templates/             # Front-end layout structures
│   └── index.html         # Interactive web layout form
└── venv/                  # Isolated Python environment dependencies (Git ignored)