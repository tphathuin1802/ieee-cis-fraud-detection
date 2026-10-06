# 💳 IEEE-CIS Fraud Detection

End-to-end machine learning project for detecting fraudulent e-commerce transactions, built on the real-world dataset provided by **Vesta Corporation** for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) competition on Kaggle.

> Goal: flag fraudulent card-not-present transactions with high accuracy while keeping false positives low, so legitimate customers don't get their cards declined at the checkout counter.

---

## 📌 Table of Contents

1. [Background](#-background)
2. [Dataset](#-dataset)
3. [Project Objectives](#-project-objectives)
4. [Project Structure](#-project-structure)
5. [Methodology](#-methodology)
6. [Results](#-results)
7. [Getting Started](#-getting-started)
8. [Tech Stack](#-tech-stack)
9. [Key Learnings](#-key-learnings)
10. [Future Work](#-future-work)
11. [Author](#-author)
12. [Acknowledgements](#-acknowledgements)

---

## 🧭 Background

Card fraud prevention saves consumers and businesses millions of dollars every year, but overly aggressive systems also block genuine purchases and hurt customer experience. The IEEE Computational Intelligence Society (IEEE-CIS), together with Vesta Corporation, released a large-scale dataset of real e-commerce transactions to benchmark better fraud detection models.

This repository is my personal take on the problem: exploring the data, engineering features, training and comparing models, and interpreting what drives fraud.

## 📊 Dataset

Source: [Kaggle – IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)

| File | Description |
|---|---|
| `train_transaction.csv` | Transaction-level data with the `isFraud` target |
| `train_identity.csv` | Identity / device / network information linked by `TransactionID` |
| `test_transaction.csv` | Transactions to predict (no label) |
| `test_identity.csv` | Identity data for the test set |
| `sample_submission.csv` | Submission format |

**Main feature groups**

- **Transaction**: `TransactionDT` (time delta), `TransactionAmt`, `ProductCD`
- **Card**: `card1`–`card6` (card type, issuer, category, etc.)
- **Address / distance**: `addr1`, `addr2`, `dist1`, `dist2`
- **Email domains**: `P_emaildomain`, `R_emaildomain`
- **Counting features**: `C1`–`C14` (e.g., number of addresses linked to a card)
- **Timedelta features**: `D1`–`D15` (e.g., days between previous transactions)
- **Match features**: `M1`–`M9` (e.g., name/card/address matches)
- **Vesta engineered features**: `V1`–`V339` (ranking, counting, and entity relations)
- **Identity**: `id_01`–`id_38`, `DeviceType`, `DeviceInfo`

**Key challenges**

- Highly **imbalanced** target (fraud is only a small percentage of transactions)
- Hundreds of features with many **missing values**
- High-cardinality categorical variables
- **Time-dependent** data → random splits can leak future information
- Anonymized / masked features → limited domain interpretability

> ⚠️ The data is not included in this repository (Kaggle terms). Download it manually, see [Getting Started](#-getting-started).

## 🎯 Project Objectives

- Perform thorough **EDA** on transactions, identity, and fraud patterns
- Build a reproducible **preprocessing and feature engineering** pipeline
- Train and compare multiple models (Logistic Regression baseline, Random Forest, XGBoost, LightGBM, CatBoost)
- Handle **class imbalance** and choose appropriate metrics
- Use a **time-based validation** strategy to avoid leakage
- Interpret the model with **SHAP** and feature importance
- *(Optional)* Deploy a small **Streamlit** demo that scores a transaction

## 🗂 Project Structure

```
ieee-cis-fraud-detection/
├── data/
│   ├── raw/                  # Kaggle CSV files (not tracked by git)
│   └── processed/            # Cleaned / merged data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_model_interpretation.ipynb
├── src/
│   ├── data_loader.py        # Load, merge, reduce memory
│   ├── features.py           # Feature engineering
│   ├── train.py              # Training & validation
│   └── evaluate.py           # Metrics & plots
├── app/
│   └── streamlit_app.py      # (Optional) demo app
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔬 Methodology

1. **Data loading & merging**: join transaction and identity tables on `TransactionID`; downcast numeric dtypes to reduce memory.
2. **EDA**: target distribution, fraud rate by `ProductCD`, card type, email domain, device type, transaction amount, and time of day/week.
3. **Preprocessing**
   - Missing value analysis; drop or flag columns with excessive nulls
   - Encode categoricals (label / frequency / target encoding)
   - Remove highly correlated `V` features
4. **Feature engineering**
   - Time features from `TransactionDT` (hour, day of week)
   - Aggregations per card / address / email (mean, std, count of `TransactionAmt`)
   - Frequency encoding of high-cardinality columns
   - Amount decimal patterns and ratios
   - Approximate **client/user ID** by combining card, address, and `D1`-based start date
5. **Validation**: time-based split (earlier months for training, later months for validation) and/or `GroupKFold`.
6. **Modeling**: baseline → tree-based gradient boosting; hyperparameter tuning with Optuna.
7. **Imbalance handling**: class weights / `scale_pos_weight`, threshold tuning (SMOTE only if it proves useful).
8. **Evaluation**: ROC-AUC (competition metric), PR-AUC, precision / recall / F1 at a chosen threshold, confusion matrix.
9. **Interpretation**: SHAP summary and dependence plots.

## 📈 Results

> Fill in after running experiments.

| Model | Validation ROC-AUC | PR-AUC | Notes |
|---|---|---|---|
| Logistic Regression (baseline) | – | – | |
| Random Forest | – | – | |
| XGBoost | – | – | |
| LightGBM | – | – | |
| CatBoost | – | – | |

**Top features (SHAP):** _to be added_

**Figures:** _add ROC curve, PR curve, and SHAP plots to `reports/figures/` and embed them here._

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the data

Using the Kaggle API (requires accepting the competition rules first):

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip data/raw/ieee-fraud-detection.zip -d data/raw
```

### 4. Run the notebooks

```bash
jupyter lab
```

Run the notebooks in order, `01` → `04`.

### 5. (Optional) Launch the demo app

```bash
streamlit run app/streamlit_app.py
```

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Data**: pandas, NumPy
- **Visualization**: matplotlib, seaborn, plotly
- **Modeling**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **Interpretation**: SHAP
- **App**: Streamlit
- **Tools**: Jupyter, Git

## 💡 Key Learnings

_Fill in as you go, for example:_

- Why random K-fold leaks information on time-ordered fraud data
- How much feature engineering around card/user identity improved AUC
- Trade-offs between precision and recall in a real fraud-prevention setting

## 🔭 Future Work

- Graph-based features linking cards, emails, and devices
- Model stacking / blending
- Cost-sensitive learning with business cost per false positive vs. missed fraud
- Model serving via FastAPI + Docker

## 👤 Author

**Huỳnh Tấn Phát**
Portfolio: [stephen-huynh.vercel.app](https://stephen-huynh.vercel.app/)

## 🙏 Acknowledgements

- [IEEE Computational Intelligence Society (IEEE-CIS)](https://cis.ieee.org/) and **Vesta Corporation** for providing the dataset
- [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) for hosting the competition

> This project is for educational and portfolio purposes only.
