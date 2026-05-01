# Datathon 2026: The Gridbreaker - Team 108bros

This repository contains all the source code, exploratory data analysis (EDA) notebooks, and forecasting results built to solve the revenue and inventory optimization challenge for **DATATHON 2026: The Gridbreakers**, hosted by Vin Telligence VinUni DS&AI Club.

## Team Members

* **Vo Gia Minh**
* **Nguyen Phung Trung Nguyen**
* **Pham Phuc Cam Chuong**
* **Than Hoang Anh Vu**

## Repository Structure

Our project is organized as follows:

```text
datathon2026-108bros/
│
├── notebooks/              # Jupyter Notebooks for Exploratory Data Analysis
│   ├── Part-1/
│   │   └── notebook.ipynb  # Initial data exploration and processing
│   └── Part-2/
│       ├── Graph1.ipynb    # Generates the "Traffic vs. Conversion Rate" chart
│       └── Graph2.ipynb    # Generates the "Gross Profit Margin Trends" chart
│
├── report/                 # Technical Report
│   └── report.pdf          # Final report in NeurIPS format
│
├── src/                    # Source code for the forecasting model
│   └── train_model.py      # Ensemble Pipeline (LightGBM + XGBoost)
│
├── submissions/            # Directory for final prediction outputs
│
├── README.md               # This documentation file
└── requirements.txt        # Python dependencies required to run the code
```

## Reproducibility & Instructions

To strictly adhere to the reproducibility requirements, our entire pipeline has been automated. Please follow these steps to replicate our results:

### Step 1: Environment Setup

Ensure you have Python 3.9+ installed. Open your terminal at the root of this repository and install the required packages:

```bash
pip install -r requirements.txt
```

### Step 2: Data Preparation

1. Download the original dataset provided by the organizers.
2. **Create a folder named `data/`** at the root of this repository (if it doesn't exist).
3. Place the `sales.csv`, `promotions.csv`, and `sales_test.csv` files inside the `data/` folder.

### Step 3: Reproduce Exploratory Data Analysis (EDA)

To view our analysis and regenerate the charts used in our technical report:

1. Start the Jupyter Notebook environment from your terminal:

   ```bash
   jupyter notebook
   ```

2. Navigate to `notebooks/Part-1/` and run `notebook.ipynb` for the initial data exploration.
3. Navigate to `notebooks/Part-2/` and run `Graph1.ipynb` and `Graph2.ipynb` to reproduce the business insight visualizations.

### Step 4: Train Model and Predict

To reproduce the forecasting results, open your terminal at the root directory and execute the training script:

```bash
python src/train_model.py
```

**This script will automatically execute the following:**

1. Load and preprocess the data from the `data/` directory.
2. Engineer domain-specific features (e.g., `is_double_day`, `is_payday`).
3. Truncate noisy historical data (pre-2019) and train the LightGBM + XGBoost ensemble model.
4. Predict `Revenue` (with inverse log transformation) and interpolate `COGS`.
5. Generate the final `submission.csv` file at the root directory, ready for Kaggle submission.

## Methodology Highlights

* **Descriptive & Diagnostic EDA:** We identified a severe inventory paradox and gross margin erosion caused by fixed promotions. This led to our prescriptive strategies: implementing "Agile Drops" and integrating "Buy Now, Pay Later" (BNPL) services.
* **Anti-Leakage Pipeline:** Our forecasting model strictly avoids temporal data leakage by excluding lag features and future operational variables.
* **Explainable AI (XAI):** We utilized SHAP values to validate that our model successfully captures market reactions to Mega Sales and Payday events.
