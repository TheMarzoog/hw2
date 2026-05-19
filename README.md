# Data Mining Project

A data mining project implementing association rules, classification, and clustering techniques with an interactive Streamlit UI.

## Datasets

Download the following datasets and place them in a `datasets/` folder:

1. **Telco Customer Churn** (Classification)
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download

2. **Customer Segmentation** (Clustering)
   https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

3. **Groceries Dataset** (Association Rules)
   https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Launch the Streamlit UI:
```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser.
