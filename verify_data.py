import pandas as pd
import os

DATA_DIR = "datasets"

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

# 1. Telco Customer Churn
print("\n[1] Telco Customer Churn (Classification)")
telco_path = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
telco = pd.read_csv(telco_path)
print(f"    Shape: {telco.shape}")
print(f"    Target column 'Churn' values: {telco['Churn'].value_counts().to_dict()}")
print(f"    First columns: {list(telco.columns[:5])}")

# 2. Mall Customers
print("\n[2] Mall Customer Segmentation (Clustering)")
mall_path = os.path.join(DATA_DIR, "Mall_Customers.csv")
mall = pd.read_csv(mall_path)
print(f"    Shape: {mall.shape}")
print(f"    Columns: {list(mall.columns)}")
print(f"    Sample:\n{mall.head(2)}")

# 3. Groceries
print("\n[3] Groceries (Association Analysis)")
groc_path = os.path.join(DATA_DIR, "Groceries_dataset.csv")
groc = pd.read_csv(groc_path)
print(f"    Shape: {groc.shape}")
print(f"    Columns: {list(groc.columns)}")
print(f"    Unique customers: {groc['Member_number'].nunique()}")
print(f"    Unique items: {groc['itemDescription'].nunique()}")

print("\n" + "=" * 60)
print("All datasets loaded successfully!")
print("=" * 60)
