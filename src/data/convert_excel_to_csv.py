import pandas as pd

# Paths
input_path = "data/raw/telco_churn.xlsx"
output_path = "data/raw/telco_churn.csv"

# Read Excel file
df = pd.read_excel(input_path)

# Save as CSV
df.to_csv(output_path, index=False)

print("Excel converted to CSV successfully.")