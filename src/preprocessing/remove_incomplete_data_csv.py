import pandas as pd

# Load the CSV file
input_csv = "clinical_expression_TMBTISAPM_LUADandLUSC.csv"  # CHANGE THIS to your CSV file path
df = pd.read_csv(input_csv)

print(f"Original number of rows: {len(df)}")

# Remove rows with missing data in the "PSMB5_fpkm_uq" column
df_filtered = df.dropna(subset=["PSMB5_fpkm_uq"])

print(f"Number of rows after removing incomplete data: {len(df_filtered)}")
print(f"Rows removed: {len(df) - len(df_filtered)}")

# Save the filtered dataframe to a new CSV
output_csv = "clinical_expression_filtered_TMBTISAPM_LUADandLUSC_complete.csv"  # Output file name
df_filtered.to_csv(output_csv, index=False)

print(f"Filtered data saved to '{output_csv}'")
