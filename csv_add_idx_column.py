import pandas as pd

file_name = "drtdf_date_G25SOC.csv"
df = pd.read_csv(file_name)
df_insert = df.copy()

# Insert row index as first column
df_insert.insert(0, "index", range(0, len(df_insert)))

output_file_name = file_name.replace(".csv", "_idx.csv")
df_insert.to_csv(output_file_name, index=False)