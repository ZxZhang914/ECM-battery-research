import pandas as pd

# Load both CSVs
df1 = pd.read_csv('fulldf_global_all.csv')
df2= pd.read_csv('fulldf_global_all1.csv')


# # Count number of rows for each unique cell_name
# df1_counts = df1['CELL'].value_counts().reset_index()
# df1_counts.columns = ['CELL', 'df1_count']

# df2_counts = df2['CELL'].value_counts().reset_index()
# df2_counts.columns = ['CELL', 'df2_count']

# # Merge counts to compare
# comparison = pd.merge(df1_counts, df2_counts, on='CELL', how='outer').fillna(0)

# # Convert counts to integers
# comparison[['df1_count', 'df2_count']] = comparison[['df1_count', 'df2_count']].astype(int)

# # Display result
# print(comparison)


# Filter both dataframes to only cell_name == 50
cols = ["CELL","Temp","R0","R1","R2","R3","C1","n1","C2","n2","C3","n3","Aw","SOH","SOC"]

df1_sub = df1.loc[:, df1.columns.intersection(cols)]
df2_sub = df2.loc[:, df2.columns.intersection(cols)]

df1_50 = df1_sub[df1_sub['CELL'] == "CELL050"]
df2_50 = df2_sub[df2_sub['CELL'] == "CELL050"]

# Compare rows to find the extra one(s)
diff_50 = pd.concat([df1_50, df2_50]).drop_duplicates(keep=False)

diff_50.to_csv("diff_cell50.csv", index=False)
print(diff_50)