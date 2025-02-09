# 1. Libraries
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 2. Loading the datasets
csv_path = os.path.join("resources", "STAXI.csv")
staxi = pd.read_csv(csv_path, delimiter=",")
print(staxi.head())

csv_path2 = os.path.join("resources", "TAS.csv")
tas = pd.read_csv(csv_path2, delimiter=",")
print(tas.head())

# 3. Cleaning the datasets
# Cleaning the TAS dataset
print(tas.columns)
tas.columns = tas.columns.str.strip().str.replace('"', '')
split_tas = tas[',TAS_Identification,TAS_Describing,TAS_ExternalThinking,TAS_OverallScore'].str.split(",", expand=True)
split_tas.columns = ["ID", "TAS_Identification", "TAS_Describing", "TAS_ExternalThinking", "TAS_OverallScore"]
print(split_tas.columns)

tas_reduced = split_tas[["ID","TAS_Identification", "TAS_Describing", "TAS_ExternalThinking"]]
print(tas_reduced.head())

tas_renamed = tas_reduced.rename(columns={
    "TAS_Identification" : "Identification",
    "TAS_Describing" : "Describing",
    "TAS_ExternalThinking" : "External_Thinking"
})

tas_renamed['Identification'] = tas_renamed['Identification'].str.strip('"').astype(int)
tas_renamed['Describing'] = tas_renamed['Describing'].str.strip('"').astype(int)
tas_renamed['External_Thinking'] = tas_renamed['External_Thinking'].str.strip('"').astype(int)
tas_renamed['ID'] = tas_renamed['ID'].str.strip('"').astype(str)

# Cleaning the STAXI dataset
print(staxi.columns)
staxi.columns = staxi.columns.str.strip().str.replace('"', '')
split_staxi = staxi[',STAXI_State_Anger,STAXI_Trait_Anger,STAXI_TAT,STAXI_TAR,STAXI_AI,STAXI_AO,STAXI_AC'].str.split(",", expand=True)
split_staxi.columns = ["ID", "STAXI_State_Anger","STAXI_Trait_Anger","STAXI_TAT","STAXI_TAR","STAXI_AI","STAXI_AO","STAXI_AC"]
print(split_staxi.columns)

staxi_reduced = split_staxi[["ID","STAXI_AC"]]
print(staxi_reduced.head())

staxi_renamed = staxi_reduced.rename(columns={
    "STAXI_AC" : "Control"
})

staxi_renamed['Control'] = staxi_renamed['Control'].str.strip('"').astype(int)
staxi_renamed['ID'] = staxi_renamed['ID'].str.strip('"').astype(str)

# Checking mising values
print(tas_renamed[['Identification', 'Describing', 'External_Thinking']].isnull().sum())
print(staxi_renamed['Control'].isnull().sum())

# Merging the datasets
# Checking whether the ID values are the same in TAS and STAXI
print(tas_renamed.tail())
print(staxi_renamed.tail())

merged_df = pd.merge(tas_renamed[['ID','Identification', 'Describing', 'External_Thinking']], staxi_renamed[['ID','Control']], on='ID')

print(merged_df.head())

# Splitting the data into features and target variable
X = merged_df[['Identification', 'Describing', 'External_Thinking']]  # Features
y = merged_df['Control']  # Target

# 4. Ensuring there is linear correlation betweeen IVs
# Visualize the linear relationship between each IV and the DV
plt.figure(figsize=(15, 10))


# Create scatter plots for each independent variable vs. Control
for i, col in enumerate(X.columns):  # Use X.columns to iterate through columns
    plt.subplot(2, 2, i+1)  # Arrange plots in a 2x2 grid (adjust as needed)
    sns.scatterplot(x=X[col], y=y)  # Plot the individual column of X vs. y
    plt.title(f"Scatter plot of {col} vs Control")
    plt.xlabel(col)
    plt.ylabel('Control')

plt.tight_layout()
plt.show()

# Pearson correlation coefficient
# Calculate the correlation between each IV and the DV
independent_vars = merged_df[['Identification', 'Describing', 'External_Thinking']]
dependent_var = merged_df['Control']

combined_df = pd.concat([independent_vars, dependent_var], axis=1)

correlation_matrix = combined_df.corr()

# Extract the correlation between each IV and 'Control'
correlations_with_control = correlation_matrix['Control'][:-1]  # Exclude 'Control' from the result

# Print the correlations
print(correlations_with_control)

