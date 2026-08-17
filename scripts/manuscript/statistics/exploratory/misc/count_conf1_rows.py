import pandas as pd

# Path to the CSV file
file_path = 'scripts/manuscript/statistics/data/combined_results_fb_score_001.csv'

# Load the CSV file with the correct delimiter and error handling
data = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip')

# Print the first few rows of the dataframe to debug
print("First few rows of the dataframe:")
print(data.head())

# Check the data type of the 'conf1' column
print("Data type of 'conf1':", data['conf1'].dtype)

# Print unique values in the 'conf1' column
print("Unique values in 'conf2':")
print(data['conf2'].unique())

# Filter rows where 'conf1' is less than 0.20
filtered_rows = data[data['conf2'] < 0.2]

# Print the filtered rows to debug
print("Filtered rows where 'conf2' < 0.20:")
print(filtered_rows)

# Count the number of such rows
count = len(filtered_rows)

print(f"Number of rows where 'conf2' < 0.20: {count}")