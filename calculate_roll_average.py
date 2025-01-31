import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, simpledialog

# Initialize tkinter
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window

# Prompt the user to select a directory containing CSV files
directory = filedialog.askdirectory(
    title="Select Directory Containing CSV Files"
)

if not directory:
    print("No directory selected. Exiting...")
    exit()

# Get the list of CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

if not csv_files:
    print("No CSV files found in the selected directory. Exiting...")
    exit()

# Let the user select a CSV file
print("Available CSV files:")
for idx, csv_file in enumerate(csv_files):
    print(f"{idx + 1}. {csv_file}")

selected_file_idx = simpledialog.askinteger(
    "Select File",
    f"Enter the number corresponding to the CSV file (1-{len(csv_files)}):",
    minvalue=1,
    maxvalue=len(csv_files)
)

if not selected_file_idx:
    print("No file selected. Exiting...")
    exit()

selected_file = csv_files[selected_file_idx - 1]
file_path = os.path.join(directory, selected_file)
print(f"Selected File: {file_path}")

# Load the selected CSV file
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()

# Display the columns available in the CSV
print("Available columns in the CSV:")
print(list(df.columns))

# Ask the user to select a column for the rolling average
rolling_column = simpledialog.askstring(
    "Rolling Average Column",
    "Enter the column name to calculate the rolling average:"
)

if rolling_column not in df.columns:
    print(f"Column '{rolling_column}' not found in the CSV. Exiting...")
    exit()

# Ask the user to input the number of rows for the rolling average
rolling_window = simpledialog.askinteger(
    "Rolling Average Window",
    "Enter the number of rows for the rolling average:",
    minvalue=1
)

if not rolling_window:
    print("Invalid rolling window. Exiting...")
    exit()

# Calculate the rolling average
df[f"{rolling_column}_rolling_avg"] = df[rolling_column].rolling(window=rolling_window).mean()

# Save the updated CSV file
output_file = os.path.join(directory, f"updated_{selected_file}")
df.to_csv(output_file, index=False)
print(f"Updated CSV saved as: {output_file}")
