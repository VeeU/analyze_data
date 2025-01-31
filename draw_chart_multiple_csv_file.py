import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# Initialize tkinter
root = tk.Tk()
root.withdraw()  # Hide the main window

# Prompt the user to select a directory
directory = filedialog.askdirectory(
    initialdir=r"C:\Users\aesavle\OneDrive - Techtronic Industries Co. Ltd\Work\5. Task_sync\2022\6_robotic mower\robotic_test\extreme temp test\raw data",
    title="Select Directory Containing CSV Files"
)

# Display the selected directory
print(f"Selected Directory: {directory}")

# Define a directory to save the charts
output_directory = os.path.join(directory, "charts")
os.makedirs(output_directory, exist_ok=True)  # Create output directory if it doesn't exist


# Function to get user input for x-axis and y-axis columns
def get_columns(df):
    print("Available columns:")
    print(list(df.columns))

    # Prompt the user for x and y columns
    x_column = input(
        "Enter the x-axis column name: ")
    y_columns = input(
        "Enter the y-axis columns, separated by commas: ")

    y_columns = [col.strip() for col in y_columns.split(",")]
    return x_column, y_columns


# Function to plot the line chart with title as the CSV filename and save it as PNG
def plot_and_save_chart(df, x_column, y_columns, file_title, count):
    for y_column in y_columns:
        if y_column in df.columns:
            plt.plot(df[x_column], df[y_column], label=y_column)
        else:
            print(f"Warning: Column '{y_column}' not found in data.")

    # Determine milestone positions for x-axis
    plt.title(file_title)
    plt.legend()

    # Save the chart
    output_path = os.path.join(output_directory, f"{file_title}.png")
    plt.savefig(output_path, bbox_inches="tight")  # Save without cutting off labels
    print(f"Chart saved as: {output_path}")

    plt.close()  # Close the plot to free memory


# Scan all CSV files in the directory and process each file
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
if csv_files:
    for count, file_name in enumerate(csv_files, start=1):
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)
        file_title = os.path.splitext(file_name)[0]  # Use the file name without extension as title
        x_column, y_columns = get_columns(df)
        plot_and_save_chart(df, x_column, y_columns, file_title, count)
else:
    print("No CSV files found in the directory.")
