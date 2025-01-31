import os
import csv

# Directory to search for text files
directory = input("input the log folder path: ")

# Output CSV file
output_csv = os.path.join(directory, "merged_1Hz log.csv")

# Search and write rows containing "bms_current"
with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Filename", "Line"])  # Headers for the CSV

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):  # Process only .txt files
                file_path = os.path.join(root, file)

                # Read each file
                with open(file_path, mode='r', encoding='utf-8') as f:
                    for line in f:
                        if "bms_current" in line:  # Check for the keyword
                            csv_writer.writerow([file, line.strip()])  # Write filename and line

print(f"Merged CSV file saved to: {output_csv}")
