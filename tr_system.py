import os
import shutil
import pandas as pd

class record_template:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.test_report_folder = r"V:\NPI\Share\1.LAB OP\9. CNB Test Report\raw\test_report"
        self.record_template_folder = r'V:\NPI\Share\1.LAB OP\9. CNB Test Report\raw\record_template'

    # Step 1: Read from CSV file
    def read_csv(self):
        self.df = pd.read_csv(self.csv_file)

    # Step 2: Create 'tr_folder' column by joining 'tr_number', 'customer_pn', 'tti_pn'
    def create_tr_folder(self):
        if all(col in self.df.columns for col in ["tr_number", "customer_pn", "tti_pn"]):
            self.df['tr_folder'] = self.df.apply(lambda row: f"{row['tr_number']}-{row['customer_pn']}-{row['tti_pn']}", axis=1)
        else:
            print("Required columns missing in the CSV file")

    # Step 3: Check if 'tr_folder' exists, if not, create it and perform further actions
    def process_folders_and_files(self):
        # Ensure 'record_template' column is treated as string
        self.df['record_template'] = self.df['record_template'].astype(str)

        for index, row in self.df.iterrows():
            tr_folder_path = os.path.join(self.test_report_folder, row['tr_folder'])

            # Step 3: Check if 'tr_folder' exists
            if not os.path.exists(tr_folder_path):
                # If the folder doesn't exist, create it
                os.makedirs(tr_folder_path)

            # Step 4: Check if 'record_template' is marked as 'done'
            if row['record_template'] != 'done':
                test_item = row['test_item']

                # Step 4.1: Search for CSV files in 'record_template' folder that include 'test_item' in their name
                for file_name in os.listdir(self.record_template_folder):
                    if file_name.endswith(".csv") and test_item in file_name: #if test item in file name, then copy
                        file_path = os.path.join(self.record_template_folder, file_name)

                        # Copy the matching CSV file to the newly created 'tr_folder'
                        shutil.copy(file_path, tr_folder_path)

                # Step 4.2: Mark the 'record_template' column as 'done' for the current row
                self.df.at[index, 'record_template'] = 'done'

    # Save the updated DataFrame back to the CSV file
    def save_csv(self):
        self.df.to_csv(self.csv_file, index=False)

    # Optional: Print the DataFrame for verification
    def print_dataframe(self):
        print(self.df[['tr_number', 'customer_pn', 'tti_pn', 'tr_folder', 'record_template']].head())

# Example usage:
record = record_template(r"V:\NPI\Share\1.LAB OP\9. CNB Test Report\raw\test_plan.csv")

# Step 1: Read the CSV file
record.read_csv()

# Step 2: Create 'tr_folder'
record.create_tr_folder()

# Step 3 and Step 4: Process the folders and files
record.process_folders_and_files()

# Save the updated CSV file
record.save_csv()

# Print the resulting DataFrame for verification
record.print_dataframe()
