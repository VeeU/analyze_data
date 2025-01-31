import osb
import re
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from openpyxl import load_workbook
import pandas as pd
import time
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image


class pack_discharge:
    @staticmethod
    def select_discharge_folders():
        # Open a dialog to let the user select multiple folders
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        folder_paths = filedialog.askdirectory(mustexist=True, title="Select ABCS Pack Discharge Folder")
        if not folder_paths:
            messagebox.showwarning("Warning", "No folder selected. Exiting...")
            return None
        return folder_paths

    @staticmethod
    def copy_folders_to_tr_path(tr_folder_path, discharge_folders):
        # Target folder where files will be copied
        target_folder = os.path.join(tr_folder_path, "battery pack discharge")
        os.makedirs(target_folder, exist_ok=True)  # Create target folder if it doesn't exist

        # Loop through all selected discharge folders and copy their entire contents, including the folder structure itself
        for folder in discharge_folders:
            if os.path.exists(folder):
                # Create the target directory within the target folder (preserve the discharge folder name)
                folder_name = os.path.basename(folder)
                target_discharge_folder = os.path.join(target_folder, folder_name)
                os.makedirs(target_discharge_folder, exist_ok=True)

                # Recursively copy all files and folders from each discharge folder to target discharge folder
                for root, dirs, files in os.walk(folder):
                    # Construct the target path by maintaining the directory structure
                    relative_path = os.path.relpath(root, folder)
                    target_dir = os.path.join(target_discharge_folder, relative_path)
                    os.makedirs(target_dir, exist_ok=True)

                    # Copy each file from source to target directory
                    for file in files:
                        source_file = os.path.join(root, file)
                        target_file = os.path.join(target_dir, file)

                        try:
                            shutil.copy2(source_file, target_file)  # Use copy2 to preserve metadata
                            print(f"Copied {source_file} to {target_file}")
                        except Exception as e:
                            print(f"Error copying {source_file} to {target_file}: {e}")
            else:
                print(f"Folder {folder} does not exist.")

    @staticmethod
    def issue_draft_report(tr_folder_path, root_folder):
        # Dynamically construct the path to the report template by joining root_folder with the relative path
        template_path = os.path.join(root_folder, "report_template", "Battery pack discharge.xlsx")

        # Ensure the target folder exists
        target_folder = os.path.join(tr_folder_path, "battery pack discharge")
        os.makedirs(target_folder, exist_ok=True)

        # Define the target path for the report template
        target_template_path = os.path.join(target_folder, "Battery pack discharge.xlsx")

        try:
            # Copy the report template to the target folder
            shutil.copy2(template_path, target_template_path)  # Use copy2 to preserve metadata
            print(f"Report template copied to {target_template_path}")
        except Exception as e:
            print(f"Error copying report template: {e}")

    @staticmethod
    def scan_and_copy_csv_to_excel(folders, excel_template_path):
        # Load the Excel workbook and select the active sheet
        headers = [
            'Sample #', 'Test condition', 'Charge Time (min)', 'Discharge Time (min)',
            'Ch. Capacity (AHr)', 'Ch. Energy (WHr)', 'Disch. Capacity (AHr)', 'Disch. Energy (WHr)',
            'Ch. Stop Voltage', 'Disch. Stop Voltage', 'Disch. Stop Temp', 'Disch. Stop Type',
            'DCIR (mOhm)', 'Disch. End Time', 'Rebound voltage'
        ]

        workbook = load_workbook(excel_template_path)
        sheet = workbook.active
        sheet.append(headers)
        workbook.save(excel_template_path)

        # Create an empty DataFrame with the specified headers
        df = pd.DataFrame()

        # Loop through each selected folder
        for folder in folders:

            # Extract the folder name and append it to the Excel sheet
            folder_name = os.path.basename(folder)

            # Regular expression to capture FW, HW, and Cdegree parts
            fw_match = re.search(r'FW[\d.]+', folder_name)
            hw_match = re.search(r'HW[\d.]+', folder_name)
            cdegree_match = re.search(r'[\d]+Cdegree', folder_name)

            # Extract matched values or set as empty if not found
            fw = fw_match.group(0) if fw_match else ''
            hw = hw_match.group(0) if hw_match else ''
            cdegree = cdegree_match.group(0) if cdegree_match else ''

            # Merge the extracted parts with hyphens
            test_condition = '-'.join(filter(None, [fw, hw, cdegree]))

            # Scan for 'data.csv' in level-1 subfolders
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)

                if os.path.isdir(subfolder_path):
                    data_csv_path = os.path.join(subfolder_path, 'data.csv')
                    if os.path.exists(data_csv_path):
                        print(f"Found 'data.csv' in {data_csv_path}")

                        # Read the CSV file into a DataFrame, including the headers
                        df_new = pd.read_csv(data_csv_path)

                        # Replace the first column of the DataFrame (Cycle #) with the subfolder name
                        df_new.iloc[:, 0] = subfolder

                        # Step 1: Extract the second column ('Test condition')
                        original_second_column = df_new['Discharge Current (A)']

                        # Step 2: Create a new second column by joining test_condition with the original data
                        new_second_column = original_second_column.apply(lambda x: f"{test_condition}-{x}A")

                        # Step 3: Drop the original second column
                        df_new.drop('Discharge Current (A)', axis=1, inplace=True)

                        # Step 4: Insert the new second column back at the same position
                        df_new.insert(1, 'Discharge Current (A)', new_second_column)

                        # Append the current DataFrame to the accumulated DataFrame
                        df = pd.concat([df_new, df], ignore_index=True)

            # Sort the DataFrame by the second column ('Discharge Current (A)')
            df_sorted = df.sort_values(by=[df.columns[1], df.columns[0]], ascending=[True, True])

        # Append the DataFrame contents row by row
        for row in df_sorted.itertuples(index=False, name=None):
            sheet.append(row)

        # Save the workbook with the new data added
        workbook.save(excel_template_path)
        print(
            f"Appended data from {data_csv_path} to Excel template with headers and folder name '{folder_name}'.")

        # Delay for 1 second to ensure the save action is completed
        time.sleep(1)

        #Chart Ahr
        # Pivot the data to show 'Disch. Capacity (AHr)' values with 'Cycle #' as columns and 'Discharge Current (A)' as rows
        analyze_Ah_dataframe = df_sorted.pivot(index="Discharge Current (A)", columns="Cycle #",
                                            values="Disch. Capacity (AHr)")

        # Create a bar chart from the transposed pivoted data
        plt.figure(figsize=(12, 8))
        analyze_Ah_dataframe.plot(kind='bar')

        # Set titles and labels
        plt.title('Disch. Capacity (AHr) by Test condition and Sample #', fontsize=14)
        plt.xlabel('Test condition', fontsize=12)
        plt.ylabel('Disch. Capacity (AHr)', fontsize=12)

        # Customize the legend and grid
        plt.legend(title='Sample #', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
        plt.grid(True)

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the plot as an image
        image_path = os.path.join(os.path.dirname(excel_template_path), "chart.png")
        plt.savefig(image_path)

        # Insert the image into the Excel template
        img = Image(image_path)
        sheet.add_image(img, 'Q18')  # Adjust the cell location as needed

        # Save the Excel file with the embedded image
        workbook.save(excel_template_path)

        # Final 1-second delay after the last save
        time.sleep(1)

        # Show the plot
        plt.show()

        #Chart Ahr
        # Pivot the data to show 'Disch. Capacity (AHr)' values with 'Cycle #' as columns and 'Discharge Current (A)' as rows
        analyze_Ah_dataframe = df_sorted.pivot(index="Discharge Current (A)", columns="Cycle #",
                                            values="Disch. Capacity (AHr)")

        # Create a bar chart from the transposed pivoted data
        plt.figure(figsize=(12, 8))
        analyze_Ah_dataframe.plot(kind='bar')

        # Set titles and labels
        plt.title('Disch. Capacity (AHr) by Test condition and Sample #', fontsize=14)
        plt.xlabel('Test condition', fontsize=12)
        plt.ylabel('Disch. Capacity (AHr)', fontsize=12)

        # Customize the legend and grid
        plt.legend(title='Sample #', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
        plt.grid(True)

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the plot as an image
        image_path = os.path.join(os.path.dirname(excel_template_path), "chart.png")
        plt.savefig(image_path)

        # Insert the image into the Excel template
        img = Image(image_path)
        sheet.add_image(img, 'R18')  # Adjust the cell location as needed

        # Save the Excel file with the embedded image
        workbook.save(excel_template_path)

        # Final 1-second delay after the last save
        time.sleep(1)

        # Show the plot
        plt.show()

        # Save the updated Excel file again after processing all folders
        workbook.save(excel_template_path)


    @staticmethod
    def select_multiple_folders():
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        selected_folders = []
        while True:
            folder_path = filedialog.askdirectory(mustexist=True, title="Select a Folder (Cancel to finish)")
            if not folder_path:
                break  # Exit if the user cancels or doesn't select a folder
            selected_folders.append(folder_path)

            # Ask if the user wants to select another folder
            if not messagebox.askyesno("Confirmation", "Do you want to select another folder?"):
                break  # Stop asking for more folders if the user says 'No'

        return selected_folders

# Main execution logic
if __name__ == "__main__":
    # Check if the three arguments (rootFolder, trFolderPath, and selectedTestItem) are passed
    if len(sys.argv) < 4:
        print("Insufficient arguments provided.")
        sys.exit(1)

    # Extract the arguments from the command line
    #root_folder = sys.argv[1]
    #tr_folder_path = sys.argv[2]
    #selected_test_item = sys.argv[3]

    # Extract the arguments from the command line
    root_folder = sys.argv[1]
    tr_folder_path = sys.argv[2]
    selected_test_item = sys.argv[3]

    # Display the passed arguments as a pop-up notification using tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Passed Arguments", f"Root Folder: {root_folder}\nTR Folder Path: {tr_folder_path}\nSelected Test Item: {selected_test_item}")

    # Step 1: Copy report template to trFolderPath/pack discharge
    pack_discharge.issue_draft_report(tr_folder_path, root_folder)

    # Step 2: Let the user select multiple pack discharge folders from ABCS pack discharge
    folders = pack_discharge.select_multiple_folders()
    if folders:
        # Join the folder paths into a string to display in a message box
        folders_str = "\n".join(folders)
        messagebox.showinfo("Selected Folders", f"The following folders were selected:\n\n{folders_str}")
    else:
        messagebox.showinfo("No Folders Selected", "No folders were selected.")
        sys.exit(1)

    # Step 3: Scan for data.csv and append its contents to the Excel template
    report_path = os.path.join(tr_folder_path, "battery pack discharge", "Battery pack discharge.xlsx")
    pack_discharge.scan_and_copy_csv_to_excel(folders, report_path)

    messagebox.showinfo("Process Complete", "Data has been copied to the report template and saved.")
