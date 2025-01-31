import os
import win32com.client as win32

def excel_to_pdf(excel_file, output_pdf):
    # Check if the Excel file exists
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"The Excel file '{excel_file}' does not exist.")

    # Start Excel application
    excel_app = win32.Dispatch("Excel.Application")

    # Open the Excel file
    workbook = excel_app.Workbooks.Open(excel_file)

    # Export as PDF
    try:
        # Save the workbook as a PDF file
        workbook.ExportAsFixedFormat(0, output_pdf)
        print(f"Excel file has been exported to PDF at: {output_pdf}")
    except Exception as e:
        print(f"Failed to convert Excel to PDF: {e}")
    finally:
        # Close the workbook and quit Excel
        workbook.Close(False)
        excel_app.Quit()

# Example usage:
excel_file = r"C:\Users\aesavle\OneDrive - Techtronic Industries Co. Ltd\Work\5. Task_sync\2022\1. Battery\data\battery_test\work\in progress\OP40806T-AMPACE JP40-130518001DG9\Test report For CNB- 20240803-OP40806T-130518001DG9.xlsx"
output_pdf = r"C:\Users\aesavle\OneDrive - Techtronic Industries Co. Ltd\Work\5. Task_sync\2022\1. Battery\data\battery_test\work\in progress\OP40806T-AMPACE JP40-130518001DG9\file.pdf"

excel_to_pdf(excel_file, output_pdf)
