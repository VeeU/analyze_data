import os


def try_open_file(file_path, keyword="bms_current"):
    # List of common encodings to try
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'utf-16', 'utf-16le', 'utf-16be']

    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    if keyword in line:
                        yield line.strip()
            break  # If reading was successful, exit the loop
        except (UnicodeDecodeError, LookupError):
            continue  # Try the next encoding if the current one fails


def find_bms_current_rows(root_folder, keyword="bms_current"):
    result_rows = []

    # Walk through all directories and subdirectories
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.txt'):
                file_path = os.path.join(dirpath, file)
                try:
                    # Attempt to read the file with multiple encodings
                    for line in try_open_file(file_path, keyword):
                        result_rows.append(line)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return result_rows


# Define the root folder path
root_folder = r"C:\Users\aesavle\OneDrive - Techtronic Industries Co. Ltd\Datalogger Record\EB 6"

# Call the function and collect the results
bms_rows = find_bms_current_rows(root_folder)

# Save the results to a file if needed
output_file = os.path.join(root_folder, "bms_current_results.txt")
with open(output_file, 'w', encoding='utf-8') as out_file:
    for row in bms_rows:
        out_file.write(row + '\n')

print(f"Collected rows containing 'bms_current' are saved in: {output_file}")
