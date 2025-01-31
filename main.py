import os
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, simpledialog  # Added simpledialog import
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

class cycle_life_analyze:
    def __init__(self):
        self.abusive_file_path = r'C:\Users\aesavle\OneDrive - Techtronic Industries Co. Ltd\Work\5. Task_sync\2022\8_app_db\data\raw\npq\abusive.csv'

    def get_milestone_values(self):
        """Pop-up dialogs to get the standard and reference milestones from the user."""
        root = Tk()
        root.withdraw()  # Hide the main window

        # Ask the user for the standard milestone
        self.standard_milestone = simpledialog.askinteger("Input", "Enter the standard milestone (e.g., 250):")

        # Ask the user for the reference milestone
        self.reference_milestone = simpledialog.askinteger("Input", "Enter the reference milestone (e.g., 500):")

        # Fallback values if the user doesn't input anything
        if not self.standard_milestone:
            self.standard_milestone = 250
        if not self.reference_milestone:
            self.reference_milestone = 500

        print(f"Standard Milestone: {self.standard_milestone}, Reference Milestone: {self.reference_milestone}")

    def run_analysis(self):
        # Select the parent folder using a file dialog
        parent_folder_path = self.browse_for_folder()
        if not parent_folder_path:
            print("No folder selected. Exiting.")
            return

        # Get milestone values from user
        self.get_milestone_values()

        # Get all level 1 child folders
        child_folders = [os.path.join(parent_folder_path, d) for d in os.listdir(parent_folder_path)
                         if os.path.isdir(os.path.join(parent_folder_path, d))]

        # Create a DataFrame to collect all data
        all_data = pd.DataFrame()

        # Loop through each child folder and look for data.csv
        for child_folder in child_folders:
            data_file_path = os.path.join(child_folder, "data.csv")
            if os.path.exists(data_file_path):
                extracted_data = self.extract_data(data_file_path, child_folder, parent_folder_path)
                if extracted_data is not None:
                    # Remove outliers from the extracted data
                    cleaned_data = self.remove_outliers(extracted_data)
                    all_data = pd.concat([all_data, cleaned_data], ignore_index=True)

        # If we collected any data, save or append it to the abusive.csv file
        if not all_data.empty:
            self.save_to_abusive(all_data)
            df_table = self.prepare_data_with_predictions_and_retention(
                all_data)  # Prepare the table with predictions and retention
            self.export_to_excel(df_table)  # Export to a new Excel workbook
            self.display_formatted_table(df_table)  # Display the table as an image
            self.draw_charts_with_prediction(all_data)  # Draw charts with predictions
        else:
            print("No valid data found to append.")

    def extract_data(self, data_file_path, parent_folder_path, grandparent_folder_path):
        try:
            # Read the CSV file and extract specific columns (Cycle #, Disch. Capacity, Disch. Energy, DCIR)
            df = pd.read_csv(data_file_path,
                             usecols=['Cycle #', 'Disch. Capacity (AHr)', 'Disch. Energy (WHr)', 'DCIR (mOhm)'])

            # Add the parent folder name and grandparent folder name as new columns
            parent_folder_name = os.path.basename(parent_folder_path)
            grandparent_folder_name = os.path.basename(os.path.dirname(parent_folder_path))

            df['Sample'] = parent_folder_name
            df['Model'] = grandparent_folder_name

            return df
        except ValueError:
            print(f"File {data_file_path} does not have the expected columns.")
            return None
        except Exception as e:
            print(f"Error processing file {data_file_path}: {e}")
            return None

    def remove_outliers(self, df):
        # Define the columns where outliers should be removed
        cols_to_check = ['Disch. Capacity (AHr)', 'Disch. Energy (WHr)', 'DCIR (mOhm)']

        for col in cols_to_check:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier range
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter out rows that are outside the bounds
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df

    def save_to_abusive(self, data):
        # Check if the abusive.csv file exists
        file_exists = os.path.exists(self.abusive_file_path)

        # Save or append the data
        if file_exists:
            data.to_csv(self.abusive_file_path, mode='a', header=False, index=False)
            print(f"Appended data to {self.abusive_file_path}")
        else:
            data.to_csv(self.abusive_file_path, mode='w', header=True, index=False)
            print(f"Created and saved data to {self.abusive_file_path}")

    def prepare_data_with_predictions_and_retention(self, data):
        # Group data by Sample and Model
        groups = data.groupby(['Sample', 'Model'])

        # Use the user-defined milestones for predictions
        prediction_cycles = np.array([self.standard_milestone, self.reference_milestone]).reshape(-1, 1)

        # Create a list to hold table data
        table_data = []

        # Iterate through each group and calculate predictions, retention, and evaluation scores
        for (sample, model), group in groups:
            # Get the last cycle data
            last_cycle_data = group.iloc[-1]

            # Get initial data (average of cycles 6 to 10)
            initial_data = group[(group['Cycle #'] >= 6) & (group['Cycle #'] <= 10)].mean()

            # Get predictions and evaluation scores (only MAE and R²)
            capacity_pred_std, capacity_pred_ref, capacity_mae, capacity_r2 = self.predict_values_with_evaluation(
                group, 'Disch. Capacity (AHr)', prediction_cycles)
            energy_pred_std, energy_pred_ref, energy_mae, energy_r2 = self.predict_values_with_evaluation(
                group, 'Disch. Energy (WHr)', prediction_cycles)
            dcir_pred_std, dcir_pred_ref, dcir_mae, dcir_r2 = self.predict_values_with_evaluation(
                group, 'DCIR (mOhm)', prediction_cycles)

            # Calculate retention percentages
            capacity_retention_last = (last_cycle_data['Disch. Capacity (AHr)'] / initial_data[
                'Disch. Capacity (AHr)']) * 100
            capacity_retention_std = (capacity_pred_std / initial_data['Disch. Capacity (AHr)']) * 100
            capacity_retention_ref = (capacity_pred_ref / initial_data['Disch. Capacity (AHr)']) * 100

            energy_retention_last = (last_cycle_data['Disch. Energy (WHr)'] / initial_data['Disch. Energy (WHr)']) * 100
            energy_retention_std = (energy_pred_std / initial_data['Disch. Energy (WHr)']) * 100
            energy_retention_ref = (energy_pred_ref / initial_data['Disch. Energy (WHr)']) * 100

            dcir_retention_last = (last_cycle_data['DCIR (mOhm)'] / initial_data['DCIR (mOhm)'] - 1) * 100
            dcir_retention_std = (dcir_pred_std / initial_data['DCIR (mOhm)'] - 1) * 100
            dcir_retention_ref = (dcir_pred_ref / initial_data['DCIR (mOhm)'] - 1) * 100

            # Add data to the table, including MAE and R², and the milestones
            table_data.append({
                'Sample': sample,
                'Model': model,
                'Initial Capacity (6-10 avg)': round(initial_data['Disch. Capacity (AHr)'], 2),
                'Initial Energy (6-10 avg)': round(initial_data['Disch. Energy (WHr)'], 2),
                'Initial DCIR (6-10 avg)': round(initial_data['DCIR (mOhm)'], 2),
                'Last Cycle': int(last_cycle_data['Cycle #']),
                'Capacity Last': round(last_cycle_data['Disch. Capacity (AHr)'], 2),
                'Energy Last': round(last_cycle_data['Disch. Energy (WHr)'], 2),
                'DCIR Last': round(last_cycle_data['DCIR (mOhm)'], 2),
                # Predictions and retention for the standard milestone
                f'Capacity Pred {self.standard_milestone}': round(capacity_pred_std, 2),
                f'Energy Pred {self.standard_milestone}': round(energy_pred_std, 2),
                f'DCIR Pred {self.standard_milestone}': round(dcir_pred_std, 2),
                f'Capacity Retention {self.standard_milestone} (%)': round(capacity_retention_std, 2),
                f'Energy Retention {self.standard_milestone} (%)': round(energy_retention_std, 2),
                f'DCIR Increase {self.standard_milestone} (%)': round(dcir_retention_std, 2),
                # Predictions and retention for the reference milestone
                f'Capacity Pred {self.reference_milestone}': round(capacity_pred_ref, 2),
                f'Energy Pred {self.reference_milestone}': round(energy_pred_ref, 2),
                f'DCIR Pred {self.reference_milestone}': round(dcir_pred_ref, 2),
                f'Capacity Retention {self.reference_milestone} (%)': round(capacity_retention_ref, 2),
                f'Energy Retention {self.reference_milestone} (%)': round(energy_retention_ref, 2),
                f'DCIR Increase {self.reference_milestone} (%)': round(dcir_retention_ref, 2),
                # Only MAE and R² for each metric
                'Capacity MAE': round(capacity_mae, 4),
                'Capacity R²': round(capacity_r2, 4),
                'Energy MAE': round(energy_mae, 4),
                'Energy R²': round(energy_r2, 4),
                'DCIR MAE': round(dcir_mae, 4),
                'DCIR R²': round(dcir_r2, 4)
            })

        # Convert to DataFrame for export and display
        return pd.DataFrame(table_data)

    def predict_values_with_evaluation(self, group, y_col, prediction_cycles):
        """Fit a linear regression model, evaluate performance, and predict values for the given cycles."""
        X = group['Cycle #'].values.reshape(-1, 1)
        y = group[y_col].values

        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the linear regression model on the training data
        model = LinearRegression().fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Evaluate the model performance on the test set (only MAE and R²)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Make predictions for the given prediction cycles (e.g., cycle 250 and 500)
        predictions = model.predict(prediction_cycles)

        # Return predictions for the 250 and 500 cycles, and the evaluation metrics (MAE and R²)
        return predictions[0], predictions[1], mae, r2

    def export_to_excel(self, df):
        """Export the rounded data to a new Excel workbook."""
        # Create a new Excel writer object
        output_file_path = r'C:\\Users\\aesavle\OneDrive - Techtronic Industries Co. Ltd\\Work\\5. Task_sync\\2022\\8_app_db\\data\\raw\\npq\\exported_data.xlsx'
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Predicted Data', index=False)

        print(f"Data exported to Excel file: {output_file_path}")

    def draw_charts_with_prediction(self, data):
        """Draw the charts for Disch. Capacity (AHr), Disch. Energy (WHr), and DCIR (mOhm) with predictions."""
        # Group data by Sample and Model
        groups = data.groupby(['Sample', 'Model'])

        # Create subplots 4for each Y axis data
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Define prediction points using user-defined milestones
        prediction_cycles = np.array([self.standard_milestone, self.reference_milestone]).reshape(-1, 1)

        # Iterate through each group and plot the values
        for (sample, model), group in groups:
            # Plot and predict Disch. Capacity (AHr)
            self.plot_with_prediction(axes[0], group, 'Disch. Capacity (AHr)', sample, model, prediction_cycles)
            axes[0].set_ylabel('Disch. Capacity (AHr)')
            axes[0].set_title(
                f'Disch. Capacity (AHr) vs Cycle # (Milestones: {self.standard_milestone}, {self.reference_milestone})')

            # Plot and predict Disch. Energy (WHr)
            self.plot_with_prediction(axes[1], group, 'Disch. Energy (WHr)', sample, model, prediction_cycles)
            axes[1].set_ylabel('Disch. Energy (WHr)')
            axes[1].set_title(
                f'Disch. Energy (WHr) vs Cycle # (Milestones: {self.standard_milestone}, {self.reference_milestone})')

            # Plot and predict DCIR (mOhm)
            self.plot_with_prediction(axes[2], group, 'DCIR (mOhm)', sample, model, prediction_cycles)
            axes[2].set_ylabel('DCIR (mOhm)')
            axes[2].set_title(
                f'DCIR (mOhm) vs Cycle # (Milestones: {self.standard_milestone}, {self.reference_milestone})')

        # Set x-axis label for the last plot
        axes[2].set_xlabel('Cycle #')

        # Add legends
        for ax in axes:
            ax.legend()

        # Display the plot
        plt.tight_layout()
        plt.show()

    def plot_with_prediction(self, ax, group, y_col, sample, model, prediction_cycles):
        """Fit a linear regression model, plot the data, and make predictions."""
        # Prepare data for linear regression
        X = group['Cycle #'].values.reshape(-1, 1)
        y = group[y_col].values

        # Plot the actual data
        ax.plot(X, y, label=f'{sample} - {model} Actual')

        # Fit the linear regression model
        linear_model = LinearRegression().fit(X, y)

        # Predict the values for the user-specified milestones
        predictions = linear_model.predict(prediction_cycles)

        # Plot the predicted values at the specified milestones
        ax.scatter(prediction_cycles, predictions, color='red', marker='x', label=f'{sample} - {model} Prediction')

        # Annotate the predictions with milestone labels
        for i, cycle in enumerate(prediction_cycles):
            ax.text(cycle, predictions[i], f'{predictions[i]:.2f} @ Cycle {cycle[0]}', fontsize=10, color='red')

    def display_formatted_table(self, df):
        """Display a well-formatted table with actual and predicted values using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 2))  # Set size based on the data

        # Hide axes
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

        # Create the table and display it
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)  # Scale table to make it more readable

        plt.tight_layout()
        plt.show()

    def browse_for_folder(self):
        # Use tkinter to open a file dialog and return the selected folder path
        root = Tk()
        root.withdraw()  # Hide the main window
        folder_selected = filedialog.askdirectory(title="Select Parent Folder")
        return folder_selected

if __name__ == "__main__":
    analyzer = cycle_life_analyze()
    analyzer.run_analysis()
