import os
import numpy as np
import lasio
import pandas as pd
import glob


current_dir = os.getcwd()
# Define the file path
file_path = current_dir +"/Data/OilSandsDB/PICKS.xls"

# Read the Excel file into a DataFrame
picks_df= pd.read_excel(file_path)

wells_path = current_dir +"/Data/OilSandsDB/WELLS.xls"
wells_df= pd.read_excel(wells_path)

picks_df['SitID'] = picks_df['SitID'].astype(str)
wells_df['UWI'] = wells_df['UWI'].astype(str)
# wells_df['UWI'] = [uwi.replace('/', '-') for uwi in wells_df['UWI'].to_list()]

wells_df['SitID'] = wells_df['SitID'].astype(str)
0
# Step 2: Merge the DataFrames
merged_df = pd.merge(picks_df, wells_df[['UWI', 'SitID']], on='SitID', how='left')


# Directory containing the .las files
las_directory = current_dir + "/Data/OilSandsDB/Logs/"

# List all .las files
las_files = glob.glob(las_directory + '*.las')

# Initialize a dictionary to hold UWI and Well Name mappings
uwi_wellname_map = {}

# Loop through each file
for las_file in las_files:
    las_data = lasio.read(las_file)
    
    # Extract UWI
    uwi = getattr(las_data.well, 'UWI', None).value if 'UWI' in las_data.well else None

    # Check if 'Well' value exists
    well_name = getattr(las_data.well, 'WELL', None).value if 'WELL' in las_data.well else uwi

    # Add to dictionary
    uwi_wellname_map[uwi] = well_name

# Map well names to the UWI in merged_df
merged_df['Well_Name'] = merged_df['UWI'].map(uwi_wellname_map)

merged_df = merged_df[merged_df['Quality'].isin([0, 1,2])]


merged_df.rename(columns={'Pick': 'Depth'}, inplace=True)
# Convert 'pick' column to numeric, coercing errors to NaN
merged_df['Depth'] = pd.to_numeric(merged_df['Depth'], errors='coerce')
merged_df.dropna(inplace=True)
output_file_path =current_dir+ '/Data/OilSandsDB/tops_map.csv'  # Replace with your desired path

merged_df.to_csv(output_file_path, index=False)


current_dir = os.getcwd()
# Define the file path
file_path = current_dir +"/Data/OilSandsDB/Logs/00-01-01-073-05W5-0.las"

# Read the LAS file
las = lasio.read(file_path)

# Access UWI - it's usually in the well section
uwi = las.well['UWI'].value
# Search for the UWI in the DataFrame
matching_rows = merged_df[merged_df['UWI'] == uwi]

print(matching_rows)


output_file_path = current_dir +"/Data/OilSandsDB/00-01-01-073-05W5-0_TOPS.csv"
matching_rows.to_csv(output_file_path, index=False)