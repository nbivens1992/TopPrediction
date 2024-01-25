import os
import lasio


current_dir = os.getcwd()
# Define the file path
file_path = current_dir +"/Data/OilSandsDB/Logs/00-01-01-073-05W5-0.las"

# Read the LAS file
las = lasio.read(file_path)

# Access UWI - it's usually in the well section
uwi = las.well['UWI'].value


# Check if UWI contains any whitespace
if uwi and any(char.isspace() for char in uwi):
    print("UWI contains whitespace.")
else:
    print("UWI does not contain whitespace.")

print("UWI:", uwi)

print("UWI:", uwi)