from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np
import os

# Path to the folder containing the .odb files
folder_path = 'E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Shear_Elastic/YZ/odb_files/'

# Output file paths
input_file_path = folder_path + '/' + 'features_V1.data'
output_file_path = folder_path + '/' + 'targets_V1.data'

# Specify the target Gauss point ID (update this with your desired ID)
target_gauss_point_id = 1  # Example: 1, replace with the desired ID

# Function to process a single .odb file and extract data for the specific Gauss point
def process_odb_file(odb, frame, target_gauss_point_id):
    # Access field outputs once
    elastic_strain_field = frame.fieldOutputs['EE']
    stress_field = frame.fieldOutputs['S']

    # Fetch instance and elements once
    instance = odb.rootAssembly.instances['CUBE-1']

    # Retrieve field values for the specific Gauss point
    elastic_strain_values = elastic_strain_field.getSubset(region=instance).values
    stress_values = stress_field.getSubset(region=instance).values

    # Filter for the target Gauss point
    for es, st in zip(elastic_strain_values, stress_values):
        if es.elementLabel == target_gauss_point_id:
            return es.data, st.data  # Return the elastic strain and stress data

    return None, None  # Return None if the target Gauss point is not found

# Loop over all .odb files in the specified folder
input_data = []
output_data = []

simulation_id = 1  # Start simulation ID from 1

for filename in os.listdir(folder_path):
    if filename.endswith('.odb'):
        odb_path = os.path.join(folder_path, filename)
        odb = openOdb(odb_path)
        last_step = odb.steps[odb.steps.keys()[-1]]
        total_frames = len(last_step.frames)
        
        for i in range(0, total_frames):  # Loop through frames
            frame = last_step.frames[i]
            elastic_strain, stress = process_odb_file(odb, frame, target_gauss_point_id)

            # Add data for the specific Gauss point if it exists
            if elastic_strain is not None and stress is not None:
                #input_data.append([simulation_id] + list(elastic_strain))
                #output_data.append([simulation_id] + list(stress))
                input_data.append(list(elastic_strain))
                output_data.append(list(stress))


        print(f"Processed {filename}")
        simulation_id += 1  # Increment simulation ID
        odb.close()

# Write the input data to a .data file
with open(input_file_path, 'w') as f:
    for row in input_data:
        f.write(', '.join(map(str, row)) + '\n')

# Write the output data to a .data file
with open(output_file_path, 'w') as f:
    for row in output_data:
        f.write(', '.join(map(str, row)) + '\n')

print(f"Input data written to {input_file_path}")
print(f"Output data written to {output_file_path}")
