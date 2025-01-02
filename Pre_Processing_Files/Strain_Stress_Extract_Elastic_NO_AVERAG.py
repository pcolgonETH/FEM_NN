from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np
import os

# Path to the folder containing the .odb files
folder_path = 'E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Shear_Elastic/YX/odb_files/'

# Output file paths
input_file_path = folder_path + '/' + 'features_V1.data'
output_file_path = folder_path + '/' + 'targets_V1.data'

# Function to process a single .odb file and extract data for every Gauss point
def process_odb_file(odb, frame):
    # Initialize arrays to collect all data
    elastic_strains = []
    stresses = []

    # Access field outputs once
    elastic_strain_field = frame.fieldOutputs['EE']
    stress_field = frame.fieldOutputs['S']

    # Fetch instance and elements once
    instance = odb.rootAssembly.instances['CUBE-1']

    # Retrieve all field values in fewer steps
    elastic_strain_values = elastic_strain_field.getSubset(region=instance).values
    stress_values = stress_field.getSubset(region=instance).values

    # Collect data for every Gauss point
    elastic_strains = [val.data for val in elastic_strain_values]
    stresses = [val.data for val in stress_values]

    # Extract Gauss point IDs
    gauss_point_ids = [val.elementLabel for val in elastic_strain_values]

    return elastic_strains, stresses, gauss_point_ids

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
            elastic_strains, stresses, gauss_point_ids = process_odb_file(odb, frame)

            # Add data for every Gauss point
            for gp_id, es, st in zip(gauss_point_ids, elastic_strains, stresses):
                # Add simulation ID and Gauss point ID
                #input_data.append([simulation_id, gp_id] + list(es))
                #output_data.append([simulation_id, gp_id] + list(st))
                input_data.append(list(es))
                output_data.append(list(st))


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
