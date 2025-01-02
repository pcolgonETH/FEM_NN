from abaqus import *
from abaqusConstants import *
from odbAccess import *
import os

# Path to the folder containing the .odb files
folder_path = 'E:/SIMULIA/Projects/Euler_Odb_Outputs/Cube_Shear_Elastic/YX/odb_files/'

# Output file paths
input_file_path = os.path.join(folder_path, 'features_V2_with_ID.data')
output_file_path = os.path.join(folder_path, 'targets_V2_with_ID.data')

# Function to process a single .odb file and extract data for a specific Gauss point
def process_odb_file(odb, frame, gp_idx=0):
    field_outputs = frame.fieldOutputs
    instance = odb.rootAssembly.instances['CUBE-1']
    field_subsets = {f'SDV{i+1}': field_outputs[f'SDV{i+1}'].getSubset(region=instance) for i in range(82)}

    # Extract data for the specified Gauss point index
    elastic_strains = [field_subsets[f'SDV{i+1}'].values[gp_idx].data for i in range(6)]
    inelastic_strains = [field_subsets[f'SDV{i+1}'].values[gp_idx].data for i in range(6, 12)]
    peeq = field_subsets['SDV13'].values[gp_idx].data
    dfgrd = [field_subsets[f'SDV{i+1}'].values[gp_idx].data for i in range(13, 22)]
    stresses = [field_subsets[f'SDV{i+1}'].values[gp_idx].data for i in range(76, 82)]
    ddsdde = [field_subsets[f'SDV{i+1}'].values[gp_idx].data for i in range(28, 64)]
    strain_inc = [field_subsets[f'SDV{i+1}'].values[gp_idx].data for i in range(70, 76)]
    strain_rate = [field_subsets[f'SDV{i+1}'].values[gp_idx].data for i in range(64, 70)]

    return elastic_strains, inelastic_strains, peeq, dfgrd, stresses, ddsdde, strain_inc, strain_rate

# Process .odb files
with open(input_file_path, 'w') as input_file, open(output_file_path, 'w') as output_file:
    odb_file_list = [f for f in os.listdir(folder_path) if f.endswith('.odb')]

    for filename in odb_file_list:
        odb_path = os.path.join(folder_path, filename)
        try:
            odb = openOdb(odb_path)
            last_step_name = list(odb.steps.keys())[-1]  # Fix: Convert keys to a list
            last_step = odb.steps[last_step_name]

            for i in range(0, len(last_step.frames)):  # Fix: Use range instead of slicing
                frame = last_step.frames[i]
                data = process_odb_file(odb, frame, gp_idx=0)
                (elastic_strains, inelastic_strains, peeq, dfgrd,
                 stresses, ddsdde, strain_inc, strain_rate) = data

                # Write input data
                input_file.write(','.join(map(str, elastic_strains)) + ',' +
                                 ','.join(map(str, inelastic_strains)) + ',' +
                                 ','.join(map(str, strain_inc)) + ',' +
                                 str(peeq) + ',' +
                                 ','.join(map(str, strain_rate)) + ',' +
                                 ','.join(map(str, dfgrd)) + '\n')

                # Write output data
                output_file.write(','.join(map(str, stresses)) + ',' +
                                  ','.join(map(str, ddsdde)) + '\n')

            print(f"Processed file: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            if 'odb' in locals():
                odb.close()

print(f"Input data written to {input_file_path}")
print(f"Output data written to {output_file_path}")
