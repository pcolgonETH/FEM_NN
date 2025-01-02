from abaqus import *
from abaqusConstants import *
from odbAccess import *
import visualization
import numpy as np
import os

# Load displacement conditions from the .data file, handling non-numeric values
#displacement_conditions_file = 'E:/SIMULIA/Projects/Euler_Inputs/Cube_Uniaxial_01/displacement_conditions.data'
#displacement_conditions = np.genfromtxt(displacement_conditions_file, delimiter=",", skip_header=0, invalid_raise=False)

# Ensure displacement conditions are valid and contain 4 columns (ID, x, y, z)
#if displacement_conditions.shape[1] != 4:
    #raise ValueError(f"Expected 4 columns (ID, x, y, z) but got {displacement_conditions.shape[1]}")

# Path to the folder containing the .odb files
folder_path = 'E:/SIMULIA/Projects/Euler_Odb_Outputs/KRATOS_PLASTIC_TRIAXIAL/odb_files/'

# Output file paths
input_file_path = folder_path + '/' + 'features_V1_with_ID.data'
output_file_path = folder_path + '/' + 'targets_V1_with_ID.data'
#input_file_path = folder_path + '/' + 'features_Job_Test.data'
#output_file_path = folder_path + '/' + 'targets_Job_Test.data'

# Function to process a single .odb file and return the average total strain and stress
def process_odb_file(odb, frame):
    
    # Initialize arrays to collect all data
    elastic_strains = []
    inelastic_strains = []
    total_strains = []
    strain_rate = []
    peeq = []
    dfgrd = []
    stresses = []
    deformations = []
    ddsdde = []
    strain_inc = []

    # Access field outputs once
    elastic_strain_field = frame.fieldOutputs['EE']
    strain_rate_field = frame.fieldOutputs['ER']
    stress_field = frame.fieldOutputs['S']
    deformation_field = frame.fieldOutputs['U']

    # Fetch the SDV fields
    sdv_fields = [frame.fieldOutputs[f'SDV{i+1}'] for i in range(82)]

    # Fetch instance and elements once
    instance = odb.rootAssembly.instances['CUBE-1']
    
    # Retrieve all field values in fewer steps
    #elastic_strain_values = elastic_strain_field.getSubset(region=instance).values
    sdv_values = [sdv_field.getSubset(region=instance).values for sdv_field in sdv_fields]
    #strain_rate_values = strain_rate_field.getSubset(region=instance).values
    #stress_values = stress_field.getSubset(region=instance).values
    deformation_values = deformation_field.getSubset(region=instance).values

    # Collect all data
    #elastic_strains.extend([val.data for val in elastic_strain_values])
    #strain_rate.extend([val.data for val in strain_rate_values])

    for idx in range(len(sdv_values[0])): # Assuming the length of sdv_values[0] equals the number of elements
        elastic_strains.append([sdv_values[i][idx].data for i in range(0, 6)])
        inelastic_strains.append([sdv_values[i][idx].data for i in range(6, 12)])  # SDV7 to SDV12 for inelastic strain
        peeq.append(sdv_values[12][idx].data)  # SDV13 for plastic equivalent strain
        dfgrd.append([sdv_values[i][idx].data for i in range(13, 22)])  # SDV14 to SDV22 for deformation gradient tensor
        ddsdde.append([sdv_values[i][idx].data for i in range(28,64)])  # SDV29 to SDV64 for DDSDDE
        strain_rate.append([sdv_values[i][idx].data for i in range(64,70)])
        stresses.append([sdv_values[i][idx].data for i in range(76,82)])
        strain_inc.append([sdv_values[i][idx].data for i in range(70,76)])
        
        
    #stresses.extend([val.data for val in stress_values])
    deformations.extend([val.data for val in deformation_values])
    #deformations.extend([np.linalg.norm(val.data) for val in deformation_values])  # Deformation magnitude

    # Convert lists to numpy arrays for efficient operations
    elastic_strains = np.array(elastic_strains)
    inelastic_strains = np.array(inelastic_strains)
    total_strains = elastic_strains + inelastic_strains
    strain_rate = np.array(strain_rate)
    peeq = np.array(peeq)
    dfgrd = np.array(dfgrd)
    stresses = np.array(stresses)
    deformations = np.array(deformations)
    ddsdde = np.array(ddsdde)
    strain_inc = np.array(strain_inc)

    # Calculate the averages
    average_elastic_strain = np.mean(elastic_strains, axis=0)
    average_inelastic_strain = np.mean(inelastic_strains, axis=0)  # Inelastic strain replaces plastic strain
    average_total_strains = np.mean(total_strains, axis=0)
    average_strain_rate = np.mean(strain_rate, axis=0)
    average_peeq = np.mean(peeq, axis=0)  # Average plastic equivalent strain
    average_dfgrd = np.mean(dfgrd, axis=0)  # Average deformation gradient tensor
    average_stress = np.mean(stresses, axis=0)
    average_deformation = np.mean(deformations, axis=0)
    average_ddsdde = np.mean(ddsdde, axis=0)
    average_strain_inc = np.mean(strain_inc, axis=0)

    return average_elastic_strain, average_inelastic_strain, average_total_strains, average_strain_rate, average_peeq, average_dfgrd, average_stress, average_deformation, average_ddsdde, average_strain_inc

# Loop over all .odb files in the specified folder
input_data = {}
output_data = {}
odb_file_list = [f for f in os.listdir(folder_path) if f.endswith('.odb')]

for idx, filename in enumerate(odb_file_list):
    if filename.endswith('.odb'):
        odb_path = os.path.join(folder_path, filename)
        odb = openOdb(odb_path)
        last_step = odb.steps[odb.steps.keys()[-1]]
        total_frames = len(last_step.frames)
        
        frame_inputs = []
        frame_outputs = []
        
        # Initialize previous strain and stress for the first frame
        #previous_elastic_strain = np.zeros(6)
        #previous_inelastic_strain = np.zeros(6)
        #previous_stress = np.zeros(6)

        for i in range(1, total_frames):
            frame = last_step.frames[i]

            # Calculate avg_elastic_strain and avg_inelastic_strain from the previous increment
            #avg_elastic_strain = previous_elastic_strain
            #avg_inelastic_strain = previous_inelastic_strain

            # Process the current frame to get the new strains, stresses, etc.
            avg_elastic_strain, avg_inelastic_strain, avg_total_strain, avg_strain_rate, avg_peeq, avg_dfgrd, avg_stress, avg_deformation, avg_ddsdde, avg_strain_inc = process_odb_file(odb, frame)

            # Calculate the increments of strain (difference between current and previous values)
            #inc_elastic_strain = current_elastic_strain - previous_elastic_strain
            #inc_inelastic_strain = current_inelastic_strain - previous_inelastic_strain
            #inc_total_strain = inc_elastic_strain + inc_inelastic_strain
            #inc_stress = current_stress - previous_stress

            # Update the previous strain and stress with the current values
            #previous_elastic_strain = current_elastic_strain
            #previous_inelastic_strain = current_inelastic_strain
            #previous_stress = current_stress

            # Store the original strain, inelastic strain, stress, and incremental values as features
            frame_inputs.append((avg_elastic_strain, avg_inelastic_strain, avg_strain_inc, avg_peeq, avg_strain_rate, avg_dfgrd))
            # Store the incremental stress as targets
            frame_outputs.append((avg_stress, avg_ddsdde))

        input_data[filename] = frame_inputs
        output_data[filename] = frame_outputs
        
        print(f"Processed {filename}")
        
        odb.close()

# Write the input data to a .data file, adding ID and x, y, z
with open(input_file_path, 'w') as f:
    for idx, (filename, frame_inputs) in enumerate(input_data.items()):
        #id_value, x, y, z = displacement_conditions[idx]  # Corresponding ID and x, y, z data from .data file
        for avg_elastic_strain, avg_inelastic_strain, avg_strain_inc, avg_peeq, avg_strain_rate, avg_dfgrd in frame_inputs:
            input_elastic_strain_str = ','.join(map(str, avg_elastic_strain))
            input_inelastic_strain_str = ','.join(map(str, avg_inelastic_strain))
            #input_total_strain_str = ','.join(map(str, avg_total_strain))
            #input_stress_str = ','.join(map(str, current_stress))
            #inc_elastic_strain_str = ','.join(map(str, inc_elastic_strain))
            #inc_inelastic_strain_str = ','.join(map(str, inc_inelastic_strain))
            #inc_total_strain_str = ','.join(map(str, inc_total_strain))
            avg_strain_inc_str = ','.join(map(str, avg_strain_inc))
            avg_peeq_str = str(avg_peeq)
            avg_strain_rate_str = ','.join(map(str, avg_strain_rate))
            avg_dfgrd_str = ','.join(map(str, avg_dfgrd))
            # Add filename ID and x, y, z to the row
            #input_str = f"{id_value},{x},{y},{z},{input_elastic_strain_str},{input_inelastic_strain_str},{input_total_strain_str},{inc_total_strain_str},{avg_peeq_str},{avg_strain_rate_str},{avg_dfgrd_str}\n"
            input_str = f"{input_elastic_strain_str},{input_inelastic_strain_str},{avg_strain_inc_str},{avg_peeq_str},{avg_strain_rate_str},{avg_dfgrd_str}\n"
            f.write(input_str)

# Write the output data to a .data file, adding ID and x, y, z
with open(output_file_path, 'w') as f:
    for idx, (filename, frame_outputs) in enumerate(output_data.items()):
        #id_value, x, y, z = displacement_conditions[idx]  # Corresponding ID and x, y, z data from .data file
        for current_stress, avg_ddsdde in frame_outputs:
            #avg_deformation_str = ','.join(map(str, avg_deformation))
            input_stress_str = ','.join(map(str, current_stress))
            #inc_stress_str = ','.join(map(str, inc_stress))
            avg_ddsdde_str = ','.join(map(str, avg_ddsdde))
            # Add filename ID and x, y, z to the row
            #output_str = f"{id_value},{x},{y},{z},{input_stress_str},{inc_stress_str},{avg_deformation_str},{avg_ddsdde_str}\n"
            output_str = f"{input_stress_str},{avg_ddsdde_str}\n"
            f.write(output_str)

print(f"Input data written to {input_file_path}")
print(f"Output data written to {output_file_path}")
