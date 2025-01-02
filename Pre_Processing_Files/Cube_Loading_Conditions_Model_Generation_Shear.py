import numpy as np
from scipy.stats import truncnorm
from abaqus import *
from abaqusConstants import *
from caeModules import *

# Parameters for displacement generation
lower_bounds = np.array([-0.2])  # Only Y1 direction displacement
upper_bounds = np.array([0.2])

# Method to generate uniform samples
def generate_uniform_samples(lower_bounds, upper_bounds, num_samples):
    samples = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_samples, 1))
    return samples

# Method to generate Gaussian samples
def generate_gaussian_samples(mean_vector, std_vector, lower_bounds, upper_bounds, num_samples):
    samples = np.zeros((num_samples, 1))
    for i in range(1):  # Single direction (Y1)
        a, b = (lower_bounds[i] - mean_vector[i]) / std_vector[i], (upper_bounds[i] - mean_vector[i]) / std_vector[i]
        samples[:, i] = truncnorm.rvs(a, b, loc=mean_vector[i], scale=std_vector[i], size=num_samples)
    return samples

# Function to create variants and modify boundary conditions
def create_variants(num_variants, displacements):
    base_model = 'Model-1-Shear-Elastic'  # Your base model name in Abaqus
    job_name = 'Job_Variant_'  # Base name for job

    # List to store displacement values for exporting to a .data file
    displacement_data = []

    for i in range(num_variants):
        # Create a copy of the base model
        new_model_name = f'{base_model}_copy_{i+1}'
        mdb.Model(name=new_model_name, objectToCopy=mdb.models[base_model])

        # Modify boundary conditions with new displacements
        modify_boundary_conditions(mdb.models[new_model_name], displacements[i])

        # Append the current displacement to the data list
        displacement_data.append(displacements[i][0])  # Only Y1 displacement

        # Use a unique job ID
        job_full_name = f'{job_name}{i+1:03d}'  # Adds leading zeros (e.g., Job_Variant_001)

        # Create input file with the updated job name
        job = mdb.Job(name=job_full_name, model=new_model_name)
        job.writeInput()

        # Delete the model after generating the input file
        del mdb.models[new_model_name]

        # Print message indicating job ID and deletion of model
        print(f"Model '{new_model_name}' created and deleted with Job ID: {i+1}")

    # Export the displacement data to a .data file
    save_displacement_data(displacement_data)

# Function to modify boundary conditions
def modify_boundary_conditions(model, displacement_values):
    # Update only the displacement in the Y1 direction
    model.boundaryConditions['Disp-Y-Shear'].setValues(u2=displacement_values[0])

# Function to save displacement data to a .data file
def save_displacement_data(displacement_data):
    with open("displacement_conditions.data", "w") as f:
        for i, value in enumerate(displacement_data):
            f.write(f"{i+1}, {value:.6f}\n")  # Save only the Y1 displacement
    print("Displacement data saved to 'displacement_conditions.data'.")

# Set the number of variants (samples) and displacement generation method here
num_variants = 150  # Define how many variants you want to generate
method = 'gaussian'  # Choose 'uniform' or 'gaussian'

# Generate displacements based on the chosen method
if method == 'uniform':
    displacements = generate_uniform_samples(lower_bounds, upper_bounds, num_variants)
elif method == 'gaussian':
    mean_vector = [0]
    std_vector = (upper_bounds - lower_bounds) / 6
    displacements = generate_gaussian_samples(mean_vector, std_vector, lower_bounds, upper_bounds, num_variants)
else:
    raise ValueError("Invalid method. Choose either 'uniform' or 'gaussian'.")

# Create model variants and generate input files
create_variants(num_variants, displacements)

print("Input files generated for all variants.")
