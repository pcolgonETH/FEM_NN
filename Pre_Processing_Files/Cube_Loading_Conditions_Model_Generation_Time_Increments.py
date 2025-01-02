import numpy as np
from scipy.stats import truncnorm
from abaqus import *
from abaqusConstants import *
from caeModules import *

# Parameters for displacement generation
lower_bounds = np.array([-0.25, -0.25, -0.25])
upper_bounds = np.array([0.25, 0.25, 0.25])

# Parameters for time increment generation
mean_time_increment = 0.001  # For Gaussian sampling
std_time_increment = 0.005
lower_bound_time_increment = 0.0001
upper_bound_time_increment = 0.05

# Method to generate uniform samples
def generate_uniform_samples(lower_bounds, upper_bounds, num_samples):
    samples = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_samples, 3))
    return samples

# Method to generate Gaussian samples
def generate_gaussian_samples(mean_vector, std_vector, lower_bounds, upper_bounds, num_samples):
    samples = np.zeros((num_samples, 3))
    for i in range(3):
        a, b = (lower_bounds[i] - mean_vector[i]) / std_vector[i], (upper_bounds[i] - mean_vector[i]) / std_vector[i]
        samples[:, i] = truncnorm.rvs(a, b, loc=mean_vector[i], scale=std_vector[i], size=num_samples)
    return samples

# Function to generate Gaussian time increments
def generate_gaussian_time_increments(mean, std, lower_bound, upper_bound, num_samples):
    a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
    time_increments = truncnorm.rvs(a, b, loc=mean, scale=std, size=num_samples)
    return time_increments

# Function to create variants and modify boundary conditions and time increments
def create_variants(num_variants, displacements, time_increments):
    base_model = 'Model-1'  # Your base model name in Abaqus
    job_name = 'Job_Variant_'  # Base name for job

    # Lists to store displacement vectors and time increments for exporting to a .data file
    displacement_data = []
    time_increment_data = []

    for i in range(num_variants):
        # Create a copy of the base model
        new_model_name = f'{base_model}_copy_{i+1}'
        mdb.Model(name=new_model_name, objectToCopy=mdb.models[base_model])

        # Modify boundary conditions with new displacements
        modify_boundary_conditions(mdb.models[new_model_name], displacements[i])

        # Modify the step with new time increment
        modify_step_time_increment(mdb.models[new_model_name], time_increments[i])

        # Append the current displacements and time increment to the data lists
        displacement_data.append(displacements[i])
        time_increment_data.append(time_increments[i])

        # Use a unique job ID
        job_full_name = f'{job_name}{i+1:03d}'  # Adds leading zeros for IDs (e.g., Job_Variant_001)

        # Create input file with the updated job name
        job = mdb.Job(name=job_full_name, model=new_model_name)
        job.writeInput()

        # Delete the model after generating the input file
        del mdb.models[new_model_name]

        # Print message indicating job ID and deletion of model
        print(f"Model '{new_model_name}' created and deleted with Job ID: {i+1}")

    # Export the displacement and time increment data to a .data file
    save_displacement_data(displacement_data, time_increment_data)

# Function to modify boundary conditions
def modify_boundary_conditions(model, displacement_values):
    # The three elements of displacement_values correspond to X, Y, and Z displacements
    model.boundaryConditions['Disp-X1'].setValues(u1=displacement_values[0])
    model.boundaryConditions['Disp-Y1'].setValues(u2=displacement_values[1])
    model.boundaryConditions['Disp-Z1'].setValues(u3=displacement_values[2])

def modify_step_time_increment(model, time_increment):
    # Assume the step is named 'Step-1', change as needed
    step_name = 'Step-1'
    if step_name in model.steps:
        step = model.steps[step_name]
        # Set maxInc to be equal to or greater than initialInc
        step.setValues(initialInc=time_increment, maxInc=time_increment)
    else:
        raise ValueError(f"Step '{step_name}' not found in the model.")

# Function to save displacement data and time increments to a .data file
def save_displacement_data(displacement_data, time_increment_data):
    with open("displacement_conditions.data", "w") as f:
        for i, (vector, time_increment) in enumerate(zip(displacement_data, time_increment_data)):
            f.write(f"{i+1}, {vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}, {time_increment:.6f}\n")
    print("Displacement and time increment data saved to 'displacement_conditions.data'.")

# Set the number of variants (samples) and displacement generation method here
num_variants = 2000  # Define how many variants you want to generate
method = 'gaussian'  # Choose 'uniform' or 'gaussian'

# Generate displacements and time increments based on the chosen method
if method == 'uniform':
    displacements = generate_uniform_samples(lower_bounds, upper_bounds, num_variants)
    time_increments = np.random.uniform(low=lower_bound_time_increment, high=upper_bound_time_increment, size=num_variants)
elif method == 'gaussian':
    mean_vector = np.array([0, 0, 0])  # For displacements
    std_vector = (upper_bounds - lower_bounds) / 6
    displacements = generate_gaussian_samples(mean_vector, std_vector, lower_bounds, upper_bounds, num_variants)
    time_increments = generate_gaussian_time_increments(mean_time_increment, std_time_increment,
                                                        lower_bound_time_increment, upper_bound_time_increment,
                                                        num_variants)
else:
    raise ValueError("Invalid method. Choose either 'uniform' or 'gaussian'.")

# Create model variants and generate input files
create_variants(num_variants, displacements, time_increments)

print("Input files generated for all variants.")
