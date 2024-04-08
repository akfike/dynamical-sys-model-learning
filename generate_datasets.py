import numpy as np

def generate_rlc_dataset(num_simulations, R, L, C, dt, time_end, initial_uncertainty_range=(-0.05, 0.05), noise_level=0.01):
    # Initialize an empty array to hold the dataset
    dataset = np.empty((0, 4))
    
    # Generate data
    for sim in range(num_simulations):
        time = np.arange(0, time_end, dt)
        V_in = np.sin(time * 2 * np.pi / time_end * np.random.uniform(0.5, 1.5))  # Randomize input frequency
        V_in_noisy = V_in + np.random.normal(0, noise_level, size=time.shape)  # Add Gaussian noise

        V_C = np.zeros_like(time)
        I_L = np.zeros_like(time)
        
        # Add uncertainty to the initial state
        initial_uncertainty_VC = np.random.uniform(*initial_uncertainty_range)
        initial_uncertainty_IL = np.random.uniform(*initial_uncertainty_range)
        V_C[0] = 5.0 + initial_uncertainty_VC  # Initial voltage across the capacitor with uncertainty
        I_L[0] = 0.0 + initial_uncertainty_IL  # Initial current through the inductor with uncertainty
        
        # Simulate using Euler's method
        for t in range(1, len(time)):
            dV_C_dt = -(1/C)*I_L[t-1]
            dI_L_dt = (1/L)*(V_in_noisy[t-1] - R*I_L[t-1] - V_C[t-1])
            V_C[t] = V_C[t-1] + dV_C_dt*dt
            I_L[t] = I_L[t-1] + dI_L_dt*dt

        V_C_noisy = V_C + np.random.normal(0, noise_level, size=time.shape)  # Adding noise to output if desired
        I_L_noisy = I_L + np.random.normal(0, noise_level, size=time.shape)  # Adding noise to output if desired

        simulation_data = np.column_stack((V_in_noisy[:-1], V_C[:-1], I_L[:-1], V_C_noisy[1:]))  # Updated for RLC
        dataset = np.vstack((dataset, simulation_data))
        # Append to the dataset: input voltage, previous capacitor voltage, previous inductor current, next capacitor voltage
    
    return dataset

# Parameters for RLC circuit
R = 1.0  # Resistance in ohms
L = 1.0  # Inductance in henrys
C = 1.0  # Capacitance in farads
dt = 0.01  # Time step in seconds
time_end = 10.0  # Total time for each simulation in seconds

# Specify the number of simulations for each dataset
num_train_simulations = 300
num_val_simulations = 100
num_test_simulations = 100
num_final_test_simulations = 100

# Generate datasets independently for RLC circuit
train_data_rlc = generate_rlc_dataset(num_train_simulations, R, L, C, dt, time_end)
val_data_rlc = generate_rlc_dataset(num_val_simulations, R, L, C, dt, time_end)
test_data_rlc = generate_rlc_dataset(num_test_simulations, R, L, C, dt, time_end)
final_test_data_rlc = generate_rlc_dataset(num_final_test_simulations, R, L, C, dt, time_end)

# Save the datasets for RLC circuit
np.save('datasets/train_data_rlc.npy', train_data_rlc)
np.save('datasets/val_data_rlc.npy', val_data_rlc)
np.save('datasets/test_data_rlc.npy', test_data_rlc)
np.save('datasets/final_test_data_rlc.npy', final_test_data_rlc)

np.savetxt('train_data_rlc.txt', train_data_rlc)
np.savetxt('val_data_rlc.txt', val_data_rlc)
np.savetxt('test_data_rlc.txt', test_data_rlc)
np.savetxt('final_test_data_rlc.txt', final_test_data_rlc)

print(f"RLC Training set size: {len(train_data_rlc)}")
print(f"RLC Validation set size: {len(val_data_rlc)}")
print(f"RLC Testing set size: {len(test_data_rlc)}")
print(f"RLC Final testing set size: {len(final_test_data_rlc)}")
