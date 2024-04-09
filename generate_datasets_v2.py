import numpy as np

def generate_rlc_dataset(num_simulations, R, L, C, dt, time_end, VC_range=(4.5, 5.5), IL_range=(-0.05, 0.05), noise_level=0.01):
    dataset = np.empty((0, 5))  
    for sim in range(num_simulations):
        time = np.arange(0, time_end, dt)
        V_in = np.sin(time * 2 * np.pi / time_end * np.random.uniform(0.5, 1.5))  
        V_in_noisy = V_in + np.random.normal(0, noise_level, size=time.shape)  

        V_C = np.zeros_like(time)
        I_L = np.zeros_like(time)
        
        V_C[0] = np.random.uniform(*VC_range)
        I_L[0] = np.random.uniform(*IL_range)
        
        for t in range(1, len(time)):
            dV_C_dt = I_L[t-1] / C
            dI_L_dt = (V_in_noisy[t-1] - I_L[t-1]*R - V_C[t-1]) / L
            V_C[t] = V_C[t-1] + dV_C_dt*dt
            I_L[t] = I_L[t-1] + dI_L_dt*dt

        V_C_noisy = V_C + np.random.normal(0, noise_level, size=time.shape)  
        I_L_noisy = I_L + np.random.normal(0, noise_level, size=time.shape)  

        simulation_data = np.column_stack((V_in_noisy[:-1], V_C[:-1], I_L[:-1], V_C_noisy[1:], I_L_noisy[1:]))
        dataset = np.vstack((dataset, simulation_data))
    
    return dataset

R = 1.0  
L = 1.0  
C = 1.0  
dt = 0.01  
time_end = 10.0  

VC_range = (4.5, 5.5)  
IL_range = (-0.05, 0.05)  

num_train_simulations = 300
num_val_simulations = 100
num_test_simulations = 100
num_final_test_simulations = 100

train_data_rlc = generate_rlc_dataset(num_train_simulations, R, L, C, dt, time_end, VC_range, IL_range)
val_data_rlc = generate_rlc_dataset(num_val_simulations, R, L, C, dt, time_end, VC_range, IL_range)
test_data_rlc = generate_rlc_dataset(num_test_simulations, R, L, C, dt, time_end, VC_range, IL_range)
final_test_data_rlc = generate_rlc_dataset(num_final_test_simulations, R, L, C, dt, time_end, VC_range, IL_range)

np.save('datasets/RLC/train_data_rlc.npy', train_data_rlc)
np.save('datasets/RLC/val_data_rlc.npy', val_data_rlc)
np.save('datasets/RLC/test_data_rlc.npy', test_data_rlc)
np.save('datasets/RLC/final_test_data_rlc.npy', final_test_data_rlc)

np.savetxt('train_data_rlc.txt', train_data_rlc)
np.savetxt('val_data_rlc.txt', val_data_rlc)
np.savetxt('test_data_rlc.txt', test_data_rlc)
np.savetxt('final_test_data_rlc.txt', final_test_data_rlc)

print(f"RLC Training set size: {len(train_data_rlc)}")
print(f"RLC Validation set size: {len(val_data_rlc)}")
print(f"RLC Testing set size: {len(test_data_rlc)}")
print(f"RLC Final testing set size: {len(final_test_data_rlc)}")
