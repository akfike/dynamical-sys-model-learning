import numpy as np
from scipy.signal import cont2discrete

k10, k12, k13, k21, k31, kd, V1 = 0.1, 0.05, 0.05, 0.07, 0.03, 0.09, 10.0  # Example values

def generate_discrete_complex_dataset(num_simulations, dt, time_end, x_range=[(-1, 1), (-1, 1), (-1, 1), (-1, 1)], noise_level=0.01):
    A = np.array([
        [-(k10 + k12 + k13), k12, k13, 0],
        [k21, -k21, 0, 0],
        [k31, 0, -k31, 0],
        [kd, 0, 0, -kd]
    ])
    B = np.array([[1/V1], [0], [0], [0]])  
    C = np.eye(4)  
    D = np.zeros((4, 1)) 
    
    system_discrete = cont2discrete((A, B, C, D), dt, method='zoh')
    Ad, Bd, _, _, _ = system_discrete
    
    dataset = np.empty((0, 8))  
    
    for sim in range(num_simulations):
        time = np.arange(0, time_end, dt)
        # u = np.sin(time * 2 * np.pi / time_end * np.random.uniform(0.5, 1.5)) 
        
        x = np.zeros((len(time), 4))  
        
        for i in range(4):
            x[0, i] = np.random.uniform(*x_range[i])
        
        for t in range(1, len(time)):
            x[t] = Ad.dot(x[t-1])
        
        x_noisy = x + np.random.normal(0, noise_level, size=x.shape)  
        
        simulation_data = np.column_stack((x_noisy[:-1], x_noisy[1:]))
        dataset = np.vstack((dataset, simulation_data))
    
    return dataset

dt = 0.01  
time_end = 10.0 

num_simulations = {'train': 300, 'val': 200, 'test': 200, 'final_test': 100}
datasets = {key: generate_discrete_complex_dataset(num, dt, time_end) for key, num in num_simulations.items()}

for key, data in datasets.items():
    np.save(f'datasets/Anesthesia/{key}_data.npy', data)
    print(f"{key.capitalize()} set size: {len(data)}")
