import numpy as np
from scipy.signal import cont2discrete

def generate_discrete_state_space_dataset(num_simulations, dt, time_end, x_range=[(-1, 1), (-1, 1)], noise_level=0.01):
    A = np.array([[-1, 1], [1, -1]])
    B = np.array([[1], [0]])  
    C = np.eye(2) 
    D = np.zeros((2, 1))  
    
    system_discrete = cont2discrete((A, B, C, D), dt, method='zoh')
    Ad, Bd, _, _, _ = system_discrete
    
    dataset = np.empty((0, 4))  # Columns: x1(t), x2(t), x1(t+dt), x2(t+dt), removed u(t)
    
    for sim in range(num_simulations):
        time = np.arange(0, time_end, dt)
        # u = np.sin(time * 2 * np.pi / time_end * np.random.uniform(0.5, 1.5))  # Not needed anymore
        
        x = np.zeros((len(time), 2))  
        
        for i in range(2):
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
datasets = {key: generate_discrete_state_space_dataset(num, dt, time_end) for key, num in num_simulations.items()}

for key, data in datasets.items():
    np.save(f'datasets/DiscreteStateSpace/{key}_data.npy', data)
    print(f"{key.capitalize()} set size: {len(data)}")
