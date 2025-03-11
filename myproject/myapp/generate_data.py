import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        layout = []
        for _ in range(10):  # 10 components
            x = np.random.uniform(0, 10)  # X position
            y = np.random.uniform(0, 10)  # Y position
            power = np.random.uniform(0.5, 5.0)  # Power constraint
            layout.extend([x, y, power])
        data.append(layout)
    
    df = pd.DataFrame(data, columns=[f"c{i}_{p}" for i in range(10) for p in ["x", "y", "power"]])
    df.to_csv("synthetic_layout_data.csv", index=False)

generate_synthetic_data()
