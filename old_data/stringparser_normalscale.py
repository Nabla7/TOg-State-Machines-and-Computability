import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
import re
import math

# Function to parse the data string
def parse_training_data(data):
    pattern = r"Sequence Length: (\d+), Hidden Size: (\d+), Accuracy: (\d\.\d+)"
    matches = re.findall(pattern, data)
    parsed_data = [{'Sequence Length': int(match[0]),
                    'Hidden Size': int(match[1]),
                    'Accuracy': float(match[2])} for match in matches]
    return parsed_data

# Example usage
data = """
Training Segment for Sequence Length 2 and Hidden Size 2
=======================================
Sequence Length: 2, Hidden Size: 2, Accuracy: 0.865

Training Segment for Sequence Length 2 and Hidden Size 5
=======================================
Sequence Length: 2, Hidden Size: 5, Accuracy: 0.835

Training Segment for Sequence Length 2 and Hidden Size 8
=======================================
Sequence Length: 2, Hidden Size: 8, Accuracy: 0.855

Training Segment for Sequence Length 2 and Hidden Size 11
=======================================
Sequence Length: 2, Hidden Size: 11, Accuracy: 0.77

Training Segment for Sequence Length 2 and Hidden Size 14
=======================================
Sequence Length: 2, Hidden Size: 14, Accuracy: 0.815

Training Segment for Sequence Length 2 and Hidden Size 17
=======================================
Sequence Length: 2, Hidden Size: 17, Accuracy: 0.835

Training Segment for Sequence Length 2 and Hidden Size 20
=======================================
Sequence Length: 2, Hidden Size: 20, Accuracy: 0.82

Training Segment for Sequence Length 2 and Hidden Size 23
=======================================
Sequence Length: 2, Hidden Size: 23, Accuracy: 0.835

Training Segment for Sequence Length 2 and Hidden Size 26
=======================================
Sequence Length: 2, Hidden Size: 26, Accuracy: 0.765

Training Segment for Sequence Length 2 and Hidden Size 29
=======================================
Sequence Length: 2, Hidden Size: 29, Accuracy: 0.86

Training Segment for Sequence Length 5 and Hidden Size 2
=======================================
Sequence Length: 5, Hidden Size: 2, Accuracy: 0.85

Training Segment for Sequence Length 5 and Hidden Size 5
=======================================
Sequence Length: 5, Hidden Size: 5, Accuracy: 0.9

Training Segment for Sequence Length 5 and Hidden Size 8
=======================================
Sequence Length: 5, Hidden Size: 8, Accuracy: 0.915

Training Segment for Sequence Length 5 and Hidden Size 11
=======================================
Sequence Length: 5, Hidden Size: 11, Accuracy: 0.935

Training Segment for Sequence Length 5 and Hidden Size 14
=======================================
Sequence Length: 5, Hidden Size: 14, Accuracy: 0.935

Training Segment for Sequence Length 5 and Hidden Size 17
=======================================
Sequence Length: 5, Hidden Size: 17, Accuracy: 0.91

Training Segment for Sequence Length 5 and Hidden Size 20
=======================================
Sequence Length: 5, Hidden Size: 20, Accuracy: 0.91

Training Segment for Sequence Length 5 and Hidden Size 23
=======================================
Sequence Length: 5, Hidden Size: 23, Accuracy: 0.94

Training Segment for Sequence Length 5 and Hidden Size 26
=======================================
Sequence Length: 5, Hidden Size: 26, Accuracy: 0.945

Training Segment for Sequence Length 5 and Hidden Size 29
=======================================
Sequence Length: 5, Hidden Size: 29, Accuracy: 0.9

Training Segment for Sequence Length 8 and Hidden Size 2
=======================================
Sequence Length: 8, Hidden Size: 2, Accuracy: 0.515

Training Segment for Sequence Length 8 and Hidden Size 5
=======================================
Sequence Length: 8, Hidden Size: 5, Accuracy: 0.91

Training Segment for Sequence Length 8 and Hidden Size 8
=======================================
Sequence Length: 8, Hidden Size: 8, Accuracy: 0.93

Training Segment for Sequence Length 8 and Hidden Size 11
=======================================
Sequence Length: 8, Hidden Size: 11, Accuracy: 0.945

Training Segment for Sequence Length 8 and Hidden Size 14
=======================================
Sequence Length: 8, Hidden Size: 14, Accuracy: 0.95

Training Segment for Sequence Length 8 and Hidden Size 17
=======================================
Sequence Length: 8, Hidden Size: 17, Accuracy: 0.945

Training Segment for Sequence Length 8 and Hidden Size 20
=======================================
Sequence Length: 8, Hidden Size: 20, Accuracy: 0.93

Training Segment for Sequence Length 8 and Hidden Size 23
=======================================
Sequence Length: 8, Hidden Size: 23, Accuracy: 0.93

Training Segment for Sequence Length 8 and Hidden Size 26
=======================================
Sequence Length: 8, Hidden Size: 26, Accuracy: 0.915

Training Segment for Sequence Length 8 and Hidden Size 29
=======================================
Sequence Length: 8, Hidden Size: 29, Accuracy: 0.915

Training Segment for Sequence Length 100 and Hidden Size 2
=======================================
Sequence Length: 100, Hidden Size: 2, Accuracy: 0.865

Training Segment for Sequence Length 100 and Hidden Size 5
=======================================
Sequence Length: 100, Hidden Size: 5, Accuracy: 0.895

Training Segment for Sequence Length 100 and Hidden Size 8
=======================================
Sequence Length: 100, Hidden Size: 8, Accuracy: 0.93

Training Segment for Sequence Length 100 and Hidden Size 11
=======================================
Sequence Length: 100, Hidden Size: 11, Accuracy: 0.805

Training Segment for Sequence Length 100 and Hidden Size 14
=======================================
Sequence Length: 100, Hidden Size: 14, Accuracy: 0.89

Training Segment for Sequence Length 100 and Hidden Size 17
=======================================
Sequence Length: 100, Hidden Size: 17, Accuracy: 0.88

Training Segment for Sequence Length 100 and Hidden Size 20
=======================================
Sequence Length: 100, Hidden Size: 20, Accuracy: 0.875

Training Segment for Sequence Length 100 and Hidden Size 23
=======================================
Sequence Length: 100, Hidden Size: 23, Accuracy: 0.88

Training Segment for Sequence Length 100 and Hidden Size 26
=======================================
Sequence Length: 100, Hidden Size: 26, Accuracy: 0.875

Training Segment for Sequence Length 100 and Hidden Size 29
=======================================
Sequence Length: 100, Hidden Size: 29, Accuracy: 0.865

Training Segment for Sequence Length 300 and Hidden Size 2
=======================================
Sequence Length: 300, Hidden Size: 2, Accuracy: 0.85

Training Segment for Sequence Length 300 and Hidden Size 5
=======================================
Sequence Length: 300, Hidden Size: 5, Accuracy: 0.875

Training Segment for Sequence Length 300 and Hidden Size 8
=======================================
Sequence Length: 300, Hidden Size: 8, Accuracy: 0.87

Training Segment for Sequence Length 300 and Hidden Size 11
=======================================
Sequence Length: 300, Hidden Size: 11, Accuracy: 0.89

Training Segment for Sequence Length 300 and Hidden Size 14
=======================================
Sequence Length: 300, Hidden Size: 14, Accuracy: 0.865

Training Segment for Sequence Length 300 and Hidden Size 17
=======================================
Sequence Length: 300, Hidden Size: 17, Accuracy: 0.845

Training Segment for Sequence Length 300 and Hidden Size 20
=======================================
Sequence Length: 300, Hidden Size: 20, Accuracy: 0.895

Training Segment for Sequence Length 300 and Hidden Size 23
=======================================
Sequence Length: 300, Hidden Size: 23, Accuracy: 0.83

Training Segment for Sequence Length 300 and Hidden Size 26
=======================================
Sequence Length: 300, Hidden Size: 26, Accuracy: 0.83

Training Segment for Sequence Length 300 and Hidden Size 29
=======================================
Sequence Length: 300, Hidden Size: 29, Accuracy: 0.855

Training Segment for Sequence Length 600 and Hidden Size 2
=======================================
Sequence Length: 600, Hidden Size: 2, Accuracy: 0.88

Training Segment for Sequence Length 600 and Hidden Size 5
=======================================
Sequence Length: 600, Hidden Size: 5, Accuracy: 0.815

Training Segment for Sequence Length 600 and Hidden Size 8
=======================================
Sequence Length: 600, Hidden Size: 8, Accuracy: 0.86

Training Segment for Sequence Length 600 and Hidden Size 11
=======================================
Sequence Length: 600, Hidden Size: 11, Accuracy: 0.915

Training Segment for Sequence Length 600 and Hidden Size 14
=======================================
Sequence Length: 600, Hidden Size: 14, Accuracy: 0.87

Training Segment for Sequence Length 600 and Hidden Size 17
=======================================
Sequence Length: 600, Hidden Size: 17, Accuracy: 0.85

Training Segment for Sequence Length 600 and Hidden Size 20
=======================================
Sequence Length: 600, Hidden Size: 20, Accuracy: 0.875

Training Segment for Sequence Length 600 and Hidden Size 23
=======================================
Sequence Length: 600, Hidden Size: 23, Accuracy: 0.855

Training Segment for Sequence Length 600 and Hidden Size 26
=======================================
Sequence Length: 600, Hidden Size: 26, Accuracy: 0.84

Training Segment for Sequence Length 600 and Hidden Size 29
=======================================
Sequence Length: 600, Hidden Size: 29, Accuracy: 0.89

Training Segment for Sequence Length 900 and Hidden Size 2
=======================================
Sequence Length: 900, Hidden Size: 2, Accuracy: 0.9

Training Segment for Sequence Length 900 and Hidden Size 5
=======================================
Sequence Length: 900, Hidden Size: 5, Accuracy: 0.875

Training Segment for Sequence Length 900 and Hidden Size 8
=======================================
Sequence Length: 900, Hidden Size: 8, Accuracy: 0.91

Training Segment for Sequence Length 900 and Hidden Size 11
=======================================
Sequence Length: 900, Hidden Size: 11, Accuracy: 0.84

Training Segment for Sequence Length 900 and Hidden Size 14
=======================================
Sequence Length: 900, Hidden Size: 14, Accuracy: 0.875

Training Segment for Sequence Length 900 and Hidden Size 17
=======================================
Sequence Length: 900, Hidden Size: 17, Accuracy: 0.875

Training Segment for Sequence Length 900 and Hidden Size 20
=======================================
Sequence Length: 900, Hidden Size: 20, Accuracy: 0.895

Training Segment for Sequence Length 900 and Hidden Size 23
=======================================
Sequence Length: 900, Hidden Size: 23, Accuracy: 0.815

Training Segment for Sequence Length 900 and Hidden Size 26
=======================================
Sequence Length: 900, Hidden Size: 26, Accuracy: 0.845

Training Segment for Sequence Length 900 and Hidden Size 29
=======================================
Sequence Length: 900, Hidden Size: 29, Accuracy: 0.855

Training Segment for Sequence Length 3000 and Hidden Size 2
=======================================
Sequence Length: 3000, Hidden Size: 2, Accuracy: 0.835

Training Segment for Sequence Length 3000 and Hidden Size 5
=======================================
Sequence Length: 3000, Hidden Size: 5, Accuracy: 0.865

Training Segment for Sequence Length 3000 and Hidden Size 8
=======================================
Sequence Length: 3000, Hidden Size: 8, Accuracy: 0.855

Training Segment for Sequence Length 3000 and Hidden Size 11
=======================================
Sequence Length: 3000, Hidden Size: 11, Accuracy: 0.895

Training Segment for Sequence Length 3000 and Hidden Size 14
=======================================
Sequence Length: 3000, Hidden Size: 14, Accuracy: 0.87

Training Segment for Sequence Length 3000 and Hidden Size 17
=======================================
Sequence Length: 3000, Hidden Size: 17, Accuracy: 0.85

Training Segment for Sequence Length 3000 and Hidden Size 20
=======================================
Sequence Length: 3000, Hidden Size: 20, Accuracy: 0.85

Training Segment for Sequence Length 3000 and Hidden Size 23
=======================================
Sequence Length: 3000, Hidden Size: 23, Accuracy: 0.845

Training Segment for Sequence Length 3000 and Hidden Size 26
=======================================
Sequence Length: 3000, Hidden Size: 26, Accuracy: 0.855

Training Segment for Sequence Length 3000 and Hidden Size 29
=======================================
Sequence Length: 3000, Hidden Size: 29, Accuracy: 0.825

Training Segment for Sequence Length 6000 and Hidden Size 2
=======================================
Sequence Length: 6000, Hidden Size: 2, Accuracy: 0.84

Training Segment for Sequence Length 6000 and Hidden Size 5
=======================================
Sequence Length: 6000, Hidden Size: 5, Accuracy: 0.845

Training Segment for Sequence Length 6000 and Hidden Size 8
=======================================
Sequence Length: 6000, Hidden Size: 8, Accuracy: 0.875

Training Segment for Sequence Length 6000 and Hidden Size 11
=======================================
Sequence Length: 6000, Hidden Size: 11, Accuracy: 0.91

Training Segment for Sequence Length 6000 and Hidden Size 14
=======================================
Sequence Length: 6000, Hidden Size: 14, Accuracy: 0.845

Training Segment for Sequence Length 6000 and Hidden Size 17
=======================================
Sequence Length: 6000, Hidden Size: 17, Accuracy: 0.88

Training Segment for Sequence Length 6000 and Hidden Size 20
=======================================
Sequence Length: 6000, Hidden Size: 20, Accuracy: 0.855

Training Segment for Sequence Length 6000 and Hidden Size 23
=======================================
Sequence Length: 6000, Hidden Size: 23, Accuracy: 0.84

Training Segment for Sequence Length 6000 and Hidden Size 26
=======================================
Sequence Length: 6000, Hidden Size: 26, Accuracy: 0.855

Training Segment for Sequence Length 6000 and Hidden Size 29
=======================================
Sequence Length: 6000, Hidden Size: 29, Accuracy: 0.885

Training Segment for Sequence Length 9000 and Hidden Size 2
=======================================
Sequence Length: 9000, Hidden Size: 2, Accuracy: 0.845

Training Segment for Sequence Length 9000 and Hidden Size 5
=======================================
Sequence Length: 9000, Hidden Size: 5, Accuracy: 0.88

Training Segment for Sequence Length 9000 and Hidden Size 8
=======================================
Sequence Length: 9000, Hidden Size: 8, Accuracy: 0.885

Training Segment for Sequence Length 9000 and Hidden Size 11
=======================================
Sequence Length: 9000, Hidden Size: 11, Accuracy: 0.875

Training Segment for Sequence Length 9000 and Hidden Size 14
"""

# Parse the data
parsed_data = parse_training_data(data)

# Convert to DataFrame
df = pd.DataFrame(parsed_data)

# Option to scale Sequence Length to log scale
use_log_scale = True  # Set to False if you do not want log scale

if use_log_scale:
    # Apply log transformation, ensuring no zero values
    df['Sequence Length'] = df['Sequence Length'].apply(lambda x: math.log(x) if x > 0 else 0)

# Extract the columns needed for the plot
seq_length = df['Sequence Length']
hidden_size = df['Hidden Size']
accuracy = df['Accuracy']

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(seq_length, hidden_size, accuracy, c=accuracy, cmap='viridis', marker='o')

# Creating a manifold
x_lin = np.linspace(seq_length.min(), seq_length.max(), 100)
y_lin = np.linspace(hidden_size.min(), hidden_size.max(), 100)
x_grid, y_grid = np.meshgrid(x_lin, y_lin)
z_grid = griddata((seq_length, hidden_size), accuracy, (x_grid, y_grid), method='linear')

# Plotting the surface
#ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, cmap='viridis', edgecolor='none')

# Labels
ax.set_xlabel('Sequence Length' + (' (Log Scale)' if use_log_scale else ''))
ax.set_ylabel('Hidden Size')
ax.set_zlabel('Accuracy')


# Color bar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Accuracy')

plt.title('3D Plot of RNN Performance')
plt.show()
