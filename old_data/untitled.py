import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

# Load the data
#df = pd.read_csv('shortsequence_nonoise_(0_1)_0{2,4}(0_1)_1{2,4}.csv')
#df = pd.read_csv('shortsequence_withnoise_(0_1)_0{2,4}(0_1)_1{2,4}.csv')


df = pd.read_csv('shortsequence_nonoise_(00_11)_.csv')


# Extract the columns needed for the plot
seq_length = df['Sequence Length']
hidden_size = df['Hidden Size']
accuracy = df['Accuracy']

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(seq_length, hidden_size, accuracy, c=accuracy, cmap='viridis', marker='o')

# Create a manifold
# Generating a meshgrid for the manifold
x_lin = np.linspace(seq_length.min(), seq_length.max(), 100)
y_lin = np.linspace(hidden_size.min(), hidden_size.max(), 100)
x_grid, y_grid = np.meshgrid(x_lin, y_lin)

# Interpolating data
z_grid = griddata((seq_length, hidden_size), accuracy, (x_grid, y_grid), method='cubic')

# Plotting the surface
#ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, cmap='viridis', edgecolor='none')

# Labels
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Hidden Size')
ax.set_zlabel('Accuracy')

# Color bar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Accuracy')

plt.title('3D Plot of RNN Performance')
plt.show()

