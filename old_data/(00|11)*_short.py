import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# Function to expand the loss vectors
def expand_loss_vectors(data):
    # Convert the string representation of loss vectors to actual lists
    data['Loss Vector'] = data['Loss Vector'].apply(ast.literal_eval)
    
    # Expand the 'Loss Vector' into separate rows
    data = data.explode('Loss Vector')
    
    # Convert 'Loss Vector' values to float
    data['Loss Vector'] = data['Loss Vector'].astype(float)
    
    return data

# Re-loading the noise-free and noisy datasets
file_path_no_noise = 'shortsequence_nonoise_(00_11)_.csv'
file_path_noise = 'shortsequence_withnoise_(00_11)_.csv'

data_no_noise = pd.read_csv(file_path_no_noise)
data_noise = pd.read_csv(file_path_noise)

# Expand the loss vectors for both datasets
data_no_noise = expand_loss_vectors(data_no_noise)
data_noise = expand_loss_vectors(data_noise)

# Group by Sequence Length and Hidden Size to get average accuracy and mean loss
avg_accuracy_no_noise = data_no_noise.groupby(['Sequence Length', 'Hidden Size']).mean()['Accuracy'].unstack()
avg_accuracy_noise = data_noise.groupby(['Sequence Length', 'Hidden Size']).mean()['Accuracy'].unstack()

mean_loss_no_noise = data_no_noise.groupby(['Sequence Length', 'Hidden Size']).mean()['Loss Vector'].unstack()
mean_loss_noise = data_noise.groupby(['Sequence Length', 'Hidden Size']).mean()['Loss Vector'].unstack()

# Plotting the four heatmaps
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Accuracy without noise
sns.heatmap(avg_accuracy_no_noise, ax=axs[0, 0], cmap='viridis')
axs[0, 0].set_title('Average Accuracy - Shallow RNN (No Noise) for (00|11)*')
axs[0, 0].set_xlabel('Hidden Size')
axs[0, 0].set_ylabel('Sequence Length')

# Accuracy with noise
sns.heatmap(avg_accuracy_noise, ax=axs[0, 1], cmap='viridis')
axs[0, 1].set_title('Average Accuracy - Shallow RNN (With Noise) for (00|11)*')
axs[0, 1].set_xlabel('Hidden Size')
axs[0, 1].set_ylabel('Sequence Length')

# Mean Loss without noise
sns.heatmap(mean_loss_no_noise, ax=axs[1, 0], cmap='coolwarm')
axs[1, 0].set_title('Mean Loss - Shallow RNN (No Noise) for (00|11)*')
axs[1, 0].set_xlabel('Hidden Size')
axs[1, 0].set_ylabel('Sequence Length')

# Mean Loss with noise
sns.heatmap(mean_loss_noise, ax=axs[1, 1], cmap='coolwarm')
axs[1, 1].set_title('Mean Loss - Shallow RNN (With Noise) for (00|11)*')
axs[1, 1].set_xlabel('Hidden Size')
axs[1, 1].set_ylabel('Sequence Length')

plt.savefig('(00|11)*_short.png')
plt.tight_layout()
plt.show()
