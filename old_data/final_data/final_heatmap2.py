import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return [np.nan]

def process_data(data):
    data['Loss Vector'] = data['Loss Vector'].apply(safe_literal_eval)
    data['Average Loss'] = data['Loss Vector'].apply(lambda x: np.mean(x) if not np.isnan(x).all() else np.nan)
    return data

def plot_heatmaps(data, file_name, full_regex, sequence_length, analysis_type, accuracy_limits, loss_limits):
    filtered_data = data[data['Sequence Length'] == sequence_length]
    relevant_data = filtered_data[['Scale Factor', 'Hidden Size', 'Accuracy', 'Average Loss']]

    avg_accuracy = relevant_data.groupby(['Scale Factor', 'Hidden Size']).mean()['Accuracy'].unstack()
    mean_loss = relevant_data.groupby(['Scale Factor', 'Hidden Size']).mean()['Average Loss'].unstack()

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(avg_accuracy, ax=axs[0], cmap='viridis', annot=True, vmin=accuracy_limits[0], vmax=accuracy_limits[1])
    axs[0].set_title(f'Average Accuracy for {full_regex} (Sequence Length {sequence_length})')
    axs[0].set_xlabel('Hidden Size')
    axs[0].set_ylabel('Scale Factor')

    sns.heatmap(mean_loss, ax=axs[1], cmap='coolwarm', annot=True, vmin=loss_limits[0], vmax=loss_limits[1])
    axs[1].set_title(f'Mean Loss for {full_regex} (Sequence Length {sequence_length})')
    axs[1].set_xlabel('Hidden Size')
    axs[1].set_ylabel('Scale Factor')

    plt.tight_layout()

    output_dir = f'png_2/{analysis_type}/{full_regex}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/heatmap_{full_regex}_sequence_{sequence_length}_{file_name}.png')
    plt.close()

# Define fixed color range limits for accuracy and loss
accuracy_limits = [0.5, 1]  # Assuming accuracy ranges from 0 to 1
loss_limits = [0.3, 1]      # Set this based on your data's loss range

# Hardcoded mapping of file names to the correct regexes
file_to_regex = {
    "Linear_asymptotic_analysis_(0_1)_0{2,4}(0_1)_1{2,4}.csv": "(0|1)*0{2,4}(0|1)*1{2,4}",
    "Linear_asymptotic_analysis_(11_00)_.csv": "(11|00)*",
    "linear_asymptotic_analysis_(abc_xyz){3}.csv": "(abc|xyz){3}",
    "log_asymptotic_analysis_(0_1)_0{2,4}(0_1)_1{2,4}.csv": "(0|1)*0{2,4}(0|1)*1{2,4}",
    "Log_asymptotic_analysis_(11_00)_.csv": "(11|00)*",
    "log_asymptotic_analysis_(abc_xyz){3}.csv": "(abc|xyz){3}"
}

for file_name, full_regex in file_to_regex.items():
    analysis_type = 'log' if 'log' in file_name.lower() else 'normal'
    data = pd.read_csv(file_name)
    processed_data = process_data(data)
    sequence_lengths = processed_data['Sequence Length'].unique()

    for sequence_length in sequence_lengths:
        plot_heatmaps(processed_data, file_name, full_regex, sequence_length, analysis_type, accuracy_limits, loss_limits)

print("Heatmap generation complete.")
