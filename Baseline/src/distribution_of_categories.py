import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
import os
from datetime import datetime
import logging



current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Specify the folder where you want to save the log and plot
save_folder = './src/distributions'
os.makedirs(save_folder, exist_ok=True)

log_file_path = os.path.join(save_folder, f'class_category_distribions_{current_time}.txt')
logging.basicConfig(filename = log_file_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()


# Load the JSON data
with open('/home/vedasri/datasets/HomerCompTraining/HomerCompTrainingReadCoco.json', 'r') as file:
    annotations = json.load(file)

# Extract category IDs and find the unique ones
categories = [annotation['category_id'] for annotation in annotations['annotations']]
unique_categories = set(categories)
num_unique_categories = len(unique_categories)

# Log the number of unique categories
logger.info(f"Number of unique categories: {num_unique_categories}")
logger.info(f"Unique categories: {unique_categories}")

unique_categories, counts = np.unique(categories, return_counts=True)

# Sort counts and corresponding unique_categories
sorted_indices = np.argsort(counts)[::-1]  # Sort indices in descending order
sorted_counts = counts[sorted_indices]
sorted_categories = unique_categories[sorted_indices]

logger.info(f"Sorted Categories: {sorted_categories}")
logger.info(f"Sorted Counts: {sorted_counts}")

# Normalize counts for colormap
norm = Normalize(vmin=sorted_counts.min(), vmax=sorted_counts.max())
norm_counts = norm(sorted_counts)
color_map = viridis(norm_counts)

# Create a bar chart
plt.figure(figsize=(10, 6))
ax = plt.gca()  # Get current axis
bars = ax.bar(np.arange(num_unique_categories), sorted_counts, color=color_map)


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval, int(yval), va='bottom', fontsize=7)  # va='bottom' for vertical alignment

plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=viridis), ax=ax, label='Number of Annotations')
plt.title('Distribution of Categories')
plt.xlabel('Category ID')
plt.ylabel('Number of Annotations')
plt.grid(axis='y', linestyle='--')

# Save the plot to a file with the current date
plot_file_path = os.path.join(save_folder, f'category_distribution_{current_time}.png')
plt.savefig(plot_file_path)

# Log the saving of the plot
logger.info(f"Plot saved to: {plot_file_path}")

# Show the plot
plt.show()