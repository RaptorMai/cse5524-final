from Dataset.FungiTastic import FungiTastic
import os

val_dataset = FungiTastic('/research/nfs_chao_209/fungi-clef-2025', split='val', usage='training')

species_dict = val_dataset.df['scientificName'].value_counts().to_dict()
# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Sort the dictionary by values in descending order and take top 20
sorted_items = sorted(species_dict.items(), key=lambda item: item[1], reverse=True)

# Extract keys and values for plotting
species_names = [item[0] for item in sorted_items]
species_counts = [item[1] for item in sorted_items]

# Create a bar chart
plt.figure(figsize=(12, 8))
# Use numeric indices for x-axis instead of species names
plt.bar(range(len(species_names)), species_counts)
plt.xticks([])  # Remove x-axis tick labels
plt.bar(species_names, species_counts)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Eval Species Distribution')
plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
# Create the directory if it doesn't exist
os.makedirs('/home/tian.855/fungi/logs', exist_ok=True)

# Save the figure
plt.savefig('/home/tian.855/fungi/logs/species_distribution.png', dpi=300)
