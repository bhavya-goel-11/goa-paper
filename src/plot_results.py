import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 1. Ensure correct command-line usage
if len(sys.argv) != 3:
    print("Usage: python plot_results.py <variant_csv_file> <output_image_name.png>")
    print("Example: python plot_results.py cgoa1_cec17_results.csv cgoa1_comparison.png")
    sys.exit(1)

variant_csv = os.path.join("..", "results", sys.argv[1])
output_image = os.path.join("..", "plots", sys.argv[2])
baseline_csv = os.path.join("..", "results", "goa_cec17_results.csv")

# 2. Check if files exist
if not os.path.exists(baseline_csv):
    print(f"Error: Could not find baseline '{baseline_csv}'. Did you run the original GOA benchmark?")
    sys.exit(1)
if not os.path.exists(variant_csv):
    print(f"Error: Could not find variant '{variant_csv}'. Ensure it is in the current directory.")
    sys.exit(1)

# 3. Load the data
df_goa = pd.read_csv(baseline_csv)
df_var = pd.read_csv(variant_csv)

# Merge the dataframes on the 'Function' column to ensure perfectly aligned rows
df_merged = pd.merge(
    df_goa[['Function', 'Mean']], 
    df_var[['Function', 'Mean']], 
    on='Function', 
    suffixes=('_GOA', '_Variant')
)

algo_name = variant_csv.split('_')[0].upper()

# 4. Print a formatted comparison table to the terminal
print("\n" + "="*65)
print(f"{f'CEC 2017 Mean Fitness: Original GOA vs {algo_name}':^65}")
print("="*65)
formatted_df = df_merged.copy()
formatted_df['Mean_GOA'] = formatted_df['Mean_GOA'].apply(lambda x: f"{x:.4e}")
formatted_df['Mean_Variant'] = formatted_df['Mean_Variant'].apply(lambda x: f"{x:.4e}")
print(formatted_df.to_string(index=False))
print("="*65 + "\n")

# 5. Generate the Side-by-Side Bar Chart
plt.figure(figsize=(16, 7))

# Set up X-axis indices and bar width
x = np.arange(len(df_merged['Function']))
width = 0.35  

# Plot both sets of bars
plt.bar(x - width/2, df_merged['Mean_GOA'], width, label='Original GOA', color='skyblue', edgecolor='black')
plt.bar(x + width/2, df_merged['Mean_Variant'], width, label=algo_name, color='lightcoral', edgecolor='black')

# Set Log scale, titles, and labels
plt.yscale('log') 
plt.title(f'Performance Comparison: Original GOA vs {algo_name} on CEC 2017', fontsize=16, fontweight='bold')
plt.xlabel('CEC17 Test Functions', fontsize=12)
plt.ylabel('Mean Fitness (Log Scale)', fontsize=12)

# Set the x-ticks to be the function names exactly in the middle of the grouped bars
plt.xticks(x, df_merged['Function'], rotation=45)

plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot for the research paper
plt.tight_layout()
plt.savefig(output_image, dpi=300)
print(f"Comparison chart successfully saved as '{output_image}'!")

# Optional: Disable if running headless
# plt.show()