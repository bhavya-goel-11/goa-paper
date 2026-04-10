import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums

# --- Configuration & Paths ---
PROPOSED_ALGO = "cgoa1"
COMPETITORS = ["goa", "pso", "gwo", "woa", "sns", "sca", "dea", "abc"]
ALL_ALGOS = [PROPOSED_ALGO] + COMPETITORS
SUITES = ["cec2014", "cec2017", "cec2020", "cec2022", "engineering"]

RESULTS_DIR = "../results"
PLOTS_DIR = "../plots"

# Create output directories
for suite in SUITES:
    os.makedirs(f"{PLOTS_DIR}/convergence/{suite}", exist_ok=True)
    os.makedirs(f"{PLOTS_DIR}/tables/{suite}", exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- Helper Function: Save DataFrame as High-Res Image Table ---
def save_df_as_image(df, filename, title="", table_type=""):
    # Calculate dimensions based on dataframe size
    fig_width = max(10, len(df.columns) * 1.8)
    fig_height = max(4, len(df) * 0.5 + 1.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    if title:
        plt.title(title, weight='bold', size=16, pad=20)
        
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     loc='center', 
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5) # Scale height to allow for multi-line text
    
    # Apply dynamic colors and formatting
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header Row Styling
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6fa') # Light purple header
        else:
            text = cell.get_text().get_text()
            
            # 1. Highlight Logic for Rank Table
            if table_type == "rank":
                if text.endswith("(1)"):
                    cell.set_facecolor('#d4edda') # Light green background
                    cell.set_text_props(weight='bold') # Bold the winning text
                    
            # 2. Highlight Logic for Wilcoxon Table
            elif table_type == "wilcoxon":
                if "(+)" in text:
                    cell.set_facecolor('#d4edda') # Light green for Win
                elif "(-)" in text:
                    cell.set_facecolor('#f8d7da') # Light red for Loss
                    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def process_suite(suite):
    print(f"\n--- Processing Suite: {suite.upper()} ---")
    
    summary_data = []
    raw_data = {}
    conv_data = {}
    
    # 1. Gather all data
    for algo in ALL_ALGOS:
        sum_path = f"{RESULTS_DIR}/{algo}/{suite}/summary.csv"
        raw_path = f"{RESULTS_DIR}/{algo}/{suite}/raw.csv"
        conv_path = f"{RESULTS_DIR}/{algo}/{suite}/convergence.csv"
        
        if os.path.exists(sum_path) and os.path.exists(raw_path):
            df_sum = pd.read_csv(sum_path)
            df_sum['Algorithm'] = algo.upper()
            summary_data.append(df_sum)
            
            df_raw = pd.read_csv(raw_path, header=None)
            raw_data[algo] = df_raw.set_index(0) 
            
            df_conv = pd.read_csv(conv_path, header=None)
            conv_data[algo] = df_conv.set_index(0)

    if not summary_data:
        print(f"  [!] No data found for {suite}. Skipping.")
        return

    master_df = pd.concat(summary_data, ignore_index=True)
    functions = master_df['Function'].unique()
    
    # --- RANKING ---
    master_df['Rank'] = 0
    for func in functions:
        func_mask = master_df['Function'] == func
        ranked = master_df[func_mask].sort_values(by=['Mean', 'StdDev'])
        master_df.loc[ranked.index, 'Rank'] = range(1, len(ranked) + 1)
        
    # --- CREATE PIVOTED MEAN/STD/RANK TABLE ---
    master_df['Display'] = master_df.apply(lambda r: f"{r['Mean']:.2e}\n±{r['StdDev']:.2e}\n({int(r['Rank'])})", axis=1)
    
    pivot_table = master_df.pivot(index='Function', columns='Algorithm', values='Display').reset_index()
    pivot_table['F_num'] = pivot_table['Function'].str.replace('F', '').astype(int)
    pivot_table = pivot_table.sort_values('F_num').drop('F_num', axis=1)
    
    cols = ['Function', PROPOSED_ALGO.upper()] + [c.upper() for c in COMPETITORS if c.upper() in pivot_table.columns]
    pivot_table = pivot_table[cols]
    
    # Save as Image with "rank" highlight logic
    mean_std_img_path = f"{PLOTS_DIR}/tables/{suite}/Mean_Std_Rank_Table.png"
    save_df_as_image(pivot_table, mean_std_img_path, f"{suite.upper()}: Mean ± StdDev (Rank)", table_type="rank")
    print(f"  -> Saved Table: {mean_std_img_path}")

    # --- WILCOXON RANK-SUM TEST ---
    wilcoxon_results = []
    win_tie_loss = {comp.upper(): {'+': 0, '=': 0, '-': 0} for comp in COMPETITORS}
    
    for func in functions:
        row = {'Function': func}
        
        if PROPOSED_ALGO in raw_data and func in raw_data[PROPOSED_ALGO].index:
            prop_runs = raw_data[PROPOSED_ALGO].loc[func].values[1:] 
            prop_mean = master_df[(master_df['Function'] == func) & (master_df['Algorithm'] == PROPOSED_ALGO.upper())]['Mean'].values[0]
            
            for comp in COMPETITORS:
                if comp in raw_data and func in raw_data[comp].index:
                    comp_runs = raw_data[comp].loc[func].values[1:]
                    comp_mean = master_df[(master_df['Function'] == func) & (master_df['Algorithm'] == comp.upper())]['Mean'].values[0]
                    
                    stat, p_value = ranksums(prop_runs, comp_runs)
                    
                    if p_value < 0.05:
                        if prop_mean < comp_mean:
                            symbol = "+"
                            win_tie_loss[comp.upper()]['+'] += 1
                        else:
                            symbol = "-"
                            win_tie_loss[comp.upper()]['-'] += 1
                    else:
                        symbol = "="
                        win_tie_loss[comp.upper()]['='] += 1
                        
                    row[comp.upper()] = f"{p_value:.2e} ({symbol})"
                else:
                    row[comp.upper()] = "N/A"
        wilcoxon_results.append(row)
        
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    wilcoxon_df['F_num'] = wilcoxon_df['Function'].str.replace('F', '').astype(int)
    wilcoxon_df = wilcoxon_df.sort_values('F_num').drop('F_num', axis=1)
    
    # Append Win/Tie/Loss summary row
    wtl_row = {'Function': 'W / T / L'}
    for comp in COMPETITORS:
        if comp.upper() in wilcoxon_df.columns:
            w = win_tie_loss[comp.upper()]['+']
            t = win_tie_loss[comp.upper()]['=']
            l = win_tie_loss[comp.upper()]['-']
            wtl_row[comp.upper()] = f"{w} / {t} / {l}"
    
    wilcoxon_df = pd.concat([wilcoxon_df, pd.DataFrame([wtl_row])], ignore_index=True)
    
    # Save as Image with "wilcoxon" highlight logic
    wilcox_img_path = f"{PLOTS_DIR}/tables/{suite}/Wilcoxon_Test_Table.png"
    save_df_as_image(wilcoxon_df, wilcox_img_path, f"{suite.upper()}: Wilcoxon Rank-Sum vs {PROPOSED_ALGO.upper()}", table_type="wilcoxon")
    print(f"  -> Saved Table: {wilcox_img_path}")

    # --- CONVERGENCE CURVES ---
    for func in functions:
        plt.figure(figsize=(8, 6))
        plot_created = False
        
        for algo in ALL_ALGOS:
            if algo in conv_data and func in conv_data[algo].index:
                y_vals = conv_data[algo].loc[func].values[1:] 
                y_vals = np.maximum(y_vals, 1e-10) 
                
                if algo == PROPOSED_ALGO:
                    plt.plot(y_vals, label=algo.upper(), linewidth=3, color='black', linestyle='-')
                else:
                    plt.plot(y_vals, label=algo.upper(), linewidth=1.5, alpha=0.8)
                plot_created = True
                
        if plot_created:
            plt.yscale('log')
            plt.xlabel('Iterations', fontweight='bold')
            plt.ylabel('Best Fitness (Log Scale)', fontweight='bold')
            plt.title(f'Convergence Curve: {suite.upper()} - {func}', fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{PLOTS_DIR}/convergence/{suite}/{func}_convergence.png", dpi=300)
            plt.close()
            
    print(f"  -> Generated Convergence Curves in plots/convergence/{suite}/")

if __name__ == "__main__":
    for s in SUITES:
        process_suite(s)
    print("\n✅ All tables and plots successfully generated with highlights!")