import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Academic Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})
sns.set_style("whitegrid", {'grid.linestyle': '--'})

# Integrator Method Styling
METHOD_STYLING = {
    'path_trace': {
        'label': 'Path Tracing (Baseline)',
        'color': '#7F8C8D',
        'marker': 'o'
    },
    'path_guide_photon': {
            'label': 'Path Guiding Photons',
            'color': '#2980B9',
            'marker': 's'
    },
    'path_guide_pss': {
        'label': 'Path Guiding PSS (Ours)',
        'color': '#C0392B',
        'marker': 'D'
    }
}

def plot_scene_results(csv_filepath, scene_name, output_dir="Plots"):
    # Create Plots Directory If It Does Not Exist Yet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create the Data Frame by Reading CSV
    df = pd.read_csv(csv_filepath)

    # 1x3 Plot
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    fig.suptitle(f'Evaluation Results: {scene_name}', fontsize=16, fontweight='bold', y=1.05)

    # Define Metrics (metric - label name - is higher better?)
    metrics = [
        ('FLIP', 'FLIP Error (Lower is Better)', False),
        ('SSIM', 'SSIM (Higher is Better)', True),
        ('RRMSE', 'RRMSE (Lower is Better)', False)
    ]

    # Loop Through Metrics
    for ax, (metric, ylabel, higher_is_better) in zip(axes, metrics):
        for method in df['Method'].unique():
            subset = df[df['Method'] == method].sort_values(by='SPP')
            if method not in METHOD_STYLING:
                continue

            config = METHOD_STYLING[method]
            ax.plot(subset['SPP'], subset[metric], label=config['label'], color=config['color'], marker=config['marker'], linewidth=2, markersize=8)
            ax.set_title(metric, fontweight='bold')
            ax.set_xlabel('Samples Per Pixel (SPP)')
            ax.set_ylabel(ylabel)

            # log2 scale for rendering samples
            ax.set_xscale('log', base=2)
            ax.set_xticks([32, 128, 512])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].legend(title='Integrator')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{scene_name}_Metrics.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved plot for {scene_name} at {save_path}')

def generate_latex_table(csv_filepaths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Combine all CSVs Into One Data Frame
    df_list = [pd.read_csv(filepath) for filepath in csv_filepaths]
    combined_df = pd.concat(df_list)

    # Map method names to pretty names
    # (Consider shortening these slightly if the text gets too small when resized)
    pretty_names = {k: v['label'] for k, v in METHOD_STYLING.items()}
    combined_df['Method'] = combined_df['Method'].map(pretty_names)

    # Format numbers to 4 decimals
    columns_to_format = ['MSE', 'RMSE', 'MRSE', 'RRMSE', 'SSIM', 'FLIP']
    for column in columns_to_format:
        combined_df[column] = combined_df[column].apply(lambda x: f"{float(x):.4f}")
    
    # Helper function to generate full LaTeX table wrapper
    def save_formatted_latex(pivot_df, filename, caption, label):
        num_data_cols = len(pivot_df.columns)
        # 'l' for Scene, 'l' for Method, 'r' for all numeric data columns
        col_alignments = 'll' + ('r' * num_data_cols)
        
        # Generate the raw table contents
        raw_latex = pivot_df.to_latex(
            multirow=True, 
            multicolumn=True,
            multicolumn_format='c', # Centers the multi-column headers (FLIP, SSIM, etc.)
            column_format=col_alignments # Left aligns text, right aligns numbers
        )
        
        # Wrap it in a table environment and resize it to fit the text width
        full_latex = f"""\\begin{{table}}[htbp]
                         \\centering
                         \\resizebox{{\\textwidth}}{{!}}{{{raw_latex}}}
                         \\caption{{{caption}}}
                         \\label{{{label}}}
                         \\end{{table}}"""
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(full_latex)

    # --- TABLE 1: MAIN METRICS ---
    pivot_main = combined_df.pivot(index=['Scene', 'Method'], columns='SPP', values=['FLIP', 'SSIM', 'RRMSE'])
    save_formatted_latex(
        pivot_main, 
        "Table_1_Main_Metrics.tex", 
        "Quantitative comparison of equal-sample rendering quality. FLIP and RRMSE are lower-is-better, SSIM is higher-is-better.", 
        "tab:main_metrics"
    )
        
    # --- TABLE 2: ABSOLUTE ERROR ---
    pivot_abs = combined_df.pivot(index=['Scene', 'Method'], columns='SPP', values=['MSE', 'RMSE'])
    save_formatted_latex(
        pivot_abs, 
        "Table_2_Absolute_Error.tex", 
        "Absolute error metrics (MSE and RMSE) across evaluated scenes.", 
        "tab:abs_error"
    )
        
    # --- TABLE 3: RELATIVE ERROR ---
    pivot_rel = combined_df.pivot(index=['Scene', 'Method'], columns='SPP', values=['MRSE', 'RRMSE'])
    save_formatted_latex(
        pivot_rel, 
        "Table_3_Relative_Error.tex", 
        "Relative error metrics (MRSE and RRMSE) across evaluated scenes.", 
        "tab:rel_error"
    )

    print(f"Saved 3 formatted LaTeX tables to {output_dir}")
    
# Main Function
if __name__ == "__main__":
    # Base Directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "EvaluationResults")
    # Save final PDF
    plots_dir = os.path.join(base_dir, "Final_Plots")
    os.makedirs(plots_dir, exist_ok=True)
    print("--- Starting Plot, and Table Generation... ---")
    # Find all CSVs inside EvaluationResults
    search_pattern = os.path.join(base_dir, "**", "*_results.csv")
    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        print(f"No CSV files found in {base_dir}")
    else:
        for csv_path in csv_files:
            parent_dir = os.path.basename(csv_path)
            scene_name = parent_dir.replace("_results.csv", "")

            print(f"Processing Scene: {scene_name}...")
            plot_scene_results(csv_path, scene_name, plots_dir)
        generate_latex_table(csv_files, plots_dir)
        print("\n--- Plot, and Table Generation Finished... ---")
        print(f"All files saved to: {plots_dir}")