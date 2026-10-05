import glob
import re
import matplotlib.pyplot as plt

cases = ["single_null", "double_null", "magnetic_island"]
basis_list = [11, 13, 15, 17, 19, 21, 23, 25]

# Initialize storage structure
data = {
    case: {
        b: {"speedup": None, "l2_error": None, "max_error": None}
        for b in basis_list
    }
    for case in cases
}

# Regex patterns to isolate numbers
regex_speedup = re.compile(r"Speed-up:\s*([\d\.]+)")
regex_l2 = re.compile(r"The relative L2 Error\s*is\s*([\d\.]+)")
regex_max = re.compile(r"The absolute Max Error\s*is\s*([\d\.]+)")

# Process all slurm*out files
for filepath in glob.glob("slurm*out"):
    with open(filepath, "r") as f:
        lines = f.readlines()
        if len(lines) < 4:
            continue
        
        last_4 = lines[-4:]
        path_line = last_4[3]
        
        found_case = None
        for c in cases:
            if c in path_line:
                found_case = c
                break
                
        basis_match = re.search(r"basis(\d+)", path_line)
        found_basis = int(basis_match.group(1)) if basis_match else None
        
        if found_case and found_basis in basis_list:
            speedup_match = regex_speedup.search(last_4[0])
            l2_match = regex_l2.search(last_4[1])
            max_match = regex_max.search(last_4[2])
            
            if speedup_match and l2_match and max_match:
                data[found_case][found_basis]["speedup"] = float(speedup_match.group(1).rstrip('.'))
                data[found_case][found_basis]["l2_error"] = float(l2_match.group(1).rstrip('.'))
                data[found_case][found_basis]["max_error"] = float(max_match.group(1).rstrip('.'))

# Generate individual semi-log plots
metrics = [
    ("speedup", "Speed-up", "Speed-up Factor (log scale)"),
    ("l2_error", "Relative L2 Error", "L2 Error % (log scale)"),
    ("max_error", "Absolute Max Error", "Max Error (log scale)")
]

for case in cases:
    for metric_key, metric_title, y_label in metrics:
        x_vals = basis_list
        y_vals = [data[case][b][metric_key] for b in x_vals]
        
        plt.figure(figsize=(7, 5))
        plt.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6)
        plt.yscale('log')
        plt.title(f"{case.replace('_', ' ').title()}: {metric_title} vs. Basis Count")
        plt.xlabel("Number of Basis Functions")
        plt.ylabel(y_label)
        plt.xticks(basis_list)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{case}_{metric_key}.png", dpi=300)
        plt.close()

# Generate scatter plots: Relative Wall Clock Time (x) vs Relative L2 Error (y)
for case in cases:
    x_vals = []
    y_vals = []
    basis_labels = []
    
    for b in basis_list:
        speedup = data[case][b]["speedup"]
        l2_err = data[case][b]["l2_error"]
        if speedup is not None and l2_err is not None and speedup != 0:
            x_vals.append(1.0 / speedup)
            y_vals.append(l2_err)
            basis_labels.append(b)
            
    plt.figure(figsize=(7, 5))
    plt.scatter(x_vals, y_vals, color='crimson', s=60, zorder=3)
    
    for x, y, r in zip(x_vals, y_vals, basis_labels):
        plt.annotate(
            f"r={r}", 
            (x, y), 
            textcoords="offset points", 
            xytext=(5, 5), 
            ha='left',
            fontsize=9
        )
        
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f"{case.replace('_', ' ').title()}: L2 Error vs. Relative Wall Clock Time")
    plt.xlabel("Relative Wall Clock Time (log scale)")
    plt.ylabel("Relative L2 Error % (log scale)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{case}_l2_vs_wc_scatter.png", dpi=300)
    plt.close()

# Write LaTeX table files for each case
for case in cases:
    # Prepare row data strings
    basis_row = " & ".join([str(b) for b in basis_list])
    
    l2_vals = []
    max_vals = []
    speedup_vals = []
    
    for b in basis_list:
        l2 = data[case][b]["l2_error"]
        m_err = data[case][b]["max_error"]
        sp = data[case][b]["speedup"]
        
        l2_vals.append(f"{l2:.2f}" if l2 is not None else "-")
        max_vals.append(f"{m_err:.4f}" if m_err is not None else "-")
        speedup_vals.append(f"{sp:.2f}" if sp is not None else "-")
        
    l2_row = " & ".join(l2_vals)
    max_row = " & ".join(max_vals)
    speedup_row = " & ".join(speedup_vals)
    
    col_align = "c" * (len(basis_list) + 1)
    
    tex_content = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}

\\begin{{document}}

\\begin{{table}}[htbp]
\\centering
\\begin{{tabular}}{{{col_align}}}
\\toprule
Basis Number & {basis_row} \\\\
\\midrule
$L_2$ Error (%) & {l2_row} \\\\
%Max Error & {max_row} \\\\
Speed-up & {speedup_row} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Performance metrics for {case.replace('_', ' ').title()}}}
\\end{{table}}

\\end{{document}}
"""
    
    tex_filename = f"{case}_results.tex"
    with open(tex_filename, "w") as tex_file:
        tex_file.write(tex_content)

print("Analysis complete. All figures and LaTeX files were saved successfully.")

