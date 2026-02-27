import pandas as pd
import os

TABLES_DIR = "tables"

def generate_latex(filename, caption, label):
    path = f"{TABLES_DIR}/{filename}"
    if not os.path.exists(path): 
        print(f"% ⚠️ WARNING: {filename} not found. Skipping.")
        return ""
    
    df = pd.read_csv(path, index_col=0)
    
    # Start Table Environment
    latex = "\\begin{table}[h!]\n\\centering\n"
    latex += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    latex += "\\resizebox{\\textwidth}{!}{\n"
    
    # Column Setup (Left align first col, Center align rest)
    cols = "l" + "c" * len(df.columns)
    latex += "\\begin{tabular}{" + cols + "}\n\\hline\n"
    
    # Header Row
    # Escape special characters like % if needed, usually bolding headers is enough
    header = " & ".join([f"\\textbf{{{c}}}" for c in df.columns])
    latex += f"\\textbf{{Metric/Model}} & {header} \\\\ \\hline\n"
    
    # Data Rows
    for index, row in df.iterrows():
        values = []
        for x in row:
            if isinstance(x, (int, float)):
                # If it's an integer (like Support), print as int, else float
                if float(x).is_integer():
                    values.append(f"{int(x)}")
                else:
                    values.append(f"{x:.4f}")
            else:
                # It's a string (like Method names in Table 6)
                # Escape common latex issues if necessary
                clean_str = str(x).replace("%", "\\%")
                values.append(clean_str)
        
        row_str = " & ".join(values)
        latex += f"\\textbf{{{index}}} & {row_str} \\\\ \n"
        
    latex += "\\hline\n\\end{tabular}\n}\n\\end{table}\n"
    return latex

if __name__ == "__main__":
    print("% ========================================================")
    print("% COPY ALL THE CODE BELOW INTO YOUR LATEX EDITOR")
    print("% ========================================================\n")
    
    # 1. Benchmark Table
    print(generate_latex(
        "Table_1_Benchmark_Comparison.csv", 
        "Quantitative Comparison of Classification Models (Sorted by Accuracy)", 
        "tab:benchmark"
    ))
    print("\n% --------------------------------------------------------\n")

    # 2. Ablation Study
    print(generate_latex(
        "Table_2_Ablation_Study.csv", 
        "Ablation Study: Impact of Modalities on Performance and Latency", 
        "tab:ablation"
    ))
    print("\n% --------------------------------------------------------\n")

    # 3. Class-Wise Performance (The New One)
    print(generate_latex(
        "Table_3_ClassWise.csv", 
        "Detailed Class-wise Detection Rate (Recall) and Support", 
        "tab:classwise"
    ))
    print("\n% --------------------------------------------------------\n")

    # 4. Computational Efficiency
    print(generate_latex(
        "Table_4_Efficiency.csv", 
        "Computational Efficiency and Suitability for Edge Deployment", 
        "tab:efficiency"
    ))
    print("\n% --------------------------------------------------------\n")

    # 5. Imbalance Handling
    print(generate_latex(
        "Table_5_Imbalance.csv", 
        "Impact of Imbalance Handling Strategies on False Positives", 
        "tab:imbalance"
    ))
    print("\n% --------------------------------------------------------\n")

    # 6. SOTA Comparison (The Literature Review One)
    print(generate_latex(
        "Table_6_SOTA_Comparison.csv", 
        "Benchmarking FabMind against Recent State-of-the-Art Studies (2020-2025)", 
        "tab:sota"
    ))
    
    print("\n% ========================================================")
    print("% END OF TABLES")
    print("% ========================================================")