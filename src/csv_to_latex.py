import pandas as pd
import os

TABLES_DIR = "tables"

def print_latex_table(filename, caption, label):
    path = f"{TABLES_DIR}/{filename}"
    if not os.path.exists(path):
        print(f"⚠️ {filename} not found.")
        return

    df = pd.read_csv(path, index_col=0)
    
    # Start LaTeX String
    latex = "\\begin{table}[h!]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
    latex += "\\resizebox{\\textwidth}{!}{\n" # Make it fit page width
    
    # Column Setup
    cols = "l" + "c" * len(df.columns) # Left align first col, center others
    latex += "\\begin{tabular}{" + cols + "}\n\\hline\n"
    
    # Header
    header = " & ".join(["\\textbf{" + c + "}" for c in df.columns])
    latex += "\\textbf{Model} & " + header + " \\\\ \\hline\n"
    
    # Rows
    for index, row in df.iterrows():
        # Format numbers to 4 decimal places
        values = []
        for x in row:
            if isinstance(x, (int, float)):
                values.append(f"{x:.4f}")
            else:
                values.append(str(x))
        
        row_str = " & ".join(values)
        latex += f"\\textbf{{{index}}} & {row_str} \\\\ \n"
        
    latex += "\\hline\n\\end{tabular}\n}\n\\end{table}\n"
    
    print("-" * 50)
    print(f"COPY THIS FOR: {filename}")
    print("-" * 50)
    print(latex)
    print("\n\n")

if __name__ == "__main__":
    print("--- LATEX TABLE GENERATOR ---\n")
    
    print_latex_table("T1_Benchmark_Metrics.csv", "Comparison of Classification Models on Semiconductor Dataset", "tab:benchmark")
    print_latex_table("T2_Ablation_Study.csv", "Ablation Study: Impact of Data Modalities on Performance", "tab:ablation")
    print_latex_table("T3_Defect_Performance.csv", "Class-wise Performance Metrics for Specific Defect Types", "tab:classwise")
    print_latex_table("T4_Computational_Cost.csv", "Computational Efficiency and Edge Deployment Suitability", "tab:efficiency")
    print_latex_table("T5_Imbalance_Strategy.csv", "Impact of Imbalance Handling Strategies on Recall", "tab:imbalance")