import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_jaccard_heatmap(jaccard_df, title="Jaccard Similarity Between Metrics"):
    """Plot heatmap of Jaccard similarity matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        jaccard_df,
        annot=True,           # Show values in cells
        fmt='.3f',            # Format to 3 decimal places
        cmap='RdYlBu',        # Color map (red-yellow-blue)
        vmin=0, vmax=1,       # Jaccard ranges from 0 to 1
        square=True,          # Make cells square
        cbar_kws={'label': 'Jaccard Similarity'},
        linewidths=0.5,       # Add lines between cells
        linecolor='white'
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def join(pths):
    df_dict = {}
    for pth in pths:
        df_dict[pth.split('\\')[-1].split("/")[-1].split('.')[0]] = pd.read_csv(pth)

    final_df = pd.concat(df_dict).reset_index(level=0).rename(columns={'level_0': 'Source_DF'}).reset_index(drop=True)

def jaccard_index(set1, set2):
    """Calculate Jaccard index between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def analyze(df, name):
    metric_cols = ["ME8", "ME4", "MI4", "MS4", "MS8", "MI8"]
    df['MS4'] = -df['MS4']
    df['MS8'] = -df['MS8']
    topk = {}
    ks = [10, 25, 50, 100, 200]

    for k in ks:
        tops = [set(df.nlargest(k, m)[['p1', 'p2']].apply(tuple, axis=1)) for m in metric_cols]

        n_metrics = len(metric_cols)
        jaccard_matrix = pd.DataFrame(index=metric_cols, columns=metric_cols, dtype=float)

        for i in range(n_metrics):
            for j in range(n_metrics):
                # Combine all sets for this metric into one set of pairs
                sets_i = tops[i]
                sets_j = tops[j]
                jaccard_matrix.iloc[i, j] = jaccard_index(sets_i, sets_j)

        print(jaccard_matrix)
        plot_jaccard_heatmap(jaccard_matrix, name + f"k{k}")

if __name__ == "__main__":
    pths = [".\Matrices\small_random.tsv", ".\Matrices\large_random.tsv", ".\Matrices\GDS5600\\100.tsv"]
    res_files = [os.path.join("./Results/", pth.split('\\')[-1].split("/")[-1].split('.')[0] + '.csv') for pth in pths]
    for pth in res_files:
        df = pd.read_csv(pth)
        print(df)
        analyze(df, "corr_" + pth.split('\\')[-1].split("/")[-1].split('.')[0])