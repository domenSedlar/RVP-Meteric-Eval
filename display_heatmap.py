import seaborn as sns
import matplotlib.pyplot as plt

def display_heatmap(jaccard_df, title="Heatmap"):
    """Plot heatmap of Jaccard similarity matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        jaccard_df,
        cmap='RdYlBu',        # Color map (red-yellow-blue)
        square=True,          # Make cells square
        cbar_kws={'label': title},
        linewidths=0.5,       # Add lines between cells
        linecolor='white'
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()