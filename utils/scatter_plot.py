import matplotlib.pyplot as plt
import seaborn as sns

def plot_parenting_attachment_scatter(data, parenting_columns, attachment_columns):
    """
    Plots scatter plots to visualize the relationship between different parenting styles and attachment styles.

    Parameters:
    - data: pandas DataFrame containing the dataset
    - parenting_columns: List of columns in the data representing different parenting styles
    - attachment_columns: List of columns in the data representing different attachment styles
    """
    # Ensure that the length of parenting_columns and attachment_columns match
    if len(parenting_columns) != len(attachment_columns):
        raise ValueError("Parenting columns and attachment columns lists must have the same length.")

    # Create a grid of scatter plots
    fig, axes = plt.subplots(len(parenting_columns), 1, figsize=(8, 5 * len(parenting_columns)))
    fig.tight_layout(pad=5.0)

    # Loop through the parenting and attachment columns
    for i, (parenting_col, attachment_col) in enumerate(zip(parenting_columns, attachment_columns)):
        ax = axes[i] if len(parenting_columns) > 1 else axes
        sns.scatterplot(x=data[parenting_col], y=data[attachment_col], ax=ax)
        ax.set_xlabel(parenting_col)
        ax.set_ylabel(attachment_col)
        ax.set_title(f"Relationship between {parenting_col} and {attachment_col}")
        
    plt.show()

# Example usage:
# Assuming 'df' is your DataFrame and you want to plot 'Parenting_Style_1', 'Parenting_Style_2' against 'Attachment_Style_1', 'Attachment_Style_2'
# plot_parenting_attachment_scatter(df, ['Parenting_Style_1', 'Parenting_Style_2'], ['Attachment_Style_1', 'Attachment_Style_2'])
