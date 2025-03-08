import numpy as np
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import get_column_plot


def plot_distributions(data, synthetic_data, metadata, columns_to_plot):
    """
    Plots the distribution of specific columns in both the real and synthetic data using bar plots.
    """
    for col in columns_to_plot:
        # Generate the plot for the current column using 'get_column_plot' from SDV
        fig = get_column_plot(
            real_data=data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            column_name=col,
            plot_type='bar'  # Specifies that the plot type is a bar plot
        )
        # Display the plot
        fig.show()


def plot_comparison_subplots(drs1_dict, drs2_dict, dqs1_dict, dqs2_dict,
                             title1="Diagnostic Scores Comparison",
                             title2="Quality Scores Comparison",
                             dict1_name="Synthesizer 1",
                             dict2_name="Synthesizer 2"):
    """
    Plots side-by-side bar charts comparing diagnostic scores and quality scores
    for two synthesizers.
    """
    # Create a subplot with 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Diagnostic Scores comparison
    labels1 = list(drs1_dict.keys())  # Labels for the first synthesizer
    drs1_values = list(drs1_dict.values())  # Values for the first synthesizer
    drs2_values = list(drs2_dict.values())  # Values for the second synthesizer
    x1 = np.arange(len(labels1))  # X-axis positions for bars
    width = 0.35  # Width of the bars

    # Plot bars for the first and second synthesizers
    rects1a = axes[0].bar(x1 - width / 2, drs1_values, width, label=dict1_name)
    rects1b = axes[0].bar(x1 + width / 2, drs2_values, width, label=dict2_name)

    # Customize the first subplot (Diagnostic Scores)
    axes[0].set_ylabel('Scores')
    axes[0].set_title(title1)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(labels1, rotation=45, ha='right')
    axes[0].legend()

    # Plot Quality Scores comparison
    labels2 = list(dqs1_dict.keys())  # Labels for the second set of scores (quality)
    dqs1_values = list(dqs1_dict.values())  # Values for the first synthesizer (quality)
    dqs2_values = list(dqs2_dict.values())  # Values for the second synthesizer (quality)
    x2 = np.arange(len(labels2))  # X-axis positions for bars

    # Plot bars for the first and second synthesizers
    rects2a = axes[1].bar(x2 - width / 2, dqs1_values, width, label=dict1_name)
    rects2b = axes[1].bar(x2 + width / 2, dqs2_values, width, label=dict2_name)

    # Customize the second subplot (Quality Scores)
    axes[1].set_ylabel('Scores')
    axes[1].set_title(title2)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2, rotation=45, ha='right')
    axes[1].legend()

    # Function to add labels on top of the bars
    def autolabel(rects, ax):
        """
        Annotates the bars with their height values for better readability.
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Apply the autolabel function to both subplots
    autolabel(rects1a, axes[0])
    autolabel(rects1b, axes[0])
    autolabel(rects2a, axes[1])
    autolabel(rects2b, axes[1])

    # Adjust layout for better spacing and display the plot
    plt.tight_layout()
    plt.show()
