import numpy as np
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import get_column_plot


def plot_distributions(data, synthetic_data, metadata, columns_to_plot):
    for col in columns_to_plot:
        fig = get_column_plot(
            real_data=data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            column_name=col,
            plot_type='bar'
        )
        fig.show()


def plot_comparison_subplots(drs1_dict, drs2_dict, dqs1_dict, dqs2_dict,
                             title1="Diagnostic Scores Comparison",
                             title2="Quality Scores Comparison",
                             dict1_name="Synthesizer 1",
                             dict2_name="Synthesizer 2"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Diagnostic Scores
    labels1 = list(drs1_dict.keys())
    drs1_values = list(drs1_dict.values())
    drs2_values = list(drs2_dict.values())
    x1 = np.arange(len(labels1))
    width = 0.35

    rects1a = axes[0].bar(x1 - width / 2, drs1_values, width, label=dict1_name)
    rects1b = axes[0].bar(x1 + width / 2, drs2_values, width, label=dict2_name)

    axes[0].set_ylabel('Scores')
    axes[0].set_title(title1)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(labels1, rotation=45, ha='right')
    axes[0].legend()

    # Plot Quality Scores
    labels2 = list(dqs1_dict.keys())
    dqs1_values = list(dqs1_dict.values())
    dqs2_values = list(dqs2_dict.values())
    x2 = np.arange(len(labels2))

    rects2a = axes[1].bar(x2 - width / 2, dqs1_values, width, label=dict1_name)
    rects2b = axes[1].bar(x2 + width / 2, dqs2_values, width, label=dict2_name)

    axes[1].set_ylabel('Scores')
    axes[1].set_title(title2)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2, rotation=45, ha='right')
    axes[1].legend()

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1a, axes[0])
    autolabel(rects1b, axes[0])
    autolabel(rects2a, axes[1])
    autolabel(rects2b, axes[1])

    plt.tight_layout()
    plt.show()



