import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase

class TextHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                      width, height, fontsize, trans):
        txt = Text(x=width/2, y=height/2, text=orig_handle.get_text(),
                  ha='center', va='center', fontsize=fontsize, 
                  fontweight='bold', transform=trans)
        return [txt]

PROJECT_ROOT = Path(__file__).parent

df = pd.read_csv(PROJECT_ROOT / "data" / "results" / "model_response_summary.csv", header=None, names=['model', 'condition', 'classification', 'value'])

data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for _, row in df.iterrows():
    data[row['model']][row['condition']][row['classification']] = row['value']

models = ['llama-3.1-8b', 'mistral-7b-v03', 'gemma-2-9b']
conditions = ['deceive_without_lying', 'deceive_without_lying_two_shots', 'lie', 'lie_two_shots']
classifications = ['deception_without_lie', 'lie', 'honest', 'invalid']

fig, axes = plt.subplots(2, 2, figsize=(5, 4.5))
colors = ["#ffe600", "#d14444", "#6AA06D", "#525252"]

axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

for idx, model in enumerate(models):
    ax = axes_flat[idx]
    
    bar_data = np.zeros((len(classifications), len(conditions)))
    for j, condition in enumerate(conditions):
        for i, classification in enumerate(classifications):
            bar_data[i, j] = data[model][condition][classification]
    
    x = np.arange(len(conditions))
    width = 0.6
    
    bottom = np.zeros(len(conditions))
    for i, classification in enumerate(classifications):
        ax.bar(x, bar_data[i], width, label=classification, 
               bottom=bottom, color=colors[i])
        bottom += bar_data[i]
    
    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['', '', '', ''])
    
    ax.text(0.5, -0.18, 'DWL', ha='center', va='top', 
            transform=ax.get_xaxis_transform(), fontsize=10, fontweight='bold')
    ax.text(2.5, -0.18, 'LIE', ha='center', va='top', 
            transform=ax.get_xaxis_transform(), fontsize=10, fontweight='bold')
    
    ax.text(0, -0.08, '0S', ha='center', va='top', 
            transform=ax.get_xaxis_transform(), fontsize=9, fontweight='bold')
    ax.text(1, -0.08, '2S', ha='center', va='top', 
            transform=ax.get_xaxis_transform(), fontsize=9, fontweight='bold')
    ax.text(2, -0.08, '0S', ha='center', va='top', 
            transform=ax.get_xaxis_transform(), fontsize=9, fontweight='bold')
    ax.text(3, -0.08, '2S', ha='center', va='top', 
            transform=ax.get_xaxis_transform(), fontsize=9, fontweight='bold')

    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

legend_ax = axes_flat[3]
legend_ax.axis('off')

legend_labels = ["deception without lie", "lie", "honest", "invalid"]
handles = []
labels = []

for i, label in enumerate(legend_labels):
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i]))
    labels.append(label)

abbreviation_data = [
    ("DWL", "Deceive-without-lying condition"),
    ("LIE", "Lie condition"),
    ("0S", "Zero shot condition"),
    ("2S", "Two shot condition")
]

for abbr, explanation in abbreviation_data:
    text_handle = Text(0, 0, abbr)
    handles.append(text_handle)
    labels.append(explanation)

legend_ax.legend(handles=handles, labels=labels, loc='center', frameon=False, 
                fontsize=9, handler_map={Text: TextHandler()})

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "figures" / "model_responses_stacked_bar_chart.png", dpi=300)