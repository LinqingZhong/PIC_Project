import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from matplotlib import rcParams

# plt.rcParams["font.family"] = "Times New Roman"

with open("/mnt/data4/zlq/PIC_Project/tools/SFT_Dataset_val_results.json", "r") as f:
    data = json.load(f)


records = []
for entry in data:
    for key, score_dict in entry.items():
        if key.endswith('_nlp'):
            model_type = 'finetuned' if key.startswith('sft') else 'vanilla'
            model_name = key.replace('vanilla_', '').replace('sft_', '').replace('_nlp', '')
            for metric, score in score_dict.items():
                records.append({
                    'model': model_name,
                    'version': model_type,
                    'metric': metric,
                    'score': score
                })

df = pd.DataFrame(records)


models = df['model'].unique()


base_colors = sns.color_palette("Set2", n_colors=len(models))
palette = {}
for i, model in enumerate(models):
    light = base_colors[i]
    dark = tuple([max(0, c - 0.3) for c in light])
    palette[(model, 'vanilla')] = light
    palette[(model, 'finetuned')] = dark


n_metrics = len(df['metric'].unique())
fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
axs = axs.flatten()

metrics = df['metric'].unique()


for i, metric in enumerate(metrics):
    ax = axs[i]
    data_metric = df[df['metric'] == metric]
    sns.barplot(
        data=data_metric,
        x="model",
        y="score",
        hue="version",
        palette=[palette[(row['model'], row['version'])] for _, row in data_metric.iterrows()],
        dodge=True,
        ax=ax,
        # ci=None
        # edgecolor='none'
    )
    ax.set_title(f"{metric.upper()} Score")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, axis='y')
    if i != 0:
        ax.get_legend().remove()
    else:
        # ax.legend(title="Version")
        ax.legend(loc='upper right')

    # for bar in ax.patches:
    #     height = bar.get_height()
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,  # 柱子中间x位置
    #         height,                            # 柱子高度y位置
    #         f'{height:.2f}',                   # 显示数字，保留2位小数
    #         ha='center',                      # 水平居中
    #         va='bottom',                      # 底部对齐
    #         fontsize=9,
    #         # fontname='Times New Roman'
    #     )

for j in range(len(metrics), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig("/mnt/data4/zlq/PIC_Project/logs/all_metrics_barplot.png", dpi=300)
plt.close()




# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x="metric", y="score", hue="version", palette=palette)
# plt.title("Boxplot of Scores by Metric (Vanilla vs Finetuned)")
# plt.ylabel("Score")
# plt.xlabel("Metric")
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.legend(title="Version")
# plt.show()
