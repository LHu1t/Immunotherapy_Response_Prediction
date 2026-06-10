import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data
data = {
    "Subtype": ["LUAD", "LUAD", "LUSC", "LUSC"],
    "Panel": ["APM", "TIS", "APM", "TIS"],
    "Correlation": [0.303, 0.314, 0.303, 0.341],
    "SD": [0.028, 0.031, 0.047, 0.046]
}

df = pd.DataFrame(data)

# Red colour palette
palette = {
    "APM": "#8B0000",   # dark red
    "TIS": "#FF6B6B"    # soft coral red
}

sns.set_theme(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(7,5))

# Barplot
ax = sns.barplot(
    data=df,
    x="Subtype",
    y="Correlation",
    hue="Panel",
    palette=palette,
    edgecolor="black"
)

# Add error bars
for i, row in df.iterrows():
    
    x_position = i // 2 + (-0.2 if row["Panel"] == "APM" else 0.2)
    
    plt.errorbar(
        x=x_position,
        y=row["Correlation"],
        yerr=row["SD"],
        fmt='none',
        ecolor="#4A0000",   # darker red error bars
        capsize=6,
        linewidth=1.5
    )

# Labels
plt.ylabel("Pearson Correlation (R)")
plt.xlabel("Cancer Subtype")
plt.title("Panel-Level Gene Expression Prediction Performance")

plt.ylim(0,0.4)

plt.legend(title="Immune Panel", loc="lower right")

plt.tight_layout()

plt.savefig("panel_correlations.pdf", bbox_inches="tight")

plt.show()