import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Configuration globale
# =========================
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.linewidth": 1.2,
})

PALETTE = {
    "Baseline": "#4c4c4c",
    "STLM-FFN": "#1f77b4",
    "STLM-Universal": "#d62728",
}

# =========================
# Chargement et nettoyage
# =========================
df = pd.read_csv("results.csv")

df["Experiment"] = df["Experiment"].str.replace(r"\(.*\)", "", regex=True).str.strip()
df["Model"] = df["Weight Sharing"].replace({
    "none": "Baseline",
    "ffn": "STLM-FFN",
    "full": "STLM-Universal"
})

df["Parameters (M)"] = df["Parameters"] / 1e6
df["VRAM (GB)"] = df["Peak Memory (MB)"] / 1024

# Filtrage raisonnable
df = df[df["Parameters (M)"] < 50]

# =========================
# FIGURE 1 — Efficacité paramétrique
# =========================
plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df,
    x="Parameters (M)",
    y="Val Loss",
    hue="Model",
    palette=PALETTE,
    marker="o"
)
plt.title("Parameter Efficiency\n(lower is better)")
plt.xlabel("Parameters (Millions)")
plt.ylabel("Validation Loss")
plt.legend(title="", frameon=False)
sns.despine()
plt.savefig("fig1_param_efficiency.png")
plt.close()

# =========================
# FIGURE 2 — Scaling en profondeur
# =========================
plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df,
    x="Depth",
    y="Val Loss",
    hue="Model",
    palette=PALETTE,
    marker="o"
)
plt.title("Scaling with Depth")
plt.xlabel("Model Depth")
plt.ylabel("Validation Loss")
plt.legend(title="", frameon=False)
sns.despine()
plt.savefig("fig2_depth_scaling.png")
plt.close()

# =========================
# FIGURE 3 — Trade-off mémoire / performance
# =========================
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df,
    x="VRAM (GB)",
    y="Val Loss",
    hue="Model",
    palette=PALETTE,
    s=80
)
plt.title("Performance vs Memory Cost")
plt.xlabel("Peak VRAM (GB)")
plt.ylabel("Validation Loss")
plt.legend(title="", frameon=False)
sns.despine()
plt.savefig("fig3_memory_tradeoff.png")
plt.close()

# =========================
# FIGURE 4 — Comparaison contrôlée (Depth = 8)
# =========================
df_d8 = df[df["Depth"] == 8]

plt.figure(figsize=(5, 4))
sns.barplot(
    data=df_d8,
    x="Model",
    y="Val Loss",
    palette=PALETTE
)
plt.title("Small-scale Comparison (Depth = 8)")
plt.ylabel("Validation Loss")
plt.xlabel("")

for i, row in df_d8.iterrows():
    plt.text(
        i,
        row["Val Loss"] + 0.015,
        f'{row["Parameters (M)"]:.1f}M',
        ha="center",
        fontsize=10
    )

sns.despine()
plt.savefig("fig4_small_scale.png")
plt.close()

# =========================
# FIGURE 5 — Impact de RoPE (séparé proprement)
# =========================
plt.figure(figsize=(6, 4))
sns.pointplot(
    data=df,
    x="Model",
    y="Val Loss",
    hue="Use RoPE",
    palette=["#999999", "#000000"],
    dodge=0.4
)
plt.title("Impact of RoPE")
plt.ylabel("Validation Loss")
plt.xlabel("")
plt.legend(title="RoPE", frameon=False)
sns.despine()
plt.savefig("fig5_rope_effect.png")
plt.close()

print("✔ Figures générées : fig1 à fig5 (paper-ready)")
