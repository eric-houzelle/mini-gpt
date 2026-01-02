import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.2,
    "font.size": 11,
})

PALETTE = {
    "Baseline": "#4c4c4c",
    "STLM-FFN": "#1f77b4",
    "STLM-Universal": "#d62728",
}

CSV_PATH = "results_v2.csv"

# =========================================================
# LOAD & CLEAN
# =========================================================
df = pd.read_csv(CSV_PATH)

df["Model"] = df["Weight Sharing"].replace({
    "none": "Baseline",
    "ffn": "STLM-FFN",
    "full": "STLM-Universal"
})

df["Parameters (M)"] = df["Parameters"] / 1e6
df["Time to Best (min)"] = df["Best Time (sec)"] / 60
df["VRAM (GB)"] = df["Peak Memory (MB)"] / 1024

# =========================================================
# FIGURE 1 — STEPS TO QUALITY (CLÉ)
# =========================================================
plt.figure(figsize=(5, 4))
sns.barplot(
    data=df,
    x="Model",
    y="Best Step",
    palette=PALETTE
)
plt.title("Steps to Reach Best Validation Loss")
plt.ylabel("Training Steps")
plt.xlabel("")
sns.despine()
plt.savefig("fig1_steps_to_quality.png")
plt.close()

# =========================================================
# FIGURE 2 — TIME TO QUALITY
# =========================================================
plt.figure(figsize=(5, 4))
sns.barplot(
    data=df,
    x="Model",
    y="Time to Best (min)",
    palette=PALETTE
)
plt.title("Time to Reach Best Validation Loss")
plt.ylabel("Time (minutes)")
plt.xlabel("")
sns.despine()
plt.savefig("fig2_time_to_quality.png")
plt.close()

# =========================================================
# FIGURE 3 — VRAM vs DEPTH (POINT CLÉ CONCEPTUEL)
# =========================================================
plt.figure(figsize=(6, 4))
sns.lineplot(
    data=df,
    x="Depth",
    y="VRAM (GB)",
    hue="Model",
    palette=PALETTE,
    marker="o"
)
plt.title("Memory Scaling with Depth\n(activations dominate)")
plt.xlabel("Model Depth")
plt.ylabel("Peak VRAM (GB)")
plt.legend(title="", frameon=False)
sns.despine()
plt.savefig("fig3_vram_vs_depth.png")
plt.close()

# =========================================================
# FIGURE 4 — PARETO (SYNTHÈSE)
# =========================================================
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df,
    x="Best Step",
    y="VRAM (GB)",
    hue="Model",
    palette=PALETTE,
    s=100
)
plt.title("Speed vs Memory Trade-off (Pareto View)")
plt.xlabel("Steps to Best Validation Loss")
plt.ylabel("Peak VRAM (GB)")
plt.legend(title="", frameon=False)
sns.despine()
plt.savefig("fig4_pareto_speed_memory.png")
plt.close()

print("✅ Figures générées : fig1 à fig4")



# =========================================================
# FIGURE — BEST VALIDATION LOSS ACHIEVED
# =========================================================
# =========================================================
# FIGURE — VAL LOSS PER CONFIGURATION
# (1 barre = 1 run, PAS d’agrégation)
# =========================================================

import numpy as np

# Préparer les labels
df_plot = df.copy()

df_plot["Params (M)"] = df_plot["Parameters"] / 1e6
df_plot["RoPE Label"] = df_plot["Use RoPE"].apply(
    lambda x: "RoPE" if bool(x) else "no-RoPE"
)

df_plot["Config Label"] = (
    df_plot["Model"]
    + " | "
    + df_plot["Params (M)"].map(lambda x: f"{x:.1f}M")
    + " | "
    + df_plot["RoPE Label"]
)

# Ordre explicite (CRUCIAL)
df_plot = df_plot.sort_values(
    by=["Model", "Params (M)", "Use RoPE"],
    ascending=[True, True, False]
).reset_index(drop=True)

x = np.arange(len(df_plot))
y = df_plot["Best Val Loss"].values
colors = df_plot["Model"].map(PALETTE).values

# Taille adaptative
plt.figure(figsize=(max(8, 0.6 * len(df_plot)), 4))

bars = plt.bar(x, y, color=colors)

plt.title("Best Validation Loss per Configuration")
plt.ylabel("Validation Loss (lower is better)")
plt.xlabel("")

plt.xticks(x, df_plot["Config Label"], rotation=45, ha="right")

# Annotations CORRECTES
for bar, val in zip(bars, y):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.004,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

sns.despine()
plt.tight_layout()
plt.savefig("fig_val_loss_per_configuration.png")
plt.close()

