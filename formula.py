import matplotlib.pyplot as plt

# Formula text
formula = r"$\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\varphi}$"

# Create figure
plt.figure(figsize=(6,2), facecolor="white")
plt.text(0.5, 0.5, formula, fontsize=28, ha="center", va="center")
plt.axis("off")
plt.tight_layout()

# Save image
path = "T2_formula.png"
plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

path
