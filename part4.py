import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os



# 1. Load all metadata files

all_data = []

# Load relax metadata from Part 2
relax_file = "relax_metadata.tsv"
if os.path.exists(relax_file):
    df = pd.read_csv(relax_file, sep="\t")
    df["stage"] = "FastRelax"
    df["label"] = "FastRelax"
    # Rename columns to be consistent
    if "relaxed_score" in df.columns:
        df = df.rename(columns={"relaxed_score": "score"})
    if "rmsd" in df.columns:
        df = df.rename(columns={"rmsd": "rmsd_to_native"})
    all_data.append(df)
    print(f"Loaded {len(df)} relax samples from {relax_file}")
else:
    print(f"WARNING: {relax_file} not found — skipping relax data")

# Load all backrub metadata from Part 3
backrub_files = sorted(glob.glob("backrub_metadata_*.tsv"))
for bf in backrub_files:
    df = pd.read_csv(bf, sep="\t")
    df["stage"] = "BackRub"
    # Create a label from the mc_kt and ntrials values
    if "mc_kt" in df.columns and "ntrials" in df.columns:
        df["label"] = df.apply(lambda r: f"BackRub kt={r['mc_kt']} nt={int(r['ntrials'])}", axis=1)
    else:
        df["label"] = bf
    # Rename score column if needed
    if "score" not in df.columns and "score_diff" in df.columns:
        df["score"] = df["score_diff"]
    all_data.append(df)
    print(f"Loaded {len(df)} backrub samples from {bf}")

if not all_data:
    print("ERROR: No metadata files found! Run Parts 2 and 3 first.")
    exit(1)

# Combine everything
data = pd.concat(all_data, ignore_index=True)
print(f"\nTotal samples: {len(data)}")



# 2. Generate Score-vs-RMSD scatter plot

print("\nGenerating Score-vs-RMSD scatter plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Plot each group with a different color
labels = data["label"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

for label, color in zip(labels, colors):
    subset = data[data["label"] == label]
    ax.scatter(
        subset["rmsd_to_native"],
        subset["score"],
        label=label,
        color=color,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        s=50
    )

ax.set_xlabel("RMSD to Native Structure (Å)", fontsize=12)
ax.set_ylabel("Rosetta Score (REU — Rosetta Energy Units)", fontsize=12)
ax.set_title("Score vs. RMSD: FastRelax and BackRub Ensembles", fontsize=14)
ax.legend(fontsize=9, loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("score_vs_rmsd.png", dpi=150)
print("Saved: score_vs_rmsd.png")



# 3. Compute Pnear metric
print("\nComputing Pnear for each sampling strategy...")
print("=" * 60)

def compute_pnear(scores, rmsds, lambda_val=1.5, kbt=1.0):
    """
    Pnear measures the quality of the folding funnel.

    Pnear = sum(exp(-rmsd^2 / lambda^2) * exp(-score/kbt)) / sum(exp(-score/kbt))

    Values close to 1.0 = strong folding funnel (low score near low RMSD)
    Values close to 0.0 = no folding funnel
    """
    scores = np.array(scores)
    rmsds = np.array(rmsds)

    # Shift scores so the minimum is 0 (avoids numerical overflow)
    scores_shifted = scores - scores.min()

    boltzmann = np.exp(-scores_shifted / kbt)
    gaussian = np.exp(-(rmsds ** 2) / (lambda_val ** 2))

    pnear = np.sum(gaussian * boltzmann) / np.sum(boltzmann)
    return pnear


results = []
for label in labels:
    subset = data[data["label"] == label]
    if "rmsd_to_native" in subset.columns and "score" in subset.columns:
        pnear = compute_pnear(subset["score"].values, subset["rmsd_to_native"].values)
        results.append({"strategy": label, "pnear": round(pnear, 4), "n_samples": len(subset)})
        print(f"  {label:40s}  Pnear = {pnear:.4f}  (n={len(subset)})")

# Save Pnear results
pnear_df = pd.DataFrame(results)
pnear_df.to_csv("pnear_results.tsv", sep="\t", index=False)
print(f"\nSaved: pnear_results.tsv")
