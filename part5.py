import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

RESIDUE_2x46 = 86    # Generic position 2x46 -> MC4R S86
RESIDUE_6x37 = 247   # Generic position 6x37 -> MC4R L247

def load_structure(filepath):
    """Load a structure from .pdb or .cif file and return biotite AtomArray."""
    if filepath.endswith(".cif"):
        file = pdbx.CIFFile.read(filepath)
        atoms = pdbx.get_structure(file, model=1)
    else:
        file = pdb.PDBFile.read(filepath)
        atoms = pdb.get_structure(file, model=1)
    return atoms


def get_ca_distance(atoms, res1, res2):
    """
    Measure the distance between C-alpha atoms of two residues.
    This is the TM6 tilt / open-distance geometry metric from GPCRdb.
    """
    # Select C-alpha of residue 1
    ca1_mask = (atoms.res_id == res1) & (atoms.atom_name == "CA")
    ca1 = atoms[ca1_mask]

    # Select C-alpha of residue 2
    ca2_mask = (atoms.res_id == res2) & (atoms.atom_name == "CA")
    ca2 = atoms[ca2_mask]

    if len(ca1) == 0:
        print(f"  WARNING: Could not find CA atom for residue {res1}")
        return None
    if len(ca2) == 0:
        print(f"  WARNING: Could not find CA atom for residue {res2}")
        return None

    # Take the first match (in case of multiple chains)
    coord1 = ca1.coord[0]
    coord2 = ca2.coord[0]

    distance = np.sqrt(np.sum((coord1 - coord2) ** 2))
    return distance


# 1. Find all structure files
print("Scanning for structure files...")

native_file = "mc4r_receptor.cif"
pdb_files = sorted(glob.glob("relaxed_*.pdb") + glob.glob("backrub_*.pdb"))

if not os.path.exists(native_file):
    print(f"ERROR: {native_file} not found!")
    exit(1)

print(f"  Native structure: {native_file}")
print(f"  Sampled structures: {len(pdb_files)} files found")

if len(pdb_files) == 0:
    print("WARNING: No relaxed_*.pdb or backrub_*.pdb files found.")
    exit(1)



# 2. Measure C-alpha distance for native structure

print(f"\nMeasuring Cα distance between residues {RESIDUE_2x46} (2x46) "
      f"and {RESIDUE_6x37} (6x37)...")
print("=" * 60)

native_atoms = load_structure(native_file)
native_distance = get_ca_distance(native_atoms, RESIDUE_2x46, RESIDUE_6x37)

if native_distance is not None:
    print(f"  Native structure distance: {native_distance:.2f} Å")
else:
    print("  ERROR: Could not measure native distance.")
    print("  Check that residue numbers match your structure.")
    print("  Open your structure in PyMOL and verify residue numbering.")
    exit(1)



# 3. Measure distance for all sampled conformations

results = []

for pdb_file in pdb_files:
    atoms = load_structure(pdb_file)
    distance = get_ca_distance(atoms, RESIDUE_2x46, RESIDUE_6x37)

    # Determine the stage (relax or backrub) from filename
    if pdb_file.startswith("relaxed_"):
        stage = "FastRelax"
    elif pdb_file.startswith("backrub_"):
        stage = "BackRub"
    else:
        stage = "Unknown"

    if distance is not None:
        results.append({
            'file': pdb_file,
            'stage': stage,
            'ca_distance_2x46_6x37': round(distance, 3),
        })
        print(f"  {pdb_file}: {distance:.2f} Å ({stage})")
    else:
        print(f"  {pdb_file}: FAILED to measure distance")

# 4. Merge with RMSD data from metadata files
print("\nMerging with RMSD data from metadata files...")

# Try to load relax metadata
rmsd_lookup = {}

relax_meta = "relax_metadata.tsv"
if os.path.exists(relax_meta):
    df = pd.read_csv(relax_meta, sep="\t")
    for _, row in df.iterrows():
        key = f"relaxed_{int(row['sample'])}.pdb"
        if "rmsd" in df.columns:
            rmsd_lookup[key] = row["rmsd"]
        elif "rmsd_to_native" in df.columns:
            rmsd_lookup[key] = row["rmsd_to_native"]


for meta_file in sorted(glob.glob("backrub_metadata_*.tsv")):
    df = pd.read_csv(meta_file, sep="\t")
    # Extract prefix from filename: backrub_metadata_kt0.7_trials1000.tsv
    parts = meta_file.replace("backrub_metadata_", "").replace(".tsv", "")
    # Try to find matching pdb files
    for _, row in df.iterrows():
        sample_idx = int(row['sample'])
        # Try various naming patterns
        for prefix_pattern in [f"backrub_test_{sample_idx}.pdb",
                               f"backrub_kt{row.get('mc_kt', '')}_{sample_idx}.pdb"]:
            if "rmsd_to_native" in df.columns:
                rmsd_lookup[prefix_pattern] = row["rmsd_to_native"]

# Add RMSD to results
for r in results:
    r['rmsd_to_native'] = rmsd_lookup.get(r['file'], None)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("functional_states.tsv", sep="\t", index=False)
print(f"Saved: functional_states.tsv")



# 5. Generate scatter plot: Open Distance Geometry vs. RMSD

print("\nGenerating scatter plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Plot by stage
colors = {"FastRelax": "blue", "BackRub": "red", "Unknown": "gray"}

for stage in results_df["stage"].unique():
    subset = results_df[results_df["stage"] == stage]
    if subset["rmsd_to_native"].notna().any():
        ax.scatter(
            subset["rmsd_to_native"],
            subset["ca_distance_2x46_6x37"],
            label=stage,
            color=colors.get(stage, "gray"),
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            s=50
        )

# Add native structure as a reference point
ax.scatter([0], [native_distance], color="green", marker="*", s=200,
           label=f"Native (8Q12)", zorder=5, edgecolors="black")

# Add reference lines for open/intermediate/closed states
# From GPCRdb: for class A, inactive (closed) state max distance is ~11.9 Å
ax.axhline(y=11.9, color="gray", linestyle="--", alpha=0.5, label="Class A inactive threshold (~11.9 Å)")

ax.set_xlabel("RMSD to Native Structure (Å)", fontsize=12)
ax.set_ylabel("Cα Distance: 2x46–6x37 (Å)\n(TM6 Tilt / Open Distance Geometry)", fontsize=12)
ax.set_title("Functional State Analysis: Open Distance vs. RMSD\nMC4R Rosetta Ensemble", fontsize=14)
ax.legend(fontsize=9, loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("open_distance_vs_rmsd.png", dpi=150)
print("Saved: open_distance_vs_rmsd.png")

print("\n" + "=" * 60)
print("DONE!")
print(f"  Native 2x46–6x37 distance: {native_distance:.2f} Å")
print(f"  Sampled structures analyzed: {len(results)}")
