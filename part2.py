import pyrosetta
import time
import csv
import os

# 1. Initialize PyRosetta
print("Initializing PyRosetta...")
pyrosetta.init()


# 2. Load the native 8Q12 structure as a pose

structure_file = "mc4r_receptor.cif"

if not os.path.exists(structure_file):
    print(f"ERROR: {structure_file} not found in current directory!")
    print(f"Current directory: {os.getcwd()}")
    print("Make sure your cleaned MC4R file from Part 1 is in this folder.")
    exit(1)

print(f"Loading structure from {structure_file}...")
pose = pyrosetta.rosetta.core.import_pose.pose_from_file(
    filename=structure_file,
    read_fold_tree=False,
    type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file
)
print(f"Loaded pose with {pose.total_residue()} residues")


# 3. Define the energy function

print("Setting up score function (ref2015)...")
sfxn = pyrosetta.create_score_function(weights_tag='ref2015')

# Score the native pose before relaxing
native_score = sfxn(pose)
print(f"Native structure score: {native_score:.2f}")


# 4. Define the constrained FastRelax protocol
print("Setting up constrained FastRelax protocol...")
fast_relax = pyrosetta.rosetta.protocols.relax.FastRelax(
    scorefxn_in=sfxn,
    standard_repeats=1
)

# Constrain the relaxed pose
fast_relax.constrain_relax_to_start_coords()
fast_relax.ramp_down_constraints(False)

# 5. Run the sampling loop

nsamples = 5  
metadata = []

print(f"\nRunning {nsamples} FastRelax samples...")
print("=" * 60)

for i in range(nsamples):
    print(f"\n--- Sample {i + 1} of {nsamples} ---")

    # Clone the native pose (so each run starts fresh)
    relaxed_pose = pose.clone()

    # Time the relaxation
    start_time = time.time()
    fast_relax.apply(relaxed_pose)
    elapsed = time.time() - start_time

    # Score the relaxed pose
    relaxed_score = sfxn(relaxed_pose)

    # Compute the heavy-atom RMSD between native and relaxed
    rmsd = pyrosetta.rosetta.core.scoring.all_atom_rmsd(pose, relaxed_pose)

    # Record metadata
    metadata.append({
        'sample': i,
        'score_diff': relaxed_score - native_score,
        'rmsd': rmsd,
        'time_seconds': round(elapsed, 2),
        'relaxed_score': round(relaxed_score, 2),
        'native_score': round(native_score, 2)
    })

    # Save the relaxed pose as a PDB file
    output_pdb = f"relaxed_{i}.pdb"
    relaxed_pose.dump_pdb(output_pdb)

    print(f"  Score difference: {relaxed_score - native_score:.2f}")
    print(f"  RMSD to native:  {rmsd:.4f}")
    print(f"  Time:            {elapsed:.1f} seconds")
    print(f"  Saved to:        {output_pdb}")


# 6. Save all metadata to a .tsv file
tsv_file = "relax_metadata.tsv"
with open(tsv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metadata[0].keys(), delimiter="\t")
    writer.writeheader()
    writer.writerows(metadata)

print("\n" + "=" * 60)
print("DONE!")
print(f"  Metadata saved to:  {tsv_file}")
