import argparse
import pyrosetta
import time
import csv
import os


def main():
  
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="BackRub ensemble sampling")
    parser.add_argument("--native_pdb", required=True,
                        help="Path to the native/original structure (e.g. mc4r_receptor.cif)")
    parser.add_argument("--input_pdb", required=True,
                        help="Path to a relaxed structure from Part 2 (e.g. relaxed_0.pdb)")
    parser.add_argument("--output_pdb", required=True,
                        help="Prefix for output PDB files (e.g. 'backrub' -> backrub_0.pdb, backrub_1.pdb, ...)")
    parser.add_argument("--ntrials", type=int, required=True,
                        help="Number of BackRub MCMC trials per sample (try 1000-10000)")
    parser.add_argument("--mc_kt", type=float, required=True,
                        help="MCMC temperature parameter (try 0.4-1.5, production=0.7)")
    parser.add_argument("--nsamples", type=int, required=True,
                        help="Number of samples to generate (try 10)")
    args = parser.parse_args()

    
    # 1. Initialize PyRosetta with MCMC parameter
   
    print(f"Initializing PyRosetta with ntrials={args.ntrials}, mc_kt={args.mc_kt}...")
    pyrosetta.init(
        extra_options=f"-backrub:ntrials {args.ntrials} -backrub:mc_kt {args.mc_kt}"
    )

    
    # 2. Load the native structure
    
    print(f"Loading native structure: {args.native_pdb}")
    if args.native_pdb.endswith(".cif"):
        native_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(
            filename=args.native_pdb,
            read_fold_tree=False,
            type=pyrosetta.rosetta.core.import_pose.FileType.CIF_file
        )
    else:
        native_pose = pyrosetta.pose_from_pdb(args.native_pdb)

    
    # 3. Load the relaxed input structure
    
    print(f"Loading input (relaxed) structure: {args.input_pdb}")
    input_pose = pyrosetta.pose_from_pdb(args.input_pdb)

    # Set up score function
    sfxn = pyrosetta.create_score_function(weights_tag='ref2015')
    input_score = sfxn(input_pose)

    
    # 4. Create the BackRub protocol
    
    print("Setting up BackRubProtocol...")
    backrub_protocol = pyrosetta.rosetta.protocols.backrub.BackrubProtocol()

   
    # 5. Run sampling loop
    
    metadata = []

    print(f"\nGenerating {args.nsamples} BackRub samples...")
    print(f"  ntrials = {args.ntrials}")
    print(f"  mc_kt   = {args.mc_kt}")
    print("=" * 60)

    for i in range(args.nsamples):
        print(f"\n--- Sample {i + 1} of {args.nsamples} ---")

        # Clone the relaxed input pose
        pose = input_pose.clone()

        # Time and apply BackRub
        start_time = time.time()
        backrub_protocol.apply(pose)
        elapsed = time.time() - start_time

        # Score the BackRub pose
        backrub_score = sfxn(pose)

        # Compute RMSD to BOTH native and input conformations
        rmsd_to_native = pyrosetta.rosetta.core.scoring.all_atom_rmsd(native_pose, pose)
        rmsd_to_input = pyrosetta.rosetta.core.scoring.all_atom_rmsd(input_pose, pose)

        # Record metadata
        metadata.append({
            'sample': i,
            'mc_kt': args.mc_kt,
            'ntrials': args.ntrials,
            'score': round(backrub_score, 2),
            'score_diff': round(backrub_score - input_score, 2),
            'rmsd_to_native': round(rmsd_to_native, 4),
            'rmsd_to_input': round(rmsd_to_input, 4),
            'time_seconds': round(elapsed, 2)
        })

        # Save the pose to disk
        output_file = f"{args.output_pdb}_{i}.pdb"
        pose.dump_pdb(output_file)

        print(f"  Score diff (vs input): {backrub_score - input_score:.2f}")
        print(f"  RMSD to native:       {rmsd_to_native:.4f}")
        print(f"  RMSD to input:        {rmsd_to_input:.4f}")
        print(f"  Time:                 {elapsed:.1f}s")
        print(f"  Saved to:             {output_file}")

    
    # 6. Save metadata to .tsv

    tsv_file = f"backrub_metadata_kt{args.mc_kt}_trials{args.ntrials}.tsv"
    with open(tsv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Metadata saved to:      {tsv_file}")
    print(f"  BackRub structures:     {args.output_pdb}_0.pdb through {args.output_pdb}_{args.nsamples - 1}.pdb")


if __name__ == "__main__":
    main()
