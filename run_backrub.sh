NATIVE="mc4r_receptor.cif"
INPUT="relaxed_0.pdb"

# test
echo "=== TEST RUN: mc_kt=0.7, ntrials=1000, nsamples=2 ==="
python sample_backrub.py \
    --native_pdb "$NATIVE" \
    --input_pdb "$INPUT" \
    --output_pdb "backrub_test" \
    --ntrials 1000 \
    --mc_kt 0.7 \
    --nsamples 2

echo ""
echo "Test run complete!"
echo ""

# --- Full parameter sweep --
 for mc_kt in 0.4 0.7 1.0 1.5; do
     for ntrials in 1000 5000 10000; do
         echo "=== Running: mc_kt=$mc_kt, ntrials=$ntrials ==="
         python sample_backrub.py \
             --native_pdb "$NATIVE" \
             --input_pdb "$INPUT" \
             --output_pdb "backrub_kt${mc_kt}_nt${ntrials}" \
             --ntrials $ntrials \
             --mc_kt $mc_kt \
             --nsamples 10
     done
 done
