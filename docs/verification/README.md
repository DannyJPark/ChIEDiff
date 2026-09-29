# Verification records

Output of the equivalence controls, committed so the claim "the released code matches the
original" is backed by a record rather than an assertion. Re-run with:

    sbatch -p <partition> -q <qos> slurm/verify_equivalence.sbatch    # cluster
    bash slurm/verify_equivalence.sbatch                              # or directly

## controls_2026-09-29_job2348339.log

| | |
|---|---|
| date | 2026-09-29 |
| Slurm job | 2348339 |
| node | node01 |
| GPU | NVIDIA GeForce RTX 3090 |
| checkpoint | iteration 1,195,000, sha256 `c0c454824585cade…` |
| result | **ALL CONTROLS PASSED** |

| Control | What it shows |
|---|---|
| A (CPU) | every loss term and all 360 gradients bitwise identical, at t = 1, 500, 999 |
| A (GPU) | informational: median 1.13x the same-model noise floor, i.e. indistinguishable from the device |
| B | the release samples like the reference, and the atom-count prior agrees exactly |
| C | the chemistry-gate assertion fires when the XS flags are absent |
| D | E(steric) + E(hydrophobic) + E(hbond) == E(full), bitwise |

Control A's GPU arm does not gate. Its own same-model floor reaches 1.7e-02 at late
timesteps, so it cannot discriminate a code difference from CUDA's order-nondeterministic
scatter atomics; the CPU arm decides and requires exact equality.

Control B compares against the reference's own run-to-run spread rather than demanding
identical molecules, because the reference does not produce identical molecules either. On
this run the reference differed from itself by 7.05 A and 37 atom types, and from the release
by 6.87 A and 47. See README section 6.4.
