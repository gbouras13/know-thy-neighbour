# Know Thy Neighbour

Code to reproduce *Know Thy Neighbour: Benchmarking Protein Language Model Embedding-Based Annotation Reveals Anisotropy Substantially Impacts Retrieval* (George Bouras and Robert A. Edwards, NeurIPS 2026 Workshop: I Can't Believe It's Not Better — Failure Modes of AI in Biology).

The paper asks whether cosine nearest-neighbour search over mean-pooled protein language model (pLM) embeddings can be thresholded to control false positives, the way E-values control them for MMseqs2 and Foldseek. It benchmarks twelve embedding methods against both tools on three tests:

| Test | Data | Question |
|---|---|---|
| Test 1 | Swiss-Prot 2026_02, Pfam 38.2 labels, 30% identity cluster split | Sensitivity for remote homologs at 1% false-positive rate, calibrated on natural hard negatives; a ladder of synthetic decoys |
| Test 2 | SCOPe/ASTRAL-40 2.08, family held out | Same-superfamily retrieval at 1% different-fold false-positive rate, and threshold-free Top-1 |
| Test 3 | SCOPe/ASTRAL-40 2.08, all-versus-all | The Foldseek SCOP sensitivity benchmark |

It also measures the anisotropy of each embedding space and tests four reference-fitted corrections: centering, all-but-the-top, diagonal whitening and ZCA whitening.

## Repository layout

```
scripts/          pipeline and analysis, numbered in rough run order
figures/R/        one R script per figure, plus a shared theme
figures/data/     the tidy CSVs every figure is drawn from
requirements.txt  Python dependencies
```

Raw data, embeddings and intermediate results are too large for git. They are archived on Zenodo (DOI to be added) with the layout described under [Data layout](#data-layout).

## Requirements

- Python 3.10 or later with `numpy`. The analysis scripts need nothing else.
- For embeddings: a GPU, and `torch`, `transformers`, `sentencepiece`, `protobuf`, `huggingface_hub`, `safetensors` (`pip install -r requirements.txt`). ESM-C needs the Biohub fork of transformers; see `requirements.txt`.
- [MMseqs2](https://github.com/soedinglab/MMseqs2) 18.8cc5c and [Foldseek](https://github.com/steineggerlab/foldseek) 10.941cd33 on the `PATH` (or set `MMSEQS` and `FOLDSEEK` to the binaries).
- [phold](https://github.com/gbouras13/phold) with its database, for ProstT5-predicted 3Di. Set `PHOLD_DB` to the phold database directory (and `PHOLD` if phold is not on the `PATH`).
- R 4.3 or later with `ggplot2`, `dplyr`, `tidyr`, `readr`, `scales`, `forcats` and `stringr`, for the figures.
- The ProtTucker head weights (`tucker_weights.pt`, [Zenodo record 14675997](https://zenodo.org/records/14675997/files/tucker_weights.pt)) in `tools/`.

All scripts run from the repository root and resolve paths relative to it.

## Reproducing the figures only

`figures/data/` holds the input of every figure, so the figures rebuild without any other data:

```bash
Rscript figures/R/make_all.R
```

Output goes to `figures/out/` as matched PDF and PNG files. Each script also runs on its own, for example `Rscript figures/R/fig1_headline.R`.

| Paper | Script | Data |
|---|---|---|
| Figure 1 | `fig1_headline.R` | `fig1_headline.csv` |
| Figure 2 | `fig2_densities.R` | `fig2_densities.csv.gz` |
| Figure 3 | `fig3b_ceiling.R` | `fig3b_ceiling.csv` |
| Figure S1 | `figS6_densities_test1_all.R` | `fig2_densities.csv.gz` |
| Figure S2 | `figS7_densities_test1_classes.R` | `figS7_densities_all_classes.csv.gz` |
| Figure S3 | `figS5_densities_test2_all.R` | `fig12_densities_test2.csv.gz` |
| Figure S4 | `figS8_length_all.R` | `fig4_length.csv` |
| Figure S5 | `figS9_hubness_all.R` | `figS2_hubness.csv` |
| Figure S6 | `fig6_ladder.R` | `fig3b_ceiling.csv` |
| Figure S7 | `figS9_ladder_test1.R` | `figS9_ladder_test1.csv` |
| Figure S8 | `figS11_ladder_top1_test1.R` | `figS9_ladder_test1.csv` |
| Figure S9 | `figS10_ladder_top1_test2.R` | `fig6_ladder.csv` |
| Figure S10 | `fig10_test3.R` | `fig10_test3.csv` |
| Figure S11 | `fig9_scope_cdf.R` | `fig9_test3_cdf.csv` |

## Reproducing the analyses from intermediate results

With the Zenodo `results/` and `data/` folders unpacked in the repository root, the analyses and figure data rebuild on a laptop. The Zenodo result tables are gzipped (`*.tsv.gz`), while the pipeline below writes plain `*.tsv`; every script accepts either form.

```bash
python3 scripts/10_threshold.py                       # Test 1 transform ladder at 1% FPR
python3 scripts/25_bootstrap_test1.py --replicates 400 --seed 0 > results/bootstrap_test1.txt
python3 scripts/25_bootstrap_test1.py --replicates 400 --seed 0 --by-stratum > results/bootstrap_test1_by_stratum.txt
python3 scripts/15_scope_compare.py                   # Test 2 at matched 1% FPR
python3 scripts/15_scope_compare.py --fpr-denominator answerable
python3 scripts/24_heldout.py                         # Test 2 split-half held-out thresholds
python3 scripts/26_hubness.py                         # reference coverage, embeddings
python3 scripts/43_hubness_alignment.py               # reference coverage, alignment arms
python3 scripts/43_test1_top1.py --models <model keys>   # Test 1 Top-1
python3 scripts/17_stratify.py > results/stratified_recovery.txt   # length strata (Fig. S4)
python3 scripts/39_test2_densities.py
python3 scripts/27_figure_data.py                     # figures/data/*.csv
python3 scripts/44_ladder_figure_data.py
python3 scripts/34_assemble_test3.py
python3 scripts/37_test3_cdf.py
Rscript figures/R/make_all.R
```

`results/bootstrap_test1.txt` also holds the achieved false-positive rate of every method against each negative class (D0–D6) at the pooled D6 threshold.

The eigenvalue spectra need the embedding matrices:

```bash
python3 scripts/28_spectrum.py --models esm2_150m esm2_650m esm2_3b esmc_300m esmc_600m esmc_6b ankh_base ankh_large prott5 prostt5 glm2_650m_embed prottucker --stem scope_reference   # spectrum, Test 2 reference
python3 scripts/28_spectrum.py --models <same keys> --stem joint_reference                                                                                                  # spectrum, Test 1 reference
```

### Robustness analyses

These produce the supplementary robustness tables. All but the last run from the intermediate results.

```bash
# Test 1, positives restricted to single-Pfam-domain queries
python3 scripts/25_bootstrap_test1.py --replicates 400 --seed 0 --pos-filter single_pfam > results/bootstrap_test1_single_pfam.txt
# Test 1, positives restricted to queries whose exact Pfam architecture occurs in the reference
python3 scripts/25_bootstrap_test1.py --replicates 400 --seed 0 --pos-filter arch_match > results/bootstrap_test1_arch_match.txt
# Test 1, threshold fitted on D6 reweighted to the positives' length distribution
python3 scripts/25_bootstrap_test1.py --replicates 400 --seed 0 --length-weighted > results/bootstrap_test1_length_weighted.txt
# Test 2, the published thresholds applied to the 566 non-answerable queries as independent negatives
python3 scripts/45_test2_nonanswerable.py            # results/test2_nonanswerable.tsv
# Test 1, Foldseek structure hits for the D6 negatives each method accepts at its 1% threshold
python3 scripts/46_d6_structural_check.py            # results/d6_structural_check*.tsv
# Test 2, what the leading principal components encode (length, composition); needs the embedding matrices
python3 scripts/47_pc_covariates.py --models esm2_650m esmc_6b prott5   # results/pc_covariates.json
```

The Test 1 threshold is the score of the (k+1)th highest D6 negative, with k = int(0.01 × n). It therefore admits 34 of 3,344 D6 proteins (1.02%). `25_bootstrap_test1.py` skips a method, with a warning, when fewer than k+1 of its calibration negatives have a score, and `46_d6_structural_check.py` stops if its Foldseek table covers no D6 protein.

## Running the full pipeline

Model keys used below: `esm2_150m esm2_650m esm2_3b esmc_300m esmc_600m esmc_6b ankh_base ankh_large prott5 prostt5 glm2_650m_embed`. Test 1 has three length arms, `small` (1–99 aa), `mid` (100–199 aa) and `control` (200–400 aa), which share one reference set.

### Test 1: Swiss-Prot

```bash
python3 scripts/01_fetch_swissprot.py --arm all     # live UniProt query; see the note below
for arm in small mid control; do python3 scripts/02_build_splits.py --arm $arm; done
python3 scripts/02c_joint_split.py                   # 30% identity, 80% coverage, 20% of clusters held out
python3 scripts/02d_measure_separation.py --prefix joint
for arm in small mid control; do
  python3 scripts/03_make_decoys.py --arm $arm --prefix joint --n 5000 --seeds 5   # D0-D5
done
python3 scripts/03b_natural_negatives.py --prefix joint --max-pident 30           # D6

# sequence search (MMseqs2)
for arm in small mid control; do REFNAME=joint ./scripts/04_run_sequence.sh $arm 16; done

# structure search: Foldseek on AlphaFold DB structures, and on ProstT5-predicted 3Di
# (07_run_structure.sh does the same on a single machine that has both a GPU and the AFDB database)
./scripts/07a_build_query_fastas.sh                  # data/armC_queries/<arm>_armC_queries.fasta
for arm in small mid control; do                     # 3Di prediction (GPU)
  phold proteins-predict -i data/armC_queries/${arm}_armC_queries.fasta \
      -o data/armC_3di/$arm -p $arm -d $PHOLD_DB -t 16 -f --mask_threshold 0 --batch_size 10
done
for arm in small mid control; do REFNAME=joint ./scripts/07b_search_structure.sh $arm 16; done
./scripts/07d_top50.sh                               # keep the 50 best hits per query
for arm in small mid control; do python3 scripts/07c_armC_hard_negatives.py --arm $arm; done

# embeddings (GPU) and cosine kNN
for m in <model keys>; do for arm in small mid control; do
  python3 scripts/05_embed.py --model $m --arm $arm --prefix joint
  python3 scripts/06_knn.py   --model $m --arm $arm --prefix joint
  for t in center diag abtt; do
    python3 scripts/18_center.py --model $m --arm $arm --transform $t
  done
  python3 scripts/18_center.py --model $m --arm $arm --transform whiten --shrink 0   # exact ZCA, as in the paper
done; done
for arm in small mid control; do python3 scripts/21_prottucker_direct.py --arm $arm; done
```

`01_fetch_swissprot.py` queries the live UniProt REST API, so a fresh run returns whatever release is current. For exact reproduction, skip it and start from the Zenodo `data/raw/` slices (Swiss-Prot release 2026_02).

`20_prottucker.sh` and `20b_parse_eat.py` run ProtTucker through the original EAT code instead, as a cross-check of `21_prottucker_direct.py`.

### Test 2: SCOPe/ASTRAL-40, family held out

```bash
python3 scripts/11_scope_build.py
./scripts/12_scope_search.sh                         # D2 shuffles and MMseqs2
python3 scripts/13_scope_analyse.py
# ProstT5 3Di for every domain (GPU): queries headed "real|<id>", reference ids bare
mkdir -p data/interim
{ awk '/^>/{print ">real|" substr($1,2); next} {print}' data/scope/scope_query.fasta
  cat data/scope/scope_reference.fasta; } > data/interim/scope_3di_input.fasta
phold proteins-predict -i data/interim/scope_3di_input.fasta -o data/scope_3di -p scope \
    -d $PHOLD_DB -t 16 -f --mask_threshold 0 --batch_size 10   # data/scope_3di/scope_{aa,3di}.fasta
./scripts/16b_scope_build_search.sh                  # Foldseek on ASTRAL coordinates and on ProstT5 3Di
./scripts/41a_predict_shuffle_3di.sh && ./scripts/41_scope_cprime_shuffles.sh
for m in <model keys>; do
  python3 scripts/14_scope_embed.py --model $m --save-embeddings
  for t in center diag abtt; do python3 scripts/18_center.py --model $m --scope --transform $t; done
  python3 scripts/18_center.py --model $m --scope --transform whiten --shrink 0      # exact ZCA, as in the paper
  python3 scripts/18_center.py --model $m --scope --transform whiten --suffix whiten_shrunk   # shrinkage ablation
done
python3 scripts/21_prottucker_direct.py --arm scope
```

The phold input must sit outside the output directory, because `-f` recreates it. `41a_predict_shuffle_3di.sh` predicts 3Di for the D2 shuffles with the same flags.

The paper reports exact ZCA throughout (`--shrink 0`). Without that flag, `18_center.py` shrinks the covariance by `a = d/n` (capped at 0.5), which is the supplementary shrinkage ablation.

### Test 3: Foldseek SCOP benchmark

Every arm is scored on the same 13,822 domains, listed in `data/scope/test3_common_ids.txt`: the domains of the 15,176-domain split that have usable ASTRAL coordinates in the Foldseek structure database. No script in this repository builds that list; it is provided in the Zenodo archive. Of these domains, 5,003 qualify as queries.

```bash
C=data/scope/test3_common_ids.txt
T=data/interim/test3; mkdir -p $T

# embedding arms, raw and ZCA-whitened (whitening fitted on the Test 2 reference only)
python3 scripts/30_test3_allvall.py \
    --models esm2_150m esm2_650m esm2_3b esmc_300m esmc_600m esmc_6b ankh_base ankh_large prott5 prostt5 glm2_650m_embed \
             esm2_150m:whiten esm2_650m:whiten esm2_3b:whiten esmc_300m:whiten esmc_600m:whiten esmc_6b:whiten \
             ankh_base:whiten ankh_large:whiten prott5:whiten prostt5:whiten glm2_650m_embed:whiten prottucker \
    --restrict $C --out results/test3_common_embed.csv --per-query results/test3_perquery_embed.csv

# MMseqs2, all versus all over the 15,176 split domains
# (data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa minus the one domain 11_scope_build.py drops)
cat data/scope/scope_reference.fasta data/scope/scope_query.fasta > $T/scope_all.fasta
mmseqs easy-search $T/scope_all.fasta $T/scope_all.fasta $T/mmseqs_hits.tsv $T/tmp_mmseqs \
    -s 7.5 --max-seqs 4000 -e 10000 --format-output query,target,evalue
python3 scripts/31_test3_score_tsv.py --hits $T/mmseqs_hits.tsv --name "sequence (MMseqs2)" --restrict $C \
    --out results/test3_common_mmseqs.csv --per-query results/test3_perquery_mmseqs.csv

# Foldseek, all versus all, on the databases 16b_scope_build_search.sh built in data/interim/scope_struct
W=data/interim/scope_struct
#   C: ASTRAL coordinates (entry names carry a .ent suffix, removed before scoring)
foldseek search $W/allDB $W/allDB $T/resC $T/tmp_c -s 9.5 -e 10000 --max-seqs 4000
foldseek convertalis $W/allDB $W/allDB $T/resC $T/fsC.m8 --format-output query,target,evalue
awk -F'\t' -v OFS='\t' '{sub(/\.ent$/,"",$1); sub(/\.ent$/,"",$2); print}' $T/fsC.m8 > $T/fsC_hits.tsv
python3 scripts/31_test3_score_tsv.py --hits $T/fsC_hits.tsv --name Foldseek --restrict $C \
    --out results/test3_common_fsC.csv --per-query results/test3_perquery_fsC.csv
#   C': ProstT5 3Di for every domain (query and reference databases concatenated)
for s in "" _ss _h; do foldseek concatdbs $W/cp_queryDB$s $W/cp_referenceDB$s $T/cpDB$s; done
foldseek search $T/cpDB $T/cpDB $T/resCp $T/tmp_cp -s 9.5 -e 10000 --max-seqs 4000
foldseek convertalis $T/cpDB $T/cpDB $T/resCp $T/fsCp_hits.tsv --format-output query,target,evalue
python3 scripts/31_test3_score_tsv.py --hits $T/fsCp_hits.tsv --name ProstT5-Foldseek --restrict $C \
    --out results/test3_common_fsCp.csv --per-query results/test3_perquery_fsCp.csv

python3 scripts/34_assemble_test3.py                 # results/test3_common_*.csv -> figures/data/fig10_test3.csv
python3 scripts/37_test3_cdf.py                      # results/test3_perquery_*.csv -> figures/data/fig9_test3_cdf.csv
```

`31_test3_score_tsv.py` appends to `--out`, so remove an old `results/test3_common_*.csv` before rescoring an arm. `34_assemble_test3.py` stops if the arms were scored on different query counts.

## Data layout

The Zenodo archive mirrors the paths the scripts expect:

```
data/raw/          Swiss-Prot slices as fetched (jsonl.gz), release 2026_02
data/queries/      joint_<arm>_query.{fasta,jsonl}: queries with Pfam, GO, length, answerable and remote flags
data/reference/    joint_reference.{fasta,jsonl}
data/decoys/       <arm>_D0..D5_*_seed1..5.fasta, <arm>_D6_natural_hard_all.{fasta,jsonl}
data/scope/        ASTRAL-40 2.08 sequences, classification, query/reference split, D2 shuffles, test3_common_ids.txt
data/embeddings/   <model>/<stem>.n<count>.f16.npy with matching .ids.txt
results/           search and kNN tables (sequence_*, structure_*, knn_*, scope_*), gzipped, plus bootstrap and analysis outputs
```

Whitened kNN tables: `knn_<arm>_<model>_whiten.tsv.gz` and `scope_knn_<model>_whiten.tsv.gz` are exact ZCA (`--shrink 0`), the estimator reported throughout the paper. `scope_knn_<model>_whiten_shrunk.tsv.gz` is the shrinkage variant used only in the supplementary shrinkage ablation.

Model checkpoints: `05_embed.py` pins a Hugging Face revision for six models. ESM-2 150M, ESM-2 650M, ProtT5, ProstT5 and gLM2-embed are unpinned, because the revision used could not be recovered. A fresh run of those five loads the current `main` branch, which may differ from the checkpoint behind the paper. The run manifest records which revision was loaded. `--require-pinned` refuses unpinned models.

## Citation

```bibtex
@inproceedings{bouras2026knowthyneighbour,
  author    = {Bouras, George and Edwards, Robert A.},
  title     = {Know Thy Neighbour: Benchmarking Protein Language Model Embedding-Based Annotation Reveals Anisotropy Substantially Impacts Retrieval},
  booktitle = {NeurIPS 2026 Workshop: I Can't Believe It's Not Better -- Failure Modes of AI in Biology},
  year      = {2026}
}
```

## Licence

MIT; see `LICENSE`.
