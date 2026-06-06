# amp-motif-discovery-pipeline
 
> Bioinformatic pipeline for antimicrobial peptide motif discovery, combination analysis and therapeutic candidate selection.
 
**BSc Biotechnology · Universitat Autònoma de Barcelona · 2025–2026**  
**Author:** Mireia Rivas Bermúdez  
**Supervisor:** Dr. Marc Torrent — SysBioLab, Dept. Biochemistry and Molecular Biology, UAB
 
---
 
## Overview
 
Starting from 25,511 AMP sequences integrated from four public databases, this pipeline:
 
1. Builds a non-redundant AMP dataset and a matched non-AMP background (UniProt sliding-window)
2. Applies independent redundancy reduction — **CD-HIT** and **MMseqs2** at 80% identity
3. Performs discriminative motif discovery — **MEME** and **STREME** (four parallel pipelines)
4. Scans motif occurrences with **FIMO** and computes enrichment (Fisher's exact test + BH FDR)
5. Groups significant motifs into families by PWM similarity (**TOMTOM**) with edit-distance re-clustering
6. Predicts membrane interaction mechanism per motif (**FMAP** + **PPM 3.0**, weighted voting)
7. Predicts MIC against *E. coli* ATCC 11775 (**APEX**) and validates against GRAMPA
8. Evaluates all pairwise motif combinations (n = 13,225) under three linker conditions
9. Selects 10 structurally diverse final candidates by composite ranking + Ward hierarchical clustering
10. Predicts 3D structures (**ColabFold/AlphaFold2**) and haemolytic profiles (**PyAMPA**)
---
 
## Repository structure
 
```
amp-motif-discovery-pipeline/
├── README.md
├── requirements.txt
├── envs/
│   ├── environment.yml           # Core analysis environment (Python 3.13)
│   └── environment_meme.yml      # MEME Suite environment (Python 3.11)
│
├── scripts/
│   ├── 01_dataset_construction/
│   ├── 02_clustering/
│   ├── 03_background_sampling/
│   ├── 04_motif_discovery/
│   ├── 05_motif_statistics/
│   ├── 06_motif_logos/
│   ├── 07_mechanism_prediction/
│   ├── 08_mic_prediction/
│   ├── 09_motif_combinations/
│   └── 10_candidate_selection/
│
└── results/
    ├── motif_statistics/         # 208 significant motifs, enrichment tables, volcano plots
    ├── motif_logos/              # Sequence logos for all 151 TOMTOM families
    └── final_candidates/         # Top 10 candidates, ranking tables, ColabFold structures
```
 
---
 
## Main outputs
 
| File | Description |
|------|-------------|
| `results/motif_statistics/` | 208 significant motifs with FDR, enrichment ratio and pipeline origin |
| `results/motif_logos/` | Sequence logos for all 151 motif families |
| `results/final_candidates/03_diverse_candidates.csv` | 10 final candidates with MIC, mechanism, haemolysis and PPT score |
| `results/final_candidates/01_master_candidates.xlsx` | Full ranked table with all analysis sheets |
 
---
 
## Execution order
 
```bash
# Stage 1 — Dataset construction
python scripts/01_dataset_construction/01_standardize_databases.py
python scripts/01_dataset_construction/02_filter_databases.py
python scripts/01_dataset_construction/03_create_master_dataset.py
python scripts/01_dataset_construction/04_background_nonamp.py
 
# Stage 2 — Redundancy reduction
python scripts/02_clustering/01_cluster_amp_cdhit.py
python scripts/02_clustering/02_cluster_amp_mmseqs2.py
 
# Stage 3 — Background sampling
python scripts/03_background_sampling/01_generate_matched_background.py
 
# Stage 4 — Motif discovery  ⚠ requires amp_meme environment
python scripts/04_motif_discovery/01_run_meme_streme.py
python scripts/04_motif_discovery/02_run_fimo_scanning.py
 
# Stage 5 — Motif statistics
python scripts/05_motif_statistics/01_fimo_enrichment_fdr.py
python scripts/05_motif_statistics/02_motif_reporting_robustness.py
python scripts/05_motif_statistics/03_tomtom_family_analysis.py
 
# Stage 6 — Motif logos
python scripts/06_motif_logos/01_generate_family_logos.py
 
# Stage 7 — Mechanism prediction
python scripts/07_mechanism_prediction/01_prepare_input.py
python scripts/07_mechanism_prediction/02_parse_fmap_output.py
python scripts/07_mechanism_prediction/03_parse_ppm_output.py
python scripts/07_mechanism_prediction/04_weighted_classification.py
 
# Stage 8 — MIC prediction
python scripts/08_mic_prediction/01_generate_variants.py
python scripts/08_mic_prediction/02_run_apex.py
python scripts/08_mic_prediction/03_validate_grampa.py
 
# Stage 9 — Motif combinations
python scripts/09_motif_combinations/01_generate_combinations.py
python scripts/09_motif_combinations/02_combination_analysis.py
python scripts/09_motif_combinations/03_linker_analysis.py
 
# Stage 10 — Candidate selection
python scripts/10_candidate_selection/01_composite_ranking.py
python scripts/10_candidate_selection/02_hierarchical_clustering.py
python scripts/10_candidate_selection/03_pyampa_haemolysis.py
```
 
All scripts resolve paths relative to the repository root.
 
---
 
## Installation
 
### Core environment
 
```bash
conda env create -f envs/environment.yml
conda activate amp_motif
```
 
### MEME Suite environment (Stage 4 only)
 
```bash
conda env create -f envs/environment_meme.yml
conda activate amp_meme
 
# Verify
meme --version && streme --version && fimo --version && tomtom --version
```
 
### External tools
 
The following tools must be installed and accessible in the corresponding environments:
 
- [CD-HIT](https://sites.google.com/view/cd-hit)
- [MMseqs2](https://github.com/soedinglab/MMseqs2)
- [MEME Suite](https://meme-suite.org/) — `meme`, `streme`, `fimo`, `tomtom`
- [FMAP](https://fmap.ame.astate.edu/)
- [PPM 3.0](https://opm.phar.umich.edu/ppm_server3)
- [APEX](https://apex.bioinformatics.net/)
- [ColabFold](https://github.com/sokrypton/ColabFold) — GPU recommended
- [PyAMPA](https://pyampa.bioinformatics.net/)
---
 
## Data sources
 
| Database | Reference |
|----------|-----------|
| [DRAMP 4.0](http://dramp.cpu-bioinfor.org/) | Ma et al., *Nucleic Acids Res* 2024 |
| [DBAASP v3](https://dbaasp.org/) | Pirtskhalava et al., *Nucleic Acids Res* 2021 |
| [dbAMP 3.0](http://140.138.77.240/~dbamp/) | Yao et al., *Nucleic Acids Res* 2025 |
| [CAMP4](http://www.camp3.bicnirrh.res.in/) | Gawde et al., *Nucleic Acids Res* 2023 |
| [UniProt](https://www.uniprot.org/) | UniProt Consortium, *Nucleic Acids Res* 2024 |
| [GRAMPA](https://github.com/zswitten/Antimicrobial-Peptides) | Witten & Witten, GitHub 2019 |
 
Raw source files are not tracked due to size. Processed results from Stage 5 onward are tracked to allow partial reproduction.
 
---
 
## Reproducibility
 
Full reproduction requires the raw source databases and all external tools listed above. From Stage 5 onward, tracked result files allow partial reproduction without raw data. Stages 7 (FMAP/PPM) and 10 (ColabFold/PyAMPA) require external tool access and are considered semi-reproducible depending on the computational environment.
