# Data Directory

This directory is intentionally empty. Run the data pipeline scripts to populate it.

## How to Download

```bash
# 1. Download FAERS data (FDA adverse event reports)
python src/data/01_download_faers.py

# 2. Preprocess raw FAERS files
python src/data/02_preprocess.py

# 3. Download OnSIDES (drug label side effects)
python src/data/04_download_onsides.py

# 4. Download external datasets (BioDEX)
python src/data/05_download_external_datasets.py

# 5. Build training and decontaminated evaluation data (Combinatorial Diversity Engine)
python src/data/03_build_training_data.py

# 6. Build evaluation data in specific format for Gemma (Base Model)
python src/data/06_build_base_eval_data.py
```

## Directory Structure After Download

```
data/
├── raw/          ← Raw FAERS quarterly files (2019Q1–2026Q1)
├── processed/    ← Preprocessed data + training pairs
└── external/     ← OnSIDES, BioDEX datasets
```

## Data Sources

- **FDA FAERS**: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- **OnSIDES**: https://github.com/tatonetti-lab/onsides
- **BioDEX**: https://huggingface.co/datasets/BioDEX/BioDEX-Reactions
