# Data

The datasets used in this project are derived from **MIMIC-IV** (Medical Information Mart for Intensive Care IV), which is a restricted-access clinical database. You must obtain credentialed access before downloading or using the data.

## Access Requirements

1. Complete the required training course at [CITI Program](https://about.citiprogram.org/), specifically the "Data or Specimens Only Research" module.
2. Apply for access to MIMIC-IV on [PhysioNet](https://physionet.org/content/mimiciv/latest/).
3. Once approved, download the MIMIC-IV dataset and place the relevant tables in your working directory.

## Data Preparation

After obtaining access to MIMIC-IV, run the preprocessing script to generate the dataset splits used for training and evaluation:

```bash
# Preprocess discharge notes into prompt format for prospective ICD-10 coding
python src/setup/preprocess_data_verl.py \
    --data_paths <path-to-mimic-iv-parquets> \
    --prompt_file src/prompts/prospective_icd_coding.csv \
    --output_folder data/icd10_standard_prospective/
```

## Directory Layout

After preprocessing, the `data/` directory contains the following splits:

| Directory | Description |
|---|---|
| `icd10_standard_prospective/` | Main dataset: prospective ICD-10 coding from admission notes |
| `icd10_standard_single_prospective/` | Single-code variant (first/principal diagnosis only) |
| `icd10/hosp/` and `icd10/icu/` | ICD-10 code lists for hospital and ICU subsets |
| `icd_10_cm_guidelines/` | ICD-10-CM official coding guidelines (per chapter, JSON) |
| `sft/` | Supervised fine-tuning warm-start data |
| `processed/` | Output of `preprocess_data_verl.py` ready for verl training |

## MeiliSearch Index

The reward function at training time resolves free-text diagnosis predictions to ICD-10 codes using a MeiliSearch full-text index. To build it:

```bash
# Start MeiliSearch
docker run -d -p 7700:7700 getmeili/meilisearch:latest

# Index ICD-10 codes from the MIMIC-IV vocabulary
python src/setup/fill_meili.py \
    --icd_codes data/icd10/hosp/icd_codes.csv \
    --index mimic_icd_10_codes
```

Set the `MEILISEARCH_URL` and `MEILI_ICD_INDEX` environment variables to point the reward function at your index (see `src/rewards.py` and the main [README](../README.md)).

## Notes

- All data files (`.parquet`, `.csv` from MIMIC) are excluded from version control via `.gitignore`.
- ICD-10-CM coding guidelines (`icd_10_cm_guidelines/`) are sourced from the [CMS official publications](https://www.cms.gov/medicare/coding-billing/icd-10-codes) and are included in the repository as they are public domain.
