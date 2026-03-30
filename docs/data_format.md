# Data Format Guide

This document describes the dataset layout expected by `zebra-prop train`.
At training step `[1/8] Validating required input files`, the code checks this format.

## 1) Required directory layout

`data_dir` must contain both `id_prop/` and `description/`.

```text
{data_dir}/
├─ id_prop/
│  └─ id_prop_{property_name}.csv
└─ description/
   └─ {task_name}/
      ├─ AtomicOrbitals.csv
      ├─ ElementProperty.csv
      └─ ... (one or more CSV files)
```

Config mapping:
- `data_dir`: root directory above
- `property_name`: part of `id_prop_{property_name}.csv`
- `task_name`: subdirectory name under `description/`

## 2) Target property file (`id_prop_{property_name}.csv`)

Location:
- `{data_dir}/id_prop/id_prop_{property_name}.csv`

Requirements:
- CSV text file
- No header row
- At least 2 columns
- Column 1: material `id`
- Column 2: target value (numeric)

Example:

```csv
1,1.12
2,0.85
3,2.31
```

Notes:
- IDs should be unique.
- IDs must match the IDs used in description CSV files.

## 3) Description files (`*.csv` except `all.csv`)

Location:
- `{data_dir}/description/{task_name}/`

How loading works:
- All `*.csv` files in the task directory are loaded.
- `all.csv` is ignored.
- Files are merged using `id` and `formula`.
- Each file's `description` column is renamed to the file stem
  (for example, `AtomicOrbitals.csv` -> `AtomicOrbitals`).

Required columns in every file:
- `id`
- `formula`
- `description`

Example:

```csv
id,formula,description
1,SiO2,Silicon oxide with tetrahedral network...
2,Al2O3,Aluminum oxide with wide band gap...
```

Notes:
- `id` values should align with `id_prop_{property_name}.csv`.
- `formula` is required as a merge key.

## 4) Fold behavior

- Train/valid/test split is done in code (k-fold index split).
- By default, all folds use the same description directory.
- You do not need `fold_0/`, `fold_1/`, ... directories for normal usage.

## 5) Common validation errors and fixes

1. `Missing target CSV: .../id_prop/id_prop_{property_name}.csv`
   - Check `data_dir` and `property_name` in config.
   - Create the file with no header and at least two columns.

2. `Missing description directory: .../description/{task_name}`
   - Check `task_name` in config.
   - Create the directory and add description CSV files.

3. `No description CSV files found in ...`
   - Add one or more `*.csv` files under the task directory.
   - Ensure files are not only named `all.csv`.

4. `... is missing columns: ...`
   - Ensure each description CSV includes `id,formula,description`.
   - Check for typos like `ID`, `Formula`, or extra spaces in headers.

5. `Target CSV must have at least 2 columns` or `Target CSV is empty`
   - Remove header rows.
   - Ensure data rows look like `id,target`.

## 6) Quick manual checks

```bash
ls -R data
head -n 5 data/id_prop/id_prop_band_gap.csv
head -n 5 data/description/human-made/AtomicOrbitals.csv
```
