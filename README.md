# Find-abnormality
## A longitudinal metagenomic framework for detecting abnormal, duplicated, and mislabeled samples

![An introduction to the method](Figure1.png)

# Overview

This computational framework identifies sample-identity errors and abnormal microbiome profiles in longitudinal metagenomic studies.

Longitudinal microbiome studies rely on repeated sampling from the same individuals. However, sample swaps, duplicated samples, and metadata errors can introduce abnormal samples that compromise downstream analyses.

The framework integrates:

- Abnormality detection
- Mislabel classification
- Optional strain-level genomic confirmation

It is designed for large-scale metagenomic datasets and provides interpretable evidence for each flagged sample.

---

# Key features

## 1. Longitudinal abnormality detection

Stage 1 identifies samples that deviate from an individual's expected microbiome trajectory. It uses:

- Bray-Curtis dissimilarity
- Within-individual distance ranking
- Mutual nearest-neighbor relationships
- Graph-based clustering

These measures separate coherent longitudinal samples from potentially abnormal samples.

## 2. Mislabeled sample identification

Stage 2 detects two major types of sample-identity errors.

### Sample duplication

Samples from different individuals that show unexpectedly high similarity are flagged as possible duplicates. Potential causes include:

- Accidental sample reuse
- Duplicate submission
- Metadata-assignment errors

### Sample swapping

Samples potentially assigned to the wrong individual are identified by comparing:

- Microbiome similarity
- Longitudinal consistency
- Candidate-subject trajectories

## 3. Strain-level identity confirmation (optional)

Stage 3 cross-validates potentially abnormal or mislabeled samples using strain-specific mutation rates. Mutation-rate matrices are generated for each species-level genome bin (SGB) with StrainPhlAn 4. The abnormal sample is then compared with longitudinal samples assigned to the recorded individual and, when testing a proposed reassignment, with samples from the candidate individual.

A low mutation rate supports a shared strain origin, whereas consistently elevated mutation rates suggest that the samples originated from different individuals. Evidence from multiple organisms is preferred: in this framework, at least two shared SGBs with concordant mutation-rate patterns are required to support the same biological source.

# Installation

## Requirements

- Python 3.7+
- `numpy`
- `pandas`
- `scipy`
- `networkx`
- `matplotlib`
- `scikit-learn`

Install the Python dependencies with:

```bash
pip install numpy pandas scipy networkx matplotlib scikit-learn
```

# Usage: Stages 1 and 2

```bash
python find_abnormality.py abundance.tsv meta.tsv -s output_prefix [-c 0.3]
```

## Input-file examples

`abundance.tsv` (tab-delimited):

| | sample1 | sample2 | sample3 |
|---|---:|---:|---:|
| s__A | 2.5 | 4.2 | 0.0 |
| s__B | 0.0 | 1.3 | 0.9 |
| s__C | 1.1 | 0.0 | 2.2 |

`meta.tsv` (tab-delimited):

| sample_id | patient | batch |
|---|---|---|
| sample1 | PAT01 | batch_0 |
| sample2 | PAT01 | batch_1 |
| sample3 | PAT02 | batch_1 |

## Output-file guide

### 1. `<prefix>_mislabeled_sns.tsv`

Lists sample pairs within each patient that are flagged as mislabeled based on anomalous distances.

- `patient_id`: Patient or subject identifier
- `sample1`, `sample2`: Samples being compared
- `rank1`, `rank2`: Rank of the distance in each sample's distance vector
- `total_samples`: Number of samples for the patient
- `flag_type`: `mislabeled`

### 2. `<prefix>_cheating_sns.tsv`

Lists sample pairs within batches that are suspiciously similar and may represent duplicates.

- `batch`: Batch name or label
- `sample1`, `sample2`: Samples identified as possible duplicates
- `bray_curtis_distance`: Bray-Curtis distance between the samples
- `cheating_group`: Reporting group, such as `C1` or `C2`

### 3. `<prefix>_true_labels.tsv`

Suggests the most likely true patient partner for each sample identified as mislabeled.

- `mislabeled_sample`: Sample flagged as problematic
- `top_candidate_patient`: Patient with the lowest mean Bray-Curtis distance to the sample
- `distance_score`: Mean distance for the candidate
- `normal_samples`: Other normal samples assigned to the patient

### 4. `<prefix>.pkl`

Python pickle file containing intermediate results for advanced users or downstream analysis.

## Notes

- Output tables are tab-delimited and include header rows.
- `<prefix>` is set with `-s`; its default value is `results`.
- If only one batch is present, only the duplicate-detection output is generated.

# Usage: Stage 3

## Generate SGB mutation-rate matrices

For each SGB listed in the StrainPhlAn link file, run StrainPhlAn 4 with mutation-rate calculation enabled:

```bash
while read -r sgb; do
    clade="t__${sgb}"
    output_dir="output_LP/output_${clade}"

    mkdir -p "${output_dir}"

    strainphlan \
        -s consensus_markers/*.pkl \
        -m "db_markers/${clade}.fna" \
        -o "${output_dir}" \
        -n 40 \
        -c "${clade}" \
        --mutation_rates \
        --marker_in_n_samples 1 \
        --sample_with_n_markers 10 \
        --phylophlan_mode accurate
done < /storeData/zhouy/01_fenbaobao/public/PRJEB38984_analyses/strainphlan/link/batch_0_strain.link
```

The `--mutation_rates` option produces an SGB-specific pairwise mutation-rate matrix. The marker filters retain markers present in at least one sample and samples containing at least 10 markers.

## Summarize mutation rates for abnormal samples

The following reusable command-line script summarizes SGB-specific mutation rates for any set of abnormal samples. It:

- accepts a glob pattern instead of project-specific paths;
- reads abnormal samples and optional candidate assignments from a TSV file;
- normalizes sample names consistently across matrix rows and columns;
- validates that each mutation matrix is square and has unique sample IDs;
- uses vectorized NumPy indexing to extract pairwise values efficiently;
- compares each abnormal sample with its recorded patient and, when supplied, a candidate patient;
- optionally includes background pairs that do not contain the abnormal sample; and
- writes one tidy, tab-delimited result table.

Create a query file such as `abnormal_samples.tsv`:

| sample_id | recorded_patient | candidate_patient |
|---|---|---|
| W0075_3 | W0075 | W0020 |
| W0069_4 | W0069 | |

Only `sample_id` is required. If `recorded_patient` is omitted, the script derives it from the portion of the sample ID before the first underscore. `candidate_patient` is optional.

```python
#!/usr/bin/env python3
"""Summarize StrainPhlAn mutation rates for potentially abnormal samples."""

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize mutation rates involving abnormal samples."
    )
    parser.add_argument(
        "--mutation-glob",
        required=True,
        help="Quoted glob matching StrainPhlAn .mutation files.",
    )
    parser.add_argument(
        "--queries",
        required=True,
        help=(
            "TSV containing sample_id and optional recorded_patient and "
            "candidate_patient columns."
        ),
    )
    parser.add_argument("--output", required=True, help="Output TSV path.")
    parser.add_argument(
        "--min-recorded-samples",
        type=int,
        default=3,
        help="Minimum number of recorded-patient samples per SGB (default: 3).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="Mutation-rate multiplier (default: 1000).",
    )
    parser.add_argument(
        "--patient-separator",
        default="_",
        help="Separator used to derive patient IDs from sample IDs (default: _).",
    )
    parser.add_argument(
        "--strip-suffix",
        default=".fastq",
        help="Suffix removed from sample IDs (default: .fastq).",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Also report selected pairs that do not contain the query sample.",
    )
    return parser.parse_args()


def normalize_id(value, suffix):
    """Remove a terminal sequencing-file suffix from one sample ID."""
    value = str(value).strip()
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value


def patient_id(sample_id, separator):
    """Derive a patient ID from a normalized sample ID."""
    return sample_id.split(separator, 1)[0]


def load_queries(path, suffix, separator):
    queries = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "sample_id" not in queries.columns:
        raise ValueError("Query TSV must contain a 'sample_id' column")

    queries["sample_id"] = queries["sample_id"].map(
        lambda value: normalize_id(value, suffix)
    )
    if "recorded_patient" not in queries.columns:
        queries["recorded_patient"] = ""
    if "candidate_patient" not in queries.columns:
        queries["candidate_patient"] = ""

    missing = queries["recorded_patient"].eq("")
    queries.loc[missing, "recorded_patient"] = queries.loc[
        missing, "sample_id"
    ].map(lambda value: patient_id(value, separator))

    if queries["sample_id"].duplicated().any():
        duplicates = queries.loc[
            queries["sample_id"].duplicated(), "sample_id"
        ].tolist()
        raise ValueError(f"Duplicate query sample IDs: {duplicates}")
    return queries


def load_mutation_matrix(path, suffix):
    matrix = pd.read_csv(path, sep="\t", index_col="ids")
    matrix.index = [normalize_id(value, suffix) for value in matrix.index]
    matrix.columns = [normalize_id(value, suffix) for value in matrix.columns]

    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("duplicate sample IDs after normalization")

    shared = matrix.index.intersection(matrix.columns, sort=False)
    if shared.empty:
        raise ValueError("row and column sample IDs do not overlap")

    matrix = matrix.loc[shared, shared].apply(pd.to_numeric, errors="coerce")
    return matrix


def summarize_query(
    matrix,
    query_sample,
    recorded_patient,
    candidate_patient,
    sgb,
    source_file,
    separator,
    scale,
    min_recorded_samples,
    include_background,
):
    names = matrix.index.to_numpy(dtype=str)
    patients = np.array([patient_id(name, separator) for name in names])

    if np.count_nonzero(names == query_sample) != 1:
        return pd.DataFrame()
    if np.count_nonzero(patients == recorded_patient) < min_recorded_samples:
        return pd.DataFrame()

    selected_patients = {recorded_patient}
    if candidate_patient:
        selected_patients.add(candidate_patient)

    keep = np.isin(patients, list(selected_patients)) | (names == query_sample)
    selected = matrix.loc[names[keep], names[keep]]
    selected_names = selected.index.to_numpy(dtype=str)
    selected_patients_array = np.array(
        [patient_id(name, separator) for name in selected_names]
    )

    row_idx, col_idx = np.triu_indices(len(selected_names), k=1)
    sample1 = selected_names[row_idx]
    sample2 = selected_names[col_idx]
    patient1 = selected_patients_array[row_idx]
    patient2 = selected_patients_array[col_idx]
    values = selected.to_numpy(dtype=float)[row_idx, col_idx]
    involves_query = (sample1 == query_sample) | (sample2 == query_sample)

    valid = ~np.isnan(values)
    if not include_background:
        valid &= involves_query

    result = pd.DataFrame(
        {
            "query_sample": query_sample,
            "recorded_patient": recorded_patient,
            "candidate_patient": candidate_patient,
            "sample1": sample1[valid],
            "sample2": sample2[valid],
            "patient1": patient1[valid],
            "patient2": patient2[valid],
            "comparison_type": np.where(
                patient1[valid] == patient2[valid], "intra", "inter"
            ),
            "involves_query": involves_query[valid],
            "mutation_rate": values[valid],
            "mutation_rate_scaled": np.round(values[valid] * scale, 3),
            "SGB": sgb,
            "source_file": source_file,
        }
    )
    return result


def main():
    args = parse_args()
    queries = load_queries(
        args.queries,
        args.strip_suffix,
        args.patient_separator,
    )
    mutation_files = sorted(glob(args.mutation_glob))
    if not mutation_files:
        raise FileNotFoundError(
            f"No mutation files matched: {args.mutation_glob}"
        )

    outputs = []
    skipped = []

    for mutation_file in mutation_files:
        sgb = Path(mutation_file).name.removesuffix(".mutation")
        try:
            matrix = load_mutation_matrix(mutation_file, args.strip_suffix)
        except (OSError, ValueError, KeyError) as error:
            skipped.append(f"{mutation_file}: {error}")
            continue

        available = set(matrix.index)
        for query in queries.itertuples(index=False):
            if query.sample_id not in available:
                continue
            summary = summarize_query(
                matrix=matrix,
                query_sample=query.sample_id,
                recorded_patient=query.recorded_patient,
                candidate_patient=query.candidate_patient,
                sgb=sgb,
                source_file=os.path.abspath(mutation_file),
                separator=args.patient_separator,
                scale=args.scale,
                min_recorded_samples=args.min_recorded_samples,
                include_background=args.include_background,
            )
            if not summary.empty:
                outputs.append(summary)

    if outputs:
        result = pd.concat(outputs, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "query_sample", "recorded_patient", "candidate_patient",
                "sample1", "sample2", "patient1", "patient2",
                "comparison_type", "involves_query", "mutation_rate",
                "mutation_rate_scaled", "SGB", "source_file",
            ]
        )

    result.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result):,} comparisons to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} invalid mutation files:")
        for message in skipped:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
```

Save the script as `summarize_mutation_rates.py`, then run:

```bash
python summarize_mutation_rates.py \
    --mutation-glob '/path/to/output_LP/*/*.mutation' \
    --queries abnormal_samples.tsv \
    --output abnormal_sample_mutation_rates.tsv
```

By default, the output contains only pairs involving an abnormal sample. Add `--include-background` to retain other within-cohort pairs for comparison. Candidate-patient samples are included automatically when `candidate_patient` is supplied in the query file.

## Visualize recorded- and candidate-patient mutation rates

The integrated `find_abnormality.py` generates one boxplot for each abnormal sample. The first box summarizes mutation rates between the abnormal sample and samples from its recorded patient. When `candidate_patient` is provided, the second box summarizes mutation rates between the abnormal sample and samples from that candidate patient. Individual observations are overlaid so that the number and distribution of SGB-level comparisons remain visible.

Mutation rates are displayed as `log10(mutations per 1,000 aligned positions)`. A dashed horizontal reference line marks 0.1 mutations per 1,000 sites, corresponding to `log10(0.1) = -1`. Rates equal to or below zero are omitted from the figure because their base-10 logarithm is undefined; the untransformed values remain available in the Stage 3 TSV output.

Run Stage 3 and generate PNG boxplots with:

```bash
python find_abnormality.py \
    --stage3-only \
    -s results \
    --stage3-mutation-glob '/path/to/output_LP/*/*.mutation' \
    --stage3-queries abnormal_samples.tsv \
    --stage3-output results_stage3_mutation_rates.tsv \
    --stage3-plot-dir results_stage3_boxplots
```

The default image format is PNG. Use `--stage3-plot-format pdf` or `--stage3-plot-format svg` for publication-oriented vector output. If `--stage3-plot-dir` is omitted, figures are written to `<suffix>_stage3_boxplots`.

Each figure is named `<sample_id>_mutation_rates.<format>`. Lower log10 values indicate greater strain similarity. The box shows the median and interquartile range, while overlaid points show the underlying comparisons across samples and shared SGBs.

## Interpretation

A low mutation rate between an abnormal sample and the other longitudinal samples from an individual supports a common strain origin. Consistently high mutation rates argue against that assignment. A same-origin conclusion should be supported by concordant evidence from at least two shared SGBs.

# License

MIT License

# Contact

zhouyong0530@outlook.com
