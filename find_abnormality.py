#!/usr/bin/env python3
"""
Find abnormal/mislabeled samples in longitudinal metagenome data.

Usage:
    python main.py metagenome_profiles.tsv meta_ci.tsv [-s SUFFIX] [-c CUTOFF]

Author: ChatGPT (2025-07-29)
"""

import argparse
import glob
import sys
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Logging Setup ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FindAbnormalSamples")


# ========== Bray-Curtis Distance ==========
def compute_bray_curtis(df):
    """
    Compute the Bray-Curtis distance matrix from a count/abundance dataframe.

    Parameters:
        df (pd.DataFrame): Feature x Sample dataframe.

    Returns:
        pd.DataFrame: Bray-Curtis distance matrix (Sample x Sample).
    """
    import numpy as np
    from scipy.spatial import distance
    import pandas as pd
    arr = np.log(df.values.T * 10000 + 1)  # Log transform to reduce bias from large counts
    dist_mat = distance.squareform(distance.pdist(arr, metric="braycurtis"))
    bray = pd.DataFrame(dist_mat, index=df.columns, columns=df.columns)
    return bray


# ========== Intra/Inter-Patient Distance ==========

def get_vector(m):
    """
    Extract upper-triangle values (excluding diagonal) from a square DataFrame.

    Parameters:
        m (pd.DataFrame): Square DataFrame.

    Returns:
        list: List of upper-triangle values.
    """
    idx = m.index
    vals = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            vals.append(m.iloc[i, j])
    return vals

def get_intra_bc(bray, meta):
    """
    Calculate intra-patient Bray-Curtis distances.

    Parameters:
        bray (pd.DataFrame): Distance matrix.
        meta (pd.DataFrame): Metadata DataFrame.

    Returns:
        list: List of intra-patient distances.
    """
    vals = []
    for pat in meta['patient'].unique():
        idx = meta[meta['patient'] == pat].index
        if len(idx) < 2:
            continue
        subm = bray.loc[idx, idx]
        vals.extend(get_vector(subm))
    return vals

def get_inter_bc(bray, meta, limit=1113):
    """
    Calculate inter-patient Bray-Curtis distances, up to a sample count limit.

    Parameters:
        bray (pd.DataFrame): Distance matrix.
        meta (pd.DataFrame): Metadata DataFrame.
        limit (int): Max number of distance pairs to collect.

    Returns:
        list: List of inter-patient distances.
    """
    patients = list(meta['patient'].unique())
    vals = []
    cnt = 0
    for i, pa in enumerate(patients):
        for pb in patients[i+1:]:
            idx_a = meta[meta['patient'] == pa].index
            idx_b = meta[meta['patient'] == pb].index
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            subm = bray.loc[idx_a, idx_b]
            vals.extend(subm.values.flatten())
            cnt += subm.size
            if cnt >= limit:
                return vals
    return vals


# ========== Mislabeled/Problematic Sample Detection ==========

def get_minimal_pair(bray, sn1, sn2):
    """
    Compute the rank of the distance between two samples (sn1, sn2).

    Returns:
        (int, int, int): (rank in sn1, rank in sn2, total samples)
    """
    bc = bray.loc[sn1, sn2]
    rank1 = bray[sn1].sort_values().tolist().index(bc)
    rank2 = bray[sn2].sort_values().tolist().index(bc)
    return rank1, rank2, bray.shape[0]

def get_problematic_patients(bray, meta):
    """
    Identify patients with sample pairs whose mutual distance is outlier (mislabeled).

    Returns:
        List of tuples: (patient_id, list of (rank1, rank2, N, "sn1-Vs-sn2"))
    """
    problems = []
    for pat in meta['patient'].unique():
        idx = meta[meta['patient'] == pat].index
        if len(idx) < 2:
            continue
        ranks = []
        sn_names = list(idx)
        for i in range(len(sn_names)):
            for j in range(i + 1, len(sn_names)):
                r1, r2, n = get_minimal_pair(bray, sn_names[i], sn_names[j])
                ranks.append((r1, r2, n, f"{sn_names[i]}-Vs-{sn_names[j]}"))
        for rank in ranks:
            if rank[0] > int(0.05 * rank[2]) and rank[1] > int(0.05 * rank[2]):
                problems.append((pat, ranks))
                break
    return problems


# ========== Cheating/Duplicate Detection ==========

def get_nearby_vector(m):
    """
    Extract values for adjacent samples (i,i+1) in the matrix.
    Used for estimating baseline duplicate distances.
    """
    idx = m.index
    vals = []
    for i in range(len(idx) - 1):
        j = i + 1
        vals.append(m.iloc[i, j])
    return vals

def get_intra_bc_nearby(bray, meta):
    """
    Get intra-patient distances between "adjacent" samples.
    """
    vals = []
    for pat in meta['patient'].unique():
        idx = meta[meta['patient'] == pat].index
        if len(idx) < 2:
            continue
        subm = bray.loc[idx, idx]
        vals.extend(get_nearby_vector(subm))
    return vals

def get_cutoff(bray, meta, problematic_pats):
    """
    Compute cutoff for flagging duplicates as cheating (mean of the lowest five intra-patient adjacent distances).

    Returns:
        float: The cutoff value.
    """
    import numpy as np
    ix = ~meta['patient'].isin(problematic_pats)
    clean_meta = meta[ix]
    vals = sorted(get_intra_bc_nearby(bray.loc[clean_meta.index, clean_meta.index], clean_meta))
    return np.mean(vals[:5]) if len(vals) >= 5 else np.mean(vals)

def get_duplicate_samples(bray, meta, batch_flags, cutoff):
    """
    Find pairs of samples within each batch with distance < cutoff.

    Returns:
        dict: batch_flag -> list of (sn1, sn2)
    """
    results = {}
    for batch in batch_flags:
        idx = meta[meta['batch'].str.endswith(batch)].index
        subm = bray.loc[idx, idx]
        pairs = []
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                if subm.iloc[i, j] < cutoff:
                    pairs.append((subm.columns[i], subm.columns[j]))
        results[batch] = pairs
    return results

def find_cheating_samples(dup_pairs):
    """
    For all batches, find sets of samples suspected of being cheating/duplicates.

    Returns:
        set: Sample names in duplicate/cheating relationships.
    """
    import networkx as nx
    sns = set()
    for pairs in dup_pairs.values():
        G = nx.Graph()
        for a, b in pairs:
            G.add_edge(a, b)
        for comp in nx.connected_components(G):
            if len(comp) > 1:
                sns |= set(comp)
    return sns


# ========== True Partner Labeling for Mislabels ==========

def identify_problemic_sn(artifact_info):
    """
    For a problematic patient, split their samples into problematic vs. normal via graph connectivity.
    """
    import networkx as nx
    mislabel_G = nx.Graph()
    for info in artifact_info[1]:
        sn1, sn2 = info[3].split("-Vs-")
        if info[0] < 10 or info[1] < 10:
            mislabel_G.add_edge(sn1, sn2)
        else:
            mislabel_G.add_node(sn1)
            mislabel_G.add_node(sn2)
    comps = list(nx.connected_components(mislabel_G))
    if not comps:
        return [], []
    normal_sns = max(comps, key=len)
    problem_sns = [sn for c in comps if c != normal_sns for sn in c]
    return list(problem_sns), list(normal_sns)

def get_sn_distance(bray, sn, pat, suffix):
    """
    Compute mean rank for sample 'sn' to all samples of patient 'pat' (with matching suffix).
    """
    ix = ~bray.index.str.contains(suffix) | (bray.index == sn)
    ranks = bray.loc[ix, sn].rank()
    match_ix = ranks.index.str.contains(pat)
    if not match_ix.any():
        return 1000
    return ranks[match_ix].mean()

def get_average_distance(bray, sn, pat_list, suffix):
    """
    For one problematic sample, compute mean distance to each patient.
    """
    return [(pat, get_sn_distance(bray, sn, pat, suffix)) for pat in pat_list]

def identify_close_patient(bray, sns, pat_list, suffix):
    """
    For a set of normal samples, find the closest patients (mean distance).
    """
    ranks = []
    for pat in pat_list:
        query_sn = pat + suffix
        if query_sn not in bray.index:
            continue
        rank = bray.loc[bray.index.str.endswith(suffix) | bray.index.isin(sns), query_sn].rank()
        ranks.append((pat, rank.loc[sns].mean()))
    return ranks

def get_true_partners(bray, meta, potential_artifacts):
    """
    For each problematic sample, suggest most likely true patient partners.
    Returns:
        (potential_partners, partners_pat): List[List[Any]], List[List[Any]]
    """
    pat_list = set(meta['patient'])
    potential_partners, partners_pat = [], []
    for art in potential_artifacts:
        prob_sns, normal_sns = identify_problemic_sn(art)
        for sn in prob_sns:
            suffix = "_" + sn.split("_")[-1]
            ranks = get_average_distance(bray, sn, pat_list, suffix)
            sorted_ranks = sorted(ranks, key=lambda x: x[1])[:5]
            potential_partners.append([sn] + sorted_ranks)
            close_pat = identify_close_patient(bray, normal_sns, pat_list, suffix)
            sorted_pat = sorted(close_pat, key=lambda x: x[1])[:5]
            partners_pat.append([tuple(normal_sns)] + sorted_pat)
    return potential_partners, partners_pat


# ========== Stage 3: Strain-level identity confirmation ==========

def normalize_sample_id(value, suffix=".fastq"):
    """Normalize a sample ID by stripping whitespace and a terminal suffix."""
    value = str(value).strip()
    return value[:-len(suffix)] if suffix and value.endswith(suffix) else value


def get_patient_id(sample_id, separator="_"):
    """Derive the patient ID from the first field of a sample ID."""
    return sample_id.split(separator, 1)[0]


def load_stage3_queries(path, strip_suffix=".fastq", patient_separator="_"):
    """Load abnormal samples and optional candidate-patient assignments."""
    queries = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "sample_id" not in queries.columns:
        raise ValueError("Stage 3 query TSV must contain a 'sample_id' column")

    queries["sample_id"] = queries["sample_id"].map(
        lambda value: normalize_sample_id(value, strip_suffix)
    )
    if "recorded_patient" not in queries.columns:
        queries["recorded_patient"] = ""
    if "candidate_patient" not in queries.columns:
        queries["candidate_patient"] = ""

    missing = queries["recorded_patient"].eq("")
    queries.loc[missing, "recorded_patient"] = queries.loc[
        missing, "sample_id"
    ].map(lambda value: get_patient_id(value, patient_separator))

    if queries["sample_id"].duplicated().any():
        duplicated = queries.loc[
            queries["sample_id"].duplicated(keep=False), "sample_id"
        ].unique().tolist()
        raise ValueError(f"Duplicate Stage 3 query sample IDs: {duplicated}")
    return queries


def load_mutation_matrix(path, strip_suffix=".fastq"):
    """Read, normalize, validate, and numerically coerce one mutation matrix."""
    matrix = pd.read_csv(path, sep="\t", index_col="ids")
    matrix.index = [normalize_sample_id(value, strip_suffix) for value in matrix.index]
    matrix.columns = [normalize_sample_id(value, strip_suffix) for value in matrix.columns]

    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("duplicate sample IDs after normalization")

    shared = matrix.index.intersection(matrix.columns, sort=False)
    if shared.empty:
        raise ValueError("row and column sample IDs do not overlap")

    return matrix.loc[shared, shared].apply(pd.to_numeric, errors="coerce")


def summarize_mutation_query(
        matrix, query_sample, recorded_patient, candidate_patient, sgb,
        source_file, patient_separator="_", scale=1000.0,
        min_recorded_samples=3, include_background=False):
    """Extract mutation-rate comparisons for one abnormal sample and SGB."""
    names = matrix.index.to_numpy(dtype=str)
    patients = np.array([
        get_patient_id(name, patient_separator) for name in names
    ])

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
    selected_patients_array = np.array([
        get_patient_id(name, patient_separator) for name in selected_names
    ])

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

    return pd.DataFrame({
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
        "source_file": os.path.abspath(source_file),
    })


def run_stage3(
        mutation_glob, query_path, output_path, min_recorded_samples=3,
        scale=1000.0, patient_separator="_", strip_suffix=".fastq",
        include_background=False):
    """Summarize StrainPhlAn mutation rates across all matching SGB files."""
    queries = load_stage3_queries(query_path, strip_suffix, patient_separator)
    mutation_files = sorted(glob.glob(mutation_glob))
    if not mutation_files:
        raise FileNotFoundError(f"No mutation files matched: {mutation_glob}")

    outputs = []
    skipped = 0
    for mutation_file in mutation_files:
        filename = Path(mutation_file).name
        sgb = filename[:-len(".mutation")] if filename.endswith(".mutation") else filename
        try:
            matrix = load_mutation_matrix(mutation_file, strip_suffix)
        except (OSError, ValueError, KeyError) as error:
            skipped += 1
            logger.warning("Skipping invalid mutation file %s: %s", mutation_file, error)
            continue

        available = set(matrix.index)
        for query in queries.itertuples(index=False):
            if query.sample_id not in available:
                continue
            summary = summarize_mutation_query(
                matrix=matrix,
                query_sample=query.sample_id,
                recorded_patient=query.recorded_patient,
                candidate_patient=query.candidate_patient,
                sgb=sgb,
                source_file=mutation_file,
                patient_separator=patient_separator,
                scale=scale,
                min_recorded_samples=min_recorded_samples,
                include_background=include_background,
            )
            if not summary.empty:
                outputs.append(summary)

    columns = [
        "query_sample", "recorded_patient", "candidate_patient", "sample1",
        "sample2", "patient1", "patient2", "comparison_type",
        "involves_query", "mutation_rate", "mutation_rate_scaled", "SGB",
        "source_file",
    ]
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=columns)
    result.to_csv(output_path, sep="\t", index=False)
    logger.info(
        "Stage 3 finished: %d comparisons from %d mutation files (%d skipped) -> %s",
        len(result), len(mutation_files), skipped, output_path,
    )
    return result


def plot_stage3_boxplots(result, output_dir, image_format="png"):
    """Plot log10 mutation rates for recorded and candidate comparisons."""
    import matplotlib.pyplot as plt

    if result.empty:
        logger.warning("Stage 3 result is empty; no boxplots were generated")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for query_sample, query_data in result.groupby("query_sample", sort=True):
        query_data = query_data[query_data["involves_query"]].copy()
        if query_data.empty:
            continue

        query_is_sample1 = query_data["sample1"].eq(query_sample)
        query_data["partner_patient"] = np.where(
            query_is_sample1, query_data["patient2"], query_data["patient1"]
        )

        recorded_patient = str(query_data["recorded_patient"].iloc[0])
        candidate_patient = str(query_data["candidate_patient"].iloc[0])
        groups = []
        labels = []

        recorded_values = query_data.loc[
            query_data["partner_patient"].eq(recorded_patient),
            "mutation_rate_scaled",
        ].dropna().to_numpy()
        recorded_values = recorded_values[recorded_values > 0]
        if recorded_values.size:
            groups.append(np.log10(recorded_values))
            labels.append(f"Recorded\n{recorded_patient}\n(n={recorded_values.size})")

        if candidate_patient:
            candidate_values = query_data.loc[
                query_data["partner_patient"].eq(candidate_patient),
                "mutation_rate_scaled",
            ].dropna().to_numpy()
            candidate_values = candidate_values[candidate_values > 0]
            if candidate_values.size:
                groups.append(np.log10(candidate_values))
                labels.append(f"Candidate\n{candidate_patient}\n(n={candidate_values.size})")

        if not groups:
            logger.warning("No plottable Stage 3 values for %s", query_sample)
            continue

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        box = ax.boxplot(
            groups,
            patch_artist=True,
            showfliers=False,
            widths=0.55,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        colors = ["#4C78A8", "#F58518"]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        for position, values in enumerate(groups, start=1):
            offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                position + offsets,
                values,
                s=22,
                color="#333333",
                alpha=0.65,
                zorder=3,
            )

        threshold = np.log10(0.1)
        ax.axhline(
            threshold,
            color="#D62728",
            linestyle="--",
            linewidth=1.4,
            label="0.1 mutations per 1,000 sites",
        )
        ax.set_title(f"Strain mutation rates for {query_sample}")
        ax.set_ylabel("log10(mutations per 1,000 aligned positions)")
        ax.legend(frameon=False, loc="best")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in query_sample
        )
        output_path = output_dir / f"{safe_name}_mutation_rates.{image_format}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(str(output_path))

    logger.info("Stage 3 boxplots written: %d -> %s", len(written), output_dir)
    return written


# ========== Main Pipeline ==========

def run_pipeline(profile_path, meta_path, suffix, cutoff):
    """
    Main workflow for mislabel/duplicate QC analysis.
    """
    import pandas as pd
    import pickle
    logger.info("Loading data...")
    df = pd.read_csv(profile_path, sep="\t", index_col=0)
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)
    assert df.shape[1] == meta.shape[0], "Sample number mismatch"
    logger.info("Data loaded: %d features x %d samples", df.shape[0], df.shape[1])

    # Ensure sample order in meta and profile match
    df = df.loc[:, meta.index]

    batch_flags = meta['batch'].apply(lambda x: "_" + x.split("_")[-1]).unique().tolist()
    logger.info("Batches: %s", batch_flags)
    logger.info("Patients: %d, Samples: %d", meta['patient'].nunique(), meta.shape[0])

    logger.info("Computing Bray-Curtis distance matrix...")
    bray = compute_bray_curtis(df)

    if len(batch_flags) == 1:
        logger.info("Only one batch: running cheating sample detection only.")
        dup_samples = get_duplicate_samples(bray, meta, batch_flags, cutoff)
        cheating_sns = find_cheating_samples(dup_samples)
        with open(f"{suffix}_cheating_sns.txt", "w") as fo:
            for batch, pairs in dup_samples.items():
                fo.write(f"cheating in {batch} group\n")
                import networkx as nx
                for comp in nx.connected_components(nx.Graph(pairs)):
                    fo.write(" == ".join(comp) + "\n")
                fo.write("###############################\n")
        logger.info("Cheating samples written: %d", len(cheating_sns))
        return

    # Step 1: Find mislabeled/problematic individuals
    logger.info("Step1: Detecting problematic patients (mislabeled individuals)...")
    prob_pats = get_problematic_patients(bray, meta)
    with open(f"{suffix}_mislabeled_sns.txt", "w") as fo:
        for pat, ranks in prob_pats:
            info = [f"{r1}|{r2}|{n}|{name}" for (r1, r2, n, name) in ranks]
            fo.write(f"{pat}\t{','.join(info)}\n")
    logger.info("Problematic patients found: %d", len(prob_pats))

    # Step 2: Cheating/duplicate sample detection
    logger.info("Step2: Detecting cheating samples...")
    dup_samples = get_duplicate_samples(bray, meta, batch_flags, cutoff)
    cheating_sns = find_cheating_samples(dup_samples)
    with open(f"{suffix}_cheating_sns.txt", "w") as fo:
        for batch, pairs in dup_samples.items():
            fo.write(f"cheating in {batch} group\n")
            import networkx as nx
            for comp in nx.connected_components(nx.Graph(pairs)):
                fo.write(" == ".join(comp) + "\n")
            fo.write("###############################\n")
    logger.info("Cheating samples written: %d", len(cheating_sns))

    # Step 3: Suggest true partners for mislabels
    logger.info("Step3: Identifying true partners for mislabels...")
    true_partners, partners_pat = get_true_partners(bray, meta, prob_pats)
    with open(f"{suffix}_true_labels.txt", "w") as fo:
        fo.write("mislabelled_sn\tnormal_sn\n")
        for k, v in zip(true_partners, partners_pat):
            info1 = str(k[0]) + "," + ",".join([f"{i[0]}|{i[1]}" for i in k[1:]])
            info2 = ("none," if len(v[0]) == 0 else ";".join(v[0]) + ",") + ",".join([f"{i[0]}|{i[1]}" for i in v[1:]])
            fo.write(f"{info1}\t{info2}\n")
    with open(f"{suffix}.pkl", "wb") as f:
        pickle.dump([prob_pats, true_partners, partners_pat], f)
    logger.info("Pipeline finished successfully.")


# ========== CLI Entrypoint ==========
def parse_args():
    parser = argparse.ArgumentParser(
        description="Find abnormal samples in longitudinal metagenomes",
        epilog="""
Example: 
    python main.py abundance.tsv meta.tsv -s results

Input File Formats:
-------------------

* abundance.tsv  (species/OTU abundance or count matrix; rows=features, columns=samples)
Example:
    \tsample1\tsample2\tsample3
    s__A\t2.5\t4.2\t0.0
    s__B\t0.0\t1.3\t0.9
    s__C\t1.1\t0.0\t2.2

* meta.tsv  (sample metadata; index=samples)
Example:
    sample_id\tpatient\tbatch
    sample1\tPAT01\tbatch_0
    sample2\tPAT01\tbatch_1
    sample3\tPAT02\tbatch_1

Note:
- The columns of abundance.tsv must match (in order and names) the sample_id of meta.tsv.
- 'batch' may be any string, typically ending in '_0', '_1', etc.
- 'patient' is the subject or individual identifier.

Outputs:
    results_mislabeled_sns.txt
    results_cheating_sns.txt
    results_true_labels.txt
    results.pkl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("metagenome_profiles", nargs="?", help="Species-level profile table (tsv, rows=features, columns=samples)")
    parser.add_argument("meta_ci", nargs="?", help="Metadata table (tsv, index=samples, columns: patient, batch)")
    parser.add_argument("-s", "--suffix", default="results", help="Suffix for output files")
    parser.add_argument("-c", "--cutoff", type=float, default=0.3, help="Distance cutoff for cheating/duplicate detection (default: 0.3)")
    parser.add_argument("--stage3-only", action="store_true", help="Skip Stages 1-2 and run only Stage 3")
    parser.add_argument("--stage3-mutation-glob", help="Quoted glob matching StrainPhlAn .mutation files")
    parser.add_argument("--stage3-queries", help="TSV with sample_id and optional recorded_patient and candidate_patient columns")
    parser.add_argument("--stage3-output", help="Stage 3 output TSV (default: <suffix>_stage3_mutation_rates.tsv)")
    parser.add_argument("--stage3-min-recorded-samples", type=int, default=3, help="Minimum recorded-patient samples per SGB (default: 3)")
    parser.add_argument("--stage3-scale", type=float, default=1000.0, help="Mutation-rate multiplier (default: 1000)")
    parser.add_argument("--stage3-patient-separator", default="_", help="Sample/patient separator (default: _)")
    parser.add_argument("--stage3-strip-suffix", default=".fastq", help="Terminal sample suffix to remove (default: .fastq)")
    parser.add_argument("--stage3-include-background", action="store_true", help="Include pairs that do not contain the query sample")
    parser.add_argument("--stage3-plot-dir", help="Directory for per-sample Stage 3 boxplots (default: <suffix>_stage3_boxplots)")
    parser.add_argument("--stage3-plot-format", choices=("png", "pdf", "svg"), default="png", help="Stage 3 boxplot format (default: png)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        if not args.stage3_only:
            if not args.metagenome_profiles or not args.meta_ci:
                raise ValueError(
                    "metagenome_profiles and meta_ci are required unless --stage3-only is used"
                )
            run_pipeline(args.metagenome_profiles, args.meta_ci, args.suffix, args.cutoff)

        stage3_requested = bool(args.stage3_mutation_glob or args.stage3_queries)
        if stage3_requested:
            if not args.stage3_mutation_glob or not args.stage3_queries:
                raise ValueError(
                    "--stage3-mutation-glob and --stage3-queries must be supplied together"
                )
            stage3_output = args.stage3_output or f"{args.suffix}_stage3_mutation_rates.tsv"
            stage3_result = run_stage3(
                mutation_glob=args.stage3_mutation_glob,
                query_path=args.stage3_queries,
                output_path=stage3_output,
                min_recorded_samples=args.stage3_min_recorded_samples,
                scale=args.stage3_scale,
                patient_separator=args.stage3_patient_separator,
                strip_suffix=args.stage3_strip_suffix,
                include_background=args.stage3_include_background,
            )
            stage3_plot_dir = args.stage3_plot_dir or f"{args.suffix}_stage3_boxplots"
            plot_stage3_boxplots(
                stage3_result,
                output_dir=stage3_plot_dir,
                image_format=args.stage3_plot_format,
            )
        elif args.stage3_only:
            raise ValueError(
                "--stage3-only requires --stage3-mutation-glob and --stage3-queries"
            )
    except Exception as e:
        logger.exception("Fatal error: %s", str(e))
        sys.exit(1)
