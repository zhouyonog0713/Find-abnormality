#!/usr/bin/env python3
"""
Find abnormal/mislabeled samples in longitudinal metagenome data.

Usage:
    python main.py metagenome_profiles.tsv meta_ci.tsv [-s SUFFIX] [-c CUTOFF]

Author: ChatGPT (2025-07-29)
"""

import argparse
import sys
import logging

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
    parser.add_argument("metagenome_profiles", help="Species-level profile table (tsv, rows=features, columns=samples)")
    parser.add_argument("meta_ci", help="Metadata table (tsv, index=samples, columns: patient, batch)")
    parser.add_argument("-s", "--suffix", default="results", help="Suffix for output files")
    parser.add_argument("-c", "--cutoff", type=float, default=0.3, help="Distance cutoff for cheating/duplicate detection (default: 0.3)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(args.metagenome_profiles, args.meta_ci, args.suffix, args.cutoff)
    except Exception as e:
        logger.exception("Fatal error: %s", str(e))
        sys.exit(1)

