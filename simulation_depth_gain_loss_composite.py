from Bio import Phylo
import pandas as pd
import numpy as np
import random
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
from synteny_tools import findSyntenyReal1, findSyntenyReal2
import random
import time


def sample_truncated_powerlaw(L0, alpha=2.0):
    k_max = max(1, L0 // 2)
    ks = np.arange(1, k_max + 1)
    weights = ks ** (-alpha)
    weights /= weights.sum()
    return int(np.random.choice(ks, p=weights))


def gene_numeric_id(gene):
    if isinstance(gene, int):
        return gene
    if not isinstance(gene, str):
        return None
    if "." in gene:
        gene = gene.split(".", 1)[1]
    try:
        return int(gene)
    except ValueError:
        return None


def build_core_gene_set(root_genome, core_fraction):
    """Deterministically mark distributed ancestral neighborhoods as conserved/core."""
    if core_fraction <= 0:
        return set()
    core_fraction = min(max(core_fraction, 0.0), 1.0)
    n_core = int(round(len(root_genome) * core_fraction))
    core = set()
    if core_fraction >= 1.0:
        for gene in root_genome:
            gid = gene_numeric_id(gene)
            if gid is not None:
                core.add(gid)
        return core
    block_size = 20
    gap_size = max(1, int(round(block_size * (1.0 - core_fraction) / core_fraction)))
    pos = 0
    while len(core) < n_core and pos < len(root_genome):
        for gene in root_genome[pos:pos + block_size]:
            if len(core) >= n_core:
                break
            gid = gene_numeric_id(gene)
            if gid is not None:
                core.add(gid)
        pos += block_size + gap_size
    return core


def is_core_gene(gene, core_gene_ids):
    gid = gene_numeric_id(gene)
    return gid in core_gene_ids if gid is not None else False


def block_core_fraction(genome, idxs, core_gene_ids):
    if not idxs or not core_gene_ids:
        return 0.0
    return sum(1 for idx in idxs if is_core_gene(genome[idx], core_gene_ids)) / len(idxs)


def should_reject_core_event(core_fraction_in_block, core_protection):
    if core_fraction_in_block <= 0 or core_protection <= 0:
        return False
    return random.random() < min(1.0, core_fraction_in_block * core_protection)


def evolve_genome_branch(
    genome, branch_length, gain_rate, loss_rate, inv_rate, trans_rate,
    gain_exp, loss_exp, inv_exp, trans_exp, next_gene_id_holder, L0_ancestral,
    core_gene_ids=None, core_protection=0.0
):
    genome = genome.copy()
    core_gene_ids = core_gene_ids or set()

    gains = np.random.poisson(gain_rate * branch_length)
    losses = np.random.poisson(loss_rate * branch_length)
    inversions = np.random.poisson(inv_rate * branch_length)
    translocations = np.random.poisson(trans_rate * branch_length)

    events = (["G"] * gains +
              ["L"] * losses +
              ["I"] * inversions +
              ["T"] * translocations)
    random.shuffle(events)

    for event in events:
        Lc = len(genome)
        loc_point = random.randrange(Lc)

        if event == "G":
            k = sample_truncated_powerlaw(L0_ancestral, gain_exp)
        elif event == "L":
            k = sample_truncated_powerlaw(L0_ancestral, loss_exp)
        elif event == "I":
            k = sample_truncated_powerlaw(L0_ancestral, inv_exp)
        else:
            k = sample_truncated_powerlaw(L0_ancestral, trans_exp)

        k = max(1, min(k, Lc))

        if event == "G":
            insert_pos = (loc_point + 1) % (Lc + 1)
            if core_gene_ids and len(genome) > 0:
                left = loc_point % len(genome)
                right = insert_pos % len(genome)
                neighboring_core = (
                    is_core_gene(genome[left], core_gene_ids)
                    or is_core_gene(genome[right], core_gene_ids)
                )
                if neighboring_core and random.random() < core_protection:
                    continue
            block = []
            for _ in range(k):
                new_gene = f"cog.{next_gene_id_holder[0]}"
                next_gene_id_holder[0] += 1
                block.append(new_gene)
            genome[insert_pos:insert_pos] = block

        elif event == "L":
            idxs = [(loc_point + t) % len(genome) for t in range(k)]
            if should_reject_core_event(
                block_core_fraction(genome, idxs, core_gene_ids),
                core_protection,
            ):
                continue
            for t in range(k):
                pos = (loc_point + t) % len(genome)
                genome[pos] = -1

        elif event == "I":
            if k > 1 and len(genome) >= 2:
                idxs = [(loc_point + t) % len(genome) for t in range(k)]
                if should_reject_core_event(
                    block_core_fraction(genome, idxs, core_gene_ids),
                    core_protection,
                ):
                    continue
                vals = [genome[p] for p in idxs][::-1]
                for p, v in zip(idxs, vals):
                    genome[p] = v

        elif event == "T":
            if len(genome) >= 2 and 0 < k < len(genome):
                Lc_now = len(genome)
                idxs = [(loc_point + t) % Lc_now for t in range(k)]
                if should_reject_core_event(
                    block_core_fraction(genome, idxs, core_gene_ids),
                    core_protection,
                ):
                    continue
                block_vals = [genome[p] for p in idxs]
                j = random.randrange(Lc_now)
                if j in idxs:
                    continue
                removed = set(idxs)
                remaining = [g for t, g in enumerate(genome) if t not in removed]
                r_before = sum(1 for t in idxs if t < j)
                j_new = j - r_before
                remaining[j_new + 1:j_new + 1] = block_vals
                genome = remaining

    return genome


def median_root_to_leaf_lengths(tree):
    return np.median([tree.distance(tree.root, leaf) for leaf in tree.get_terminals()])


def branch_depth_to_nearest_leaf(tree, child, branch_length):
    """Depth from branch distal end to nearest descendant leaf plus half branch length."""
    descendant_leaves = child.get_terminals()
    if not descendant_leaves:
        distal_to_leaf = 0.0
    else:
        distal_to_leaf = min(tree.distance(child, leaf) for leaf in descendant_leaves)
    return distal_to_leaf + 0.5 * (branch_length or 0.0)


def depth_dependent_gain_loss_multiplier(depth, scale=2700.0, decay=3300.0):
    return 1.0 + scale * np.exp(-decay * depth)


def evolve_genome(
    tree, root_genome, per_gene_gain_rate, per_gene_loss_rate,
    per_gene_inv_rate, per_gene_trans_rate, gain_exp, loss_exp, inv_exp,
    trans_exp, next_gene_id_holder, core_fraction=0.0, core_protection=0.0,
    terminal_rate_multiplier=1.0, internal_rate_multiplier=1.0,
    use_depth_dependent_gain_loss=False,
    depth_gain_loss_scale=2700.0,
    depth_gain_loss_decay=3300.0,
):
    genomes = {}
    num_genes = len(root_genome)
    median_path_length = median_root_to_leaf_lengths(tree)

    branch_gain_rate = (num_genes * per_gene_gain_rate) / median_path_length
    branch_loss_rate = (num_genes * per_gene_loss_rate) / median_path_length
    branch_inv_rate = (num_genes * per_gene_inv_rate) / median_path_length
    branch_trans_rate = (num_genes * per_gene_trans_rate) / median_path_length

    L0_ancestral = len(root_genome)
    core_gene_ids = build_core_gene_set(root_genome, core_fraction)

    genomes[tree.root] = root_genome.copy()
    for parent in tree.find_clades(order="level"):
        for child in parent.clades:
            parent_genome = genomes[parent]
            rate_multiplier = (
                terminal_rate_multiplier if child.is_terminal()
                else internal_rate_multiplier
            )
            gain_loss_multiplier = rate_multiplier
            if use_depth_dependent_gain_loss:
                branch_depth = branch_depth_to_nearest_leaf(
                    tree, child, child.branch_length
                )
                gain_loss_multiplier *= depth_dependent_gain_loss_multiplier(
                    branch_depth,
                    scale=depth_gain_loss_scale,
                    decay=depth_gain_loss_decay,
                )
            child_genome = evolve_genome_branch(
                parent_genome, child.branch_length,
                branch_gain_rate * gain_loss_multiplier,
                branch_loss_rate * gain_loss_multiplier,
                branch_inv_rate * rate_multiplier,
                branch_trans_rate * rate_multiplier,
                gain_exp, loss_exp, inv_exp, trans_exp,
                next_gene_id_holder,
                L0_ancestral,
                core_gene_ids=core_gene_ids,
                core_protection=core_protection
            )
            genomes[child] = child_genome
    return genomes


# def synteny_blocks(genome1, genome2):
#     pos_in_B = {gene: idx for idx, gene in enumerate(genome2)}
#     blocks = []
#     i = 0
#     while i < len(genome1):
#         gene = genome1[i]
#         if gene in pos_in_B:
#             j = pos_in_B[gene]
#             block_len = 1
#             while i + block_len < len(genome1):
#                 next_gene = genome1[i + block_len]
#                 if next_gene in pos_in_B and pos_in_B[next_gene] == j + block_len:
#                     block_len += 1
#                 else:
#                     break
#             blocks.append(block_len)
#             i += block_len
#         else:
#             i += 1
#     return blocks

def get_real_genomes_from_cc(path):
    df = pd.read_csv(path, names=[
        "gene_ID", "genome_ID", "protein_ID", "protein_length",
        "atgc_cog_footprint", "atgc_cog_footprint_length",
        "atgc_cog_ID", "protein_cluster_ID", "match_class"
    ])
    df = df.dropna(subset=["atgc_cog_ID"])
    return df.groupby("genome_ID")["atgc_cog_ID"].apply(list).to_dict()


def run_simulation(
    tree, root_genome,
    per_gene_gain_rate, per_gene_loss_rate, per_gene_inv_rate,
    per_gene_trans_rate=0.0,
    gain_exp=2.0, loss_exp=2.0, inv_exp=2.0, trans_exp=2.0,
    core_fraction=0.0, core_protection=0.0,
    terminal_rate_multiplier=1.0, internal_rate_multiplier=1.0,
    use_depth_dependent_gain_loss=False,
    depth_gain_loss_scale=2700.0,
    depth_gain_loss_decay=3300.0,
):
    next_gene_id_holder = [len(root_genome) + 1]

    genomes = evolve_genome(
        tree, root_genome,
        per_gene_gain_rate, per_gene_loss_rate, per_gene_inv_rate, per_gene_trans_rate,
        gain_exp, loss_exp, inv_exp, trans_exp,
        next_gene_id_holder,
        core_fraction=core_fraction,
        core_protection=core_protection,
        terminal_rate_multiplier=terminal_rate_multiplier,
        internal_rate_multiplier=internal_rate_multiplier,
        use_depth_dependent_gain_loss=use_depth_dependent_gain_loss,
        depth_gain_loss_scale=depth_gain_loss_scale,
        depth_gain_loss_decay=depth_gain_loss_decay,
    )

    leaves = tree.get_terminals()

    pairwise_blocks = {}

    start_time = time.time()

    for leaf1, leaf2 in combinations(leaves, 2):
        g1 = genomes[leaf1].copy()
        g2 = genomes[leaf2].copy()
        # start_time = time.time()
        # blocks1 = findSyntenyReal1(g1.copy(), g2.copy())
        # print("first algo:" + str(time.time() - start_time))
        # start_time = time.time()
        blocks = findSyntenyReal2(g1.copy(), g2.copy())
        #print("second algo:" + str(time.time() - start_time))
        block_lengths = [abs(b[4]) for b in blocks if len(b) > 4]
        pairwise_blocks[(leaf1.name, leaf2.name)] = block_lengths


    return pairwise_blocks

# optional plot
# start_time = time.time()
# sim_pairs = run_simulation(tree, root_genome, per_gene_gain_rate=0.1, per_gene_loss_rate=0.1, per_gene_inv_rate=0.0)
# end_time = time.time()
# print(end_time - start_time)
# all_sim_lengths = [length for blocks in sim_pairs.values() for length in blocks]
# plt.figure(figsize=(10, 5))
# plt.hist(all_sim_lengths, bins=range(1, max(all_sim_lengths)+2), color="orange", edgecolor="black", alpha=0.7)
# plt.title("Distribution of Simulated Synteny Block Lengths")
# plt.xlabel("Synteny Block Length")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()
