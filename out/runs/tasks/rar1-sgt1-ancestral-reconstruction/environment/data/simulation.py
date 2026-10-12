import random
import numpy as np
from Bio.Phylo.BaseTree import Tree, Clade
from io import StringIO

SEED = 42
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_to_idx = {aa: i for i, aa in enumerate(AA)}

RAR1_LEN = 150
SGT1_LEN = 200
RAR1_INTERFACE = 75
SGT1_INTERFACES = [100, 130, 160]

root_interface = {'RAR1_75': 'D', 'SGT1_100': 'K', 'SGT1_130': 'R', 'SGT1_160': 'Y'}
fern_ancestor_interface = {'RAR1_75': 'D', 'SGT1_100': 'K', 'SGT1_130': 'S', 'SGT1_160': 'Y'}
core_fern_ancestor_interface = {'RAR1_75': 'K', 'SGT1_100': 'E', 'SGT1_130': 'S', 'SGT1_160': 'Y'}

def binds(rar1_75, sgt1_100, sgt1_130, sgt1_160):
    if sgt1_160 != 'Y':
        return False
    if sgt1_130 == 'R':
        return rar1_75 == 'D'
    elif sgt1_130 == 'S':
        if sgt1_100 == 'E':
            return rar1_75 == 'K'
        else:
            return rar1_75 in ['D', 'K']
    else:
        return False

def generate_tree():
    seed_leaves = [Clade(name=f"Seed_{i:02d}", branch_length=0.1) for i in range(1, 21)]
    fern_leaves = [Clade(name=f"Fern_{i:02d}", branch_length=0.1) for i in range(1, 21)]
    
    seed_clade = Clade(name="SeedClade", branch_length=0.1)
    seed_clade.clades = seed_leaves
    
    early_ferns = fern_leaves[:5]
    core_ferns = fern_leaves[5:]
    
    early_clade = Clade(name="EarlyFernClade", branch_length=0.1)
    early_clade.clades = early_ferns
    
    core_ancestor = Clade(name="CoreFernAncestor", branch_length=0.1)
    core_ancestor.clades = core_ferns
    
    fern_ancestor = Clade(name="FernAncestor", branch_length=0.1)
    fern_ancestor.clades = [early_clade, core_ancestor]
    
    root = Clade(name="Root", branch_length=0.0)
    root.clades = [seed_clade, fern_ancestor]
    
    tree = Tree(root=root)
    
    for clade in tree.find_clades():
        if clade.name != 'Root':
            clade.branch_length = random.uniform(0.05, 0.3)
    return tree

def simulate_sequences(tree):
    random.seed(SEED)
    np.random.seed(SEED)
    
    root_seq_rar1 = [random.choice(AA) for _ in range(RAR1_LEN)]
    root_seq_sgt1 = [random.choice(AA) for _ in range(SGT1_LEN)]
    
    for pos, aa in root_interface.items():
        if pos.startswith('RAR1'):
            idx = int(pos.split('_')[1]) - 1
            root_seq_rar1[idx] = aa
        else:
            idx = int(pos.split('_')[1]) - 1
            root_seq_sgt1[idx] = aa
    
    node_seqs = {}
    
    def assign_seq(clade, parent_rar1, parent_sgt1):
        if clade.name == 'Root':
            rar1 = root_seq_rar1[:]
            sgt1 = root_seq_sgt1[:]
        elif clade.name == 'SeedClade':
            rar1 = parent_rar1[:]
            sgt1 = parent_sgt1[:]
            for pos, aa in root_interface.items():
                if pos.startswith('RAR1'):
                    idx = int(pos.split('_')[1]) - 1
                    rar1[idx] = aa
                else:
                    idx = int(pos.split('_')[1]) - 1
                    sgt1[idx] = aa
        elif clade.name == 'FernAncestor':
            rar1 = parent_rar1[:]
            sgt1 = parent_sgt1[:]
            for pos, aa in fern_ancestor_interface.items():
                if pos.startswith('RAR1'):
                    idx = int(pos.split('_')[1]) - 1
                    rar1[idx] = aa
                else:
                    idx = int(pos.split('_')[1]) - 1
                    sgt1[idx] = aa
        elif clade.name == 'EarlyFernClade':
            rar1 = parent_rar1[:]
            sgt1 = parent_sgt1[:]
            for pos, aa in fern_ancestor_interface.items():
                if pos.startswith('RAR1'):
                    idx = int(pos.split('_')[1]) - 1
                    rar1[idx] = aa
                else:
                    idx = int(pos.split('_')[1]) - 1
                    sgt1[idx] = aa
        elif clade.name == 'CoreFernAncestor':
            rar1 = parent_rar1[:]
            sgt1 = parent_sgt1[:]
            for pos, aa in core_fern_ancestor_interface.items():
                if pos.startswith('RAR1'):
                    idx = int(pos.split('_')[1]) - 1
                    rar1[idx] = aa
                else:
                    idx = int(pos.split('_')[1]) - 1
                    sgt1[idx] = aa
        else:
            rar1 = parent_rar1[:]
            sgt1 = parent_sgt1[:]
        
        rate = 0.5
        if clade.branch_length is not None and clade.branch_length > 0:
            for i in range(RAR1_LEN):
                if i == RAR1_INTERFACE - 1:
                    continue
                if random.random() < 1 - np.exp(-rate * clade.branch_length):
                    rar1[i] = random.choice(AA)
            for i in range(SGT1_LEN):
                if i + 1 in SGT1_INTERFACES:
                    continue
                if random.random() < 1 - np.exp(-rate * clade.branch_length):
                    sgt1[i] = random.choice(AA)
        
        if clade.is_terminal():
            for pos in list(root_interface.keys()):
                if random.random() < 0.1:
                    if pos.startswith('RAR1'):
                        idx = int(pos.split('_')[1]) - 1
                        rar1[idx] = random.choice(AA)
                    else:
                        idx = int(pos.split('_')[1]) - 1
                        sgt1[idx] = random.choice(AA)
        
        node_seqs[clade] = (rar1, sgt1)
        
        for child in clade.clades:
            assign_seq(child, rar1, sgt1)
    
    assign_seq(tree.root, None, None)
    return node_seqs

def get_interface_states(seq_rar1, seq_sgt1):
    states = {}
    states['RAR1_75'] = seq_rar1[RAR1_INTERFACE - 1]
    states['SGT1_100'] = seq_sgt1[SGT1_INTERFACES[0] - 1]
    states['SGT1_130'] = seq_sgt1[SGT1_INTERFACES[1] - 1]
    states['SGT1_160'] = seq_sgt1[SGT1_INTERFACES[2] - 1]
    return states

def get_binding(states):
    return binds(states['RAR1_75'], states['SGT1_100'], states['SGT1_130'], states['SGT1_160'])
