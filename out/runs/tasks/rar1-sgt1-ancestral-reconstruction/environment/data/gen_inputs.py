import random
import os
import Bio.Phylo
from simulation import generate_tree, simulate_sequences, get_interface_states, get_binding, binds

def main():
    random.seed(42)
    tree = generate_tree()
    node_seqs = simulate_sequences(tree)
    
    with open('/app/data/species_tree.nwk', 'w') as f:
        Bio.Phylo.write(tree, f, 'newick')
    
    with open('/app/data/rar1_alignment.fasta', 'w') as f:
        for clade in tree.get_terminals():
            seq, _ = node_seqs[clade]
            f.write(f">{clade.name}\n")
            f.write(''.join(seq) + "\n")
    
    with open('/app/data/sgt1_alignment.fasta', 'w') as f:
        for clade in tree.get_terminals():
            _, seq = node_seqs[clade]
            f.write(f">{clade.name}\n")
            f.write(''.join(seq) + "\n")
    
    with open('/app/data/interface_positions.tsv', 'w') as f:
        f.write("protein\tposition\tdescription\n")
        f.write("RAR1\t75\tkey residue\n")
        f.write("SGT1\t100\tlock residue 1\n")
        f.write("SGT1\t130\tlock residue 2\n")
        f.write("SGT1\t160\tlock residue 3\n")
    
    terminals = tree.get_terminals()
    species_states = {}
    for clade in terminals:
        rar1, sgt1 = node_seqs[clade]
        states = get_interface_states(rar1, sgt1)
        species_states[clade.name] = states
    
    pairs = []
    pairs.append(("Seed_01", "Seed_01"))
    pairs.append(("Seed_05", "Seed_05"))
    pairs.append(("Fern_01", "Fern_01"))
    pairs.append(("Fern_02", "Fern_02"))
    pairs.append(("Fern_10", "Fern_10"))
    pairs.append(("Fern_15", "Fern_15"))
    pairs.append(("Seed_01", "Fern_01"))
    pairs.append(("Fern_01", "Seed_01"))
    pairs.append(("Fern_10", "Seed_01"))
    pairs.append(("Seed_01", "Fern_10"))
    pairs.append(("Fern_10", "Fern_01"))
    pairs.append(("Fern_01", "Fern_10"))
    pairs.append(("Seed_01", "Fern_18"))
    pairs.append(("Fern_18", "Seed_01"))
    pairs.append(("Seed_10", "Fern_05"))
    
    with open('/app/data/binding_assays.tsv', 'w') as f:
        f.write("species_A\tspecies_B\tRAR1_75\tSGT1_100\tSGT1_130\tSGT1_160\tbinding\n")
        for sp_A, sp_B in pairs:
            states_A = species_states[sp_A]
            states_B = species_states[sp_B]
            rar1_75 = states_A['RAR1_75']
            sgt1_100 = states_B['SGT1_100']
            sgt1_130 = states_B['SGT1_130']
            sgt1_160 = states_B['SGT1_160']
            binding = binds(rar1_75, sgt1_100, sgt1_130, sgt1_160)
            f.write(f"{sp_A}\t{sp_B}\t{rar1_75}\t{sgt1_100}\t{sgt1_130}\t{sgt1_160}\t{int(binding)}\n")

if __name__ == '__main__':
    main()
