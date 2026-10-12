import random
from simulation import generate_tree, simulate_sequences, get_interface_states, get_binding, binds

def generate_gold():
    random.seed(42)
    tree = generate_tree()
    node_seqs = simulate_sequences(tree)
    
    root_clade = tree.root
    fern_ancestor = tree.find_any(name='FernAncestor')
    core_fern_ancestor = tree.find_any(name='CoreFernAncestor')
    
    def get_states(clade):
        rar1, sgt1 = node_seqs[clade]
        return get_interface_states(rar1, sgt1)
    
    gold = {
        'ancestral_states': {
            'Root': get_states(root_clade),
            'FernAncestor': get_states(fern_ancestor),
            'CoreFernAncestor': get_states(core_fern_ancestor)
        },
        'extant_binding': {},
        'mutation_order': ['SGT1_130', 'RAR1_75', 'SGT1_100'],
        'binding_rule': [
            {'RAR1_75': 'D', 'SGT1_100': 'K', 'SGT1_130': 'R', 'SGT1_160': 'Y'},
            {'RAR1_75': 'D', 'SGT1_100': 'K', 'SGT1_130': 'S', 'SGT1_160': 'Y'},
            {'RAR1_75': 'K', 'SGT1_100': 'K', 'SGT1_130': 'S', 'SGT1_160': 'Y'},
            {'RAR1_75': 'K', 'SGT1_100': 'E', 'SGT1_130': 'S', 'SGT1_160': 'Y'}
        ]
    }
    
    for clade in tree.get_terminals():
        rar1, sgt1 = node_seqs[clade]
        states = get_interface_states(rar1, sgt1)
        binding = get_binding(states)
        gold['extant_binding'][clade.name] = binding
    
    return gold
