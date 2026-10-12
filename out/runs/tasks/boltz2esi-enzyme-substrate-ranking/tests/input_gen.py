import os, sys, random, math
from rdkit import Chem

def generate_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    random.seed(20260730)
    cores = ['c1ccccc1', 'c1ccncc1', 'c1ccoc1', 'c1ccsc1', 'c1ccc2ccccc2c1', 'c1ccc2[nH]ccc2c1', 'c1ccc2ncccc2c1', 'c1ccc2ccccc2n1', 'C1CCCCC1', 'C1CCOCC1', 'C1CCNC1', 'C1CCNCC1', 'C1CCOC1', 'c1ccc(cc1)c2ccccc2', 'c1cc2ccccc2cc1', 'c1ccc(cc1)C', 'c1ccc(cc1)CC', 'c1ccc(cc1)CCC', 'c1ccoc1C', 'c1ccsc1C']
    subs = ['O', 'N', 'C', 'F', 'Cl', 'Br', 'C(=O)O', 'C(=O)N', 'C#N', '[N+](=O)[O-]', 'C(F)(F)F', 'OC', 'S', 'C(=O)C', 'CC']
    all_smiles = []
    for core in cores:
        for sub in subs:
            smi = core + sub
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                all_smiles.append(Chem.MolToSmiles(mol))
    all_smiles = sorted(set(all_smiles))
    if len(all_smiles) < 150:
        raise RuntimeError(f'Only generated {len(all_smiles)} valid SMILES, need 150')
    selected = all_smiles[:150]
    with open(os.path.join(outdir, 'substrates.smi'), 'w') as f:
        for i, smi in enumerate(selected, 1):
            f.write(f'SUB{i:03d}\t{smi}\n')
    families = ['CYP', 'GT', 'MT', 'PT', 'NRPS']
    organisms = ['Homo sapiens', 'Arabidopsis thaliana', 'Escherichia coli', 'Streptomyces coelicolor', 'Saccharomyces cerevisiae', 'Aspergillus niger']
    aa_list = list('ACDEFGHIKLMNPQRSTVWY')
    enzymes = {}
    annotations = []
    for i in range(1, 31):
        eid = f'ENZ{i:03d}'
        fam = random.choice(families)
        if fam == 'CYP':
            length = random.randint(350, 550)
        elif fam == 'GT':
            length = random.randint(350, 500)
        elif fam == 'MT':
            length = random.randint(300, 450)
        elif fam == 'PT':
            length = random.randint(350, 600)
        else:
            length = random.randint(500, 780)
        seq = [random.choice(aa_list) for _ in range(length)]
        positions = sorted(random.sample(range(1, length+1), 3))
        res_types = random.choices(['H','D','S','C','E','K','R','Y'], k=3)
        for pos, res in zip(positions, res_types):
            seq[pos-1] = res
        seq_str = ''.join(seq)
        enzymes[eid] = seq_str
        catal = ';'.join(f'{res}{pos}' for pos, res in zip(positions, res_types))
        annotations.append({'enzyme_id': eid, 'family': fam, 'catalytic_residues': catal, 'organism': random.choice(organisms), 'length': length})
    with open(os.path.join(outdir, 'enzymes.fasta'), 'w') as f:
        for eid, seq in enzymes.items():
            f.write(f'>{eid}\n{seq}\n')
    import pandas as pd
    pd.DataFrame(annotations).to_csv(os.path.join(outdir, 'enzyme_annotations.tsv'), sep='\t', index=False)
    sub_ids = [f'SUB{i:03d}' for i in range(1, 151)]
    sub_smiles = dict(zip(sub_ids, selected))
    from rdkit.Chem import Descriptors, Crippen
    mw_logp = {}
    for sid, smi in sub_smiles.items():
        mol = Chem.MolFromSmiles(smi)
        mw_logp[sid] = (Descriptors.MolWt(mol), Crippen.MolLogP(mol))
    ref_rows = []
    for eid in enzymes:
        selected_subs = random.sample(sub_ids, 10)
        for sid in selected_subs:
            mw, logp = mw_logp[sid]
            score = 0.5 + 0.3 * math.tanh((mw - 250)/100) + 0.2 * (logp - 2.0)/5.0 + random.uniform(-0.1, 0.1)
            score = max(0.0, min(1.0, score))
            ref_rows.append({'enzyme_id': eid, 'substrate_id': sid, 'interaction_score': round(score, 4)})
    pd.DataFrame(ref_rows).to_csv(os.path.join(outdir, 'reference_pairs.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    generate_all(outdir)
