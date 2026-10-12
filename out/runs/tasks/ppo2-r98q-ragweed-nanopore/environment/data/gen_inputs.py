#!/usr/bin/env python3
import argparse, gzip, json, random, os, sys

def generate_reference(seed, config):
    rng = random.Random(seed)
    length = config['reference_length']
    bases = list('ACGT')
    seq = [rng.choice(bases) for _ in range(length)]
    seq[0:3] = list('ATG')
    seq[-3:] = list('TAA')
    seq[291:294] = list('CGA')
    return ''.join(seq)

def mutate_allele(seq, allele, site):
    idx = site - 1
    if allele == 'R98L':
        seq = seq[:idx] + 'T' + seq[idx+1:]
    elif allele == 'R98Q':
        seq = seq[:idx] + 'A' + seq[idx+1:]
    return seq

def simulate_read(seq, rng, error_sub, error_ins, error_del):
    bases = 'ACGT'
    out = []
    for b in seq:
        r = rng.random()
        if r < error_del:
            continue
        elif r < error_del + error_ins:
            out.append(rng.choice(bases))
            out.append(b)
        else:
            if rng.random() < error_sub:
                alt = [x for x in bases if x != b]
                out.append(rng.choice(alt))
            else:
                out.append(b)
    return ''.join(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    config = json.loads(args.config)
    rng = random.Random(args.seed)
    ref = generate_reference(args.seed, config)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'reads'), exist_ok=True)
    with open(os.path.join(args.outdir, 'reference.fasta'), 'w') as f:
        f.write('>PPO2_CDS\n')
        for i in range(0, len(ref), 60):
            f.write(ref[i:i+60] + '\n')
    for pop, pop_cfg in config['populations'].items():
        n = pop_cfg['n_reads']
        freqs = pop_cfg['freqs']
        alleles = list(freqs.keys())
        weights = [freqs[a] for a in alleles]
        out_path = os.path.join(args.outdir, 'reads', f'{pop}.fastq.gz')
        with gzip.open(out_path, 'wt') as f:
            for i in range(n):
                allele = rng.choices(alleles, weights=weights, k=1)[0]
                amplicon = ref[config['amplicon_start']-1:config['amplicon_end']]
                site_in_amplicon = config['mutation_site'] - config['amplicon_start'] + 1
                amplicon = mutate_allele(amplicon, allele, site_in_amplicon)
                read_seq = simulate_read(amplicon, rng, config['error_rate_sub'], config['error_rate_ins'], config['error_rate_del'])
                qual = ''.join(chr(33 + rng.randint(5, 30)) for _ in range(len(read_seq)))
                f.write(f'@read_{pop}_{i} length={len(read_seq)}\n')
                f.write(read_seq + '\n')
                f.write('+\n')
                f.write(qual + '\n')
    print('Done')

if __name__ == '__main__':
    main()
