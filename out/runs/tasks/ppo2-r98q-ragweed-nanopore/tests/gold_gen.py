import random

SEED = 12345
CONFIG = {
    "reference_length": 1700,
    "amplicon_start": 1,
    "amplicon_end": 600,
    "mutation_site": 293,
    "populations": {
        "S": {"n_reads": 2000, "freqs": {"WT_R98": 1.0, "R98L": 0.0, "R98Q": 0.0}},
        "R1": {"n_reads": 2500, "freqs": {"WT_R98": 0.05, "R98L": 0.75, "R98Q": 0.20}},
        "R2": {"n_reads": 2500, "freqs": {"WT_R98": 0.03, "R98L": 0.02, "R98Q": 0.95}}
    },
    "error_rate_sub": 0.03,
    "error_rate_ins": 0.01,
    "error_rate_del": 0.02
}

def generate_reference(seed=SEED, config=CONFIG):
    rng = random.Random(seed)
    length = config['reference_length']
    bases = list('ACGT')
    seq = [rng.choice(bases) for _ in range(length)]
    seq[0:3] = list('ATG')
    seq[-3:] = list('TAA')
    seq[291:294] = list('CGA')
    return ''.join(seq)
