import os, random, itertools, math, hashlib
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

SEED = 20260729
random.seed(SEED)
np.random.seed(SEED)

def random_dna(n, rng=random):
    return "".join(rng.choice("ACGT") for _ in range(n))

def mutate(seq, rate, rng=random):
    out = []
    for ch in seq:
        if rng.random() < rate:
            out.append(rng.choice([b for b in "ACGT" if b != ch]))
        else:
            out.append(ch)
    return "".join(out)

def main(outdir="."):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "models"), exist_ok=True)
    PHYLA = [f"Phy{i}" for i in range(1,6)]
    FAM_PER_PHY = 10
    GEN_PER_FAM = 12
    N_REF_PER_GEN = 5
    N_SEEN_QUERIES = 600
    N_NOVEL_QUERIES = 600

    taxonomy = []
    for p in PHYLA:
        for fi in range(FAM_PER_PHY):
            fam = f"{p}_F{fi}"
            for gi in range(GEN_PER_FAM):
                gen = f"{fam}_G{gi}"
                taxonomy.append({"phylum": p, "family": fam, "genus": gen})

    fam_motif5 = {}
    fam_motif3 = {}
    for tax in taxonomy:
        fam = tax["family"]
        if fam not in fam_motif5:
            h = int(hashlib.md5(fam.encode()).hexdigest()[:8], 16)
            rng = random.Random(h)
            fam_motif5[fam] = random_dna(30, rng)
            fam_motif3[fam] = random_dna(30, rng)

    phy_block5 = {}
    phy_block3 = {}
    for p in PHYLA:
        h = int(hashlib.md5(p.encode()).hexdigest()[:8], 16)
        rng = random.Random(h)
        phy_block5[p] = random_dna(120, rng)
        phy_block3[p] = random_dna(120, rng)

    ITS3 = "GCATCGATGAAGAACGCAGC"
    ITS4 = "TCCTCCGCTTATTGATATGC"
    ITS4_rc = str(Seq(ITS4).reverse_complement())

    def make_seq(tax, rng):
        left = mutate(phy_block5[tax["phylum"]], 0.01, rng) + mutate(fam_motif5[tax["family"]], 0.02, rng)
        right = mutate(fam_motif3[tax["family"]], 0.02, rng) + mutate(phy_block3[tax["phylum"]], 0.01, rng)
        h = int(hashlib.md5(tax["genus"].encode()).hexdigest()[:8], 16)
        grng = random.Random(h)
        its1 = random_dna(180, grng)
        s58 = random_dna(140, grng)
        its2 = random_dna(200, grng)
        return left + its1 + s58 + ITS3 + its2 + ITS4_rc + right

    ref_records = []
    for i, tax in enumerate(taxonomy):
        for rep in range(N_REF_PER_GEN):
            rid = f"REF_{i:04d}_{rep}"
            rng = random.Random(SEED + i*100 + rep)
            seq = make_seq(tax, rng)
            seq = mutate(seq, 0.005, rng)
            ref_records.append({"id": rid, "tax": tax, "seq": seq})

    with open(os.path.join(outdir, "reference.fasta"), "w") as f:
        for rec in ref_records:
            f.write(f">{rec['id']}\n{rec['seq']}\n")

    with open(os.path.join(outdir, "reference_taxonomy.tsv"), "w") as f:
        f.write("id\tphylum\tfamily\tgenus\n")
        for rec in ref_records:
            t = rec["tax"]
            f.write(f"{rec['id']}\t{t['phylum']}\t{t['family']}\t{t['genus']}\n")

    query_records = []
    for i, tax in enumerate(taxonomy):
        qid = f"Q_SEEN_{i:04d}"
        rng = random.Random(SEED + 100000 + i)
        seq = make_seq(tax, rng)
        seq = mutate(seq, 0.01, rng)
        query_records.append({"id": qid, "tax": tax, "stratum": "seen", "seq": seq})

    for i in range(N_NOVEL_QUERIES):
        p = random.choice(PHYLA)
        fam = f"{p}_NF{i}"
        gen = f"{fam}_NG{i}"
        tax = {"phylum": p, "family": fam, "genus": gen}
        qid = f"Q_NOVEL_{i:04d}"
        rng = random.Random(SEED + 200000 + i)
        seq = make_seq(tax, rng)
        seq = mutate(seq, 0.01, rng)
        query_records.append({"id": qid, "tax": tax, "stratum": "novel", "seq": seq})

    with open(os.path.join(outdir, "queries.fasta"), "w") as f:
        for rec in query_records:
            f.write(f">{rec['id']}\n{rec['seq']}\n")

    with open(os.path.join(outdir, "queries_meta.tsv"), "w") as f:
        f.write("query_id\tphylum\tfamily\tgenus\tstratum\n")
        for rec in query_records:
            t = rec["tax"]
            f.write(f"{rec['id']}\t{t['phylum']}\t{t['family']}\t{t['genus']}\t{rec['stratum']}\n")

    with open(os.path.join(outdir, "primers.tsv"), "w") as f:
        f.write("name\tsequence\n")
        f.write(f"ITS3\t{ITS3}\n")
        f.write(f"ITS4\t{ITS4}\n")

    def kmer_features(seq, k=3, n_left=150, n_right=150):
        left = seq[:n_left]
        right = seq[-n_right:]
        kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
        kmap = {kmer: i for i, kmer in enumerate(kmers)}
        def counts(s):
            vec = np.zeros(len(kmers))
            for i in range(len(s)-k+1):
                kmer = s[i:i+k]
                if kmer in kmap:
                    vec[kmap[kmer]] += 1
            return vec
        return np.concatenate([counts(left), counts(right)])

    X = []
    y = []
    for rec in ref_records:
        X.append(kmer_features(rec["seq"], k=3))
        y.append(rec["tax"]["family"])
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    cnn = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    cnn.fit(X, y_enc)
    joblib.dump({"model": cnn, "label_encoder": le, "k": 3, "n_left": 150, "n_right": 150},
                os.path.join(outdir, "models", "mycoai_cnn.joblib"))

    def kmer_features4(seq, k=4, n_left=200, n_right=200):
        left = seq[:n_left]
        right = seq[-n_right:]
        kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
        kmap = {kmer: i for i, kmer in enumerate(kmers)}
        def counts(s):
            vec = np.zeros(len(kmers))
            for i in range(len(s)-k+1):
                kmer = s[i:i+k]
                if kmer in kmap:
                    vec[kmap[kmer]] += 1
            return vec
        return np.concatenate([counts(left), counts(right)])

    X4 = []
    for rec in ref_records:
        X4.append(kmer_features4(rec["seq"], k=4))
    trans = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=43)
    trans.fit(X4, y_enc)
    joblib.dump({"model": trans, "label_encoder": le, "k": 4, "n_left": 200, "n_right": 200},
                os.path.join(outdir, "models", "mycoai_transformer.joblib"))

    predict_code = '''import sys, argparse, joblib, numpy as np, itertools
from Bio import SeqIO
from Bio.Seq import Seq

def kmer_features(seq, k, n_left, n_right):
    left = seq[:n_left]
    right = seq[-n_right:]
    kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
    kmap = {kmer: i for i, kmer in enumerate(kmers)}
    def counts(s):
        vec = np.zeros(len(kmers))
        for i in range(len(s)-k+1):
            kmer = s[i:i+k]
            if kmer in kmap:
                vec[kmap[kmer]] += 1
        return vec
    return np.concatenate([counts(left), counts(right)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnn","transformer"])
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.model == "cnn":
        bundle = joblib.load("/app/data/models/mycoai_cnn.joblib")
    else:
        bundle = joblib.load("/app/data/models/mycoai_transformer.joblib")
    model = bundle["model"]
    le = bundle["label_encoder"]
    k = bundle["k"]
    n_left = bundle["n_left"]
    n_right = bundle["n_right"]
    with open(args.out, "w") as out:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            feat = kmer_features(str(rec.seq), k, n_left, n_right).reshape(1, -1)
            proba = model.predict_proba(feat)[0]
            idx = int(np.argmax(proba))
            fam = le.inverse_transform([idx])[0]
            conf = float(proba[idx])
            out.write(f"{rec.id}\t{fam}\t{conf:.6f}\n")

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(outdir, "models", "predict_mycoai.py"), "w") as f:
        f.write(predict_code)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")