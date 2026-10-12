#!/usr/bin/env python3
import os, json, csv, math, sys

DATA = '/app/data'
OUT = '/app/output'
RESULT = '/logs/verifier/reward.json'

def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                continue
            seq.append(line.strip())
    return ''.join(seq)

def load_genes(path):
    genes = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            if parts[2] == 'CDS':
                start = int(parts[3])
                end = int(parts[4])
                strand = parts[6]
                name = None
                for kv in parts[8].split(';'):
                    if kv.startswith('Name='):
                        name = kv[5:]
                    elif kv.startswith('ID='):
                        name = kv[3:]
                genes.append((name, start, end, strand))
    genes.sort(key=lambda x: x[1])
    return genes

def load_manifest(path):
    samples = []
    with open(path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            samples.append(dict(zip(header, parts)))
    samples.sort(key=lambda x: int(x['timepoint']))
    return samples

def load_coverage(sample):
    cov = {}
    path = os.path.join(DATA, sample['coverage_path'])
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            cov[int(parts[1])] = int(parts[2])
    return cov

def load_vcf_records(samples):
    records = []
    for s in samples:
        sid = s['sample_id']
        vcf_path = os.path.join(DATA, s['vcf_path'])
        with open(vcf_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue
                chrom, pos, vid, ref, alt, qual, filt, info = parts[:8]
                records.append({
                    'sample_id': sid,
                    'chrom': chrom,
                    'pos': int(pos),
                    'ref': ref,
                    'alt': alt,
                    'filter': filt,
                    'variant_id': f'{chrom}:{pos}:{ref}:{alt}',
                })
    return records

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def translate(codon):
    return CODON_TABLE.get(codon.upper(), 'X')

def annotate(pos, ref_base, alt, ref_seq, genes):
    for name, start, end, strand in genes:
        if start <= pos <= end:
            if strand == '+':
                offset = pos - start
                codon_start = start + (offset // 3) * 3
                codon_pos = offset // 3 + 1
                ref_codon = ref_seq[codon_start-1:codon_start+2]
                alt_codon = list(ref_codon)
                idx = offset % 3
                alt_codon[idx] = alt
                ref_aa = translate(ref_codon)
                alt_aa = translate(''.join(alt_codon))
                return name, f'{name}:{ref_aa}{codon_pos}{alt_aa}'
            else:
                pass
    return 'NA', 'NA'

def compute_expected():
    ref_seq = read_fasta(os.path.join(DATA, 'reference.fa'))
    genes = load_genes(os.path.join(DATA, 'genes.gff3'))
    samples = load_manifest(os.path.join(DATA, 'manifest.tsv'))
    coverage = {s['sample_id']: load_coverage(s) for s in samples}
    records = load_vcf_records(samples)

    total_records = len(records)
    low_quality_records = sum(1 for r in records if r['filter'] != 'PASS')
    low_coverage_records = 0
    pass_records = 0

    variant_samples = {}
    variant_cov = {}
    for r in records:
        sid = r['sample_id']
        pos = r['pos']
        vid = r['variant_id']
        cov = coverage[sid].get(pos, 0)
        if r['filter'] != 'PASS':
            continue
        if cov < 50:
            low_coverage_records += 1
            continue
        pass_records += 1
        variant_samples.setdefault(vid, set()).add(sid)
        variant_cov.setdefault(vid, {})[sid] = cov

    expected_rows = {}
    sample_order = [s['sample_id'] for s in samples]
    timepoint_of = {s['sample_id']: int(s['timepoint']) for s in samples}

    for vid, sset in variant_samples.items():
        chrom, pos_s, ref_b, alt = vid.split(':')
        pos = int(pos_s)
        gene, aa = annotate(pos, ref_b, alt, ref_seq, genes)
        present_samples = sorted(sset, key=lambda sid: timepoint_of[sid])
        first_sid = present_samples[0]
        last_sid = present_samples[-1]
        first_tp = timepoint_of[first_sid]
        last_tp = timepoint_of[last_sid]
        n_present = len(present_samples)
        var_type = 'original' if first_tp == 0 else 'emerging'
        temp_class = 'persistent' if n_present >= 2 else 'transient'
        covs = [variant_cov[vid][sid] for sid in present_samples]
        mean_cov = round(sum(covs)/len(covs), 1)
        expected_rows[vid] = {
            'variant_id': vid,
            'pos': pos,
            'ref': ref_b,
            'alt': alt,
            'gene': gene,
            'aa_change': aa,
            'first_sample': first_sid,
            'first_timepoint': first_tp,
            'last_sample': last_sid,
            'last_timepoint': last_tp,
            'n_present': n_present,
            'variant_type': var_type,
            'temporal_class': temp_class,
            'mean_coverage': mean_cov,
            'present_samples': ','.join(present_samples),
            'presence': {sid: (1 if sid in sset else 0) for sid in sample_order},
        }
    expected_summary = {
        'total_variants': len(expected_rows),
        'original_variants': sum(1 for r in expected_rows.values() if r['variant_type'] == 'original'),
        'emerging_variants': sum(1 for r in expected_rows.values() if r['variant_type'] == 'emerging'),
        'persistent_variants': sum(1 for r in expected_rows.values() if r['temporal_class'] == 'persistent'),
        'transient_variants': sum(1 for r in expected_rows.values() if r['temporal_class'] == 'transient'),
        'coding_variants': sum(1 for r in expected_rows.values() if r['gene'] != 'NA'),
        'noncoding_variants': sum(1 for r in expected_rows.values() if r['gene'] == 'NA'),
        'n_vcf_records': total_records,
        'n_pass_records': pass_records,
        'n_low_quality_records': low_quality_records,
        'n_low_coverage_records': low_coverage_records,
    }
    return expected_rows, expected_summary, sample_order

def load_agent_variants(path):
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows[row['variant_id']] = row
    return rows

def load_agent_matrix(path):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        matrix = {}
        order = []
        for row in reader:
            if not row:
                continue
            vid = row[0]
            order.append(vid)
            matrix[vid] = {header[i]: int(row[i]) for i in range(1, len(header))}
    return matrix, header[1:], order

def main():
    expected_rows, expected_summary, sample_order = compute_expected()
    checks = []
    ok = True
    try:
        agent_variants = load_agent_variants(os.path.join(OUT, 'variants.tsv'))
        agent_matrix, agent_samples, agent_order = load_agent_matrix(os.path.join(OUT, 'presence_matrix.tsv'))
        with open(os.path.join(OUT, 'summary.json')) as f:
            agent_summary = json.load(f)
    except Exception as e:
        ok = False
        print(f'Error loading outputs: {e}')
    checks.append(('files_and_parse', 1.0 if ok else 0.0))
    if not ok:
        score = 0.0
        reward = {'score': score, 'reward': score}
        os.makedirs('/logs/verifier', exist_ok=True)
        with open(RESULT, 'w') as f:
            json.dump(reward, f)
        return

    with open(os.path.join(OUT, 'variants.tsv')) as f:
        header = f.readline().strip().split('\t')
    expected_cols = ['variant_id','pos','ref','alt','gene','aa_change','first_sample','first_timepoint','last_sample','last_timepoint','n_present','variant_type','temporal_class','mean_coverage','present_samples']
    cols_ok = (header == expected_cols)
    checks.append(('variants_columns', 1.0 if cols_ok else 0.0))

    matrix_cols_ok = (agent_samples == sample_order)
    checks.append(('presence_matrix_columns', 1.0 if matrix_cols_ok else 0.0))

    exp_ids = set(expected_rows.keys())
    agt_ids = set(agent_variants.keys())
    if not exp_ids and not agt_ids:
        set_score = 1.0
    else:
        inter = len(exp_ids & agt_ids)
        union = len(exp_ids | agt_ids)
        set_score = inter / union if union > 0 else 0.0
    checks.append(('variant_set_jaccard', set_score))

    common = exp_ids & agt_ids
    if not common:
        aa_score = temp_score = cov_score = 0.0
    else:
        aa_correct = 0
        temp_correct = 0
        cov_correct = 0
        for vid in common:
            e = expected_rows[vid]
            a = agent_variants[vid]
            if a.get('gene') == e['gene'] and a.get('aa_change') == e['aa_change']:
                aa_correct += 1
            temp_fields = ['first_sample','first_timepoint','last_sample','last_timepoint','n_present','variant_type','temporal_class']
            if all(str(a.get(f)) == str(e[f]) for f in temp_fields):
                temp_correct += 1
            try:
                av = float(a.get('mean_coverage', 'nan'))
                ev = float(e['mean_coverage'])
                if abs(av - ev) <= 0.11:
                    cov_correct += 1
            except:
                pass
        aa_score = aa_correct / len(common)
        temp_score = temp_correct / len(common)
        cov_score = cov_correct / len(common)
    checks.append(('amino_acid_annotation', aa_score))
    checks.append(('temporal_classification', temp_score))
    checks.append(('mean_coverage', cov_score))

    if not common:
        pres_score = 0.0
    else:
        correct_bits = 0
        total_bits = 0
        for vid in common:
            e_pres = expected_rows[vid]['presence']
            a_pres = agent_matrix.get(vid, {})
            for sid in sample_order:
                total_bits += 1
                if a_pres.get(sid, 0) == e_pres[sid]:
                    correct_bits += 1
        pres_score = correct_bits / total_bits if total_bits > 0 else 0.0
    checks.append(('presence_matrix', pres_score))

    summary_keys = list(expected_summary.keys())
    if not agent_summary:
        sum_score = 0.0
    else:
        correct = 0
        for k in summary_keys:
            if k in agent_summary and agent_summary[k] == expected_summary[k]:
                correct += 1
        sum_score = correct / len(summary_keys)
    checks.append(('summary_json', sum_score))

    agent_rows_list = list(agent_variants.values())
    try:
        sorted_ok = True
        for i in range(1, len(agent_rows_list)):
            p1 = (int(agent_rows_list[i-1]['pos']), agent_rows_list[i-1]['ref'], agent_rows_list[i-1]['alt'])
            p2 = (int(agent_rows_list[i]['pos']), agent_rows_list[i]['ref'], agent_rows_list[i]['alt'])
            if p1 > p2:
                sorted_ok = False
                break
        order_ok = (agent_order == [r['variant_id'] for r in agent_rows_list])
        order_score = 1.0 if (sorted_ok and order_ok) else 0.0
    except:
        import traceback as _tb, sys as _sys
        _err = _tb.format_exc()
        try:
            import os as _os
            _os.makedirs('/logs/verifier', exist_ok=True)
            open('/logs/verifier/grader_error.txt','w').write(_err)
        except Exception:
            pass
        print('GRADER_EXCEPTION\n' + _err, file=_sys.stderr, flush=True)
        order_score = 0.0
    checks.append(('ordering', order_score))

    score = sum(s for _, s in checks) / len(checks)
    reward = {'score': score, 'reward': score}
    os.makedirs('/logs/verifier', exist_ok=True)
    with open(RESULT, 'w') as f:
        json.dump(reward, f)
    for name, s in checks:
        print(f'{name}: {s:.4f}')
    print(f'RAW_SCORE: {score:.4f}')

if __name__ == '__main__':
    main()
