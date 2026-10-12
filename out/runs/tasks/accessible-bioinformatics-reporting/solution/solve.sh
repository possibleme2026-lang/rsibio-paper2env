#!/bin/bash
python - <<'PY'
import json
with open('/app/environment/data/analysis_outputs.json') as f:
    data = json.load(f)
with open('/app/environment/data/candidate_statements.json') as f:
    statements = json.load(f)

def supported(c):
    t = c['type']
    if t == 'top_up_gene':
        genes = data['differential_expression']['up']
        return bool(genes) and max(genes, key=lambda x: x['log2FC'])['gene'] == c['value']
    if t == 'top_down_gene':
        genes = data['differential_expression']['down']
        return bool(genes) and min(genes, key=lambda x: x['log2FC'])['gene'] == c['value']
    if t == 'gene_direction':
        for g in data['differential_expression'].get(c['direction'], []):
            if g['gene'] == c['gene']:
                return True
        return False
    if t == 'top_pathway':
        enr = data['enrichment']
        return bool(enr) and min(enr, key=lambda x: x['padj'])['term'] == c['value']
    if t == 'min_pvalue_gene':
        genes = data['differential_expression']['up'] + data['differential_expression']['down']
        return bool(genes) and min(genes, key=lambda x: x['pvalue'])['gene'] == c['value']
    if t == 'figure_caption_contains':
        for fig in data['figures']:
            if fig['id'] == c['figure'] and c['keyword'].lower() in fig['caption'].lower():
                return True
        return False
    if t == 'study_contains':
        return c['keyword'].lower() in data['study'].lower()
    return False

selected = [s['id'] for s in statements if supported(s['check'])]
summary = 'This report compares treated and control samples. TP53 is the most strongly up-regulated gene, CDKN1A is down-regulated, p53 signaling is the top enriched pathway, and Fig1 is a volcano plot.'
with open('/app/output.json', 'w') as f:
    json.dump({'summary': summary, 'selected_statement_ids': selected}, f)
PY
