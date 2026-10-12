#!/bin/bash
set -euo pipefail
mkdir -p /app/output
python - <<'PY'
import pandas as pd, numpy as np, scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
train = pd.read_csv('/app/data/tear_proteomics_train.tsv', sep='\t')
blind = pd.read_csv('/app/data/blind_test_proteomics.tsv', sep='\t')
paths = pd.read_csv('/app/data/pathways.tsv', sep='\t')
meta = ['sample_id','participant_id','cohort','condition','timepoint']
prots = [c for c in train.columns if c not in meta]
keep=[]
for p in prots:
    ok=True
    for (c,cond), g in train.groupby(['cohort','condition']):
        if g[p].isna().mean()>0.2:
            ok=False; break
    if ok: keep.append(p)
prots=keep
records=[]
for c in ['A','B']:
    for cond in ['stimulated','control']:
        sub=train[(train.cohort==c)&(train.condition==cond)]
        pre=sub[sub.timepoint=='pre'].set_index('participant_id')
        post=sub[sub.timepoint=='post'].set_index('participant_id')
        common=pre.index.intersection(post.index)
        for p in prots:
            a=pre.loc[common,p].astype(float); b=post.loc[common,p].astype(float)
            m=~(a.isna()|b.isna())
            if m.sum()<10: continue
            d=(b[m]-a[m]).values
            fc=d.mean()
            t,pv=stats.ttest_rel(b[m],a[m])
            records.append({'cohort':c,'condition':cond,'protein':p,'log2FC':fc,'pval':pv})
res=pd.DataFrame(records)
for (c,cond), idx in res.groupby(['cohort','condition']).groups.items():
    pv=res.loc[idx,'pval'].values
    n=len(pv); order=np.argsort(pv); ranked=pv[order]
    q=ranked*n/(np.arange(1,n+1)); q=np.minimum.accumulate(q[::-1])[::-1]; q=np.clip(q,0,1)
    out=np.empty(n); out[order]=q
    res.loc[idx,'FDR']=out
sig=[]; stats_rows=[]
for p in prots:
    r=res[res.protein==p]
    def get(c,cond):
        x=r[(r.cohort==c)&(r.condition==cond)]
        return x.iloc[0] if len(x)>0 else None
    sa=get('A','stimulated'); sb=get('B','stimulated'); ca=get('A','control'); cb=get('B','control')
    if None in (sa,sb,ca,cb): continue
    if abs(sa.log2FC)>=1.0 and abs(sb.log2FC)>=1.0 and sa.FDR<0.05 and sb.FDR<0.05 and np.sign(sa.log2FC)==np.sign(sb.log2FC) and ca.FDR>=0.05 and cb.FDR>=0.05:
        sig.append(p)
        stats_rows.append({'protein':p,'log2FC_A_stim':sa.log2FC,'FDR_A_stim':sa.FDR,'log2FC_B_stim':sb.log2FC,'FDR_B_stim':sb.FDR,'direction':'up' if sa.log2FC>0 else 'down','log2FC_A_ctrl':ca.log2FC,'FDR_A_ctrl':ca.FDR,'log2FC_B_ctrl':cb.log2FC,'FDR_B_ctrl':cb.FDR})
pd.DataFrame(stats_rows).to_csv('/app/output/signature.tsv',sep='\t',index=False)
universe=set(prots)
enr=[]
for path,g in paths.groupby('pathway'):
    pp=set(g.protein)&universe
    if not pp: continue
    ov=set(sig)&pp
    pv=stats.hypergeom.sf(len(ov)-1, len(universe), len(pp), len(sig))
    enr.append({'pathway':path,'p_value':pv,'overlap_count':len(ov)})
enr=pd.DataFrame(enr).sort_values(['p_value','pathway']).head(3)
enr.to_csv('/app/output/enrichment.tsv',sep='\t',index=False)
stim=train[train.condition=='stimulated']
X=stim[sig].values
y=(stim.timepoint=='post').astype(int).values
clf=make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegression(solver='liblinear', penalty='l2', C=1.0, random_state=42))
clf.fit(X,y)
Xb=blind[sig].values
prob=clf.predict_proba(Xb)[:,1]
pd.DataFrame({'sample_id':blind.sample_id,'post_probability':prob}).to_csv('/app/output/predictions.csv',index=False)
PY
