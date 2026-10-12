FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.4.2
COPY data/ /app/data/
RUN mkdir -p /app/output
RUN python - <<'PY'
import numpy as np, pandas as pd
SEED=20260731
TRUE=['PRTN3','LGALS3','CTSD','IGHG2','AZGP1','LTF','LCN1','LYZ','S100A8','S100A9','CST3','TIMP1','MMP9','SERPINA1']
BG=['ALB','APOA1','APOA2','APOB','APOC1','APOC2','APOC3','APOE','B2M','C3','C4A','C4B','C5','C6','C7','C8A','C8B','C8G','C9','CFB','CFD','CFH','CFI','CLU','CRP','F2','FGA','FGB','FGG','FN1','HP','HPX','IGHA1','IGHA2','IGHD','IGHE','IGHG1','IGHG3','IGHG4','IGHM','IGKC','IGLC1','IGLC2','IGLC3','IGLC4','IGLC5','IGLC6','IGLC7','JCHAIN','KNG1']
ALL=TRUE+BG
if len(ALL)<200:
    ALL += [f'SYN{i:03d}' for i in range(len(ALL),200)]
ALL=ALL[:200]
rng=np.random.RandomState(SEED)
n=len(ALL)
base=np.clip(rng.normal(20,2.5,n),15,28)
cohort_eff={'A':np.zeros(n),'B':rng.normal(0,0.4,n)}
true_eff={}
for i,p in enumerate(TRUE):
    true_eff[p]=(1 if i%2==0 else -1)*(1.7+0.05*(i%3))
def make(ps):
    rows=[]
    for pid,c,cond in ps:
        part_eff=rng.normal(0,1.0,n)
        for tp in ['pre','post']:
            vals=base+part_eff+cohort_eff[c]
            if tp=='post' and cond=='stimulated':
                for p,eff in true_eff.items():
                    idx=ALL.index(p)
                    vals[idx]+=eff+rng.normal(0,0.15)
            vals+=rng.normal(0,0.6,n)
            mr=rng.uniform(0.01,0.08,n)
            for p in TRUE:
                idx=ALL.index(p)
                mr[idx]=0.01
            vals[rng.rand(n)<mr]=np.nan
            row={'sample_id':f'{pid}_{tp}','participant_id':pid,'cohort':c,'condition':cond,'timepoint':tp}
            for i,p in enumerate(ALL): row[p]=vals[i]
            rows.append(row)
    return pd.DataFrame(rows)
train_ps=[]
for c in ['A','B']:
    for cond in ['stimulated','control']:
        count=56 if cond=='stimulated' else 28
        for i in range(count):
            train_ps.append((f'TR_{c}_{cond}_{i:03d}',c,cond))
blind_ps=[]
for c in ['A','B']:
    for cond in ['stimulated','control']:
        count=18 if cond=='stimulated' else 10
        for i in range(count):
            blind_ps.append((f'BL_{c}_{cond}_{i:03d}',c,cond))
train_df=make(train_ps)
blind_df=make(blind_ps)
blind_out=blind_df.drop(columns=['participant_id','cohort','condition','timepoint'])
train_df.to_csv('/app/data/tear_proteomics_train.tsv',sep='\t',index=False)
blind_out.to_csv('/app/data/blind_test_proteomics.tsv',sep='\t',index=False)
paths=[]
all_paths=['Innate Immune Response','Inflammatory Response','Host Defense','Extracellular Matrix','Complement Cascade','Glycolysis','Oxidative Phosphorylation','Protein Folding','Cytoskeleton','Cell Adhesion','Apoptosis','Autophagy','Lysosomal Function','Antigen Presentation','Iron Homeostasis','Lipid Metabolism','Wound Healing','Neuroprotection','Angiogenesis','Coagulation']
for i,p in enumerate(TRUE):
    if i<5: paths.append(('Innate Immune Response',p))
    if i<8: paths.append(('Inflammatory Response',p))
    if i<11: paths.append(('Host Defense',p))
    if i>=11: paths.append(('Extracellular Matrix',p))
for p in ALL:
    if p in TRUE: continue
    for path in rng.choice(all_paths, rng.randint(1,4), replace=False):
        paths.append((path,p))
pd.DataFrame(paths, columns=['pathway','protein']).drop_duplicates().to_csv('/app/data/pathways.tsv',sep='\t',index=False)
PY
