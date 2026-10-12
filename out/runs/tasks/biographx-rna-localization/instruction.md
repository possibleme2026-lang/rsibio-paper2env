# BioGraphX-RNA: Physicochemical Graph Encoding for RNA Localization Inference

You are given a synthetic but realistic RNA dataset and a pre-trained BioGraphX-RNA gated fusion model. Your job is to implement the full inference pipeline: build multi-scale interaction graphs from RNA sequences using RNAfold secondary structure, compute graph-derived features, and combine them with frozen RiNALMo-like sequence embeddings via an interpretable gated fusion layer to predict subcellular localization probabilities.

## Inputs
- `/app/data/sequences.fasta` — 300 RNA sequences. Headers are `>seq_id|rna_type` where `rna_type` is one of `mRNA`, `miRNA`, `lncRNA`. Sequences range from 20 nt (miRNA) to 800 nt (mRNA).
- `/app/data/embeddings.csv` — frozen 128-dimensional embeddings for each `seq_id` (columns `emb_0` ... `emb_127`).
- `/app/data/model_weights.json` — pre-trained weights for the gated fusion model. Contains `graph_features` (list of 12 feature names), `classes` (`["nucleus","cytoplasm","exosome"]`), `graph_weights` (3x12), `seq_weights` (3x128), `gate_weights` (12), `gate_bias` (scalar), `class_bias` (3).
- All input files are read-only.

## Task
Implement the BioGraphX-RNA inference pipeline in `/app/solve.py` (or any language) and run it to produce the following outputs in `/app/output/`:

1. **Graph construction**: For each sequence, run ViennaRNA `RNAfold --noPS` with default parameters (37°C) to obtain the minimum free energy (MFE) secondary structure in dot-bracket notation. Build an undirected simple graph with nodes at nucleotide positions 1..L. Add edge types:
   - `backbone`: between consecutive positions i and i+1.
   - `basepair`: between positions i and j if the dot-bracket has a matching pair `(` at i and `)` at j.
   - `stacking`: for every base pair (i, j) with i < j, if (i+1, j-1) is also a base pair, add an edge between i and j-1.
   - `physchem`: for every pair i < j with 4 <= j-i <= 50, add an edge if the nucleotide pair (S[i], S[j]) has an interaction score >= 0.5. Score table: A-U=1.0, G-C=1.0, G-U=0.7, A-G=0.5, C-U=0.5, all others=0.0. The score is symmetric.
   Multiple edge types between the same node pair are merged into one edge; the edge's type set records all types.

2. **Feature extraction**: For each sequence, compute exactly these 12 features:
   - `length`: sequence length L.
   - `gc_content`: (count of G + C) / L.
   - `purine_content`: (count of A + G) / L.
   - `mfe_norm`: MFE from RNAfold divided by L.
   - `unpaired_fraction`: number of unpaired positions ('.' in dot-bracket) / L.
   - `basepair_count`: number of '(' in dot-bracket.
   - `backbone_edges`: L - 1.
   - `stacking_edges`: number of unique node pairs whose edge type set contains `stacking`.
   - `physchem_edges`: number of unique node pairs whose edge type set contains `physchem`.
   - `avg_degree`: average number of unique neighbors over all nodes.
   - `clustering_coeff`: average local clustering coefficient over nodes with degree >= 2. For a node v with degree k_v, let e_v be the number of edges among its neighbors. C_v = 2 * e_v / (k_v * (k_v - 1)).
   - `max_degree`: maximum degree (number of unique neighbors) over all nodes.

3. **Gated fusion inference**: For each sequence, let `f[0..11]` be the feature vector in the order above, and `e[0..127]` the embedding vector. Compute:
   - `gate = sigmoid(sum(f[i] * gate_weights[i]) + gate_bias)`
   - For each class c in `classes`:
     `seq_logit[c] = sum(e[j] * seq_weights[c][j]) + class_bias[c]`
     `graph_logit[c] = sum(f[i] * graph_weights[c][i])`
     `fused_logit[c] = gate * seq_logit[c] + (1 - gate) * graph_logit[c]`
   - `probs = softmax(fused_logit)`.

4. **Output files**:
   - `/app/output/graph_features.csv`: header `seq_id,length,gc_content,purine_content,mfe_norm,unpaired_fraction,basepair_count,backbone_edges,stacking_edges,physchem_edges,avg_degree,clustering_coeff,max_degree`, one row per sequence in FASTA order.
   - `/app/output/gates.csv`: header `seq_id,gate`, one row per sequence in FASTA order.
   - `/app/output/predictions.csv`: header `seq_id,nucleus,cytoplasm,exosome`, one row per sequence in FASTA order, probabilities in [0,1] summing to 1 within 1e-5.

## Constraints
1. The `graph_features.csv` must have exactly 300 data rows in the same `seq_id` order as `sequences.fasta`, with the exact header specified.
2. Sequence-level features (`length`, `gc_content`, `purine_content`, `backbone_edges`) must match the reference computation within 1e-6.
3. Structural features (`mfe_norm`, `unpaired_fraction`, `basepair_count`) must match the reference RNAfold-based computation within 1e-6.
4. Graph features (`stacking_edges`, `physchem_edges`, `avg_degree`, `clustering_coeff`, `max_degree`) must match the reference graph computation within 1e-6.
5. `gates.csv` must have the exact header and one row per sequence; every gate value must be in [0,1] and match the reference gated fusion formula within 1e-6.
6. `predictions.csv` must have the exact header and one row per sequence; all probabilities must be in [0,1] and each row must sum to 1 within 1e-5. The root-mean-square error (RMSE) against the reference probabilities must be <= 0.01.

## Scoring
The grader checks constraints 1–6 independently. Score is a weighted sum: constraints 1–5 are worth 15% each (binary pass/fail per constraint), and constraint 6 is worth 25% based on RMSE (score = max(0, 1 - RMSE/0.05)). The final reward is the total score. Partial credit is given for each constraint.
