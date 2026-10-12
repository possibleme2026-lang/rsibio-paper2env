# Tear proteomics music stimulation dataset

Synthetic-but-realistic tear proteomics dataset for a two-cohort study of musical stimulation.

Files in /app/data:
- tear_proteomics_train.tsv: discovery data, 336 samples x 200 proteins.
- blind_test_proteomics.tsv: blinded validation data, 112 samples x 200 proteins.
- pathways.tsv: protein-pathway annotations.

Proteins are log2 LFQ intensities. Missing values are NA.
The data were generated with a protected seed; true protein labels are not provided.
