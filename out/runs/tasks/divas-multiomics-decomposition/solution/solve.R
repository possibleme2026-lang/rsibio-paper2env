library(DIVAS)
library(data.table)

T <- as.matrix(fread('/app/data/transcriptomics.csv'))
P <- as.matrix(fread('/app/data/proteomics.csv'))
M <- as.matrix(fread('/app/data/metabolomics.csv'))
T <- T[, -1]
P <- P[, -1]
M <- M[, -1]
T <- scale(T)
P <- scale(P)
M <- scale(M)
data <- list(transcriptomics = T, proteomics = P, metabolomics = M)

# Placeholder for actual DIVAS call - adjust according to package API.
# res <- divas(data, n_components = 15, ...)
# Extract components, scores, and loadings, then write to /app/results/.
# This reference skeleton demonstrates the expected multi-step workflow.
