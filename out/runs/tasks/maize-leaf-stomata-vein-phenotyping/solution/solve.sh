#!/bin/bash
set -euo pipefail
cat > /app/solution/analyze.py << 'PYEOF'
import os, json
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from scipy import stats
from skimage.morphology import skeletonize

meta = pd.read_csv('/app/data/image_metadata.csv')
geno = pd.read_csv('/app/data/genotypes.csv')
marker_info = pd.read_csv('/app/data/marker_info.csv')

traits_list = []
for _, row in meta.iterrows():
    img_path = f"/app/data/images/{row['image_id']}.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    ppm = row['pixels_per_mm']
    area_mm2 = (img.shape[0] / ppm) * (img.shape[1] / ppm)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    veins = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    skeleton = skeletonize(veins > 0).astype(np.uint8)
    vein_length_px = np.sum(skeleton)
    vein_density = vein_length_px / ppm / area_mm2
    vein_thickness = (np.sum(veins > 0) / vein_length_px) / ppm if vein_length_px > 0 else 0
    stomata_mask = thresh.copy()
    stomata_mask[veins > 0] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(stomata_mask, connectivity=8)
    stomata_centers = []
    stomata_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 20 < area < 500:
            stomata_centers.append(centroids[i])
            stomata_areas.append(area)
    stomatal_density = len(stomata_centers) / area_mm2
    stomatal_size = np.mean(stomata_areas) / (ppm**2) if stomata_areas else 0
    if len(stomata_centers) > 1:
        tree = cKDTree(stomata_centers)
        dists, _ = tree.query(stomata_centers, k=2)
        nnd = dists[:, 1] / ppm
        nnd_mean = np.mean(nnd)
        nnd_cv = np.std(nnd) / np.mean(nnd)
    else:
        nnd_mean = 0
        nnd_cv = 0
    traits_list.append({
        'image_id': row['image_id'],
        'ril_id': row['ril_id'],
        'stomatal_density': stomatal_density,
        'stomatal_size': stomatal_size,
        'vein_density': vein_density,
        'vein_thickness': vein_thickness,
        'nnd_mean': nnd_mean,
        'nnd_cv': nnd_cv
    })

traits_df = pd.DataFrame(traits_list)
traits_df.to_csv('/app/output/traits.csv', index=False)
ril_traits = traits_df.groupby('ril_id').mean(numeric_only=True).reset_index()
ril_traits.to_csv('/app/output/ril_traits.csv', index=False)

qtl_results = []
trait_cols = ['stomatal_density', 'stomatal_size', 'vein_density', 'vein_thickness', 'nnd_mean', 'nnd_cv']
merged = pd.merge(ril_traits, geno, on='ril_id')
for trait in trait_cols:
    y = merged[trait].values
    for _, mrow in marker_info.iterrows():
        marker = mrow['marker_id']
        x = merged[marker].values
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        lod = -np.log10(p_val) if p_val > 0 else 10
        if lod > 3.0:
            qtl_results.append({
                'trait': trait,
                'marker_id': marker,
                'chromosome': int(mrow['chromosome']),
                'position': float(mrow['position']),
                'lod': float(lod)
            })
with open('/app/output/qtl_results.json', 'w') as f:
    json.dump(qtl_results, f)
PYEOF
python /app/solution/analyze.py
