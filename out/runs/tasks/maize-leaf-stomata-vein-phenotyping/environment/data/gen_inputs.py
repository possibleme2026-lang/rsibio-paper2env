import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import cKDTree
import os

SEED = 20240501
N_RIL = 50
N_IMAGES_PER_RIL = 2
IMG_SIZE = 512
PIXELS_PER_MM = 512
AREA_MM2 = (IMG_SIZE / PIXELS_PER_MM) ** 2

def main():
    rng = np.random.default_rng(SEED)
    os.makedirs('/app/data/images', exist_ok=True)
    n_chrom = 10
    markers_per_chrom = 20
    marker_info = []
    for chrom in range(1, n_chrom+1):
        pos = 0.0
        for i in range(markers_per_chrom):
            pos += rng.uniform(2, 10)
            marker_info.append({'marker_id': f'M{chrom}_{i+1}', 'chromosome': chrom, 'position': round(pos, 2)})
    marker_df = pd.DataFrame(marker_info)
    marker_df.to_csv('/app/data/marker_info.csv', index=False)
    geno = rng.integers(0, 2, size=(N_RIL, len(marker_df)))
    geno_df = pd.DataFrame(geno, columns=marker_df['marker_id'])
    geno_df.insert(0, 'ril_id', [f'RIL_{i+1:03d}' for i in range(N_RIL)])
    geno_df.to_csv('/app/data/genotypes.csv', index=False)
    qtl_sd = [('M1_3', 25.0), ('M4_7', -18.0)]
    qtl_vd = [('M2_5', 0.4), ('M7_10', -0.25)]
    qtl_nnd = [('M5_8', 0.12), ('M9_4', -0.09)]
    base_sd = 120.0
    base_vd = 2.0
    base_nnd = 0.6
    ril_means = {}
    for trait, qtl_list, base in [('stomatal_density', qtl_sd, base_sd), ('vein_density', qtl_vd, base_vd), ('nnd_cv', qtl_nnd, base_nnd)]:
        vals = np.full(N_RIL, base)
        for marker_id, eff in qtl_list:
            idx = list(marker_df['marker_id']).index(marker_id)
            vals += eff * geno[:, idx]
        vals += rng.normal(0, base*0.08, N_RIL)
        ril_means[trait] = vals
    metadata = []
    for ril_idx in range(N_RIL):
        ril_id = f'RIL_{ril_idx+1:03d}'
        for img_idx in range(N_IMAGES_PER_RIL):
            sd = max(30, ril_means['stomatal_density'][ril_idx] + rng.normal(0, 8))
            vd = max(0.5, ril_means['vein_density'][ril_idx] + rng.normal(0, 0.1))
            nnd_cv_target = max(0.2, ril_means['nnd_cv'][ril_idx] + rng.normal(0, 0.05))
            n_stomata = int(round(sd * AREA_MM2))
            n_veins = max(1, int(round(vd)))
            vein_thickness_px = rng.normal(15, 3)
            img_array = rng.normal(180, 10, (IMG_SIZE, IMG_SIZE))
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            draw = ImageDraw.Draw(img)
            for v in range(n_veins):
                y_center = int((v + 0.5) * IMG_SIZE / n_veins + rng.normal(0, 5))
                thickness = int(vein_thickness_px + rng.normal(0, 2))
                y0 = max(0, y_center - thickness//2)
                y1 = min(IMG_SIZE, y_center + thickness//2)
                draw.rectangle([0, y0, IMG_SIZE, y1], fill=50)
            img = img.filter(ImageFilter.GaussianBlur(1))
            n_clusters = max(1, int(10 + 20 * nnd_cv_target))
            cluster_centers = rng.uniform(0, IMG_SIZE, (n_clusters, 2))
            points = []
            for _ in range(n_stomata):
                center = cluster_centers[rng.integers(n_clusters)]
                x = int(center[0] + rng.normal(0, 25))
                y = int(center[1] + rng.normal(0, 25))
                x = np.clip(x, 5, IMG_SIZE-5)
                y = np.clip(y, 5, IMG_SIZE-5)
                points.append((x, y))
            for (x, y) in points:
                major = int(rng.normal(12, 2))
                minor = int(rng.normal(6, 1))
                major = max(4, major)
                minor = max(2, minor)
                draw.ellipse([x-minor//2, y-major//2, x+minor//2, y+major//2], fill=30)
            img_id = f'{ril_id}_img{img_idx+1}'
            img.save(f'/app/data/images/{img_id}.png')
            metadata.append({'image_id': img_id, 'ril_id': ril_id, 'pixels_per_mm': PIXELS_PER_MM})
    pd.DataFrame(metadata).to_csv('/app/data/image_metadata.csv', index=False)

if __name__ == '__main__':
    main()
