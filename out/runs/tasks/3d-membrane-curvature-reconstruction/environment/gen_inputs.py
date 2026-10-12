import numpy as np, os
os.makedirs('/app/data/volumes', exist_ok=True)
N = 97
coords = np.arange(N, dtype=np.float64) - 48.0
X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
def sdf_sphere(X, Y, Z): return np.sqrt(X**2 + Y**2 + Z**2) - 20.0
def sdf_cylinder(X, Y, Z): return np.maximum(np.sqrt(X**2 + Y**2) - 12.0, np.abs(Z) - 30.0)
def sdf_torus(X, Y, Z):
    q = np.sqrt(X**2 + Y**2) - 25.0
    return np.sqrt(q**2 + Z**2) - 8.0
def sdf_catenoid(X, Y, Z):
    a = 8.0
    return np.sqrt(X**2 + Y**2) - a * np.cosh(Z / a)
def sdf_wavy(X, Y, Z):
    A = 5.0
    lam = 32.0
    return Z - A * np.sin(2 * np.pi * X / lam) * np.cos(2 * np.pi * Y / lam)
shapes = [
    ('vol_00', sdf_sphere),
    ('vol_01', sdf_cylinder),
    ('vol_02', sdf_torus),
    ('vol_03', sdf_catenoid),
    ('vol_04', sdf_wavy),
]
rng = np.random.default_rng(12345)
for name, fn in shapes:
    sdf = fn(X, Y, Z)
    intensity = np.exp(-(sdf / 1.2)**2) + 0.05
    intensity += rng.normal(0, 0.10, intensity.shape)
    np.save(f'/app/data/volumes/{name}.npy', intensity.astype(np.float32))
with open('/app/data/metadata.csv', 'w') as f:
    f.write('volume_id,voxel_size_nm,roi_expr\n')
    f.write('vol_00,1.0,all\n')
    f.write('vol_01,1.0,abs(z)<=25\n')
    f.write('vol_02,1.0,all\n')
    f.write('vol_03,1.0,abs(z)<=8\n')
    f.write('vol_04,1.0,(abs(x)<=24)&(abs(y)<=24)\n')
