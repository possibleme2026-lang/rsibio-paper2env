# Review notes

- Data are synthetic but realistic: 97^3 voxel volumes with 1 nm isotropic spacing, generated from analytic SDFs with additive Gaussian noise. The generator is embedded in the build context and deleted after image build, so the agent cannot read the clean SDF or noise seed.
- Shortcut probe: The agent cannot read the clean SDF or gold metrics; they are only in tests/. The Dockerfile and gen_inputs.py are not present in the final container.
- Tool chain: numpy, scipy, scikit-image, trimesh for geometric curvature. The task focuses on the reconstruction algorithm rather than calling a single domain-specific binary.
- Known limitations: The task uses analytic surfaces rather than experimental EM data; however, the noise, voxelization, and curvature estimation make the reconstruction non-trivial and require multi-step reasoning.
