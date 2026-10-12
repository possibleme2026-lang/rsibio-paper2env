# Acoustic Feature Descriptions

The dataset contains 12 acoustic features extracted from environmental sound clips. All features are normalized to have zero mean and unit variance across the dataset.

- `f1`: Spectral centroid (weighted mean of frequencies)
- `f2`: Spectral rolloff (frequency below which 85% of energy is contained)
- `f3`: Spectral flatness (ratio of geometric mean to arithmetic mean of power spectrum)
- `f4`: RMS energy (root mean square of the signal)
- `f5`: Zero-crossing rate (rate of sign changes in the signal)
- `f6`: Mel-frequency cepstral coefficient 1 (MFCC1)
- `f7`: MFCC2
- `f8`: MFCC3
- `f9`: Spectral bandwidth
- `f10`: Spectral contrast
- `f11`: Chroma energy
- `f12`: Temporal centroid (temporal location of the signal's energy)

Metadata columns:
- `category`: source category (dog, rain, car, speech, music)
- `distortion`: acoustic distortion applied (none, lowpass, clipping, time_reverse)
- `n_sources`: number of concurrent sound sources (1, 2, or 3)
- `snr_db`: signal-to-noise ratio in dB (0, 10, 20, or 30)
