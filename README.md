# MMInfant

Code for the paper **“A Video-Audio Fused Infant Cry Detection System in the Noisy Clinical Environment.”**

Infant cry detection using audio, body motion, and facial features. Includes classification code, de-identified datasets, and feature extraction.

## Installation

Classification runs on Linux with a CPU. Run all commands from the repository root.

```bash
conda create -n mminfant python=3.9.25 pip -y
conda activate mminfant
python -m pip install -r requirements.txt
```

## Data

`dataset/` contains precomputed features, labels, and fixed ten-fold splits, grouped by infant.

| Dataset | Infants | Recordings | Windows |
|---|---:|---:|---:|
| NEWBORN200 | 200 | 200 | 3,236 |
| NICU50 | 50 | 63 | 21,504 |

Each recording has four NPZ files in `Features/`, with one row per window:

| File | Fields | Contents |
|---|---|---|
| `*_acoustic.npz` | `acoustic`, `acoustic_feature_names` | 102 acoustic features and column names |
| `*_motion.npz` | `motion`, `motion_feature_names` | 25 motion features and column names |
| `*_face.npz` | `face`, `face_feature_names` | 15 MAR/EAR features and column names; classification uses the first five MAR columns |
| `*_metadata.npz` | `label`, `scene` | Cry and scenario labels |

`label`: 0 = no target-infant cry, 1 = target-infant cry. `scene`: 0 = unannotated (NEWBORN200); 1–8 = NICU50 scenarios S1–S8 defined in the paper's Scenario Categorization section.

Raw clinical recordings are private.

## Training and evaluation

Run the main classification experiments and paired statistical comparisons:

```bash
python scripts/run_main_experiments.py
```

This evaluates SVM, random forest, and LightGBM with Audio, Motion, Face, Early Fusion, and Late Fusion. Results are saved to `results/main_experiment/`; rerunning replaces them.

After training, plot the **paired F1-score comparison** (LGBM Early Fusion vs. best LGBM single modality):

```bash
python scripts/plot_paired_macro_f1.py
```

## Feature extraction

To process your own recordings, follow [InfantVision](InfantVision/README.md). Skip this step when using the supplied features.

## Acknowledgements

See [third-party sources and licenses](InfantVision/README.md#third-party-sources-and-licenses).
