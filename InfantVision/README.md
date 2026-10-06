# Video-to-feature extraction

Extract features from your own recordings. For supplied features, skip this guide. Run commands from the repository root.

## Installation

Requires Linux, an NVIDIA GPU, a compatible driver and C++ build tools. Also install `mminfant` using the [main README](../README.md).

Create the visual environment:

```bash
conda create -n mminfant-vision python=3.8.20 pip -y
conda activate mminfant-vision
python -m pip install -r InfantVision/requirements.txt
```

Install the CUDA build tools:

```bash
conda install -c nvidia/label/cuda-12.1.1 \
  cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1 -y
```

Build and check MMCV. `8.9` targets RTX 4090 / RTX 6000 Ada; adjust for your GPU.

```bash
CUDA_HOME="$CONDA_PREFIX" MMCV_WITH_OPS=1 \
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
python -m pip install \
  --no-cache-dir --no-binary=mmcv --no-build-isolation mmcv==2.2.0
python -c "import mmcv._ext"
```


## Model weights

**Body parsing** — official ModelScope weights and configuration:

```bash
mkdir -p weights/body
curl --fail --location \
  'https://modelscope.cn/api/v1/models/iic/cv_resnet101_image-multiple-human-parsing/repo?Revision=v1.0.1&FilePath=pytorch_model.pt' \
  --output weights/body/pytorch_model.pt
curl --fail --location \
  'https://modelscope.cn/api/v1/models/iic/cv_resnet101_image-multiple-human-parsing/repo?Revision=v1.0.1&FilePath=configuration.json' \
  --output weights/body/configuration.json
```

**Face detection** — YOLOv5l-face weights distributed by CodeFormer; only the detector weights are used.

```bash
curl --fail --location \
  'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5l-face.pth' \
  --output weights/yolov5l-face.pth
```

**Face landmarks** — our trained HRNet checkpoint, hosted on [Hugging Face](https://huggingface.co/ShaneXan/MMInfant-HRNet-R90JT):

```bash
curl --fail --location \
  'https://huggingface.co/ShaneXan/MMInfant-HRNet-R90JT/resolve/0cbe8508978eb7d3d784fabedfa78454cc086618/hrnet-r90jt.pth' \
  --output weights/hrnet-r90jt.pth
```

## Process video

Use a video and synchronized WAV with matching basenames.

Extract body motion:

```bash
conda activate mminfant-vision
python InfantVision/body_seg_5.py \
  --video input/recording.mp4 --model-dir weights/body \
  --output-dir work/Body
```

Extract facial landmarks in the same environment:

```bash
python InfantVision/facial_landmark_68.py \
  --video input/recording.mp4 \
  --landmark-weights weights/hrnet-r90jt.pth \
  --detector-weights weights/yolov5l-face.pth \
  --output-dir work/Face
```

Outputs: per-frame JSON and previews. Face also supports `--device cpu`.

## Generate features

```bash
conda activate mminfant
python scripts/extract_window_features.py \
  --audio input/recording.wav --label input/recording.txt \
  --body-json work/Body/recording_motion_features.json \
  --face-json work/Face/recording_face_landmarks.json \
  --output-dir work/Features --record-id recording
```

Labels: headerless TSV with `start_seconds`, `end_seconds`, `cry_label` (0/1). Optional `--scene` uses the same format with IDs 1–8, within the label range.

Outputs: metadata, acoustic, motion and face NPZ files; 2.5-second windows, 1.5-second steps. Use 30 FPS; incompatible Body/Face frame counts (e.g. at 29.97 FPS) are rejected. All extraction commands refuse to overwrite existing outputs.

## Third-party sources and licenses

- Face detector: [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face) — [GPL-3.0](licenses/YOLOv5-face-GPL-3.0.txt).
- Face landmarks: [InfAnFace](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) — [MIT](licenses/InfAnFace-MIT.txt).
- Body parsing: [ModelScope](https://modelscope.cn/models/iic/cv_resnet101_image-multiple-human-parsing).

Third-party code and weights retain their original licenses.
