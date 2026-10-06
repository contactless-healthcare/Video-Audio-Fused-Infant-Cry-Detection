#!/usr/bin/env python3
"""Extract four NPZ files for one explicitly specified, synchronized recording.

Inputs are a WAV, Body/Face JSON, and tab-separated start/end/class intervals
in seconds from the same recording origin. Cry labels use 0/1. Optional scene
intervals use IDs 1..8 and must stay within the label file's time range;
without --scene metadata uses 0, matching the loader.
Output rows follow the existing 2.5 s window / 1.5 s step in code/config.py.
The face cache retains all 15 candidate descriptors; classification selects MAR.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def validate_intervals(path, allowed_codes, bounds=None):
    count = 0
    min_start, max_end = float("inf"), 0.0
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) != 3:
                raise ValueError(f'{path}:{number}: expected three tab-separated columns')
            start, end = float(parts[0]), float(parts[1])
            code = int(parts[2])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError(f'{path}:{number}: invalid interval in seconds')
            if code not in allowed_codes:
                raise ValueError(f'{path}:{number}: unsupported annotation code {code}')
            if bounds is not None and not (bounds[0] <= start and end <= bounds[1]):
                raise ValueError(f'{path}:{number}: scene intervals must lie within the label range [{bounds[0]}, {bounds[1]}] seconds')
            min_start, max_end = min(min_start, start), max(max_end, end)
            count += 1
    if not count:
        raise ValueError(f'Annotation file is empty: {path}')
    return min_start, max_end


def validate_visual_alignment(body_path, face_path, window_seconds, step_seconds):
    summaries = []
    for path, frame_key in ((body_path, 'features'), (face_path, 'frames')):
        with path.open() as stream:
            data = json.load(stream)
        fps = data.get('video_info', {}).get('fps')
        frames = data.get(frame_key)
        if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'{path}: video_info.fps must be a positive number')
        if not isinstance(frames, list) or not frames:
            raise ValueError(f'{path}: {frame_key} must contain frame records')
        declared_count = data.get('video_info', {}).get('frame_count')
        if declared_count is not None and declared_count != len(frames):
            raise ValueError(f'{path}: frame_count differs from the number of frame records')
        summaries.append((fps, len(frames)))
    if summaries[0] != summaries[1]:
        raise ValueError('Body and Face JSON must have identical FPS and frame counts')
    fps = summaries[0][0]
    rounded = tuple(int(round(fps * seconds)) for seconds in (window_seconds, step_seconds))
    truncated = tuple(int(fps * seconds) for seconds in (window_seconds, step_seconds))
    if rounded != truncated or min(truncated) < 1:
        raise ValueError('This FPS gives different Body/Face window or step frame counts; use synchronized inputs with matching frame counts (30 FPS satisfies this check)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audio', type=Path, required=True, help='WAV synchronized with the input video.')
    parser.add_argument('--label', type=Path, required=True, help='TSV: start_seconds, end_seconds, cry 0/1.')
    parser.add_argument('--body-json', type=Path, required=True, help='Body motion JSON for this recording.')
    parser.add_argument('--face-json', type=Path, required=True, help='Face landmarks JSON for this recording.')
    parser.add_argument('--scene', type=Path, help='Optional TSV: start_seconds, end_seconds, scene ID 1..8.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for the four generated NPZ files.')
    parser.add_argument('--record-id', help='Output filename stem; defaults to the WAV filename stem.')
    args = parser.parse_args()
    for path in (args.audio, args.label, args.body_json, args.face_json, args.scene):
        if path is not None and not path.is_file():
            parser.error(f'Input file does not exist: {path}')
    record_id = args.record_id or args.audio.stem
    if record_id in {'.', '..'} or not record_id or any(c in record_id for c in '/\\\r\n\t'):
        parser.error('--record-id must be a filename stem, not a path')
    outputs = {kind: args.output_dir / f'{record_id}_{kind}.npz'
               for kind in ('metadata', 'acoustic', 'motion', 'face')}
    if any(path.exists() for path in outputs.values()):
        parser.error('An output already exists; choose a new output directory or record ID.')
    try:
        label_bounds = validate_intervals(args.label, {0, 1})
        if args.scene is not None:
            validate_intervals(args.scene, set(range(1, 9)), bounds=label_bounds)
    except ValueError as error:
        parser.error(str(error))

    # Import after argument validation so --help needs only the standard library.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
    import numpy as np
    import config
    try:
        validate_visual_alignment(args.body_json, args.face_json, config.slidingWindows, config.step)
    except ValueError as error:
        parser.error(str(error))
    from features.Feature_Extraction_Audio import NICUWav2Segments, acoustic_features_and_spectrogram
    from features.Feature_Extraction_Body import body_features
    from features.Feature_Extraction_Face import facial_features

    sample = NICUWav2Segments(str(args.audio), str(args.label), str(args.scene) if args.scene else None)
    if args.scene is None:
        sample['scene'] = [0] * len(sample['label'])
    acoustic, acoustic_names = acoustic_features_and_spectrogram(sample['data'])
    motion, motion_names = body_features(str(args.body_json), config.slidingWindows, config.step, str(args.label))
    face, face_names = facial_features(str(args.face_json), config.slidingWindows, config.step, str(args.label))
    n_windows = len(sample['label'])
    for name, values, names, width in (('acoustic', acoustic, acoustic_names, 102),
                                      ('motion', motion, motion_names, 25),
                                      ('face', face, face_names, 15)):
        if n_windows == 0 or values.shape != (n_windows, width) or len(names) != width:
            raise ValueError(f'{name} shape {values.shape} does not match {n_windows} windows x {width}; check synchronization and annotation range')
    if len(sample['scene']) != n_windows:
        raise ValueError('Scene and cry labels have different window counts')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(outputs['metadata'], label=sample['label'], scene=sample['scene'])
    np.savez(outputs['acoustic'], acoustic=acoustic, acoustic_feature_names=acoustic_names)
    np.savez(outputs['motion'], motion=motion, motion_feature_names=motion_names)
    np.savez(outputs['face'], face=face, face_feature_names=face_names)
    print(f'Saved {n_windows} windows for {record_id}: audio 102D, motion 25D, face 15D (classification uses MAR 5D).')
    for path in outputs.values():
        print(path)


if __name__ == '__main__':
    main()
