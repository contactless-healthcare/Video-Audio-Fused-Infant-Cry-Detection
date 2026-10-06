import numpy as np
import utils
from typing import List, Dict, Tuple, Callable
import warnings

# Suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)


MOTION_GRANULARITY_PARTS = {
    'global': ('WholeFrameMotion',),
    'somatic': ('WholeBody',),
    'segmental': ('Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg'),
}


def get_body_parts(granularity: str) -> Tuple[str, ...]:
    """Return the JSON motion fields used by a named granularity."""
    normalized = str(granularity).lower()
    if normalized not in MOTION_GRANULARITY_PARTS:
        choices = ', '.join(MOTION_GRANULARITY_PARTS)
        raise ValueError(
            f"Unknown motion granularity {granularity!r}; expected one of: {choices}."
        )
    return MOTION_GRANULARITY_PARTS[normalized]


def extract_amplitude_values(window_data: List[Dict], field_name: str) -> np.ndarray:
    """
    Extract motion amplitude values (r) from window data
    
    Args:
        window_data: List of frame data within the window
        field_name: Field name to extract (e.g., 'Face', 'Left-arm')
    
    Returns:
        Processed amplitude values array (outliers filtered)
    """
    amplitude_values = []
    for frame in window_data:
        if field_name in frame and frame[field_name]:
            candidates = frame[field_name]
            try:
                # WholeFrameMotion is emitted as [dx, dy, r, angle] and has no
                # parsing confidence. All mask-based fields use
                # [dx, dy, r, angle, confidence].
                if field_name == 'WholeFrameMotion':
                    best = candidates[0]
                else:
                    best = max(candidates, key=lambda x: x[4])
                amplitude = float(best[2])
                amplitude_values.append(
                    amplitude if np.isfinite(amplitude) else np.nan
                )
            except (IndexError, TypeError, ValueError):
                # A malformed frame-level measurement is missing, not a
                # sentinel value. The window statistics below use valid
                # frames only.
                amplitude_values.append(np.nan)
        else:
            amplitude_values.append(np.nan)
    
    amplitude_array = np.array(amplitude_values)
    return utils.filter_outliers(amplitude_array)

def get_feature_calculators() -> List[Tuple[str, Callable]]:
    """
    Get list of feature calculation functions
    
    Returns:
        List of (feature_name, calculation_function) tuples
    """
    return [
        ('std', lambda arr: np.nan if np.all(np.isnan(arr)) else np.nanstd(arr)),
        ('median', lambda arr: np.nan if np.all(np.isnan(arr)) else np.nanmedian(arr)),
        ('mean', lambda arr: np.nan if np.all(np.isnan(arr)) else np.nanmean(arr)),
        ('max', lambda arr: np.nan if np.all(np.isnan(arr)) else np.nanmax(arr)),
        ('min', lambda arr: np.nan if np.all(np.isnan(arr)) else np.nanmin(arr)),
    ]

def body_features_from_data(
    data: Dict,
    window_size_sec: float,
    step_size_sec: float,
    label_path: str,
    granularity: str = 'segmental',
) -> Tuple[np.ndarray, List[str]]:
    """Extract motion-window features from an already-loaded motion JSON."""
    body_parts = get_body_parts(granularity)
    features = data['features']
    fps = data['video_info']['fps']

    min_start_time, max_end_time = utils.get_valid_time_range(label_path)
    features = utils.crop_data_by_time(features, fps, min_start_time, max_end_time)

    window_size, step_size = utils.calculate_window_params(
        fps, window_size_sec, step_size_sec
    )
    window_start_indices = np.arange(0, len(features) - window_size + 1, step_size)

    feature_calculators = get_feature_calculators()
    feature_names = [
        f"{part}_{stat_name}"
        for part in body_parts
        for stat_name, _ in feature_calculators
    ]

    num_windows = len(window_start_indices)
    num_features = len(feature_names)
    features_array = np.zeros((num_windows, num_features))

    for i, start_idx in enumerate(window_start_indices):
        end_idx = start_idx + window_size
        window_data = features[start_idx:end_idx]

        window_features = []
        for part in body_parts:
            amplitude_values = extract_amplitude_values(window_data, part)
            for _, calculator in feature_calculators:
                window_features.append(calculator(amplitude_values))

        features_array[i] = window_features

    return features_array, feature_names


def body_features(
    input_path: str,
    window_size_sec: float,
    step_size_sec: float,
    label_path: str,
    granularity: str = 'segmental',
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract body motion features (based on motion amplitude) from input file

    Args:
        input_path: Path to input JSON file
        window_size_sec: Window size in seconds
        step_size_sec: Step size in seconds
        label_path: Path to label file (for determining valid time range)
        granularity: One of global, somatic, or segmental

    Returns:
        features_array: Extracted feature array (one row per window)
        feature_names: Corresponding feature names list
    """
    data = utils.load_and_validate_json(input_path)
    return body_features_from_data(
        data,
        window_size_sec,
        step_size_sec,
        label_path,
        granularity,
    )
