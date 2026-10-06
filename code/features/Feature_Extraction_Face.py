import numpy as np
from typing import List, Dict, Tuple, Callable
from math import isnan
import utils

# Facial landmark indices
MOUTH_POINTS = (48, 54, 56, 58, 50, 52)
LEFT_EYE_POINTS = range(36, 42)
RIGHT_EYE_POINTS = range(42, 48)

# Error tracking statistics
_face_stats = {
    'total': 0,
    'mar_errors': 0,
    'left_ear_errors': 0,
    'right_ear_errors': 0,
    'errors': 0
}

# Confidence filtering thresholds
FACE_CONFIDENCE_THRESHOLD = 0.80
MAR_AVG_CONF_THRESHOLD = 0.40  # Minimum mean confidence across mouth landmarks.
EAR_AVG_CONF_THRESHOLD = 0.40

def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_mar(landmarks: List[List[float]]) -> float:
    """Calculate Mouth Aspect Ratio (MAR)"""
    try:
        if len(landmarks) < 68: return np.nan
        p48, p54, p56, p58, p50, p52 = [landmarks[i] for i in MOUTH_POINTS]
        mouth_width = calculate_distance(p54, p48)
        if mouth_width < 1e-6: return np.nan
        
        left_lip = calculate_distance(p58, p50)
        right_lip = calculate_distance(p56, p52)
        mar = (left_lip + right_lip) / (2 * mouth_width)
        return round(float(mar), 4) if not isnan(mar) else np.nan
    except Exception:
        return np.nan

def calculate_eye_aspect_ratio(eye_points: List[List[float]]) -> float:
    """Calculate Eye Aspect Ratio (EAR) for single eye"""
    A = calculate_distance(eye_points[1], eye_points[5])
    B = calculate_distance(eye_points[2], eye_points[4])
    C = calculate_distance(eye_points[0], eye_points[3])
    return (A + B) / (2 * C) if C > 1e-6 else np.nan

def calculate_ear(landmarks: List[List[float]]) -> Tuple[float, float]:
    """Calculate EAR for both eyes (left, right)"""
    try:
        if len(landmarks) < 68: return (np.nan, np.nan)
        left_eye = [landmarks[i] for i in LEFT_EYE_POINTS]
        right_eye = [landmarks[i] for i in RIGHT_EYE_POINTS]
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        return (
            round(float(left_ear), 4) if not isnan(left_ear) else np.nan,
            round(float(right_ear), 4) if not isnan(right_ear) else np.nan
        )
    except Exception:
        return (np.nan, np.nan)

def _check_confidence(frame: Dict, indices: Tuple[int, ...], face_conf_thresh: float, part_conf_thresh: float) -> List[List[float]]:
    """Confidence filtering: face detection + landmark quality"""
    landmarks = frame.get('landmarks')
    face_conf = frame.get('face_confidence')
    landmark_confs = frame.get('landmark_confidences')
    
    if landmarks is None or face_conf is None or landmark_confs is None:
        return []

    # Face detection confidence filter
    if face_conf < face_conf_thresh:
        return []
        
    # Landmark confidence filter
    critical_confs = [landmark_confs[i] for i in indices]
    part_conf_score = np.median(critical_confs)
    
    if part_conf_score < part_conf_thresh:
        return []
    
    return landmarks

def extract_window_features(window_frames: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract window features with confidence filtering and error tracking"""
    num_frames = len(window_frames)
    _face_stats['total'] += num_frames

    # Feature calculation
    raw_mars_list = []
    raw_left_ears_list = []
    raw_right_ears_list = []

    for frame in window_frames:
        # MAR feature
        landmarks_mar = _check_confidence(frame, MOUTH_POINTS, FACE_CONFIDENCE_THRESHOLD, MAR_AVG_CONF_THRESHOLD)
        raw_mars_list.append(calculate_mar(landmarks_mar))
        
        # Left EAR feature
        landmarks_left_ear = _check_confidence(frame, LEFT_EYE_POINTS, FACE_CONFIDENCE_THRESHOLD, EAR_AVG_CONF_THRESHOLD)
        raw_left_ears_list.append(calculate_ear(landmarks_left_ear)[0])

        # Right EAR feature
        landmarks_right_ear = _check_confidence(frame, RIGHT_EYE_POINTS, FACE_CONFIDENCE_THRESHOLD, EAR_AVG_CONF_THRESHOLD)
        raw_right_ears_list.append(calculate_ear(landmarks_right_ear)[1])

    # Convert to NumPy arrays
    raw_feats = {
        'mars': np.array(raw_mars_list),
        'left_ears': np.array(raw_left_ears_list),
        'right_ears': np.array(raw_right_ears_list)
    }

    # Error statistics
    feats = {}
    is_frame_error = np.zeros(num_frames, dtype=bool)
    STAT_KEYS = {
        'mars': 'mar_errors',
        'left_ears': 'left_ear_errors',
        'right_ears': 'right_ear_errors'
    }
    
    for k, v in raw_feats.items():
        error_key = STAT_KEYS[k]

        cleaned = utils.filter_outliers(v.copy())
        feats[k] = cleaned

        final_nans_mask = np.isnan(cleaned)
        final_nans_count = np.sum(final_nans_mask)

        _face_stats[error_key] += final_nans_count
        is_frame_error = is_frame_error | final_nans_mask   
        
    _face_stats['errors'] += np.sum(is_frame_error)

    return feats

def create_feature_calculator(func: Callable, feature_key: str) -> Callable:
    """Factory function to create feature calculators"""
    def calculator(data: Dict[str, np.ndarray]) -> float:
        values = data[feature_key]
        return np.nan if np.all(np.isnan(values)) else func(values)

    return calculator

def get_facial_feature_calculators() -> List[Tuple[str, Callable]]:
    """Generate statistical feature calculators (mean, median, std, etc.)"""
    FEATURES = {
        'mar': 'mars', 'left_ear': 'left_ears', 'right_ear': 'right_ears'
    }
    STATS = {
        'mean': np.nanmean, 'median': np.nanmedian, 'std': np.nanstd, 
        'min': np.nanmin, 'max': np.nanmax
    }
    
    calculators = []
    for feature_name, data_key in FEATURES.items():
        for stat_name, stat_func in STATS.items():
            name = f"{feature_name}_{stat_name}"
            calculator = create_feature_calculator(stat_func, data_key)
            calculators.append((name, calculator))
    
    return calculators

def facial_features(
    input_path: str,
    window_size_sec: float,
    step_size_sec: float,
    label_path: str
) -> Tuple[np.ndarray, List[str]]:
    """Main pipeline for facial feature extraction"""
    try:
        data = utils.load_and_validate_json(input_path)
        fps = data['video_info']['fps']
        frames = utils.crop_data_by_time(
            data.get('frames', []), fps, *utils.get_valid_time_range(label_path)
        )
        
        window_size = int(fps * window_size_sec)
        step_size = int(fps * step_size_sec)
        n_frames = len(frames)
        
        window_starts = range(0, n_frames - window_size + 1, step_size)
        calculators = get_facial_feature_calculators()
        feature_names = [name for name, _ in calculators]
        
        # Extract features for all windows
        all_window_features = [
            extract_window_features(frames[start:start+window_size])
            for start in window_starts
        ]

        # Apply statistical calculators to window features
        features = np.array([
            [calc(window_feats) for _, calc in calculators]
            for window_feats in all_window_features
        ])

        return features, feature_names
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        n_features = len(get_facial_feature_calculators())
        return np.zeros((0, n_features)), []