import os
import glob

import numpy as np
from tqdm import tqdm

from features.Feature_Extraction_Audio import acoustic_features_and_spectrogram, NICUWav2Segments
from features.Feature_Extraction_Body import body_features
from features.Feature_Extraction_Face import facial_features, _face_stats
import config
import utils

def extract_audio_features(dataDir):
    """Extract audio features"""
    labelDir = utils.get_label_file_path(dataDir)
    sceneDir = utils.get_scene_file_path(dataDir)
    if not os.path.exists(sceneDir):
        sample = NICUWav2Segments(dataDir, labelDir, None)
        sample['scene'] = [0] * len(sample['label']) 
    else:
        sample = NICUWav2Segments(dataDir, labelDir, sceneDir)
    acoustic_feature, acoustic_feature_name = acoustic_features_and_spectrogram(sample["data"])
    return acoustic_feature, acoustic_feature_name, sample["label"], sample["scene"]

def extract_body_motion_features(dataDir):
    """Extract body motion features"""
    file = utils.get_motion_feature_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    motion_feature, motion_feature_name = body_features(file, config.slidingWindows, config.step, labelDir)
    return motion_feature, motion_feature_name

def extract_facial_features(dataDir):
    """Extract facial features"""
    file = utils.get_face_landmark_file_path(dataDir)
    labelDir = utils.get_label_file_path(dataDir)
    facial_feature, facial_feature_name = facial_features(file, config.slidingWindows, config.step, labelDir)
    return facial_feature, facial_feature_name

def load_or_extract_features(audio_files):
    """
    Loads or extracts feature data from audio files, including feature names.
    
    Args:
        audio_files (list): List of paths to audio files (WAV format)
    
    Returns:
        dict: Dictionary mapping subject IDs to lists of feature dictionaries
    """
    # Get parent directory and create features directory
    parent_dir = os.path.dirname(audio_files[0])
    features_dir = os.path.join(os.path.dirname(parent_dir), 'Features')
    os.makedirs(features_dir, exist_ok=True)
    
    # Initialize dictionary to store subject features
    subject_features = {}
    
    # Process each audio file with progress bar
    for file_path in tqdm(audio_files, desc="Processing files"):
        # 1. Define multi-file paths and initialize data structure
        base_filename = utils.generate_feature_filename(file_path).replace('.npz', '')
        
        metadata_file_path = os.path.join(features_dir, f"{base_filename}_metadata.npz")
        acoustic_file_path = os.path.join(features_dir, f"{base_filename}_acoustic.npz")
        motion_file_path = os.path.join(features_dir, f"{base_filename}_motion.npz")
        face_file_path = os.path.join(features_dir, f"{base_filename}_face.npz")

        subject_id = utils.extract_subject_id(file_path)
        
        loaded_data = {}
        
        # --- 2. Core Load/Extract/Save Logic: Metadata -> Acoustic -> Motion -> Face ---

        # --- A. Metadata (Label/Scene): Must be loaded or extracted first ---
        if os.path.exists(metadata_file_path):
            try:
                data = np.load(metadata_file_path, allow_pickle=True)
                loaded_data.update({'label': data['label'], 'scene': data['scene']})
            except Exception as error:
                print(f"Metadata load error, regenerating: {error}")
                loaded_data = {} # Force full regeneration if metadata fails

        if 'label' not in loaded_data:
            # Check if wav exists before trying to extract
            if not os.path.exists(file_path):
                print(f"Warning: Features missing for {file_path} and wav not found. Skipping.")
                continue
            # Extract acoustic to get segment definitions (label/scene)
            acoustic_features, acoustic_feature_names, label, scene = extract_audio_features(file_path)
            
            # Save Metadata
            np.savez(metadata_file_path, label=label, scene=scene)
            loaded_data.update({'label': label, 'scene': scene})
            
            # Save Acoustic features (extracted simultaneously with metadata)
            np.savez(acoustic_file_path, acoustic=acoustic_features, acoustic_feature_names=acoustic_feature_names)
            loaded_data.update({'acoustic': acoustic_features, 'acoustic_feature_names': acoustic_feature_names})
        
        # Skip if metadata extraction failed
        if 'label' not in loaded_data:
            print(f"Fatal error: Could not obtain metadata for {file_path}. Skipping.")
            continue


        # --- B. Acoustic Features (Only regenerate if file is missing/corrupted) ---
        if 'acoustic' not in loaded_data:
            if os.path.exists(acoustic_file_path):
                try:
                    data = np.load(acoustic_file_path, allow_pickle=True)
                    loaded_data.update({'acoustic': data['acoustic'], 'acoustic_feature_names': data['acoustic_feature_names'].tolist()})
                except Exception as error:
                    print(f"Acoustic load error, regenerating: {error}")

        if 'acoustic' not in loaded_data:
            # Extract only features (label/scene already handled)
            acoustic_features, acoustic_feature_names, _, _ = extract_audio_features(file_path)
            
            np.savez(acoustic_file_path, acoustic=acoustic_features, acoustic_feature_names=acoustic_feature_names)
            loaded_data.update({'acoustic': acoustic_features, 'acoustic_feature_names': acoustic_feature_names})


        # --- C. Body Motion Features ---
        if 'motion' not in loaded_data:
            if os.path.exists(motion_file_path):
                try:
                    data = np.load(motion_file_path, allow_pickle=True)
                    loaded_data.update({'motion': data['motion'], 'motion_feature_names': data['motion_feature_names'].tolist()})
                except Exception as error:
                    print(f"Motion load error, regenerating: {error}")

        if 'motion' not in loaded_data:
            motion_features, motion_feature_names = extract_body_motion_features(file_path)
            
            np.savez(motion_file_path, motion=motion_features, motion_feature_names=motion_feature_names)
            loaded_data.update({'motion': motion_features, 'motion_feature_names': motion_feature_names})


        # --- D. Facial Features ---
        if 'face' not in loaded_data:
            if os.path.exists(face_file_path):
                try:
                    data = np.load(face_file_path, allow_pickle=True)
                    loaded_data.update({'face': data['face'], 'face_feature_names': data['face_feature_names'].tolist()})
                except Exception as error:
                    print(f"Face load error, regenerating: {error}")

        if 'face' not in loaded_data:
            facial_features, facial_feature_names = extract_facial_features(file_path)
            
            np.savez(face_file_path, face=facial_features, face_feature_names=facial_feature_names)
            loaded_data.update({'face': facial_features, 'face_feature_names': facial_feature_names})

        # 3. Store the final result or skip if incomplete
        required_keys = ['acoustic', 'motion', 'face', 'label', 'scene']
        if all(key in loaded_data for key in required_keys): 
            subject_features.setdefault(subject_id, []).append(loaded_data)
        else:
            print(f"Warning: Incomplete data for {file_path}. Skipping. Keys found: {loaded_data.keys()}")
    
    # Counts include repeated frames in overlapping windows, after IQR filtering.
    if _face_stats['total'] > 0:
        t = _face_stats['total']
        print(f"Face window observations | Total: {t}, valid ratio: {1 - _face_stats['errors'] / t:.2%}")
        print(f"  - MAR valid ratio: {1 - _face_stats['mar_errors'] / t:.2%}")
        print(f"  - L-EAR valid ratio: {1 - _face_stats['left_ear_errors'] / t:.2%}")
        print(f"  - R-EAR valid ratio: {1 - _face_stats['right_ear_errors'] / t:.2%}")

    return subject_features

def prepare_feature_matrices(subject_data, enable_scene_printing=False):
    """Prepare feature matrices and print dataset statistics."""
    print("\nPreparing feature matrices...")
    
    # Lists to store features and labels
    acoustic_features = []
    motion_features = []
    facial_features = []
    labels = []
    scenes = []
    subject_ids = []
    
    # Get feature names (assumes all samples have the same feature names)
    first_sample = next(iter(subject_data.values()))[0]
    acoustic_feature_names = first_sample.get('acoustic_feature_names', None)
    motion_feature_names = first_sample.get('motion_feature_names', None)
    facial_feature_names = first_sample.get('face_feature_names', None)
    
    if None in [acoustic_feature_names, motion_feature_names, facial_feature_names]:
        raise ValueError("Feature names not properly loaded! Please check the feature extraction function.")
    
    # Track infant crying status by base ID
    crying_infants = set()
    non_crying_infants = set()
    infant_crying_status = {}  # Maps base infant ID to whether they have cried
    
    # Track segment crying statistics
    crying_segment_count = 0
    non_crying_segment_count = 0
    
    # Track scene distribution
    scene_counts = {}  # {scene_type: count}
    
    for subj_id_with_timestamp, samples in tqdm(subject_data.items(), desc="Processing subjects"):
        # Extract the infant ID before the record suffix.
        base_infant_id = subj_id_with_timestamp.split('_')[0]
        
        # Initialize infant crying status if not already recorded
        if base_infant_id not in infant_crying_status:
            infant_crying_status[base_infant_id] = False
        
        has_crying = False
        for sample in samples:
            window_counts = {
                key: len(sample[key])
                for key in ('label', 'scene', 'acoustic', 'motion', 'face')
            }
            if len(set(window_counts.values())) != 1:
                raise ValueError(
                    f"Window count mismatch for {subj_id_with_timestamp}: "
                    f"{window_counts}"
                )

            frame_count = window_counts['label']
            for i in range(frame_count):
                label = sample['label'][i]
                scene = sample['scene'][i]  
                
                if scene not in scene_counts:
                    scene_counts[scene] = 0
                scene_counts[scene] += 1
                
                if label == 1:  # 1 indicates crying
                    crying_segment_count += 1
                    has_crying = True
                else:
                    non_crying_segment_count += 1
                                
                # Append features and labels for each modality
                acoustic_features.append(sample['acoustic'][i])
                motion_features.append(sample['motion'][i])
                facial_features.append(sample['face'][i])
                labels.append(label)
                scenes.append(scene)
                subject_ids.append(subj_id_with_timestamp)
        
        # Update infant crying status
        if has_crying:
            infant_crying_status[base_infant_id] = True
    
    # Count infants based on crying status
    for base_infant_id, has_cried in infant_crying_status.items():
        if has_cried:
            crying_infants.add(base_infant_id)
        else:
            non_crying_infants.add(base_infant_id)
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total infants: {len(infant_crying_status)}")
    print(f"Crying infants: {len(crying_infants)}")
    print(f"Non-crying infants: {len(non_crying_infants)}")
    print(f"\nTotal segments: {len(labels)}")
    print(f"Crying segments: {crying_segment_count}")
    print(f"Non-crying segments: {non_crying_segment_count}")
    print(f"Crying segment proportion: {crying_segment_count/len(labels):.2%}")

    print("\n=== Feature Dimensions ===")
    print(f"Acoustic features: {len(acoustic_feature_names)}")
    print(f"Motion features: {len(motion_feature_names)}")
    print(f"Facial features: {len(facial_feature_names)}")
    
    if not np.all(np.array(scenes) == 0) and enable_scene_printing:
        print("\n=== Scene Distribution ===")
        total_scenes = sum(scene_counts.values())
        for scene in sorted(scene_counts.keys()):
            count = scene_counts[scene]
            print(f"Scene {scene}: {count} segments ({count/total_scenes:.2%})")
    
    return (
        subject_ids,
        acoustic_features,
        motion_features,
        facial_features,
        acoustic_feature_names,
        motion_feature_names,
        facial_feature_names,
        labels,
        scenes,
    )

def load_data(enable_scene_printing=False):
    """Load and prepare all data.

    Supports two modes:
    1. With sample_manifest.txt: Read sample names from manifest (parent of dataDir/)
    2. With wav files: Scan *.wav files in dataDir (backward compatible fallback)
    """
    prefix = config.dataDir
    manifest_path = os.path.join(os.path.dirname(prefix), 'sample_manifest.txt')

    # Try manifest first, fall back to wav scanning
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            sample_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(sample_names)} samples from manifest: {manifest_path}")
        # Construct wav paths for feature extraction
        wav_files = [os.path.join(prefix, f"{name}.wav") for name in sample_names]
    else:
        wav_files = glob.glob(os.path.join(prefix, "*.wav"))
        if not wav_files:
            raise FileNotFoundError(
                f"No wav files found in {prefix} and no manifest at {manifest_path}."
            )
        print(f"Loaded {len(wav_files)} samples from wav files")

    subject_data = load_or_extract_features(wav_files)
    print(f"Total subjects: {len(subject_data)}")
    
    (subject_ids, acoustic_features, motion_features, 
     face_features, acoustic_feature_names,
     motion_feature_names, face_feature_names, labels, scenes) = prepare_feature_matrices(subject_data, enable_scene_printing)

    # The cache retains 15 MAR/EAR columns; classification uses the first five MAR statistics.
    face_feature_names = face_feature_names[:5]
    face_features[:] = [row[:5] for row in face_features]

    return (
        subject_ids,
        (acoustic_features, motion_features, face_features),  
        (acoustic_feature_names, motion_feature_names, face_feature_names),
        labels,
        scenes,
    )