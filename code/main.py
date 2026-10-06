"""Evaluate the selected modalities on fixed outer folds."""

def execute_evaluations(model_evaluator, feature_sets, target_labels, participant_ids, model_type, tasks):
    """
    Execute the evaluations selected by tasks.
    Returns fold metrics for each task.
    """
    # Unpack feature sets for clarity
    audio_features, motion_features, facial_features = feature_sets
    combined_feature_set = (audio_features, motion_features, facial_features)
    audio_dim = len(audio_features[0])
    motion_dim = len(motion_features[0])
    face_dim = len(facial_features[0])
    fold_metrics = {}
    
    if 'audio' in tasks:
        fold_metrics['audio'] = model_evaluator.evaluate_feature_combination(
            audio_features,
            target_labels,
            "Audio",
            participant_ids,
            model_type,
        )
    if 'motion' in tasks:
        fold_metrics['motion'] = model_evaluator.evaluate_feature_combination(
            motion_features,
            target_labels,
            "Motion",
            participant_ids,
            model_type,
            missing_indicator_indices=range(0, motion_dim, 5),
        )
    if 'face' in tasks:
        fold_metrics['face'] = model_evaluator.evaluate_feature_combination(
            facial_features,
            target_labels,
            "Face",
            participant_ids,
            model_type,
            missing_indicator_indices=range(0, face_dim, 5),
        )
    if 'late_fusion' in tasks:
        _, _, fold_metrics['late_fusion'] = model_evaluator.evaluate_multimodal_fusion(
            combined_feature_set, 
            target_labels, 
            "Multimodal-Late Fusion (Stacking)", 
            participant_ids, 
            model_type, 
            model_type, 
            model_type
        )
    if 'early_fusion' in tasks:
        fold_metrics['early_fusion'] = model_evaluator.evaluate_feature_combination(
            combined_feature_set, 
            target_labels, 
            "Multimodal-Early Fusion (Concatenation)", 
            participant_ids, 
            model_type,
            missing_indicator_indices=[
                *range(audio_dim, audio_dim + motion_dim, 5),
                *range(
                    audio_dim + motion_dim,
                    audio_dim + motion_dim + face_dim,
                    5,
                ),
            ],
        )
    return fold_metrics
