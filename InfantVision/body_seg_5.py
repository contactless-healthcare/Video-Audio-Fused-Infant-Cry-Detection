import os
import argparse
import warnings
import logging
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from reproducibility_utils import set_seed

MIN_SCORE_THRESHOLD = 0.7  
MIN_PIXEL_AREA = 3000

# === Configuration for Reproducibility ===
warnings.filterwarnings("ignore") 
logging.disable(logging.CRITICAL) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def simple_white_balance_with_gain_limit(image, limit=10.0):
    """Simple white balance based on Gray World assumption, with gain limit."""
    avg_b, avg_g, avg_r = np.mean(image, axis=(0, 1))
    max_avg = max(avg_b, avg_g, avg_r)

    blue_ratio = min(max_avg / avg_b, limit) if avg_b > 0 else limit
    green_ratio = min(max_avg / avg_g, limit) if avg_g > 0 else limit
    red_ratio = min(max_avg / avg_r, limit) if avg_r > 0 else limit

    wb_image = image.copy().astype(np.float32)
    wb_image[:, :, 0] = np.clip(wb_image[:, :, 0] * blue_ratio, 0, 255)
    wb_image[:, :, 1] = np.clip(wb_image[:, :, 1] * green_ratio, 0, 255)
    wb_image[:, :, 2] = np.clip(wb_image[:, :, 2] * red_ratio, 0, 255)

    return wb_image.astype(np.uint8)


def save_img(balanced_frame, visual_mask, output_dir, video_name_without_ext):
    """Save the first frame's segmentation results (image, masked image, mask only)."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    frame_rgb = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2RGB)
    mask_rgb = visual_mask[:, :, :3]

    axs[0].imshow(frame_rgb)
    axs[0].set_title('Image (White Balanced)')
    axs[0].axis('off')

    axs[1].imshow(frame_rgb)
    axs[1].imshow(mask_rgb, alpha=0.5) 
    axs[1].set_title('Image with Mask (Alpha=0.5)')
    axs[1].axis('off')

    axs[2].imshow(mask_rgb) 
    axs[2].set_title('Color Mask Only')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{video_name_without_ext}_seg_result.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_visual_mask(frame, part_masks_dict, body_part_colors):
    """Generate the visual mask (RGBA) and the blended frame (BGR)."""
    height, width = frame.shape[:2]
    visual_mask = np.zeros((height, width, 4), dtype=np.float32) # RGBA (0-1 float32)
    
    for part_name, masks in part_masks_dict.items():
        r, g, b, a = body_part_colors[part_name]
        color_float = (r / 255.0, g / 255.0, b / 255.0, a)

        for mask in masks:
            visual_mask[mask > 0] = color_float
    
    # Convert RGBA (0-1) to BGR (0-255) for OpenCV blending
    mask_bgr_255 = (visual_mask[:, :, [2, 1, 0]] * 255).astype(np.uint8)
    
    # Blend original frame and mask (50% transparency)
    blended_frame = cv2.addWeighted(frame, 0.5, mask_bgr_255, 0.5, 0)
    
    return visual_mask, blended_frame


def Generate_Intime_Mask_AVI(input_video_path, segmentation_pipeline, body_part_colors, output_dir=None):
    """
    Main function to process video: WB, Segmentation, Optical Flow, Feature Extraction.
    Saves a masked video (AVI) and motion features (JSON).
    """
    # Setup output paths
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(input_video_path)), 'Body')
    os.makedirs(output_dir, exist_ok=True)
    video_name_without_ext = os.path.splitext(os.path.basename(input_video_path))[0]
    output_json_path = os.path.join(output_dir, f"{video_name_without_ext}_motion_features.json")
    output_video_path = os.path.join(output_dir, f"{video_name_without_ext}_masked.avi")
    
    # Skip if output files already exist
    if (os.path.exists(output_video_path) and os.path.exists(output_json_path) and
        any(f.startswith(f"{video_name_without_ext}_seg_result") and f.endswith('.png') for f in os.listdir(output_dir))):
        print(f"\nOutput files exist, skipping: {os.path.basename(output_video_path)}")
        return
            
    # Video capture and info extraction
    video_capture = cv2.VideoCapture(input_video_path)
    print(f"Processing video: {input_video_path}")
    
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize JSON output structure
    motion_analysis_results = {
        "video_info": {"path": input_video_path, "fps": float(fps), "frame_count": total_frames, "resolution": [float(width), float(height)]},
        "features": []
    }
    
    # Video writer setup (MJPG codec for AVI)
    video_codec = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(output_video_path, video_codec, fps, (width, height))     

    progress_bar = tqdm(total=total_frames, desc="Processing Video", unit='frame')
    
    current_frame_index = 0
    previous_gray_frame = None
    
    while True:
        ret, current_frame = video_capture.read()
        if not ret:
            break
        
        # 1. White Balance
        balanced_frame = simple_white_balance_with_gain_limit(current_frame.copy())
        
        # 2. Segmentation (Human Parsing)
        part_masks_dict = {part_name: [] for part_name in body_part_colors.keys()}
        part_scores_dict = {part_name: [] for part_name in body_part_colors.keys()}     
        segmentation_result = segmentation_pipeline(balanced_frame)
        
        for label, mask, score in zip(segmentation_result[OutputKeys.LABELS], segmentation_result['masks'], segmentation_result["scores"]):
            if label in body_part_colors.keys():
                mask_area = np.sum(mask > 0)
                if score >= MIN_SCORE_THRESHOLD and mask_area >= MIN_PIXEL_AREA: 
                    part_masks_dict[label].append(mask)
                    part_scores_dict[label].append(score)

        # 3. Generate visualization & Save first frame
        visual_mask, blended_frame = generate_visual_mask(balanced_frame, part_masks_dict, body_part_colors)
        
        if current_frame_index == 0:
            save_img(balanced_frame, visual_mask, output_dir, video_name_without_ext)

        # 4. Optical Flow (Farneback)
        gray_frame = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2GRAY)
        optical_flow = None
        if previous_gray_frame is not None:
            optical_flow = cv2.calcOpticalFlowFarneback(previous_gray_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        previous_gray_frame = gray_frame

        # 5. Face Bounding Box and size
        head_width = head_height = 0.0
        if "Face" in part_masks_dict and len(part_masks_dict["Face"]) > 0:
            face_mask = part_masks_dict["Face"][0]
            rows = np.any(face_mask, axis=1)
            cols = np.any(face_mask, axis=0)
            
            if np.any(rows) and np.any(cols):
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                cv2.rectangle(blended_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                head_height = float(y_max - y_min)
                head_width = float(x_max - x_min)
        
        video_writer.write(blended_frame)

        # 6. Initialize frame features
        frame_motion_features = {
            "Frame": current_frame_index, "head": [head_width, head_height],
            "Face": [], "Left-arm": [], "Right-arm": [], "Left-leg": [], 
            "Right-leg": [], "Torso-skin": [], "WholeBody": [], "WholeFrameMotion": [],
        }
        
        # 7. Calculate motion features for each part
        if optical_flow is not None:
            for part_name, masks in part_masks_dict.items():
                for mask_idx, mask in enumerate(masks):
                    if mask.sum() > 0:
                        mask_indices = mask > 0
                        shift_x, shift_y = np.mean(optical_flow[mask_indices], axis=0)
                        shift_r = np.sqrt(shift_x ** 2 + shift_y ** 2)
                        shift_a = np.degrees(np.arctan2(shift_y, shift_x))
                        current_score = part_scores_dict[part_name][mask_idx]
                        
                        motion_vector = [float(shift_x), float(shift_y), float(shift_r), float(shift_a), float(current_score)]
                        frame_motion_features[part_name].append(motion_vector)

            # 8. WholeBody motion (combined human parts)
            combined_mask = np.zeros_like(gray_frame, dtype=bool)
            target_parts = ['Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Torso-skin']
            combined_scores = []
            
            for part_name in target_parts:
                if part_name in part_masks_dict:
                    for mask in part_masks_dict[part_name]:
                        combined_mask = combined_mask | (mask > 0)
                    if part_name in part_scores_dict:
                        combined_scores.extend(part_scores_dict[part_name])

            if combined_mask.any():
                shift_x, shift_y = np.mean(optical_flow[combined_mask], axis=0)
                shift_r = np.sqrt(shift_x**2 + shift_y**2)
                shift_a = np.degrees(np.arctan2(shift_y, shift_x))
                avg_score = np.mean(combined_scores) if combined_scores else 0.0
                
                frame_motion_features["WholeBody"].append([
                    float(shift_x), float(shift_y), float(shift_r), float(shift_a), float(avg_score)
                ])
                
            # 9. WholeFrameMotion (Global motion)
            shift_x_global, shift_y_global = np.mean(optical_flow, axis=(0,1))
            shift_r_global = np.sqrt(shift_x_global**2 + shift_y_global**2)
            shift_a_global = np.degrees(np.arctan2(shift_y_global, shift_x_global))
            
            frame_motion_features["WholeFrameMotion"].append([
                float(shift_x_global), float(shift_y_global), float(shift_r_global), float(shift_a_global)
            ])

        # 10. Append frame features
        motion_analysis_results["features"].append(frame_motion_features)     

        current_frame_index += 1
        progress_bar.update(1)

    # 11. Release resources and save JSON
    video_capture.release()
    video_writer.release()
    progress_bar.close()
    
    with open(output_json_path, 'w') as f:
        json.dump(motion_analysis_results, f, indent=2)
    
    print(f"\nProcessing completed. Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract body motion JSON from one video.')
    parser.add_argument('--video', required=True, help='Input video; timestamps start at zero.')
    parser.add_argument('--output-dir', required=True, help='Directory for Body JSON and previews.')
    parser.add_argument('--model-dir', required=True, help='Downloaded ModelScope model directory.')
    parser.add_argument('--device', default='cuda', help='ModelScope device (default: cuda).')
    args = parser.parse_args()
    if not os.path.isfile(args.video):
        parser.error(f'Input video does not exist: {args.video}')
    if not os.path.isdir(args.model_dir):
        parser.error(f'Model directory does not exist: {args.model_dir}')
    record_stem = os.path.splitext(os.path.basename(args.video))[0]
    output_suffixes = ('_motion_features.json', '_masked.avi', '_seg_result.png')
    if any(os.path.exists(os.path.join(args.output_dir, record_stem + suffix)) for suffix in output_suffixes):
        parser.error('An output already exists for this video; choose a new output directory.')
    set_seed(42)
    segmentation_pipeline = pipeline(
        Tasks.image_segmentation,
        args.model_dir,
        device=args.device
    )

    # Body parts and their corresponding colors (R, G, B, A), 0-255 range, A=1
    BODY_PART_COLORS = {
        'Left-arm': (255, 0, 0, 1),
        'Right-arm': (255, 0, 0, 1),
        'Left-leg': (0, 0, 255, 1),
        'Right-leg': (0, 0, 255, 1),
        'Face': (255, 255, 255, 1),
        'Torso-skin': (0, 255, 0, 1),
    }

    # === Process video ===
    Generate_Intime_Mask_AVI(
        input_video_path=args.video,
        segmentation_pipeline=segmentation_pipeline,
        body_part_colors=BODY_PART_COLORS,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()