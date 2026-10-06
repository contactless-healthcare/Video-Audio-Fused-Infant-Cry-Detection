import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import glob
import cv2
import json
from tqdm import tqdm
from PIL import Image
import numpy as np

from reproducibility_utils import set_seed

import lib.models as models
from lib.config import config, update_config
from lib.utils.transforms import crop
from lib.core.evaluation import decode_preds
from face_detector import FaceDetector

class FaceAlignmentProcessor:
    def __init__(self, config_path, model_path, face_detector_path, device="cuda"):
        self.device = torch.device(device)
        self.args = self._parse_args(config_path, model_path)
        self.model = self._initialize_model()
        # Initializing the custom face detector
        self.face_detector = FaceDetector(face_detector_path, device=device)
        
    @staticmethod
    def _parse_args(config_path, model_path):
        # Setting up arguments for config and model file paths
        parser = argparse.ArgumentParser(description='Train Face Alignment')
        parser.add_argument('--cfg', default=config_path, help='experiment configuration filename', type=str)
        parser.add_argument('--model_file', default=model_path, type=str)
        args = parser.parse_args([])
        update_config(config, args)
        return args
    
    def _initialize_model(self):
        # Update model config
        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()
        
        # Load model and set to DataParallel
        model = models.get_face_alignment_net(config)
        gpus = [config.GPUS] if isinstance(config.GPUS, int) else list(config.GPUS)
        model = nn.DataParallel(model, device_ids=gpus if self.device.type == "cuda" else []).to(self.device)
        
        # Load weights
        state_dict = torch.load(self.args.model_file, map_location=self.device)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def prepare_input(self, image, bbox, image_size):
        # Prepares the input image crop for the face alignment model
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
        center = torch.Tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        scale *= 1.25
        
        img = np.array(image.convert('RGB'), dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img = crop(img, center, scale, image_size, rot=0)
        img = (img / 255.0 - mean) / std
        img = torch.Tensor(img.transpose([2, 0, 1])).unsqueeze(0)
        
        return img, center, scale
    
    def predict_landmarks(self, image, bbox):
        # Predicts facial landmarks
        inp, center, scale = self.prepare_input(image, bbox, config.MODEL.IMAGE_SIZE)
        
        with torch.no_grad():
            output = self.model(inp.to(self.device))
        
        score_map = output.data.cpu()
        preds = decode_preds(score_map, center, scale, [64, 64])

        flattened_scores = score_map.view(score_map.size(0), score_map.size(1), -1)
        max_scores = flattened_scores.max(2)[0]
        landmark_confs = max_scores[0].cpu().numpy().tolist()
        
        return preds.numpy(), landmark_confs
    
    def draw_results(self, image, preds):
        # Draws landmarks and a bounding box based on the landmarks
        for point in preds[0, :, :]:
            cv2.circle(image, tuple(map(int, point)), 2, (255, 255, 0), 1)

        x_min, y_min = map(int, np.min(preds[0, :, :], axis=0))
        x_max, y_max = map(int, np.max(preds[0, :, :], axis=0))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return image
    
    def process_frame(self, frame):
        # Processes a single frame: detects face, predicts landmarks, and draws results
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        try:
            # Use the custom face detector
            face_bbox, face_conf = self.face_detector.detect_face(frame)
            if face_bbox is None:
                return frame, None, None, None
            
            # Predict landmarks
            preds, landmark_conf = self.predict_landmarks(pil_image, face_bbox)
            processed_frame = self.draw_results(frame.copy(), preds)
            
            # Return processed frame and landmarks as a list
            landmarks = preds[0].tolist()  # shape (68, 2)
            return processed_frame, landmarks, face_conf, landmark_conf
            
        except Exception:
            return frame, None, None, None
    
    def process_image(self, image_path, output_path):
        """Processes a single image."""
        cv_image = cv2.imread(image_path)
        result, landmarks = self.process_frame(cv_image)
        if landmarks is not None:
            with open(output_path.replace('.png', '.json'), 'w') as f:
                json.dump({'landmarks': landmarks}, f)
        cv2.imwrite(output_path, result)
        print(f"Result saved to {output_path}")
    
    def process_video(self, video_path, output_dir=None):
        """Processes a video file, saving the output video and landmark JSON."""
        # Setup output paths
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'Face')
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{base_name}_face_landmarks.avi")
        output_json_path = os.path.join(output_dir, f"{base_name}_face_landmarks.json")
        output_img_path = os.path.join(output_dir, f"{base_name}_face_landmarks.png")

        # Check if output files exist to skip processing
        if (os.path.exists(output_video_path) and 
            os.path.exists(output_json_path) and
            os.path.exists(output_img_path)):
            print(f"\nOutput files exist, skipping: {os.path.basename(video_path)}")
            return {
                'video_path': output_video_path,
                'json_path': output_json_path,
                'image_path': output_img_path,
                'status': 'skipped'
            }
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Initialize results dictionary
        results = {
            'video_info': {
                'path': video_path,
                'fps': fps,
                'frame_count': total_frames,
                'resolution': f"{width}x{height}"
            },
            'frames': []
        }
        
        # Process frames
        first_frame_saved = False
        for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, landmarks, face_conf, landmark_confs = self.process_frame(frame)            
            out.write(processed_frame)
            
            # Save frame data to results
            results['frames'].append({
                'frame_number': frame_idx,
                'timestamp': frame_idx / fps,
                'landmarks': landmarks,
                'face_confidence': face_conf,
                'landmark_confidences': landmark_confs
            })

            # Save the first frame with detected landmarks
            if not first_frame_saved and landmarks is not None:
                cv2.imwrite(output_img_path, processed_frame)
                print(f"\nFirst face detected at frame: {frame_idx}, saved as: {os.path.basename(output_img_path)}")
                first_frame_saved = True
        
        # Release resources
        cap.release()
        out.release()
        
        # Save JSON results
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessing completed for {os.path.basename(video_path)}")
        return {
            'video_path': output_video_path,
            'json_path': output_json_path,
            'image_path': output_img_path if first_frame_saved else None,
            'status': 'completed' if total_frames > 0 else 'failed'
        }

def main():
    parser = argparse.ArgumentParser(description='Extract facial landmarks from one video.')
    parser.add_argument('--video', required=True, help='Input video; timestamps start at zero.')
    parser.add_argument('--output-dir', required=True, help='Directory for Face JSON and previews.')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='Inference device (default: cuda).')
    parser.add_argument('--landmark-weights', required=True, help='HRNet landmark weights.')
    parser.add_argument('--detector-weights', required=True, help='YOLOv5-face detector weights.')
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config/hrnet-r90jt.yaml'),
                        help='HRNet configuration; defaults to the bundled configuration.')
    args = parser.parse_args()
    for path in (args.video, args.landmark_weights, args.detector_weights, args.config):
        if not os.path.isfile(path):
            parser.error(f'Input file does not exist: {path}')
    record_stem = os.path.splitext(os.path.basename(args.video))[0]
    output_suffixes = ('_face_landmarks.json', '_face_landmarks.avi', '_face_landmarks.png')
    if any(os.path.exists(os.path.join(args.output_dir, record_stem + suffix)) for suffix in output_suffixes):
        parser.error('An output already exists for this video; choose a new output directory.')
    set_seed(42)
    processor = FaceAlignmentProcessor(args.config, args.landmark_weights, args.detector_weights, device=args.device)
    processor.process_video(args.video, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
