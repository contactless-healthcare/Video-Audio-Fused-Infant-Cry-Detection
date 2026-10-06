import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression_face, scale_coords, xyxy2xywh
from utils.datasets import letterbox
from reproducibility_utils import set_seed

class FaceDetector:
    """
    Estimates face orientation and provides the bounding box of the largest detected face.
    """
    def __init__(self, weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        set_seed(seed=42)
        self.device = device
        self.model = attempt_load(weights_path, map_location=device)
        self.img_size = 800
        self.conf_thres = 0.02
        self.iou_thres = 0.5
        
    def _preprocess(self, frame):
        """Preprocesses the frame for model inference."""
        img0 = frame.copy()
        h0, w0 = frame.shape[:2]
        r = self.img_size / max(h0, w0)
        
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img

    @torch.no_grad()
    def _get_detections(self, img_tensor, original_shape):
        """Runs inference and applies NMS to get detections scaled to original image."""
        pred = self.model(img_tensor)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        
        if len(pred[0]) == 0:
            return None
        
        # Sort by confidence
        det = pred[0]
        det = det[det[:, 4].argsort(descending=True)]
        
        # Scale coords back to original image size
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], original_shape).round()
        
        return det

    def detect_face(self, image):
        """
        Detects faces and returns the bounding box [x1, y1, x2, y2] of the largest face.
        This is the primary method used by FaceAlignmentProcessor.
        """
        img_tensor = self._preprocess(image)
        detections = self._get_detections(img_tensor, image.shape)
        
        if detections is None:
            return None, None
        
        max_area = 0
        best_face = None
        best_conf = None
        
        # Find the largest face among all detections
        for det in detections:
            x1, y1, x2, y2, conf = det[:5].cpu().numpy().astype(float)
            area = (x2 - x1) * (y2 - y1)
            
            if area > max_area:
                max_area = area
                best_face = [int(x1), int(y1), int(x2), int(y2)]
                best_conf = float(conf)
        
        return best_face, best_conf