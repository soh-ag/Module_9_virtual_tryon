import torch
import cv2
import numpy as np
import requests
import base64
from PIL import Image
import io
from inference_sdk import InferenceHTTPClient
from ultralytics import SAM
import os
from typing import List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VirtualTryOn:
    def __init__(self, roboflow_api_key: str, segmind_api_key: str):
        """
        Initialize the Virtual Try-On system
        
        Args:
            roboflow_api_key: API key for Roboflow inference
            segmind_api_key: API key for Segmind inpainting
        """
        self.roboflow_api_key = roboflow_api_key
        self.segmind_api_key = segmind_api_key
        
        # Setup device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # Initialize Roboflow client
        self.roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=self.roboflow_api_key
        )
        
        # Initialize SAM model
        self.sam_model = None
        self._load_sam_model()
        
        # Define clothing regions
        self.upper_region = ['Tshirt', 'jacket', 'shirt', 'dress', 'sweater']
        self.lower_region = ['pants', 'skirt', 'short']
        
    def _load_sam_model(self):
        """Load SAM model on GPU"""
        try:
            self.sam_model = SAM("sam2.1_b.pt")
            self.sam_model.to(self.device)
            logger.info("SAM model loaded successfully on GPU")
        except Exception as e:
            logger.error(f"Error loading SAM model: {e}")
            raise
    
    def detect_clothing(self, image_path: str) -> dict:
        """
        Detect clothing items in the image using Roboflow
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Detection results from Roboflow
        """
        try:
            result = self.roboflow_client.infer(image_path, model_id="main-fashion-wmyfk/1")
            return result
        except Exception as e:
            logger.error(f"Error in clothing detection: {e}")
            raise
    
    def detect_region(self, detection_results: dict) -> Tuple[List[str], List[List[int]]]:
        """
        Categorize detected clothing into upper/lower regions and extract bounding boxes
        
        Args:
            detection_results: Results from Roboflow detection
            
        Returns:
            Tuple of (regions, bboxes)
        """
        regions = []
        predictions = detection_results["predictions"]
        
        for prediction in predictions:
            if prediction['class'] in self.upper_region:
                regions.append('upper')
            elif prediction['class'] in self.lower_region:
                regions.append('lower')
        
        bboxes = [
            [
                int(pred["x"] - pred["width"] / 2),   # x_min
                int(pred["y"] - pred["height"] / 2),  # y_min
                int(pred["x"] + pred["width"] / 2),   # x_max
                int(pred["y"] + pred["height"] / 2)   # y_max
            ]
            for pred in predictions
        ]
        
        return regions, bboxes
    
    def get_bbox_by_region(self, regions: List[str], bboxes: List[List[int]], choice: str) -> Optional[List[int]]:
        """
        Get bounding box for specified region
        
        Args:
            regions: List of detected regions
            bboxes: List of bounding boxes
            choice: 'upper' or 'lower'
            
        Returns:
            Bounding box for the specified region or None
        """
        try:
            if choice in regions:
                index = regions.index(choice)
                return bboxes[index]
            return None
        except Exception as e:
            logger.error(f"Error getting bbox by region: {e}")
            return None
    
    def segment_clothing(self, image_path: str, bbox: List[int]) -> np.ndarray:
        """
        Segment clothing using SAM model
        
        Args:
            image_path: Path to the input image
            bbox: Bounding box for segmentation
            
        Returns:
            Segmentation result
        """
        try:
            segment_result = self.sam_model(image_path, bboxes=[bbox])
            return segment_result
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            raise
    
    def save_binary_mask(self, segment_result, save_path: str = "mask.png") -> str:
        """
        Save binary mask from segmentation result
        
        Args:
            segment_result: Result from SAM segmentation
            save_path: Path to save the mask
            
        Returns:
            Path to saved mask
        """
        try:
            masks = segment_result[0].masks.data.cpu().numpy()
            
            if len(masks) > 0:
                # Take the first mask
                mask = masks[0]
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                
                # Save as PNG
                cv2.imwrite(save_path, binary_mask)
                return save_path
            else:
                raise ValueError("No masks found in segmentation result")
        except Exception as e:
            logger.error(f"Error saving binary mask: {e}")
            raise
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise
    
    def generate_tryon_image(self, original_image_path: str, mask_path: str, prompt: str) -> Image.Image:
        """
        Generate try-on image using Segmind inpainting API
        
        Args:
            original_image_path: Path to original image
            mask_path: Path to mask image
            prompt: Text prompt for generation
            
        Returns:
            Generated PIL Image
        """
        try:
            url = "https://api.segmind.com/v1/sdxl-inpaint"
            
            data = {
                "image": self.image_to_base64(original_image_path),
                "mask": self.image_to_base64(mask_path),
                "prompt": prompt,
                "negative_prompt": "bad quality, painting, blur",
                "samples": 1,
                "scheduler": "DDIM",
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "seed": 12467,
                "strength": 0.9,
                "base64": False
            }
            
            headers = {'x-api-key': self.segmind_api_key}
            
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            # Convert response to PIL Image
            image_bytes = response.content
            image_stream = io.BytesIO(image_bytes)
            image = Image.open(image_stream)
            
            return image
        except Exception as e:
            logger.error(f"Error generating try-on image: {e}")
            raise
    
    def process_virtual_tryon(self, image_path: str, region_choice: str, prompt: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """
        Complete virtual try-on pipeline
        
        Args:
            image_path: Path to input image
            region_choice: 'upper' or 'lower'
            prompt: Generation prompt
            
        Returns:
            Tuple of (original_image, masked_image, generated_image)
        """
        try:
            # Step 1: Detect clothing
            detection_results = self.detect_clothing(image_path)
            
            # Step 2: Get regions and bboxes
            regions, bboxes = self.detect_region(detection_results)
            
            # Step 3: Get bbox for specified region
            bbox = self.get_bbox_by_region(regions, bboxes, region_choice)
            if bbox is None:
                raise ValueError(f"No {region_choice} clothing detected in the image")
            
            # Step 4: Segment clothing
            segment_result = self.segment_clothing(image_path, bbox)
            
            # Step 5: Save mask
            mask_path = "temp_mask.png"
            self.save_binary_mask(segment_result, mask_path)
            
            # Step 6: Generate try-on image
            generated_image = self.generate_tryon_image(image_path, mask_path, prompt)
            
            # Load original image for return
            original_image = Image.open(image_path)
            
            # Create masked image for visualization
            mask_image = Image.open(mask_path).convert('L')
            
            # Clean up temporary files
            if os.path.exists(mask_path):
                os.remove(mask_path)
            
            return original_image, mask_image, generated_image
            
        except Exception as e:
            logger.error(f"Error in virtual try-on process: {e}")
            raise