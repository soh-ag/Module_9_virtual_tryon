from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
from PIL import Image
import io
import base64
from virtual_try_on import VirtualTryOn
from dotenv import load_dotenv
import logging
from typing import Optional

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Virtual Try-On API",
    description="API for virtual clothing try-on using AI",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize VirtualTryOn class
try:
    virtual_tryon = VirtualTryOn(
        roboflow_api_key=os.getenv("ROBOFLOW_API_KEY"),
        segmind_api_key=os.getenv("SEGMIND_API_KEY")
    )
    logger.info("Virtual Try-On system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Virtual Try-On system: {e}")
    virtual_tryon = None

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Virtual Try-On API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    gpu_available = virtual_tryon.device.type == 'cuda' if virtual_tryon else False
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "models_loaded": virtual_tryon is not None
    }

@app.post("/virtual-tryon")
async def virtual_tryon_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    region: str = Form(...)
):
    """
    Virtual try-on endpoint
    
    Args:
        file: Image file to process
        prompt: Text prompt for generation
        region: 'upper' or 'lower' clothing region
    
    Returns:
        JSON response with base64 encoded images
    """
    if virtual_tryon is None:
        raise HTTPException(status_code=500, detail="Virtual Try-On system not initialized")
    
    # Validate region input
    if region not in ['upper', 'lower']:
        raise HTTPException(status_code=400, detail="Region must be 'upper' or 'lower'")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] or ".jpg"
    temp_image_path = f"uploads/{file_id}{file_extension}"
    
    try:
        # Save uploaded file
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing image: {temp_image_path}")
        logger.info(f"Region: {region}, Prompt: {prompt}")
        
        # Process virtual try-on
        original_image, mask_image, generated_image = virtual_tryon.process_virtual_tryon(
            image_path=temp_image_path,
            region_choice=region,
            prompt=prompt
        )
        
        # Convert images to base64
        original_b64 = pil_to_base64(original_image)
        mask_b64 = pil_to_base64(mask_image.convert('RGB'))  # Convert mask to RGB for display
        generated_b64 = pil_to_base64(generated_image)
        
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return JSONResponse({
            "success": True,
            "message": "Virtual try-on completed successfully",
            "data": {
                "original_image": original_b64,
                "mask_image": mask_b64,
                "generated_image": generated_b64,
                "region": region,
                "prompt": prompt
            }
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        logger.error(f"Error processing virtual try-on: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/detect-clothing")
async def detect_clothing_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to detect clothing items in an image
    
    Args:
        file: Image file to analyze
        
    Returns:
        Available clothing regions
    """
    if virtual_tryon is None:
        raise HTTPException(status_code=500, detail="Virtual Try-On system not initialized")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] or ".jpg"
    temp_image_path = f"uploads/{file_id}{file_extension}"
    
    try:
        # Save uploaded file
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Detect clothing
        detection_results = virtual_tryon.detect_clothing(temp_image_path)
        regions, bboxes = virtual_tryon.detect_region(detection_results)
        
        # Get unique regions
        available_regions = list(set(regions))
        
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return JSONResponse({
            "success": True,
            "data": {
                "available_regions": available_regions,
                "detected_items": [pred['class'] for pred in detection_results["predictions"]]
            }
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        logger.error(f"Error detecting clothing: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)