from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import uuid
import cv2
import numpy as np
# Try to import Real-ESRGAN, but continue without it if it fails
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Real-ESRGAN import failed: {e}")
    print("🔄 Running in simple resize mode")
    REALESRGAN_AVAILABLE = False

app = FastAPI()

print("🚀 AI Image Upscaler Backend Starting...")
print("📦 Loading Real-ESRGAN model...")

# Initialize the Real-ESRGAN model
upscaler = None

if REALESRGAN_AVAILABLE:
    try:
        # Define the model architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # Get the model path
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'RealESRGAN_x4plus.pth')
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            print("⚠️  Falling back to simple resize mode")
            upscaler = None
        else:
            # Initialize RealESRGANer
            upscaler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0,  # No tiling for better quality
                tile_pad=10,
                pre_pad=0,
                half=False,  # Use full precision
                gpu_id=None  # Use CPU (safer for compatibility)
            )
            
            print("✅ Real-ESRGAN model loaded successfully!")
            print("🎯 Ready for 4x image upscaling!")
        
    except Exception as e:
        print(f"❌ Error loading Real-ESRGAN model: {e}")
        print("⚠️  Falling back to simple resize mode")
        upscaler = None
else:
    print("⚠️  Real-ESRGAN not available, using simple resize mode")
    upscaler = None

@app.get("/")
def read_root():
    return {
        "message": "AI Image Upscaler Backend Running",
        "model_status": "Real-ESRGAN loaded" if upscaler else "Simple resize mode",
        "scale_factor": "4x" if upscaler else "2x"
    }

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await file.read()
        
        # Convert to PIL Image
        input_image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL to OpenCV format (BGR)
        img_array = np.array(input_image)
        if len(img_array.shape) == 3:
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv2 = img_array
        
        print(f"📤 Processing image: {input_image.size} -> ", end="")
        
        if upscaler:
            # Use Real-ESRGAN for high-quality upscaling
            try:
                output, _ = upscaler.enhance(img_cv2, outscale=4)
                print(f"{output.shape[1]}x{output.shape[0]} (Real-ESRGAN 4x)")
                
                # Convert back to RGB for PIL
                if len(output.shape) == 3:
                    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                else:
                    output_rgb = output
                
                upscaled_image = Image.fromarray(output_rgb)
                
            except Exception as e:
                print(f"❌ Real-ESRGAN failed: {e}")
                print("🔄 Falling back to simple resize...")
                # Fallback to simple resize
                width, height = input_image.size
                upscaled_image = input_image.resize((width * 2, height * 2), Image.LANCZOS)
                
        else:
            # Fallback: Simple resize if model failed to load
            width, height = input_image.size
            upscaled_image = input_image.resize((width * 2, height * 2), Image.LANCZOS)
            print(f"{width * 2}x{height * 2} (Simple 2x resize)")
        
        # Save upscaled image temporarily
        output_filename = f"upscaled_{uuid.uuid4().hex}.png"
        output_path = f"/tmp/{output_filename}"
        upscaled_image.save(output_path, "PNG", quality=95)
        
        print("✅ Upscaling completed successfully!")
        
        return FileResponse(
            output_path, 
            filename=f"upscaled_{file.filename}",
            media_type="image/png"
        )
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/status")
def get_status():
    return {
        "model_loaded": upscaler is not None,
        "model_type": "Real-ESRGAN" if upscaler else "Simple Resize",
        "scale_factor": "4x" if upscaler else "2x",
        "backend_running": True
    }

if __name__ == "__main__":
    import uvicorn
    print("🌐 Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)