from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import uuid
import torch
import numpy as np
import cv2
import requests
from pathlib import Path

app = FastAPI()

print("🚀 AI Image Upscaler Backend Starting...")
print("📦 Loading SwinIR model...")

# Initialize SwinIR
upscaler = None
device = None

def download_swinir_model():
    """Download SwinIR model if not exists"""
    model_dir = Path("../models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "SwinIR_classical_SR_x4.pth"
    
    if not model_path.exists():
        print("📥 Downloading correct SwinIR model (first time only)...")
        # Try the classical SR model instead
        url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Model downloaded: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return None
    else:
        print(f"✅ Model found: {model_path}")
        return model_path

class SwinIRModel:
    def __init__(self, model_path):
        try:
            # Import SwinIR architecture
            from swinir_arch import SwinIR
            
            # Set device
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon
                print("🔥 Using Apple Silicon MPS acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("🔥 Using CUDA GPU acceleration")
            else:
                self.device = torch.device('cpu')
                print("💻 Using CPU (slower but works)")
            
            # Initialize model with EXACT parameters for classical SR model
            self.model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=48,  # Changed to 48 for classical SR
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],  # 6 layers for medium model
                embed_dim=180,  # 180 for medium model
                num_heads=[6, 6, 6, 6, 6, 6],  # 6 heads for medium model
                mlp_ratio=2,
                upsampler='pixelshuffledirect',  # Direct pixelshuffle for classical SR
                resi_connection='1conv'
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['params_ema'] if 'params_ema' in checkpoint else checkpoint)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            print("✅ SwinIR model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load SwinIR: {e}")
            raise e
    
    def upscale(self, image):
        """Upscale image using SwinIR"""
        try:
            # Convert PIL to tensor
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Handle large images by tiling if necessary
            _, _, h, w = img_tensor.shape
            
            # For images larger than 512x512, use tiling to avoid memory issues
            if h > 512 or w > 512:
                return self._tile_process(img_tensor)
            else:
                # Process smaller images directly
                with torch.no_grad():
                    output = self.model(img_tensor)
                
                # Convert back to PIL
                output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
                
                return Image.fromarray(output)
                
        except Exception as e:
            print(f"❌ SwinIR upscaling failed: {e}")
            raise e
    
    def _tile_process(self, img_tensor, tile_size=256, tile_pad=10):
        """Process large images using tiling"""
        try:
            batch, channel, height, width = img_tensor.shape
            output_height = height * 4
            output_width = width * 4
            
            # Initialize output tensor
            output = torch.zeros((batch, channel, output_height, output_width), 
                               dtype=img_tensor.dtype, device=self.device)
            
            tiles_x = (width + tile_size - 1) // tile_size
            tiles_y = (height + tile_size - 1) // tile_size
            
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Calculate tile boundaries
                    start_x = x * tile_size
                    start_y = y * tile_size
                    end_x = min(start_x + tile_size, width)
                    end_y = min(start_y + tile_size, height)
                    
                    # Extract tile with padding
                    start_x_pad = max(start_x - tile_pad, 0)
                    start_y_pad = max(start_y - tile_pad, 0)
                    end_x_pad = min(end_x + tile_pad, width)
                    end_y_pad = min(end_y + tile_pad, height)
                    
                    tile = img_tensor[:, :, start_y_pad:end_y_pad, start_x_pad:end_x_pad]
                    
                    # Process tile
                    with torch.no_grad():
                        tile_output = self.model(tile)
                    
                    # Calculate output coordinates
                    output_start_x = start_x * 4
                    output_start_y = start_y * 4
                    output_end_x = end_x * 4
                    output_end_y = end_y * 4
                    
                    # Calculate crop coordinates for removing padding
                    crop_start_x = (start_x - start_x_pad) * 4
                    crop_start_y = (start_y - start_y_pad) * 4
                    crop_end_x = crop_start_x + (end_x - start_x) * 4
                    crop_end_y = crop_start_y + (end_y - start_y) * 4
                    
                    # Place tile in output
                    output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                        tile_output[:, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            
            # Convert to PIL
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
            
            return Image.fromarray(output)
            
        except Exception as e:
            print(f"❌ Tile processing failed: {e}")
            raise e

def enhanced_fallback_upscale(image):
    """Fallback enhanced upscaling if SwinIR fails"""
    try:
        from PIL import ImageFilter, ImageEnhance
        
        # Multi-step upscaling for better quality
        width, height = image.size
        
        # First 2x with LANCZOS
        img_2x = image.resize((width * 2, height * 2), Image.LANCZOS)
        
        # Apply sharpening
        enhancer = ImageEnhance.Sharpness(img_2x)
        img_2x_sharp = enhancer.enhance(1.2)
        
        # Another 2x for total 4x
        img_4x = img_2x_sharp.resize((width * 4, height * 4), Image.LANCZOS)
        
        # Final enhancement
        enhancer2 = ImageEnhance.Sharpness(img_4x)
        final_image = enhancer2.enhance(1.1)
        
        # Slight contrast boost
        contrast_enhancer = ImageEnhance.Contrast(final_image)
        result = contrast_enhancer.enhance(1.05)
        
        return result
    except Exception as e:
        print(f"❌ Fallback upscaling failed: {e}")
        # Last resort: simple resize
        width, height = image.size
        return image.resize((width * 4, height * 4), Image.LANCZOS)

# Initialize SwinIR
try:
    model_path = download_swinir_model()
    if model_path and os.path.exists("swinir_arch.py"):
        upscaler = SwinIRModel(model_path)
        model_status = "SwinIR AI Model (4x)"
        print("🎯 SwinIR ready for state-of-the-art upscaling!")
    else:
        if not os.path.exists("swinir_arch.py"):
            print("⚠️  swinir_arch.py not found - using fallback")
        upscaler = None
        model_status = "Enhanced Fallback"
except Exception as e:
    print(f"⚠️  SwinIR initialization failed: {e}")
    upscaler = None
    model_status = "Enhanced Fallback"

@app.get("/")
def read_root():
    return {
        "message": "AI Image Upscaler Backend Running",
        "model_status": model_status,
        "scale_factor": "4x"
    }

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data))
        
        print(f"📤 Processing image: {input_image.size} -> ", end="")
        
        if upscaler:
            # Use SwinIR AI upscaling
            try:
                upscaled_image = upscaler.upscale(input_image)
                method = "SwinIR AI 4x"
            except Exception as e:
                print(f"❌ SwinIR failed: {e}")
                print("🔄 Falling back to enhanced mode...")
                upscaled_image = enhanced_fallback_upscale(input_image)
                method = "Enhanced Fallback 4x"
        else:
            # Use enhanced fallback
            upscaled_image = enhanced_fallback_upscale(input_image)
            method = "Enhanced Fallback 4x"
        
        print(f"{upscaled_image.size} ({method})")
        
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
        "model_type": model_status,
        "scale_factor": "4x",
        "backend_running": True
    }

if __name__ == "__main__":
    import uvicorn
    print("🌐 Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)