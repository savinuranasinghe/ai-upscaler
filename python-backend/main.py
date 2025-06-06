from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import uuid
import torch
import numpy as np
import requests
from pathlib import Path

app = FastAPI()

print("🚀 AI Image Upscaler Backend Starting...")
print("📦 Loading Universal SwinIR model...")

upscaler = None

def download_swinir_model():
    """Download SwinIR model if not exists"""
    model_dir = Path("../models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "SwinIR_classical_SR_x4.pth"
    
    if not model_path.exists():
        print("📥 Downloading SwinIR model...")
        url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
        
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

class UniversalSwinIRModel:
    def __init__(self, model_path):
        try:
            from swinir_arch import SwinIR
            
            # Set device
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("🔥 Using Apple Silicon MPS acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("🔥 Using CUDA GPU acceleration")
            else:
                self.device = torch.device('cpu')
                print("💻 Using CPU (slower but works)")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
                print("✅ Using EMA parameters")
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
                print("✅ Using regular parameters")
            else:
                state_dict = checkpoint
            
            # Initialize model
            self.model = SwinIR(
                upscale=4,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='nearest+conv',
                resi_connection='1conv',
                patch_norm=True
            )
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Model requirements
            self.window_size = 8
            self.scale_factor = 4
            
            print("✅ Universal SwinIR model loaded - supports ANY image size!")
            
        except Exception as e:
            print(f"❌ Failed to load SwinIR: {e}")
            raise e
    
    def make_divisible(self, dimension, divisor=8):
        """Make dimension divisible by divisor"""
        return ((dimension + divisor - 1) // divisor) * divisor
    
    def pad_image(self, image):
        """Pad image to make dimensions compatible"""
        width, height = image.size
        
        new_width = self.make_divisible(width, self.window_size)
        new_height = self.make_divisible(height, self.window_size)
        
        if new_width != width or new_height != height:
            padded = Image.new('RGB', (new_width, new_height), color=(0, 0, 0))
            padded.paste(image, (0, 0))
            return padded, (width, height)
        
        return image, (width, height)
    
    def crop_output(self, image, original_size):
        """Crop upscaled image to correct proportions"""
        orig_width, orig_height = original_size
        target_width = orig_width * self.scale_factor
        target_height = orig_height * self.scale_factor
        return image.crop((0, 0, target_width, target_height))
    
    def process_large_image_with_tiles(self, image, tile_size=512):
        """Process large images using overlapping tiles"""
        width, height = image.size
        
        if width <= tile_size and height <= tile_size:
            return self.process_single_image(image)
        
        print(f"🧩 Processing large image with tiling: {width}x{height}")
        
        overlap = 64
        output_width = width * self.scale_factor
        output_height = height * self.scale_factor
        output_image = Image.new('RGB', (output_width, output_height))
        
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        
        print(f"📦 Processing {tiles_x}x{tiles_y} tiles...")
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile boundaries with overlap
                start_x = max(0, x * tile_size - overlap if x > 0 else 0)
                start_y = max(0, y * tile_size - overlap if y > 0 else 0)
                end_x = min(width, (x + 1) * tile_size + overlap)
                end_y = min(height, (y + 1) * tile_size + overlap)
                
                # Extract and process tile
                tile = image.crop((start_x, start_y, end_x, end_y))
                
                try:
                    upscaled_tile = self.process_single_image(tile)
                    
                    # Calculate output position (accounting for overlap)
                    output_x = start_x * self.scale_factor
                    output_y = start_y * self.scale_factor
                    
                    # Crop overlap if needed
                    if x > 0 or y > 0:
                        crop_left = overlap * self.scale_factor if x > 0 else 0
                        crop_top = overlap * self.scale_factor if y > 0 else 0
                        
                        tile_w, tile_h = upscaled_tile.size
                        cropped_tile = upscaled_tile.crop((
                            crop_left, crop_top, tile_w, tile_h
                        ))
                        output_x += crop_left
                        output_y += crop_top
                        upscaled_tile = cropped_tile
                    
                    # Paste into output
                    output_image.paste(upscaled_tile, (output_x, output_y))
                    
                    print(f"  ✅ Tile {y*tiles_x + x + 1}/{tiles_x*tiles_y} completed")
                    
                except Exception as e:
                    print(f"  ❌ Tile {y*tiles_x + x + 1} failed: {e}")
                    continue
        
        return output_image
    
    def process_single_image(self, image):
        """Process single image with automatic padding"""
        padded_image, original_size = self.pad_image(image)
        
        # Convert to tensor
        img_array = np.array(padded_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Process with SwinIR
        with torch.no_grad():
            output_tensor = self.model(img_tensor)
        
        # Convert back to PIL
        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        upscaled_padded = Image.fromarray(output)
        
        # Crop to correct size
        return self.crop_output(upscaled_padded, original_size)
    
    def upscale(self, image):
        """Universal upscale - works with ANY image size"""
        try:
            if image.mode != 'RGB':
                print("🔄 Converting to RGB")
                image = image.convert('RGB')
            
            width, height = image.size
            print(f"📐 Input: {width}x{height}")
            
            # Choose processing method
            if width <= 1024 and height <= 1024:
                print("🔧 Direct processing")
                result = self.process_single_image(image)
            else:
                print("🧩 Tile processing")
                result = self.process_large_image_with_tiles(image)
            
            final_width, final_height = result.size
            print(f"✅ Output: {final_width}x{final_height}")
            return result
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise e

def enhanced_fallback_upscale(image):
    """High-quality fallback upscaling"""
    try:
        from PIL import ImageFilter, ImageEnhance
        
        width, height = image.size
        
        # Multi-step upscaling
        img_2x = image.resize((width * 2, height * 2), Image.LANCZOS)
        enhancer = ImageEnhance.Sharpness(img_2x)
        img_2x_sharp = enhancer.enhance(1.2)
        
        img_4x = img_2x_sharp.resize((width * 4, height * 4), Image.LANCZOS)
        
        # Final enhancement
        enhancer2 = ImageEnhance.Sharpness(img_4x)
        final_image = enhancer2.enhance(1.1)
        
        contrast_enhancer = ImageEnhance.Contrast(final_image)
        return contrast_enhancer.enhance(1.05)
        
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
        width, height = image.size
        return image.resize((width * 4, height * 4), Image.LANCZOS)

# Initialize Universal SwinIR
try:
    model_path = download_swinir_model()
    if model_path and os.path.exists("swinir_arch.py"):
        upscaler = UniversalSwinIRModel(model_path)
        model_status = "Universal SwinIR AI (ANY SIZE)"
        print("🎯 Universal SwinIR ready - supports images of ANY size!")
    else:
        upscaler = None
        model_status = "Enhanced Fallback"
        print("⚠️ SwinIR not available - using fallback")
except Exception as e:
    print(f"⚠️ SwinIR initialization failed: {e}")
    upscaler = None
    model_status = "Enhanced Fallback"

@app.get("/")
def read_root():
    return {
        "message": "Universal AI Image Upscaler",
        "model_status": model_status,
        "scale_factor": "4x",
        "supports": "ANY image size"
    }

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data))
        
        print(f"📤 Processing: {input_image.size} -> ", end="")
        
        if upscaler:
            try:
                upscaled_image = upscaler.upscale(input_image)
                method = "Universal SwinIR AI 4x"
            except Exception as e:
                print(f"❌ SwinIR failed: {e}")
                print("🔄 Using fallback...")
                upscaled_image = enhanced_fallback_upscale(input_image)
                method = "Enhanced Fallback 4x"
        else:
            upscaled_image = enhanced_fallback_upscale(input_image)
            method = "Enhanced Fallback 4x"
        
        print(f"{upscaled_image.size} ({method})")
        
        # Save result
        output_filename = f"upscaled_{uuid.uuid4().hex}.png"
        output_path = f"/tmp/{output_filename}"
        upscaled_image.save(output_path, "PNG", quality=95)
        
        print("✅ Completed successfully!")
        
        return FileResponse(
            output_path, 
            filename=f"upscaled_{file.filename}",
            media_type="image/png"
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/status")
def get_status():
    return {
        "model_loaded": upscaler is not None,
        "model_type": model_status,
        "scale_factor": "4x",
        "supports_any_size": True,
        "backend_running": True
    }

if __name__ == "__main__":
    import uvicorn
    print("🌐 Starting Universal AI Upscaler on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)