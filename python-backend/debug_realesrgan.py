import os
print("🔍 Debugging Real-ESRGAN loading...")

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import cv2
    print("✅ OpenCV imported successfully")
except Exception as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")

try:
    from realesrgan import RealESRGANer
    print("✅ RealESRGANer imported successfully")
except Exception as e:
    print(f"❌ RealESRGANer import failed: {e}")

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print("✅ RRDBNet imported successfully")
except Exception as e:
    print(f"❌ RRDBNet import failed: {e}")

# Test 2: Check model path
print("\n2. Testing model path...")
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'RealESRGAN_x4plus.pth')
abs_path = os.path.abspath(model_path)
print(f"Model path: {abs_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path)} bytes")

# Test 3: Try creating the model
print("\n3. Testing model creation...")
try:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    print("✅ RRDBNet model created successfully")
except Exception as e:
    print(f"❌ RRDBNet creation failed: {e}")

# Test 4: Try initializing RealESRGANer
print("\n4. Testing RealESRGANer initialization...")
try:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None
    )
    print("✅ RealESRGANer initialized successfully!")
except Exception as e:
    print(f"❌ RealESRGANer initialization failed: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()

print("\n🏁 Debug complete!")