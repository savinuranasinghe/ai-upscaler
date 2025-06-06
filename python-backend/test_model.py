import os

# This mimics the path used in main.py
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'RealESRGAN_x4plus.pth')
abs_path = os.path.abspath(model_path)

print('Model path:', abs_path)
print('File exists:', os.path.exists(model_path))
if os.path.exists(model_path):
    print('File size:', os.path.getsize(model_path), 'bytes')
else:
    print('❌ Model file not found!')
    print('Expected location:', abs_path)