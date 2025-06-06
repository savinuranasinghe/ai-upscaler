import torch
from pathlib import Path

model_path = Path("../models/SwinIR_classical_SR_x4.pth")

if model_path.exists():
    print("🔍 Analyzing model architecture...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get the state dict
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint
    
    print("\n📊 Model Statistics:")
    print(f"Total parameters: {len(state_dict)}")
    
    # Analyze conv_last to determine embed_dim
    if 'conv_last.weight' in state_dict:
        conv_last_shape = state_dict['conv_last.weight'].shape
        print(f"\nconv_last.weight shape: {conv_last_shape}")
        print(f"This suggests embed_dim or num_feat = {conv_last_shape[1]}")
    
    # Check which upsampler is used
    print("\n🔧 Upsampler type detection:")
    if 'upsample.0.weight' in state_dict:
        print("✅ Uses 'pixelshuffle' or 'pixelshuffledirect' upsampler")
    elif 'conv_up1.weight' in state_dict and 'conv_up2.weight' in state_dict:
        print("✅ Uses 'nearest+conv' upsampler")
    
    # Check for conv_before_upsample
    if 'conv_before_upsample.0.weight' in state_dict:
        print("✅ Has conv_before_upsample (suggests 'pixelshuffle' or 'nearest+conv')")
    
    # List all unique layer prefixes
    print("\n📝 Unique layer prefixes:")
    prefixes = set()
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        prefixes.add(prefix)
    
    for prefix in sorted(prefixes):
        # Count parameters with this prefix
        count = sum(1 for k in state_dict.keys() if k.startswith(prefix))
        print(f"  - {prefix}: {count} parameters")
    
    # Check for patch_embed norm
    if 'patch_embed.norm.weight' in state_dict:
        print("\n✅ Model has patch_embed normalization")
    
    # Analyze depths by counting RSTB layers
    print("\n🏗️ RSTB layer analysis:")
    max_layer = -1
    for key in state_dict.keys():
        if key.startswith('layers.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                layer_idx = int(parts[1])
                max_layer = max(max_layer, layer_idx)
    
    if max_layer >= 0:
        print(f"Number of RSTB layers: {max_layer + 1}")
        
else:
    print("❌ Model file not found!")