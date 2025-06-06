import torch
from pathlib import Path
import json

def analyze_swinir_model():
    model_path = Path("../models/SwinIR_classical_SR_x4.pth")
    
    if not model_path.exists():
        print("❌ Model file not found!")
        return
    
    print("🔍 Comprehensive SwinIR Model Analysis")
    print("=" * 50)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get the state dict
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
        print("✅ Using EMA parameters")
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
        print("✅ Using regular parameters")
    else:
        state_dict = checkpoint
        print("✅ Using direct state dict")
    
    print(f"\n📊 Total parameters: {len(state_dict)}")
    
    # Analyze key patterns
    print("\n🔍 Key Analysis:")
    layer_counts = {}
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) > 0:
            layer_type = parts[0]
            if layer_type in layer_counts:
                layer_counts[layer_type] += 1
            else:
                layer_counts[layer_type] = 1
    
    for layer_type, count in sorted(layer_counts.items()):
        print(f"  {layer_type}: {count} parameters")
    
    # Check for specific missing layers
    missing_layers = []
    expected_layers = [
        'conv_first.weight',
        'conv_after_body.weight',
        'patch_embed.norm.weight',
        'layers.0.residual_group.blocks.0.norm1.weight'
    ]
    
    print("\n🔧 Layer Existence Check:")
    for layer in expected_layers:
        if layer in state_dict:
            print(f"  ✅ {layer}")
        else:
            print(f"  ❌ {layer}")
            missing_layers.append(layer)
    
    # Analyze conv_last to determine architecture
    if 'conv_last.weight' in state_dict:
        conv_last_shape = state_dict['conv_last.weight'].shape
        print(f"\n🎯 conv_last.weight shape: {conv_last_shape}")
        print(f"   Input channels: {conv_last_shape[1]}")
        print(f"   Output channels: {conv_last_shape[0]}")
    
    # Determine upsampler type
    print("\n🔧 Upsampler Detection:")
    if any('upsample.' in key for key in state_dict.keys()):
        print("  ✅ Uses 'pixelshuffle' upsampler")
        upsampler = 'pixelshuffle'
    elif any('conv_up1' in key for key in state_dict.keys()):
        print("  ✅ Uses 'nearest+conv' upsampler")
        upsampler = 'nearest+conv'
    else:
        print("  ❓ Unknown upsampler")
        upsampler = 'unknown'
    
    # Count RSTB layers
    max_layer = -1
    for key in state_dict.keys():
        if key.startswith('layers.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                layer_idx = int(parts[1])
                max_layer = max(max_layer, layer_idx)
    
    num_layers = max_layer + 1 if max_layer >= 0 else 0
    print(f"\n🏗️ Number of RSTB layers: {num_layers}")
    
    # Analyze first layer to get embed_dim
    embed_dim = None
    if 'conv_first.weight' in state_dict:
        embed_dim = state_dict['conv_first.weight'].shape[0]
        print(f"📐 Embed dimension: {embed_dim}")
    
    # Try to determine depths
    depths = []
    for i in range(num_layers):
        max_block = -1
        pattern = f'layers.{i}.residual_group.blocks.'
        for key in state_dict.keys():
            if key.startswith(pattern):
                parts = key.split('.')
                if len(parts) > 4 and parts[4].isdigit():
                    block_idx = int(parts[4])
                    max_block = max(max_block, block_idx)
        if max_block >= 0:
            depths.append(max_block + 1)
    
    print(f"📏 Depths per layer: {depths}")
    
    # Generate configuration
    print("\n🛠️ Suggested SwinIR Configuration:")
    config = {
        'upscale': 4,
        'in_chans': 3,
        'img_size': 64,
        'window_size': 8,
        'img_range': 1.0,
        'embed_dim': embed_dim or 180,
        'depths': depths or [6] * num_layers,
        'num_heads': [6] * num_layers,
        'mlp_ratio': 2,
        'upsampler': upsampler,
        'resi_connection': '1conv',
        'patch_norm': 'patch_embed.norm.weight' in state_dict
    }
    
    print(json.dumps(config, indent=2))
    
    # Check for attention patterns
    print(f"\n🎯 Attention Analysis:")
    attention_keys = [k for k in state_dict.keys() if 'attn' in k and 'relative_position_bias_table' in k]
    if attention_keys:
        first_attn = attention_keys[0]
        bias_shape = state_dict[first_attn].shape
        print(f"  Relative position bias shape: {bias_shape}")
        print(f"  Suggests window_size: {int((bias_shape[0] + 1) ** 0.5 / 2)}")
    
    return config, missing_layers

if __name__ == "__main__":
    try:
        config, missing = analyze_swinir_model()
        if missing:
            print(f"\n⚠️ Missing layers: {missing}")
            print("💡 Try loading with strict=False")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()