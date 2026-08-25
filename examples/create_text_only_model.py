#!/usr/bin/env python3
"""
Create text-only variant of Gemma 4 26B by removing vision weights.

This script:
1. Loads each safetensors shard
2. Filters out vision/image weights
3. Saves text-only weights
4. Updates config to remove vision_config
5. Copies other necessary files

Memory savings: ~1.5-2 GB GPU memory
"""

from safetensors.torch import load_file, save_file
import json
import shutil
import os
import argparse

def create_text_only_model(model_path, output_path):
    """Create text-only model from multimodal Gemma 4."""

    print("=" * 70)
    print("Creating Text-Only Gemma 4 Model")
    print("=" * 70)
    print(f"Source: {model_path}")
    print(f"Target: {output_path}")
    print()

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    total_original_size = 0
    total_new_size = 0
    total_removed = 0

    # Process each safetensors file
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]

    for shard in sorted(safetensors_files):
        input_file = os.path.join(model_path, shard)
        output_file = os.path.join(output_path, shard)

        print(f"Processing {shard}...")

        # Load weights
        print("  Loading weights...")
        weights = load_file(input_file)

        # Filter out vision weights
        text_weights = {}
        vision_weights_removed = 0

        for key, tensor in weights.items():
            # Check if this is a vision-related weight
            if any(keyword in key.lower() for keyword in ['vision', 'image', 'visual']):
                vision_weights_removed += 1
                if vision_weights_removed <= 5:  # Show first 5
                    print(f"    Removing: {key}")
            else:
                text_weights[key] = tensor

        if vision_weights_removed > 5:
            print(f"    ... and {vision_weights_removed - 5} more vision tensors")

        print(f"  Kept: {len(text_weights)} tensors")
        print(f"  Removed: {vision_weights_removed} vision tensors")

        # Save text-only weights
        print("  Saving text-only weights...")
        save_file(text_weights, output_file)

        # Calculate size savings
        original_size = os.path.getsize(input_file) / (1024**3)
        new_size = os.path.getsize(output_file) / (1024**3)
        savings = original_size - new_size

        total_original_size += original_size
        total_new_size += new_size
        total_removed += vision_weights_removed

        print(f"  Original: {original_size:.2f} GB")
        print(f"  New: {new_size:.2f} GB")
        print(f"  Savings: {savings:.2f} GB")
        print()

    # Copy and modify config
    print("Updating config.json...")
    config_path = os.path.join(model_path, "config.json")
    output_config_path = os.path.join(output_path, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Remove vision and audio configs
    original_config = config.copy()
    if "vision_config" in config:
        config["vision_config"] = None
        print("  Removed vision_config")
    if "audio_config" in config:
        config["audio_config"] = None
        print("  Removed audio_config")
    if "video_config" in config:
        config["video_config"] = None
        print("  Removed video_config")

    # Save modified config
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("  ✓ Config updated")
    print()

    # Update model index if it exists
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        print("Updating model.safetensors.index.json...")
        with open(index_path, "r") as f:
            index = json.load(f)

        # Remove vision weight references from weight_map
        if "weight_map" in index:
            original_count = len(index["weight_map"])
            index["weight_map"] = {
                k: v for k, v in index["weight_map"].items()
                if not any(keyword in k.lower() for keyword in ['vision', 'image', 'visual'])
            }
            removed_count = original_count - len(index["weight_map"])
            print(f"  Removed {removed_count} vision weight entries")

        output_index_path = os.path.join(output_path, "model.safetensors.index.json")
        with open(output_index_path, "w") as f:
            json.dump(index, f, indent=2)
        print("  ✓ Index updated")
        print()

    # Copy other necessary files
    print("Copying other files...")
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "processor_config.json",
        ".gitattributes",
        "README.md"
    ]

    for filename in files_to_copy:
        src = os.path.join(model_path, filename)
        dst = os.path.join(output_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Copied: {filename}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total original size: {total_original_size:.2f} GB")
    print(f"Total new size: {total_new_size:.2f} GB")
    print(f"Total savings: {total_original_size - total_new_size:.2f} GB")
    print(f"Vision tensors removed: {total_removed}")
    print()
    print("✓ Text-only model created successfully!")
    print()
    print(f"Location: {output_path}")
    print()
    print("To use this model, update your config:")
    print(f"  MODEL_PATH={output_path}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create text-only variant of Gemma 4 by removing vision weights"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it",
        help="Path to original multimodal Gemma 4 model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/nvmedata/hf_checkpoints/gemma-4-26B-A4B-it-text-only",
        help="Path for text-only model output"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        exit(1)

    if not os.path.exists(os.path.join(args.model_path, "config.json")):
        print(f"Error: No config.json found in: {args.model_path}")
        exit(1)

    # Create text-only model
    create_text_only_model(args.model_path, args.output_path)
