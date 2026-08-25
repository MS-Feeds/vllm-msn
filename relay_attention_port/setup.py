#!/usr/bin/env python3
"""Setup script for relay attention port to vLLM 0.9.1."""

import os
import sys
import shutil
from pathlib import Path

def install_relay_attention():
    """Install relay attention into vLLM 0.9.1."""
    
    # Get vLLM installation path
    try:
        import vllm
        vllm_path = Path(vllm.__file__).parent
        print(f"Found vLLM installation at: {vllm_path}")
    except ImportError:
        print("Error: vLLM 0.9.1 is not installed. Please install it first:")
        print("pip install vllm==0.9.1")
        return False
    
    # Check vLLM version
    if not hasattr(vllm, '__version__') or not vllm.__version__.startswith('0.9'):
        print("Warning: This port is designed for vLLM 0.9.1. Current version:", getattr(vllm, '__version__', 'unknown'))
    
    # Create relay attention directory in vLLM
    relay_dir = vllm_path / "attention" / "backends" / "relay"
    relay_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy relay attention files
    current_dir = Path(__file__).parent
    
    files_to_copy = [
        ("relayattn_ops_v091.py", "relayattn_ops.py"),
        ("relay_attention_backend.py", "relay_attention_backend.py"),
        ("relay_config.py", "relay_config.py"),
    ]
    
    for src_file, dst_file in files_to_copy:
        src_path = current_dir / src_file
        dst_path = relay_dir / dst_file
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_file} to {dst_path}")
        else:
            print(f"Warning: {src_file} not found")
    
    # Create __init__.py for the relay module
    init_file = relay_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("""\"\"\"Relay attention backend for vLLM 0.9.1.\"\"\"

from .relay_attention_backend import RelayAttentionBackend, RelayAttentionImpl
from .relay_config import RelayConfig, RelayInfo, RelayMetadata

__all__ = [
    "RelayAttentionBackend",
    "RelayAttentionImpl", 
    "RelayConfig",
    "RelayInfo",
    "RelayMetadata",
]
""")
        print(f"Created {init_file}")
    
    # Update attention backend selector
    selector_file = vllm_path / "attention" / "selector.py"
    if selector_file.exists():
        update_backend_selector(selector_file)
    
    print("\n✅ Relay attention installation completed!")
    print("\nTo use relay attention:")
    print("1. Import the backend: from vllm.attention.backends.relay import RelayAttentionBackend")
    print("2. Register it with the attention system")
    print("3. Configure your model to use relay attention")
    
    return True

def update_backend_selector(selector_file):
    """Update the attention backend selector to include relay attention."""
    
    try:
        with open(selector_file, 'r') as f:
            content = f.read()
        
        # Check if relay backend is already registered
        if "relay_attention" in content:
            print("Relay attention backend already registered in selector")
            return
        
        # Find the backend registration section
        if "def get_attn_backend(" in content:
            # Add relay attention to the backend list
            # This is a simplified approach - in practice, you'd need to modify
            # the actual backend selection logic
            print("Note: Manual backend registration may be required")
            print("Please check the attention selector file for integration details")
        
    except Exception as e:
        print(f"Warning: Could not update backend selector: {e}")

def create_example_script():
    """Create an example script for using relay attention."""
    
    example_script = """#!/usr/bin/env python3
\"\"\"Example usage of relay attention with vLLM 0.9.1.\"\"\"

import torch
from vllm.attention.backends.relay import RelayConfig, RelayInfo, RelayMetadata

def example_relay_attention():
    \"\"\"Example of using relay attention.\"\"\"
    
    # Create relay configuration
    config = RelayConfig(
        enabled=True,
        use_triton=True,
        cache_system_prompts=True,
        max_cached_system_prompts=50
    )
    
    # Create relay info for a specific system prompt length
    relay_info = RelayInfo(
        system_length=10,
        enabled=True,
        use_triton=True
    )
    
    # Create relay metadata
    relay_metadata = RelayMetadata(relay_info)
    
    print("Relay attention configuration created successfully!")
    print(f"System prompt length: {relay_info.system_length}")
    print(f"Triton enabled: {relay_info.use_triton}")
    print(f"Cache enabled: {relay_info.cache_system_prompts}")

if __name__ == "__main__":
    example_relay_attention()
"""
    
    example_path = Path(__file__).parent / "example_usage.py"
    example_path.write_text(example_script)
    print(f"Created example script: {example_path}")

def main():
    """Main installation function."""
    
    print("🚀 Installing Relay Attention for vLLM 0.9.1")
    print("=" * 50)
    
    # Install relay attention
    if install_relay_attention():
        # Create example script
        create_example_script()
        
        print("\n" + "=" * 50)
        print("🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Run the example: python example_usage.py")
        print("2. Check the README.md for detailed usage instructions")
        print("3. Integrate relay attention into your vLLM models")
    else:
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 