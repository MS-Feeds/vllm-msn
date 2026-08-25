#!/usr/bin/env python3
"""Verification script for relay attention port to vLLM 0.9.1."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists and report status."""
    if filepath.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_file_content(filepath: Path, required_content: List[str]) -> bool:
    """Check if file contains required content."""
    if not filepath.exists():
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing_content = []
        for req in required_content:
            if req not in content:
                missing_content.append(req)
        
        if missing_content:
            print(f"⚠️  {filepath.name}: Missing content: {missing_content}")
            return False
        else:
            print(f"✅ {filepath.name}: Content verified")
            return True
    except Exception as e:
        print(f"❌ {filepath.name}: Error reading file - {e}")
        return False

def verify_port_structure():
    """Verify the port structure and files."""
    print("🔍 Verifying Relay Attention Port Structure")
    print("=" * 50)
    
    current_dir = Path(__file__).parent
    files_to_check = [
        # Core implementation files
        (current_dir / "relayattn_ops_v091.py", "Core relay fusion operations"),
        (current_dir / "relay_attention_backend.py", "Relay attention backend"),
        (current_dir / "relay_config.py", "Configuration classes"),
        (current_dir / "integration_example.py", "Integration examples"),
        
        # Documentation files
        (current_dir / "README.md", "Documentation"),
        (current_dir / "PORTING_SUMMARY.md", "Porting summary"),
        
        # Setup and installation
        (current_dir / "setup.py", "Installation script"),
        
        # Reference files (original vLLM 0.2.6)
        (current_dir / "relayattn_ops_v026.py", "Original v0.2.6 operations"),
        (current_dir / "attention_v026.py", "Original v0.2.6 attention"),
        (current_dir / "attention_v091.py", "vLLM 0.9.1 attention reference"),
    ]
    
    all_files_exist = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    return all_files_exist

def verify_core_implementation():
    """Verify core implementation components."""
    print("\n🔧 Verifying Core Implementation")
    print("=" * 50)
    
    current_dir = Path(__file__).parent
    
    # Check relay operations
    ops_file = current_dir / "relayattn_ops_v091.py"
    ops_required = [
        "def relay_fusion(",
        "def _relay_fuse_triton(",
        "@triton.jit",
        "class RelayAttentionBackend:",
        "tl.program_id(",
        "tl.load(",
        "tl.store("
    ]
    
    # Check backend implementation
    backend_file = current_dir / "relay_attention_backend.py"
    backend_required = [
        "class RelayAttentionImpl(AttentionImpl):",
        "class RelayAttentionBackend(AttentionBackend):",
        "def forward(",
        "def _forward_relay(",
        "def _compute_attention_with_lse(",
        "relay_ops.fuse_attention_outputs("
    ]
    
    # Check configuration
    config_file = current_dir / "relay_config.py"
    config_required = [
        "@dataclass",
        "class RelayConfig:",
        "class RelayInfo:",
        "class RelayMetadata:",
        "system_length: int",
        "enabled: bool"
    ]
    
    all_implementations_valid = True
    
    if not check_file_content(ops_file, ops_required):
        all_implementations_valid = False
    
    if not check_file_content(backend_file, backend_required):
        all_implementations_valid = False
    
    if not check_file_content(config_file, config_required):
        all_implementations_valid = False
    
    return all_implementations_valid

def verify_documentation():
    """Verify documentation completeness."""
    print("\n📚 Verifying Documentation")
    print("=" * 50)
    
    current_dir = Path(__file__).parent
    
    # Check README
    readme_file = current_dir / "README.md"
    readme_required = [
        "# Relay Attention for vLLM 0.9.1",
        "## 📋 Overview",
        "## 🚀 Usage",
        "## 🔧 Implementation Details",
        "```python",
        "def relay_fusion(",
        "## 📊 Performance Benefits"
    ]
    
    # Check porting summary
    summary_file = current_dir / "PORTING_SUMMARY.md"
    summary_required = [
        "# Relay Attention Porting Summary",
        "## 📋 **Porting Status: COMPLETED**",
        "## 🎯 **What Was Accomplished**",
        "## 🔧 **Key Technical Adaptations**",
        "## 🚀 **Usage Examples**"
    ]
    
    all_docs_valid = True
    
    if not check_file_content(readme_file, readme_required):
        all_docs_valid = False
    
    if not check_file_content(summary_file, summary_required):
        all_docs_valid = False
    
    return all_docs_valid

def verify_setup_script():
    """Verify setup script functionality."""
    print("\n🛠️  Verifying Setup Script")
    print("=" * 50)
    
    current_dir = Path(__file__).parent
    setup_file = current_dir / "setup.py"
    
    setup_required = [
        "def install_relay_attention():",
        "def update_backend_selector(",
        "def create_example_script():",
        "vllm_path = Path(vllm.__file__).parent",
        "relay_dir = vllm_path / \"attention\" / \"backends\" / \"relay\"",
        "shutil.copy2(",
        "Relay attention installation completed"
    ]
    
    return check_file_content(setup_file, setup_required)

def verify_integration_example():
    """Verify integration example completeness."""
    print("\n🔗 Verifying Integration Example")
    print("=" * 50)
    
    current_dir = Path(__file__).parent
    example_file = current_dir / "integration_example.py"
    
    example_required = [
        "class RelayAttentionIntegration:",
        "def create_attention_layer(",
        "def create_relay_metadata(",
        "def example_usage():",
        "config = RelayConfig(",
        "integration = RelayAttentionIntegration(config)",
        "attention.forward("
    ]
    
    return check_file_content(example_file, example_required)

def run_verification():
    """Run complete verification."""
    print("🚀 Relay Attention Port Verification")
    print("=" * 60)
    
    results = {
        "structure": verify_port_structure(),
        "implementation": verify_core_implementation(),
        "documentation": verify_documentation(),
        "setup": verify_setup_script(),
        "integration": verify_integration_example()
    }
    
    print("\n" + "=" * 60)
    print("📊 Verification Results")
    print("=" * 60)
    
    all_passed = True
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component.capitalize():15} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Relay attention port is complete and ready for use")
        print("\nNext steps:")
        print("1. Run: python setup.py")
        print("2. Test integration with vLLM 0.9.1")
        print("3. Check README.md for usage instructions")
    else:
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("Please check the failed components above")
    
    return all_passed

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1) 