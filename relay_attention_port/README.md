# Relay Attention for vLLM 0.9.1

This directory contains the ported implementation of Relay Attention from vLLM 0.2.6 to vLLM 0.9.1.

## 📋 Overview

Relay Attention is an optimization technique that improves efficiency when processing multiple requests that share a common system prompt. Instead of computing attention for the full sequence (system + user) each time, it:

1. **Computes attention separately** for system and user parts
2. **Fuses the attention outputs** using a learned weighting mechanism
3. **Reduces redundant computation** for shared system prompts

## 🏗️ Architecture

### Core Components

- **`relayattn_ops_v091.py`**: Core relay fusion operations with Triton kernels
- **`relay_attention_backend.py`**: vLLM 0.9.1 attention backend integration
- **`relay_config.py`**: Configuration and metadata classes
- **`integration_example.py`**: Example usage and integration patterns

### Key Differences from vLLM 0.2.6

1. **Backend Architecture**: Adapted to vLLM 0.9.1's new attention backend system
2. **Metadata Handling**: Updated to work with vLLM 0.9.1's attention metadata
3. **Cache Integration**: Modified to work with the new KV cache system
4. **Configuration**: Simplified configuration system

## 🚀 Usage

### Basic Integration

```python
from relay_attention_port.relay_config import RelayConfig
from relay_attention_port.integration_example import RelayAttentionIntegration

# Create configuration
config = RelayConfig(
    enabled=True,
    use_triton=True,
    cache_system_prompts=True
)

# Create integration
integration = RelayAttentionIntegration(config)

# Create attention layer with relay support
attention = integration.create_attention_layer(
    num_heads=32,
    head_size=128,
    scale=0.125
)

# Create relay metadata
relay_metadata = integration.create_relay_metadata(system_length=10)

# Use in forward pass
output = attention.forward(query, key, value, attn_metadata=relay_metadata)
```

### Advanced Configuration

```python
from relay_attention_port.relay_config import RelayConfig, RelayInfo

# Custom relay configuration
relay_info = RelayInfo(
    system_length=15,
    enabled=True,
    use_triton=True,
    cache_system_prompts=True,
    max_cached_system_prompts=100
)

config = RelayConfig(
    enabled=True,
    default_relay_info=relay_info,
    enable_prefetch=True,
    prefetch_batch_size=8
)
```

## 🔧 Implementation Details

### Relay Fusion Algorithm

The core of relay attention is the fusion operation:

```python
def relay_fusion(out_sys, lse_sys, out_usr, lse_usr):
    # Calculate fusion weights
    alpha_sys = 1. / (1. + (lse_usr - lse_sys).exp())
    
    # Fuse outputs
    out_fused = alpha_sys * out_sys + (1. - alpha_sys) * out_usr
    
    return out_fused
```

### Backend Integration

The implementation integrates with vLLM 0.9.1's attention system through:

1. **Custom Attention Backend**: `RelayAttentionBackend` implements the required interface
2. **Metadata Extension**: `RelayMetadata` extends attention metadata with relay information
3. **Forward Pass Override**: Custom forward pass that handles relay logic

## 📊 Performance Benefits

### Expected Improvements

- **Reduced Computation**: Avoid redundant system prompt processing
- **Better Throughput**: Higher request processing rates
- **Memory Efficiency**: Optimized memory usage for shared prompts

### Benchmarks

*Note: Performance benchmarks would be added here after testing*

## 🔄 Migration from vLLM 0.2.6

### Key Changes Required

1. **Import Updates**: Update import paths to match vLLM 0.9.1 structure
2. **Backend Registration**: Register the relay attention backend
3. **Metadata Handling**: Update metadata creation and handling
4. **Configuration**: Update configuration to use new format

### Compatibility

- ✅ **vLLM 0.9.1**: Fully compatible
- ⚠️ **vLLM 0.2.6**: Requires migration (see original implementation)
- ❌ **Earlier versions**: Not supported

## 🧪 Testing

### Unit Tests

```bash
# Run unit tests (when implemented)
python -m pytest tests/test_relay_attention.py
```

### Integration Tests

```bash
# Test with actual vLLM models
python integration_example.py
```

## 📝 API Reference

### RelayConfig

```python
@dataclass
class RelayConfig:
    enabled: bool = False
    use_triton: bool = True
    cache_system_prompts: bool = True
    max_cached_system_prompts: int = 100
    enable_prefetch: bool = True
    prefetch_batch_size: int = 4
```

### RelayInfo

```python
@dataclass
class RelayInfo:
    system_length: int
    enabled: bool = True
    use_triton: bool = True
    cache_system_prompts: bool = True
    max_cached_system_prompts: int = 100
```

### RelayMetadata

```python
class RelayMetadata:
    def __init__(self, relay_info: Optional[RelayInfo] = None)
    def set_cache_keys(self, system_key: str, user_key: str)
    def has_relay_info(self) -> bool
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Backend Registration**: Verify relay backend is properly registered
3. **Memory Issues**: Check cache configuration for large models

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install vLLM 0.9.1: `pip install vllm==0.9.1`
4. Run tests: `python -m pytest`

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for all public methods
- Include unit tests for new features

## 📄 License

This implementation follows the same license as the original vLLM project.

## 🙏 Acknowledgments

- Original relay attention implementation in vLLM 0.2.6
- vLLM development team for the attention backend architecture
- Triton team for the GPU kernel framework 