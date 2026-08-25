"""Configuration classes for relay attention in vLLM 0.9.1."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RelayInfo:
    """Information about relay attention configuration."""
    
    # Length of the system prompt
    system_length: int
    
    # Whether to enable relay attention
    enabled: bool = True
    
    # Whether to use Triton kernels for fusion
    use_triton: bool = True
    
    # Cache configuration for system prompts
    cache_system_prompts: bool = True
    
    # Maximum number of cached system prompts
    max_cached_system_prompts: int = 100


@dataclass
class RelayConfig:
    """Configuration for relay attention."""
    
    # Whether to enable relay attention globally
    enabled: bool = False
    
    # Default relay info
    default_relay_info: Optional[RelayInfo] = None
    
    # Whether to use Triton kernels by default
    use_triton: bool = True
    
    # Cache configuration
    cache_system_prompts: bool = True
    max_cached_system_prompts: int = 100
    
    # Performance tuning
    enable_prefetch: bool = True
    prefetch_batch_size: int = 4
    
    def create_relay_info(self, system_length: int) -> RelayInfo:
        """Create a RelayInfo instance with the given system length."""
        return RelayInfo(
            system_length=system_length,
            enabled=self.enabled,
            use_triton=self.use_triton,
            cache_system_prompts=self.cache_system_prompts,
            max_cached_system_prompts=self.max_cached_system_prompts,
        )


class RelayMetadata:
    """Metadata for relay attention operations."""
    
    def __init__(self, relay_info: Optional[RelayInfo] = None):
        self.relay_info = relay_info
        self.system_cache_key: Optional[str] = None
        self.user_cache_key: Optional[str] = None
        
    def set_cache_keys(self, system_key: str, user_key: str):
        """Set cache keys for system and user parts."""
        self.system_cache_key = system_key
        self.user_cache_key = user_key
        
    def has_relay_info(self) -> bool:
        """Check if relay info is available."""
        return self.relay_info is not None and self.relay_info.enabled 