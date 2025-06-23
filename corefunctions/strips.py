from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import yaml
import os

@dataclass
class LEDStrip:
    """Represents a single LED strip with its properties and metadata"""
    
    # Basic properties
    id: str                           # Unique identifier for the strip
    length: int                       # Number of LEDs in strip
      # Primary group this strip belongs to
    groups: List[str] = field(default_factory=list)  # All groups this strip belongs to

    # Physical properties
    indices: Optional[np.ndarray] = None  # Physical indices if they differ from sequential
    distance: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None
    # Additional custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize derived properties"""
        # Ensure positions has correct shape
        # If indices not provided, create sequential indices
        if self.indices is None:
            self.indices = np.arange(self.length)
        
        self.distance=self.indices/self.length
        


class StripManager:
    """Manages a collection of LED strips with query capabilities"""
    
    def __init__(self):
        self.strips: Dict[str, LEDStrip] = {}
        self._groups_cache: Dict[str, List[str]] = {}  # Cache for group lookups
        self._themes_cache: Dict[str, List[str]] = {}  # Cache for theme lookups
    
    def add_strip(self, strip: LEDStrip) -> None:
        """Add a strip to the collection"""
        self.strips[strip.id] = strip
        self._invalidate_caches()
    
    def remove_strip(self, strip_id: str) -> None:
        """Remove a strip from the collection"""
        if strip_id in self.strips:
            del self.strips[strip_id]
            self._invalidate_caches()
    
    def get_strip(self, strip_id: str) -> LEDStrip:
        """Get a specific strip by ID"""
        if strip_id not in self.strips:
            raise KeyError(f"Strip {strip_id} not found")
        return self.strips[strip_id]
    
    def get_strip_ids(self) -> List[str]:
        """Get all strip IDs"""
        return list(self.strips.keys())
    
    def get_all_strips(self) -> Dict[str, LEDStrip]:
        """Get all strips"""
        return self.strips
    
    def get_strips_by_group(self, group: str) -> Dict[str, LEDStrip]:
        """Get all strips that belong to a specific group"""
        if group not in self._groups_cache:
            self._groups_cache[group] = [
                strip_id for strip_id, strip in self.strips.items()
                if group in strip.groups
            ]
        
        return {
            strip_id: self.strips[strip_id]
            for strip_id in self._groups_cache[group]
        }
    
    def get_strips_by_theme(self, theme: str) -> Dict[str, LEDStrip]:
        """Get all strips with a specific theme"""
        if theme not in self._themes_cache:
            self._themes_cache[theme] = [
                strip_id for strip_id, strip in self.strips.items()
                if strip.theme == theme
            ]
        
        return {
            strip_id: self.strips[strip_id]
            for strip_id in self._themes_cache[theme]
        }
    
    def get_spatial_strips(self, center: Tuple[float, float, float], 
                          radius: float) -> Dict[str, LEDStrip]:
        """Get strips that have LEDs within a sphere of specified radius from center"""
        center_array = np.array(center)
        result = {}
        
        for strip_id, strip in self.strips.items():
            # Calculate distances from center to each LED in the strip
            distances = np.linalg.norm(strip.positions - center_array, axis=1)
            
            # If any LED is within radius, include this strip
            if np.any(distances <= radius):
                result[strip_id] = strip
        
        return result
    
    def _invalidate_caches(self) -> None:
        """Clear caches when strips are modified"""
        self._groups_cache.clear()
        self._themes_cache.clear()
    
    def create_buffers(self) -> Dict[str, np.ndarray]:
        """Create a buffer dictionary with arrays for each strip (for rendering)"""
        return {
            strip_id: np.zeros((strip.length, 4), dtype=np.float32)  # RGBA
            for strip_id, strip in self.strips.items()
        }
    
class StripLoader:
    """Utilities for loading strip definitions from files"""
    
    @staticmethod
    def from_json(file_path: str) -> StripManager:
        """Load strip definitions from a JSON file"""
        manager = StripManager()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for strip_data in data.get('strips', []):
            manager.add_strip(StripLoader._create_strip_from_dict(strip_data))
        
        return manager
    
    @staticmethod
    def from_yaml(file_path: str) -> StripManager:
        """Load strip definitions from a YAML file"""
        manager = StripManager()
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        for strip_data in data.get('strips', []):
            manager.add_strip(StripLoader._create_strip_from_dict(strip_data))
        
        return manager
    
    @staticmethod
    def _create_strip_from_dict(data: Dict[str, Any]) -> LEDStrip:
        """Create a LEDStrip instance from a dictionary"""
        # Extract required fields
        strip_id = data['id']
        length = data['length']
        # Handle positions - could be inline or reference a file
        positions = data.get('positions')
        if isinstance(positions, str) and os.path.exists(positions):
            # Load positions from file (CSV, NPY, etc.)
            if positions.endswith('.npy'):
                positions = np.load(positions)
            elif positions.endswith('.csv'):
                positions = np.loadtxt(positions, delimiter=',')
            else:
                raise ValueError(f"Unsupported position file format: {positions}")
        else:
            # Convert from JSON array to numpy array
            positions = np.array(positions)
        
        # Create the strip with other metadata
        return LEDStrip(
            id=strip_id,
            length=length,
            groups=data.get('groups', []),
            metadata=data.get('metadata', {})
        )

class BufferManager:
    """Manages multiple output buffers for different graphic generators"""
    
    def __init__(self, strip_manager: StripManager):
        self.strip_manager = strip_manager
        self.generators = {}  # Dictionary to store generator names and their buffers
        self.generator_alphas = {}

    def register_generator(self, generator_name: str) -> None:
        """
        Register a new graphic generator and create its buffers
        
        Args:
            generator_name: Name of the generator
            alpha: Global alpha value for this generator (0.0 - 1.0)
        """
        # Skip if generator already exists
        if generator_name in self.generators:
            # Update alpha value if generator exists
            #self.generator_alphas[generator_name] = alpha
            return
            
        # Create buffers for this generator (one per strip)
        self.generators[generator_name] = {
            strip_id: np.zeros((strip.length, 4), dtype=np.float32)  # RGBA
            for strip_id, strip in self.strip_manager.strips.items()
        }
        
        # Store alpha value for this generator
        self.generator_alphas[generator_name] = 1
    
    
    def get_buffer(self, generator_name: str, strip_id: str) -> np.ndarray:
        """Get a specific buffer for a generator and strip"""
        if generator_name not in self.generators:
            raise KeyError(f"Generator '{generator_name}' not registered")
        
        if strip_id not in self.generators[generator_name]:
            raise KeyError(f"Strip '{strip_id}' not found in generator '{generator_name}'")
        
        return self.generators[generator_name][strip_id]
    
    def get_all_buffers(self, generator_name: str) -> Dict[str, np.ndarray]:
        """Get all buffers for a specific generator"""
        if generator_name not in self.generators:
            raise KeyError(f"Generator '{generator_name}' not registered")
        
        return self.generators[generator_name]
    
    def clear_buffer(self, generator_name: str, strip_id: str = None) -> None:
        """Clear buffer(s) for a generator, optionally for a specific strip only"""
        if generator_name not in self.generators:
            raise KeyError(f"Generator '{generator_name}' not registered")
        
        if strip_id is not None:
            # Clear specific strip buffer
            if strip_id in self.generators[generator_name]:
                self.generators[generator_name][strip_id].fill(0)
        else:
            # Clear all buffers for this generator
            for buffer in self.generators[generator_name].values():
                buffer.fill(0)
    
    def merge_buffers(self, output_buffers: Dict[str, np.ndarray], blend_mode: str = 'alpha') -> None:
        """
        Merge all generator buffers into the output buffers using specified blend mode
        
        Args:
            output_buffers: Target buffers to merge into (typically the final output)
            blend_mode: How to combine pixels ('alpha', 'add', 'max', etc.)
        """
        # For each strip
        for strip_id in output_buffers:
            if strip_id not in self.strip_manager.strips:
                continue
                
            # Clear the output buffer
            output_buffers[strip_id].fill(0)
            
            # For each generator, blend its buffer into the output
            for generator_name, gen_buffers in self.generators.items():
                # Skip generators with zero alpha
                generator_alpha = self.generator_alphas.get(generator_name, 1.0)
                if generator_alpha <= 0.0:
                    continue
                    
                if strip_id not in gen_buffers:
                    continue
                    
                source_buffer = gen_buffers[strip_id]
                target_buffer = output_buffers[strip_id]
                
                if blend_mode == 'alpha':
                    # Alpha blending
                    for i in range(len(target_buffer)):
                        # Apply generator's global alpha to the pixel alpha
                        alpha = source_buffer[i, 3] * generator_alpha
                        if alpha > 0:
                            # Only blend if source has some opacity
                            target_buffer[i, :3] = (1 - alpha) * target_buffer[i, :3] + alpha * source_buffer[i, :3]
                            target_buffer[i, 3] = max(target_buffer[i, 3], alpha)
                            
                elif blend_mode == 'add':
                    # Additive blending with generator alpha
                    if generator_alpha < 1.0:
                        # Scale source by generator alpha
                        scaled_source = source_buffer * generator_alpha
                        target_buffer += scaled_source
                    else:
                        target_buffer += source_buffer
                    np.clip(target_buffer, 0, 1, out=target_buffer)
                    
                elif blend_mode == 'max':
                    # Maximum value blending with generator alpha
                    if generator_alpha < 1.0:
                        # Scale source by generator alpha
                        scaled_source = source_buffer * generator_alpha
                        np.maximum(target_buffer, scaled_source, out=target_buffer)
                    else:
                        np.maximum(target_buffer, source_buffer, out=target_buffer)

strip_manager = StripLoader.from_json("C:\\Users\\diete\\Desktop\\devel-local\\Out The Other\\OTO\\strips.json")
buffer_manager = BufferManager(strip_manager)

# Register different graphic generators
buffer_manager.register_generator("rainbow")
buffer_manager.register_generator("sparkle") 
buffer_manager.register_generator("wave")
output_buffers = strip_manager.create_buffers()
def hsv_to_rgb(h, s, v):
    # Simple HSV to RGB conversion
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if h_i == 0:
        return v, t, p
    elif h_i == 1:
        return q, v, p
    elif h_i == 2:
        return p, v, t
    elif h_i == 3:
        return p, q, v
    elif h_i == 4:
        return t, p, v
    else:
        return v, p, q
def generate_rainbow(buffers):
    for strip_id, buffer in buffers.items():
        # Fill buffer with rainbow pattern
        for i in range(len(buffer)):
            hue = (i / len(buffer)) % 1.0
            # Convert HSV to RGB
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
            buffer[i] = [r, g, b, 1.0]  # Full opacity

generate_rainbow(buffer_manager.get_all_buffers("rainbow"))

buffer_manager.merge_buffers(output_buffers, blend_mode='alpha')