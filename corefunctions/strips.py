from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
import numpy as np
import json
import yaml
import os
import math

@dataclass
class LEDStrip:
    """Represents a single LED strip with its properties and metadata"""
    
    # Basic properties
    id: str                           # Unique identifier for the strip
    length: int                       # Number of LEDs in strip
    type: Literal["line", "circle", "arc"]   # Type of strip (line, circle, or arc)
    groups: List[str] = field(default_factory=list)  # All groups this strip belongs to

    # Physical properties
    indices: Optional[np.ndarray] = None  # Physical indices if they differ from sequential
    coordinates: Optional[np.ndarray] = None  # 3D coordinates of each LED
    distance: Optional[np.ndarray] = None  # Normalized distance along the strip (0-1)
    
    # Position data (stored for reference)
    position_data: Dict[str, Any] = field(default_factory=dict)
    
    # Additional custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize derived properties"""
        # Ensure positions has correct shape
        # If indices not provided, create sequential indices
        if self.indices is None:
            self.indices = np.arange(self.length)
        
        # Initialize distance (normalized position along strip)
        self.distance = self.indices / (self.length - 1) if self.length > 1 else np.zeros(1)
        
        # Generate 3D coordinates based on type and position data
        self._generate_coordinates()
    
    def _generate_coordinates(self):
        """Generate 3D coordinates for each LED based on strip type"""
        if self.type == "line":
            self._generate_line_coordinates()
        elif self.type == "circle":
            self._generate_circle_coordinates()
        elif self.type == "arc":
            self._generate_arc_coordinates()
        else:
            raise ValueError(f"Unsupported strip type: {self.type}")
    
    def _generate_line_coordinates(self):
        """Generate coordinates for a line strip"""
        if "start" not in self.position_data or "end" not in self.position_data:
            # Create default coordinates if position data not provided
            self.coordinates = np.zeros((self.length, 3))
            return
            
        start = np.array(self.position_data["start"])
        end = np.array(self.position_data["end"])
        
        # Generate points along the line
        self.coordinates = np.zeros((self.length, 3))
        for i in range(self.length):
            t = i / (self.length - 1) if self.length > 1 else 0
            self.coordinates[i] = start + t * (end - start)
    
    def _generate_circle_coordinates(self):
        """Generate coordinates for a circular strip"""
        if "center" not in self.position_data or "radius" not in self.position_data:
            # Create default coordinates if position data not provided
            self.coordinates = np.zeros((self.length, 3))
            return
            
        center = np.array(self.position_data["center"])
        radius = self.position_data["radius"]
        normal = np.array(self.position_data.get("normal", [0, 1, 0]))
        start_angle = self.position_data.get("start_angle", 0.0)
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Find perpendicular vectors to the normal to define the circle plane
        # First perpendicular vector
        if abs(normal[0]) < abs(normal[1]):
            perp1 = np.array([1, 0, 0])
        else:
            perp1 = np.array([0, 1, 0])
            
        perp1 = perp1 - np.dot(perp1, normal) * normal
        perp1 = perp1 / np.linalg.norm(perp1)
        
        # Second perpendicular vector (cross product)
        perp2 = np.cross(normal, perp1)
        
        # Generate points around the circle
        self.coordinates = np.zeros((self.length, 3))
        for i in range(self.length):
            angle = start_angle + (i / self.length) * 2 * np.pi
            point = center + radius * (perp1 * np.cos(angle) + perp2 * np.sin(angle))
            self.coordinates[i] = point
    
    def _generate_arc_coordinates(self):
        """Generate coordinates for an arc strip"""
        if ("start" not in self.position_data or 
            "end" not in self.position_data or 
            "arc_degrees" not in self.position_data):
            # Create default coordinates if position data not provided
            self.coordinates = np.zeros((self.length, 3))
            return
            
        start = np.array(self.position_data["start"])
        end = np.array(self.position_data["end"])
        arc_degrees = self.position_data["arc_degrees"]
        normal = np.array(self.position_data.get("normal", [0, 1, 0]))
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate arc parameters
        chord = end - start
        chord_length = np.linalg.norm(chord)
        
        # Convert arc degrees to radians
        arc_radians = math.radians(arc_degrees)
        
        # Calculate the radius of the arc
        # For a circular arc: chord_length = 2 * radius * sin(arc_angle/2)
        radius = chord_length / (2 * math.sin(arc_radians / 2)) if arc_radians != 0 else float('inf')
        
        # Find the center of the arc
        # The center is at a distance 'h' from the midpoint of the chord
        # where h = radius * cos(arc_angle/2)
        midpoint = (start + end) / 2
        
        # Direction from midpoint to center is perpendicular to chord and in plane defined by normal
        chord_normalized = chord / chord_length
        
        # Direction vector perpendicular to both chord and normal
        perp_dir = np.cross(normal, chord_normalized)
        perp_dir = perp_dir / np.linalg.norm(perp_dir)
        
        # Direction from midpoint to center (perpendicular to chord, in plane)
        center_dir = np.cross(chord_normalized, perp_dir)
        center_dir = center_dir / np.linalg.norm(center_dir)
        
        # Distance from midpoint to center
        h = radius * math.cos(arc_radians / 2)
        
        # If arc_degrees > 180, flip the center direction
        if arc_degrees > 180:
            h = -h
            
        # Calculate center
        center = midpoint + h * center_dir
        
        # Calculate start angle and sweep angle
        # Vector from center to start point
        vec_to_start = start - center
        vec_to_start = vec_to_start / np.linalg.norm(vec_to_start)
        
        # Find perpendicular vectors to define the arc plane
        # First perpendicular vector is the normalized vector to start
        perp1 = vec_to_start
        
        # Second perpendicular vector (cross product of normal and perp1)
        perp2 = np.cross(normal, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # The sweep direction should be from start to end
        # Calculate the sign of the sweep by checking which side of the plane the end point is on
        vec_to_end = end - center
        vec_to_end = vec_to_end / np.linalg.norm(vec_to_end)
        
        # Dot product with perp2 to determine direction
        dot_product = np.dot(vec_to_end, perp2)
        sweep_sign = -1 if dot_product < 0 else 1
        
        # Generate points along the arc
        self.coordinates = np.zeros((self.length, 3))
        
        # Start angle is 0 (vec_to_start is already our x-axis equivalent)
        start_angle = 0
        
        # Calculate the total sweep angle
        sweep_angle = sweep_sign * arc_radians
        
        for i in range(self.length):
            t = i / (self.length - 1) if self.length > 1 else 0
            angle = start_angle + t * sweep_angle
            
            # Calculate point position using parametric equation of circle
            point = center + radius * (perp1 * math.cos(angle) + perp2 * math.sin(angle))
            self.coordinates[i] = point

class StripManager:
    """Manages a collection of LED strips with query capabilities"""
    
    def __init__(self):
        self.strips: Dict[str, LEDStrip] = {}
        self._groups_cache: Dict[str, List[str]] = {}  # Cache for group lookups
    
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
    
    def get_spatial_strips(self, center: Tuple[float, float, float], 
                          radius: float) -> Dict[str, LEDStrip]:
        """Get strips that have LEDs within a sphere of specified radius from center"""
        center_array = np.array(center)
        result = {}
        
        for strip_id, strip in self.strips.items():
            if strip.coordinates is None:
                continue
                
            # Calculate distances from center to each LED in the strip
            distances = np.linalg.norm(strip.coordinates - center_array, axis=1)
            
            # If any LED is within radius, include this strip
            if np.any(distances <= radius):
                result[strip_id] = strip
        
        return result
    
    def _invalidate_caches(self) -> None:
        """Clear caches when strips are modified"""
        self._groups_cache.clear()
    
    def create_buffers(self) -> Dict[str, np.ndarray]:
        """Create a buffer dictionary with arrays for each strip (for rendering)"""
        return {
            strip_id: np.zeros((strip.length, 4), dtype=np.float32)  # RGBA
            for strip_id, strip in self.strips.items()
        }

    def create_dmx_senders(self) -> Dict[str, Any]:
        """Create DMX senders for each unique IP address in strip metadata"""
        from corefunctions.ImageToDMX import SACNPixelSender
        
        # Group strips by IP address
        ip_strips = {}
        for strip_id, strip in self.strips.items():
            ip = strip.metadata.get('REC_IP')
            if ip:
                if ip not in ip_strips:
                    ip_strips[ip] = []
                ip_strips[ip].append(strip)
        
        # Create receivers config for each IP
        senders = {}
        for ip, strips in ip_strips.items():
            # Calculate total pixel count for this IP
            total_pixels = sum(strip.length for strip in strips)
            
            # Sort strips by Strip_num to ensure correct order
            strips.sort(key=lambda s: s.metadata.get('Strip_num', 0))
            
            # Create receiver configuration with strip mapping info
            receiver_config = {
                'ip': ip,
                'pixel_count': total_pixels,
                'strip_info': [(strip.id, strip.length, strip.metadata.get('Direction', 1)) 
                            for strip in strips]
            }
            
            # Create sender for this IP
            try:
                sender = SACNPixelSender([receiver_config])
                senders[ip] = {
                    'sender': sender,
                    'config': receiver_config
                }
            except Exception as e:
                print(f"Error creating DMX sender for {ip}: {e}")
        
        return senders

    def send_dmx(self, output_buffers, dmx_senders):
        """Send buffer data to DMX receivers"""
        for ip, sender_info in dmx_senders.items():
            sender = sender_info['sender']
            config = sender_info['config']
            
            # Send to DMX using the direct output buffers
            try:
                sender.send_from_buffers(output_buffers, config['strip_info'])
            except Exception as e:
                print(f"Error sending DMX data to {ip}: {e}")


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
        strip_type = data.get('type', 'line')  # Default to line if type not specified
        
        # Get position data
        position_data = data.get('position', {})
        
        # Create the strip with other metadata
        return LEDStrip(
            id=strip_id,
            length=length,
            type=strip_type,
            groups=data.get('groups', []),
            position_data=position_data,
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
        """
        # Skip if generator already exists
        if generator_name in self.generators:
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
                    # Vectorized alpha blending
                    # Calculate combined alpha values (source alpha * generator alpha)
                    alphas = source_buffer[:, 3] * generator_alpha
                    
                    # Create a mask for pixels with alpha > 0
                    mask = alphas > 0
                    
                    if np.any(mask):  # Only process if there are pixels to blend
                        # Calculate the blended colors for RGB channels
                        one_minus_alpha = 1 - alphas[mask, np.newaxis]
                        target_buffer[mask, :3] = one_minus_alpha * target_buffer[mask, :3] + \
                                                alphas[mask, np.newaxis] * source_buffer[mask, :3]
                        
                        # Update the alpha channel with the maximum value
                        target_buffer[mask, 3] = np.maximum(target_buffer[mask, 3], alphas[mask])
                            
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


# Helper functions for testing
def hsv_to_rgb(h, s, v):
    """Simple HSV to RGB conversion"""
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
    """Generate a rainbow pattern in the buffers"""
    for strip_id, buffer in buffers.items():
        # Fill buffer with rainbow pattern
        for i in range(len(buffer)):
            hue = (i / len(buffer)) % 1.0
            # Convert HSV to RGB
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
            buffer[i] = [r, g, b, 1.0]  # Full opacity