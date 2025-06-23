import time
import numpy as np
from skimage import color
from pathlib import Path

ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'

def test(instate, outstate):
    name='test'
    buffers=outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        instate['hue_offset'] = 0.0
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    remaining_time = instate['duration'] - instate['elapsed_time']
    
    # If less than 10 seconds remain, fade the generator alpha
    if remaining_time < 10.0:
        # Linear fade from 1.0 to 0.0 over 10 seconds
        generator_alpha = remaining_time / 10.0
        # Ensure we don't go below 0
        generator_alpha = max(0.0, generator_alpha)
        
        # Update the generator's alpha value
        buffers.generator_alphas[name] = generator_alpha
    else:
        # Full intensity
        buffers.generator_alphas[name] = 1.0

    # Get all strip buffers for our generator
    all_buffers = buffers.get_all_buffers(name)
    
    # Calculate time-based hue shift (0.05 per second)
    hue_shift_speed = 0.05
    instate['hue_offset'] = (instate['hue_offset'] + 
                            hue_shift_speed * (outstate['current_time'] - outstate['last_time'])) % 1.0
    
    # Fill each buffer with a rainbow pattern
    for strip_id, buffer in all_buffers.items():
        strip_length = len(buffer)
        
        for i in range(strip_length):
            # Calculate hue based on position and time
            # Position creates the rainbow, offset makes it move
            position_hue = i / strip_length
            hue = (position_hue + instate['hue_offset']) % 1.0
            
            # Convert HSV to RGB
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
            
            # Set the pixel color with full opacity
            buffer[i] = [r, g, b, 0.25]

def OTO_blink(instate, outstate):
    """
    Generator that randomly selects pixels to light up and then fades all pixels.
    Each frame, 10 new pixels are activated and all pixels fade by 5%.
    Vectorized implementation for efficiency.
    """
    name = 'blink'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return
    remaining_time = instate['duration'] - instate['elapsed_time']
    
    # If less than 10 seconds remain, fade the generator alpha
    if remaining_time < 10.0:
        # Linear fade from 1.0 to 0.0 over 10 seconds
        generator_alpha = remaining_time / 10.0
        # Ensure we don't go below 0
        generator_alpha = max(0.0, generator_alpha)
        
        # Update the generator's alpha value
        buffers.generator_alphas[name] = generator_alpha
    else:
        # Full intensity
        buffers.generator_alphas[name] = 1.0

    # Get all strip buffers for our generator
    all_buffers = buffers.get_all_buffers(name)
    
    # First, fade all existing pixels by 5%
    fade_factor = 0.95  # 5% reduction per frame
    
    for strip_id, buffer in all_buffers.items():
        # Vectorized fade: multiply alpha channel by fade factor
        buffer[:, 3] *= fade_factor
    
    # Add 10 new random pixels
    for _ in range(10):
        # Select a random strip
        strip_id = np.random.choice(list(all_buffers.keys()))
        buffer = all_buffers[strip_id]
        
        # Select a random pixel
        pixel_idx = np.random.randint(0, len(buffer))
        
        # Generate a random color (RGB) with full opacity
        random_color = np.append(np.random.random(3), 1.0)
        
        # Set the pixel in the buffer
        buffer[pixel_idx] = random_color

def hsv_to_rgb(h, s, v):
    """Convert HSV color to RGB color"""
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