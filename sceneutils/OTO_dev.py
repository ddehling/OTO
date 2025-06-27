import time
import numpy as np
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
    

# ... existing code ...

def OTO_point_traveler(instate, outstate):
    """
    Generator that illuminates a single point moving down the length of left_wall.
    The point has a random color and pixels fade slowly after the point passes.
    """
    name = 'point_traveler'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        # Initialize position at the start of the strip
        instate['position'] = 0
        # Generate a random color for this run
        instate['color'] = np.append(np.random.random(3), 1.0)
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
    
    # First, fade all existing pixels by a small amount (slower decay)
    fade_factor = 0.98  # 2% reduction per frame
    
    for strip_id, buffer in all_buffers.items():
        # Vectorized fade: multiply alpha channel by fade factor
        buffer[:, 3] *= fade_factor
    
    # Focus on the left_wall strip
    if 'right_spine' in all_buffers:
        buffer = all_buffers['right_spine']
        strip_length = len(buffer)
        
        # Set the current position to the current color
        if instate['position'] < strip_length:
            buffer[instate['position']] = instate['color']
        
        # Move the position for the next frame
        instate['position'] += 1
        
        # If we've reached the end, reset position and choose a new color
        if instate['position'] >= strip_length:
            instate['position'] = 0
            instate['color'] = np.append(np.random.random(3), 1.0)

# ... existing code ...

def OTO_heartbeat(instate, outstate):
    """
    Generator that creates a heartbeat pattern on objects tagged with the 'heart' category.
    Objects tagged with 'left' and 'right' groups will pulse in alternating phases.
    For circular strips, creates a rotating wave pattern that follows the circle.
    Heart rate and strength are configurable, and color can be varied.
    """
    name = 'heartbeat'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name,1)
        
        # Initialize heartbeat parameters with defaults
        instate['heart_rate'] = 60  # beats per minute
        instate['strength'] = 1.0   # intensity of the pulse (0.0 to 1.0)
        instate['base_color'] = [0.8, 0.0, 0.0, 0.1]  # default: red with low base intensity
        instate['peak_color'] = [1.0, 0.0, 0.0, 1.0]  # default: bright red at peak
        instate['phase'] = 0.0      # current phase of the heartbeat cycle
        
        # Parameters for circular patterns
        instate['wave_freq'] = 6.0  # Number of waves around the circle
        instate['rotation_speed'] = 0.8  # Rotation speed (revolutions per second)
        
        # Print debug information
        print(f"OTO_heartbeat: Initialized with heart_rate={instate['heart_rate']}, strength={instate['strength']}")
        return

    if instate['count'] == -1:
        buffers.generator_alphas[name] = 0
        return

    # Set generator alpha to full (ensure visibility)
    buffers.generator_alphas[name] = 1.0

    # Get all strip buffers for our generator
    all_buffers = buffers.get_all_buffers(name)
    
    # Directly access the strip_manager from the buffer manager
    strip_manager = buffers.strip_manager
    
    # Fallback to outstate if not available directly
    if strip_manager is None and 'strip_manager' in outstate:
        strip_manager = outstate['strip_manager']
    
    if strip_manager is None:
        print("OTO_heartbeat: No strip_manager available!")
        return
    
    # Update phase based on heart rate
    # heart_rate is in BPM, so divide by 60 to get beats per second
    beats_per_second = instate['heart_rate'] / 60.0
    phase_increment = beats_per_second * (outstate['current_time'] - outstate['last_time'])
    instate['phase'] = (instate['phase'] + phase_increment) % 1.0
    
    # Calculate rotation for circular patterns
    rotation_increment = instate['rotation_speed'] * (outstate['current_time'] - outstate['last_time'])
    if 'rotation_phase' not in instate:
        instate['rotation_phase'] = 0.0
    instate['rotation_phase'] = (instate['rotation_phase'] + rotation_increment) % 1.0
    
    # Check if we should update the base color from external conditions
    if 'heart_color' in outstate:
        # Update base and peak colors based on external input
        base_color = outstate['heart_color'].copy()
        base_color[3] = 0.1  # Low alpha for base
        instate['base_color'] = base_color
        
        peak_color = outstate['heart_color'].copy()
        peak_color[3] = 1.0  # Full alpha for peak
        instate['peak_color'] = peak_color
    
    # Check if heart rate should be updated from external conditions
    if 'control_speed' in outstate:
        instate['heart_rate'] = outstate['control_speed']
    
    # Check if strength should be updated from external conditions
    if 'control_intensity' in outstate:
        instate['strength'] = outstate['control_intensity']/100.0
    
    # Process each buffer
    for strip_id, buffer in all_buffers.items():
        # Get the actual strip object
        try:
            strip = strip_manager.get_strip(strip_id)
        except KeyError:
            # Skip if strip doesn't exist
            continue
        
        # Check if strip is tagged with 'heart' category
        if 'heart' not in strip.groups:
            continue
        
        # Determine if this is a left or right heart component based on tags
        is_left = 'left' in strip.groups
        is_right = 'right' in strip.groups
        is_circle = 'circle' in strip.groups or strip.type == "circle"
        
        # If neither left nor right, default to left phase
        if not (is_left or is_right):
            is_left = True
        
        # Phase offset: left and right sides beat in alternating phases
        phase_offset = 0.0 if is_left else 0.5
        current_phase = (instate['phase'] + phase_offset) % 1.0
        
        # Create heartbeat waveform (global intensity)
        if current_phase < 0.1:
            # Sharp rise (systole)
            global_intensity = current_phase / 0.1
        elif current_phase < 0.4:
            # First decline (early diastole)
            global_intensity = 1.0 - 0.7 * ((current_phase - 0.1) / 0.3)
        else:
            # Second, more gradual decline (late diastole)
            global_intensity = 0.3 - 0.3 * ((current_phase - 0.4) / 0.6)
        
        # Apply strength modifier to global intensity
        global_intensity *= instate['strength']
        
        # Now handle different patterns based on strip type
        strip_length = len(buffer)
        
        if is_circle:
            # For circular strips, create a rotating wave pattern
            # Apply the circular pattern to individual LEDs
            for i in range(strip_length):
                # Calculate position around circle (0 to 1)
                pos = i / strip_length
                
                # Create multiple sinusoidal waves that rotate around the circle
                # Primary wave (follows the heartbeat rhythm)
                primary_wave = 0.5 + 0.5 * np.sin(2 * np.pi * (pos + instate['rotation_phase']))
                
                # Secondary wave (higher frequency for texture)
                secondary_wave = 0.5 + 0.5 * np.sin(2 * np.pi * instate['wave_freq'] * pos + 4 * np.pi * instate['rotation_phase'])
                
                # Combine waves with global heartbeat intensity
                local_intensity = global_intensity * (primary_wave * secondary_wave)
                
                # Ensure intensity is in valid range
                local_intensity = np.clip(local_intensity, 0.0, 1.0)
                
                # Interpolate between base and peak colors based on intensity
                r = instate['base_color'][0] + (instate['peak_color'][0] - instate['base_color'][0]) * local_intensity
                g = instate['base_color'][1] + (instate['peak_color'][1] - instate['base_color'][1]) * local_intensity
                b = instate['base_color'][2] + (instate['peak_color'][2] - instate['base_color'][2]) * local_intensity
                a = instate['base_color'][3] + (instate['peak_color'][3] - instate['base_color'][3]) * local_intensity
                
                # Set individual pixel
                buffer[i] = [r, g, b, a]
        else:
            # For non-circular strips, use uniform intensity
            # Interpolate between base and peak colors based on intensity
            r = instate['base_color'][0] + (instate['peak_color'][0] - instate['base_color'][0]) * global_intensity
            g = instate['base_color'][1] + (instate['peak_color'][1] - instate['base_color'][1]) * global_intensity
            b = instate['base_color'][2] + (instate['peak_color'][2] - instate['base_color'][2]) * global_intensity
            a = instate['base_color'][3] + (instate['peak_color'][3] - instate['base_color'][3]) * global_intensity
            
            # Apply the color to all pixels in the strip
            buffer[:] = [r, g, b, a]