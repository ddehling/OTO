from sceneutils.imgutils import *  # noqa: F403
import numpy as np
from pathlib import Path


# ... existing code ...

def OTO_inactive_pattern_cycle(instate, outstate):
    """
    Generator that cycles between multiple pattern generators over time.
    
    Each pattern has its own buffer and alpha value determined by its proximity
    to the current active pattern position. Patterns more than 1 unit away from
    the current position have zero alpha and are skipped for efficiency.
    """
    name = 'pattern_cycle'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        # Register our main generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['num_patterns'] = 5  # Total number of patterns
        instate['cycle_duration'] = 60.0  # Time to complete full cycle (seconds)
        instate['current_pattern'] = 0.0  # Current pattern position (floating point)
        instate['pattern_names'] = []  # Will store names of sub-pattern generators
        
        # Register individual pattern buffers
        for i in range(instate['num_patterns']):
            pattern_name = f"{name}_pattern_{i}"
            instate['pattern_names'].append(pattern_name)
            buffers.register_generator(pattern_name)
        
        # Initialize pattern-specific state variables
        instate['twinkle_stars'] = {}  # For pattern 0 (twinkle)
        instate['movers'] = {}         # For pattern 1 (moving dots)
        instate['cycle_phases'] = {}   # For pattern 2 (brightness cycling)
        instate['rainbow_offset'] = 0.0  # For pattern 3 (rainbow waves)
        instate['pulse_positions'] = {}  # For pattern 4 (pulse)
        
        return

    if instate['count'] == -1:
        # Cleanup all pattern buffers
        buffers.generator_alphas[name] = 0
        for pattern_name in instate['pattern_names']:
            buffers.generator_alphas[pattern_name] = 0
        return

    # Set main generator alpha to full
    buffers.generator_alphas[name] = 1.0
    global_alpha=outstate.get('control_mode_inactive', 1)
    if global_alpha<0.01:   
        return
    
    # Update current pattern position based on time
    time_position = (outstate['current_time'] % instate['cycle_duration']) / instate['cycle_duration']
    instate['current_pattern'] = time_position * instate['num_patterns']
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        # Fade out main generator and all patterns
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha

    # Calculate alpha for each pattern based on distance from current_pattern
    for i, pattern_name in enumerate(instate['pattern_names']):
        # Calculate distance to current pattern (accounting for wraparound)
        direct_distance = abs(instate['current_pattern'] - i)
        wrap_distance = instate['num_patterns'] - direct_distance
        distance = min(direct_distance, wrap_distance)
        
        # If distance > 1, set alpha to 0 and skip rendering
        if distance > 1.0:
            buffers.generator_alphas[pattern_name] = 0.0
            continue
        
        # Calculate alpha based on distance (1.0 at distance=0, 0.0 at distance=1)
        pattern_alpha = 1.0 - distance
        buffers.generator_alphas[pattern_name] = pattern_alpha*global_alpha
        
        # Get buffers for this pattern
        pattern_buffers = buffers.get_all_buffers(pattern_name)
        
        # Delta time for animation calculations
        delta_time = outstate['current_time'] - outstate['last_time']
        
        # Render this pattern
        if i == 0:
            # Pattern 0: Slow Twinkle
            # Stars appear randomly, gradually brighten, then fade out
            
            # First, fade existing stars
            for strip_id, buffer in pattern_buffers.items():
                if strip_id not in instate['twinkle_stars']:
                    instate['twinkle_stars'][strip_id] = []
                
                # Update existing stars
                new_stars = []
                for star in instate['twinkle_stars'][strip_id]:
                    # Update star life
                    star['life'] += delta_time / star['duration']
                    
                    # Calculate brightness based on life cycle
                    if star['life'] < 0.5:
                        # Fade in
                        brightness = star['life'] * 2
                    else:
                        # Fade out
                        brightness = 2.0 - (star['life'] * 2)
                    
                    brightness = max(0.0, min(1.0, brightness))
                    
                    # Set pixel color with calculated brightness
                    if star['life'] < 1.0:
                        buffer[star['position']] = [
                            star['color'][0] * brightness,
                            star['color'][1] * brightness, 
                            star['color'][2] * brightness,
                            brightness
                        ]
                        new_stars.append(star)
                
                # Replace star list with active stars
                instate['twinkle_stars'][strip_id] = new_stars
                
                # Randomly add new stars (more rarely for a slow twinkle)
                if np.random.random() < 0.05:  # 5% chance per frame per strip
                    position = np.random.randint(0, len(buffer))
                    color = [
                        0.7 + 0.3 * np.random.random(),  # Slightly randomized white
                        0.7 + 0.3 * np.random.random(),
                        0.7 + 0.3 * np.random.random()
                    ]
                    instate['twinkle_stars'][strip_id].append({
                        'position': position,
                        'color': color,
                        'duration': 3.0 + np.random.random() * 5.0,  # 3-8 second duration
                        'life': 0.0  # 0.0 to 1.0 life cycle
                    })
        
        elif i == 1:
            # Pattern 1: Moving Dots with Fade Trails
            # Dots that move along strips and leave fading trails
            
            # Initialize movers if needed
            for strip_id, buffer in pattern_buffers.items():
                if strip_id not in instate['movers']:
                    instate['movers'][strip_id] = []
                    # Create 2 initial movers per strip
                    for _ in range(2):
                        instate['movers'][strip_id].append({
                            'position': np.random.random() * len(buffer),
                            'speed': 5.0 + np.random.random() * 10.0,  # 5-15 pixels/sec
                            'color': [
                                0.5 + 0.5 * np.random.random(),
                                0.5 + 0.5 * np.random.random(),
                                0.5 + 0.5 * np.random.random()
                            ]
                        })
                
                # Fade all pixels slightly
                fade_factor = 0.95  # 5% fade per frame
                buffer[:, 3] *= fade_factor
                
                # Update movers
                for mover in instate['movers'][strip_id]:
                    # Update position
                    mover['position'] += mover['speed'] * delta_time
                    
                    # Wrap around at strip end
                    if mover['position'] >= len(buffer):
                        mover['position'] = 0
                        # New random color when wrapping
                        mover['color'] = [
                            0.5 + 0.5 * np.random.random(),
                            0.5 + 0.5 * np.random.random(),
                            0.5 + 0.5 * np.random.random()
                        ]
                    
                    # Set pixel and create small trail
                    pos = int(mover['position'])
                    buffer[pos] = [mover['color'][0], mover['color'][1], mover['color'][2], 1.0]
                    
                    # Add a short trail
                    for i in range(1, 5):
                        trail_pos = (pos - i) % len(buffer)
                        trail_alpha = 1.0 - (i / 5.0)  # Fade based on distance
                        buffer[trail_pos] = [
                            mover['color'][0] * trail_alpha,
                            mover['color'][1] * trail_alpha,
                            mover['color'][2] * trail_alpha,
                            trail_alpha
                        ]
        
        elif i == 2:
            # Pattern 2: Global Slow Brightness Cycling
            # Each strip cycles brightness at slightly different rates
            
            # Initialize cycle phases if needed
            for strip_id, buffer in pattern_buffers.items():
                if strip_id not in instate['cycle_phases']:
                    # Random starting phase and period for each strip
                    instate['cycle_phases'][strip_id] = {
                        'phase': np.random.random(),
                        'period': 8.0 + np.random.random() * 4.0  # 8-12 second cycle
                    }
                
                # Update phase
                instate['cycle_phases'][strip_id]['phase'] += delta_time / instate['cycle_phases'][strip_id]['period']
                if instate['cycle_phases'][strip_id]['phase'] > 1.0:
                    instate['cycle_phases'][strip_id]['phase'] -= 1.0
                
                # Calculate brightness using sinusoidal pattern
                phase = instate['cycle_phases'][strip_id]['phase']
                brightness = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi))
                
                # Apply a uniform dim color to the entire strip
                # Using a warm white/amber
                buffer[:] = [brightness, brightness * 0.8, brightness * 0.5, brightness]
        
        elif i == 3:
            # Pattern 3: Rainbow Waves
            # Flowing rainbow pattern with gentle movement
            
            # Update global rainbow offset
            instate['rainbow_offset'] += delta_time * 0.2  # Slow movement
            if instate['rainbow_offset'] > 1.0:
                instate['rainbow_offset'] -= 1.0
            
            for strip_id, buffer in pattern_buffers.items():
                # Apply rainbow pattern across the strip
                strip_length = len(buffer)
                
                for j in range(strip_length):
                    # Calculate hue based on position and offset
                    # This creates a flowing rainbow effect
                    hue = (j / strip_length + instate['rainbow_offset']) % 1.0
                    
                    # Convert HSV to RGB (saturation and value fixed for vibrant colors)
                    r, g, b = hsv_to_rgb(hue, 0.8, 0.7)
                    
                    # Set pixel color
                    buffer[j] = [r, g, b, 0.7]  # Slightly transparent
        
        elif i == 4:
            # Pattern 4: Pulses
            # Waves of light that travel along strips
            
            # Initialize pulse positions if needed
            for strip_id, buffer in pattern_buffers.items():
                if strip_id not in instate['pulse_positions']:
                    instate['pulse_positions'][strip_id] = []
                    # Create an initial pulse
                    instate['pulse_positions'][strip_id].append({
                        'position': 0,
                        'speed': 30.0,  # Pixels per second
                        'width': 10,    # Width of pulse in pixels
                        'hue': np.random.random()  # Random color
                    })
                
                # Clear buffer
                buffer[:] = [0, 0, 0, 0]
                
                # Update existing pulses
                new_pulses = []
                for pulse in instate['pulse_positions'][strip_id]:
                    # Update position
                    pulse['position'] += pulse['speed'] * delta_time
                    
                    # If pulse has left the strip, don't keep it
                    if pulse['position'] - pulse['width'] > len(buffer):
                        continue
                    
                    # Draw the pulse
                    for j in range(pulse['width'] * 2):
                        pos = int(pulse['position']) - j
                        if 0 <= pos < len(buffer):
                            # Calculate intensity based on distance from pulse center
                            intensity = 1.0 - (j / (pulse['width'] * 2))
                            
                            # Convert pulse color from HSV to RGB
                            r, g, b = hsv_to_rgb(pulse['hue'], 1.0, 1.0)
                            
                            # Set pixel
                            buffer[pos] = [r * intensity, g * intensity, b * intensity, intensity]
                    
                    new_pulses.append(pulse)
                
                # Replace with active pulses
                instate['pulse_positions'][strip_id] = new_pulses
                
                # Randomly add new pulses
                if np.random.random() < 0.02:  # 2% chance per frame per strip
                    instate['pulse_positions'][strip_id].append({
                        'position': 0,
                        'speed': 30.0 + np.random.random() * 20.0,  # 30-50 pixels/sec
                        'width': 5 + int(np.random.random() * 10),  # 5-15 pixels wide
                        'hue': np.random.random()  # Random color
                    })



# ... existing code ...

def OTO_generic_pattern(instate, outstate):
    """
    Generic pattern generator template with controllable alpha level.
    
    This function provides a starting point for creating new patterns
    with proper initialization, cleanup, and alpha control.
    """
    name = 'generic_pattern'
    buffers = outstate['buffers']

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['param1'] = 5.0  # Example parameter
        instate['param2'] = 0.0  # Example state variable
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get alpha level from outstate or use default 1.0
    alpha = outstate.get('generic', 1.0)
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = alpha
    
    # Skip rendering if alpha is too low
    if alpha < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        # Fade out over the last 10 seconds
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * alpha
        
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update state variables
    instate['param2'] += delta_time * 0.5  # Example state update
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Render pattern to each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Example rendering code
        for i in range(len(buffer)):
            # Calculate pixel values based on position and time
            intensity = 0.5 + 0.5 * np.sin((i / 10.0) + instate['param2'])
            
            # Set pixel color (RGBA format)
            buffer[i] = [intensity, intensity * 0.8, intensity * 0.5, intensity]


# ... existing code ...

def OTO_awaken(instate, outstate):
    """
    Awakening pattern generator with movement between base strips and ear animations.
    
    Features:
    1. Base groups: Movement flows between bottom, middle, and top strips
    2. Ear strips: Spinning movement when control_sensor is active
    3. Alpha level controlled by outstate['control_mode_awaken']
    
    The pattern creates a flow of energy moving from bottom strips to middle strips
    to top strips, simulating an awakening or activation sequence.
    """
    name = 'awaken'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['flow_position'] = 0.0  # Position in flow cycle (0 to 1)
        instate['flow_speed'] = 0.3     # Speed of the flow (units per second)
        instate['ear_rotation'] = 0.0   # Rotation angle for ear spinning effect
        
        # Track which strips belong to each section of the base
        instate['base_bottom_strips'] = []
        instate['base_middle_strips'] = []
        instate['base_top_strips'] = []
        instate['ear_strips'] = []
        
        # Identify and categorize strips
        for strip_id in strip_manager.strips:
            strip = strip_manager.get_strip(strip_id)
            
            if 'base' in strip.groups:
                if 'bottom' in strip.groups:
                    instate['base_bottom_strips'].append(strip_id)
                elif 'middle' in strip.groups:
                    instate['base_middle_strips'].append(strip_id)
                elif 'top' in strip.groups:
                    instate['base_top_strips'].append(strip_id)
            
            if 'ear' in strip.groups:
                instate['ear_strips'].append(strip_id)
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get alpha level from outstate
    alpha = outstate.get('control_mode_awaken', 0.0)
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = alpha
    
    # Skip rendering if alpha is too low
    if alpha < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * alpha
        
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update flow position (loops from 0 to 1)
    instate['flow_position'] += instate['flow_speed'] * delta_time
    if instate['flow_position'] > 1.0:
        instate['flow_position'] -= 1.0
    
    # Check if ear spinning should be active
    ear_spinning_active = outstate.get('control_sensor', 0.0) > 0.01
    
    # Update ear rotation angle if spinning is active
    if ear_spinning_active:
        rotation_speed = 1.5  # rotations per second
        instate['ear_rotation'] += rotation_speed * 2 * np.pi * delta_time
        # Keep rotation angle within 0 to 2π
        if instate['ear_rotation'] > 2 * np.pi:
            instate['ear_rotation'] -= 2 * np.pi
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Calculate intensities for each section based on flow position
    # Flow moves from bottom (0.0) to middle (0.33) to top (0.67) and back to bottom
    
    # First determine which phase of the flow we're in
    if instate['flow_position'] < 0.33:  # Bottom to Middle phase
        # Bottom: Bright to dim
        bottom_intensity = 1.0 - (instate['flow_position'] / 0.33)
        # Middle: Dim to bright
        middle_intensity = instate['flow_position'] / 0.33
        # Top: Dim
        top_intensity = 0.1
    elif instate['flow_position'] < 0.67:  # Middle to Top phase
        # Bottom: Dim
        bottom_intensity = 0.1
        # Middle: Bright to dim
        middle_intensity = 1.0 - ((instate['flow_position'] - 0.33) / 0.33)
        # Top: Dim to bright
        top_intensity = (instate['flow_position'] - 0.33) / 0.33
    else:  # Top to Bottom phase
        # Bottom: Dim to bright
        bottom_intensity = (instate['flow_position'] - 0.67) / 0.33
        # Middle: Dim
        middle_intensity = 0.1
        # Top: Bright to dim
        top_intensity = 1.0 - ((instate['flow_position'] - 0.67) / 0.33)
    
    # Apply minimum intensity to ensure all sections have some illumination
    bottom_intensity = max(0.1, bottom_intensity)
    middle_intensity = max(0.1, middle_intensity)
    top_intensity = max(0.1, top_intensity)
    
    # Render pattern to each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Base strip processing - determine which section it belongs to
        if strip_id in instate['base_bottom_strips']:
            # Bottom strips - pulsing cyan-blue with intensity based on flow
            intensity = bottom_intensity * 0.8  # Scale for visual balance
            
            # Create a cyan-blue color with intensity modulation
            r_val = 0.0
            g_val = intensity * 0.6
            b_val = intensity
            a_val = intensity
            
            # Set uniform color for the strip
            buffer[:] = [r_val, g_val, b_val, a_val]
            
            # Add random sparkles for more visual interest when at peak intensity
            if bottom_intensity > 0.7:
                sparkle_count = int(strip_length * 0.1)  # 10% of pixels get sparkles
                sparkle_indices = np.random.choice(strip_length, sparkle_count, replace=False)
                
                for idx in sparkle_indices:
                    sparkle_intensity = 0.8 + 0.2 * np.random.random()
                    buffer[idx] = [
                        0.1 * sparkle_intensity,  # Touch of white
                        0.7 * sparkle_intensity, 
                        sparkle_intensity, 
                        sparkle_intensity
                    ]
        
        elif strip_id in instate['base_middle_strips']:
            # Middle strips - brighter cyan with intensity based on flow
            intensity = middle_intensity * 0.9  # Scale for visual balance
            
            # Create a bright cyan color with intensity modulation
            r_val = 0.0
            g_val = intensity * 0.8
            b_val = intensity
            a_val = intensity
            
            # Set uniform color for the strip
            buffer[:] = [r_val, g_val, b_val, a_val]
            
            # Add more pronounced sparkles for peak intensity
            if middle_intensity > 0.7:
                sparkle_count = int(strip_length * 0.15)  # 15% of pixels get sparkles
                sparkle_indices = np.random.choice(strip_length, sparkle_count, replace=False)
                
                for idx in sparkle_indices:
                    sparkle_intensity = 0.9 + 0.1 * np.random.random()
                    buffer[idx] = [
                        0.2 * sparkle_intensity,  # More white
                        0.8 * sparkle_intensity, 
                        sparkle_intensity, 
                        sparkle_intensity
                    ]
        
        elif strip_id in instate['base_top_strips']:
            # Top strips - brightest blue-white with intensity based on flow
            intensity = top_intensity * 1.0  # Full intensity
            
            # Create a blue-white color with intensity modulation
            r_val = intensity * 0.3  # More white content
            g_val = intensity * 0.8
            b_val = intensity
            a_val = intensity
            
            # Set uniform color for the strip
            buffer[:] = [r_val, g_val, b_val, a_val]
            
            # Add bright white sparkles for peak intensity
            if top_intensity > 0.7:
                sparkle_count = int(strip_length * 0.2)  # 20% of pixels get sparkles
                sparkle_indices = np.random.choice(strip_length, sparkle_count, replace=False)
                
                for idx in sparkle_indices:
                    sparkle_intensity = 1.0  # Full brightness
                    buffer[idx] = [
                        0.7 * sparkle_intensity,  # Almost white
                        0.9 * sparkle_intensity, 
                        sparkle_intensity, 
                        sparkle_intensity
                    ]
        
        elif strip_id in instate['ear_strips'] and ear_spinning_active:
            # Ear groups with spinning effect when sensor is active
            for i in range(strip_length):
                # Calculate angle position in the circular pattern
                angle_pos = (i / strip_length) * 2 * np.pi
                
                # Combine with current rotation angle
                combined_angle = angle_pos + instate['ear_rotation']
                
                # Create an intensity pattern with multiple peaks
                num_peaks = 3  # Number of bright spots in the rotation
                intensity = 0.5 + 0.5 * np.cos(combined_angle * num_peaks)
                
                # Add color variation based on angle
                hue = (combined_angle / (2 * np.pi)) % 1.0
                r, g, b = hsv_to_rgb(hue, 0.7, intensity)
                
                buffer[i] = [r, g, b, intensity * 0.8]
        
        elif strip_id in instate['ear_strips'] and not ear_spinning_active:
            # Ear groups with subtle pulsing when sensor is not active
            pulse_rate = 0.5  # Pulses per second
            pulse_phase = (outstate['current_time'] * pulse_rate) % 1.0
            
            # Simple pulsing effect
            intensity = 0.2 + 0.3 * np.sin(pulse_phase * 2 * np.pi)
            
            # Gentle blue pulse
            buffer[:] = [0.0, intensity * 0.5, intensity, intensity]
        
        else:
            # Other strips - subtle blue pulsing
            pulse_rate = 0.3  # Slower pulse for background strips
            pulse_phase = (outstate['current_time'] * pulse_rate) % 1.0
            
            # Simple pulsing effect
            intensity = 0.1 + 0.1 * np.sin(pulse_phase * 2 * np.pi)
            
            # Very subtle blue glow
            buffer[:] = [0.0, intensity * 0.3, intensity, intensity * 0.5]

# ... existing code ...