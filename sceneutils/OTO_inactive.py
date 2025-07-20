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
# ... existing code ...

def OTO_sunrise_joy(instate, outstate):
    """
    A joyful sunrise-themed pattern generator.
    
    Creates a rising sun effect in the middle of each strip in the "base" groups,
    transitioning from warm orange/yellow colors to light blue sky colors.
    The sun rises from the bottom to top strips over time, with cloud effects
    in the upper sky and rippling water reflections away from the sun.
    
    Optimized with vectorized operations where possible.
    """
    name = 'sunrise_joy'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['sun_position'] = 0.0  # Position of the sun (0.0 to 1.0)
        instate['cycle_duration'] = 100.0  # Time for a complete sunrise cycle (seconds)
        instate['base_strips'] = []  # Will store IDs of strips in "base" groups
        
        # Cloud effect parameters
        instate['cloud_positions'] = {}  # Store cloud positions for each strip
        instate['cloud_speeds'] = {}     # Store cloud speeds for each strip
        
        # Ocean wave parameters
        instate['wave_offsets'] = {}     # Random offsets for wave patterns
        instate['wave_speeds'] = {}      # Wave movement speeds
        
        # Identify strips in "base" groups using strip_manager
        for strip_id, strip in strip_manager.strips.items():
            if 'base' in strip.groups:
                instate['base_strips'].append(strip_id)
                
                # Initialize clouds for this strip
                strip_length = strip.length
                instate['cloud_positions'][strip_id] = []
                instate['cloud_speeds'][strip_id] = []
                
                # Create 3-5 clouds per strip
                num_clouds = np.random.randint(3, 6)
                for _ in range(num_clouds):
                    # Random position
                    pos = np.random.uniform(0, strip_length)
                    # Random width (5-15% of strip length)
                    width = np.random.uniform(0.05, 0.15) * strip_length
                    # Random density (0.2-0.6)
                    density = np.random.uniform(0.2, 0.6)
                    # Random speed (0.5-2 pixels per second)
                    speed = np.random.uniform(0.5, 2.0)
                    
                    instate['cloud_positions'][strip_id].append({
                        'pos': pos,
                        'width': width,
                        'density': density
                    })
                    instate['cloud_speeds'][strip_id].append(speed)
                
                # Initialize wave parameters for this strip
                instate['wave_offsets'][strip_id] = np.random.uniform(0, 2 * np.pi, 3)  # 3 wave components with random phases
                instate['wave_speeds'][strip_id] = np.random.uniform(0.2, 0.5, 3)  # Different speeds for each wave component
        
        # Sort base strips from bottom to top based on their groups
        def get_strip_order(strip_id):
            strip = strip_manager.get_strip(strip_id)
            if 'bottom' in strip.groups:
                return 0
            elif 'middle' in strip.groups:
                return 1
            elif 'top' in strip.groups:
                return 2
            return 3  # For any other base strips without specific vertical position
        
        instate['base_strips'].sort(key=get_strip_order)
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get alpha level from outstate or use default 1.0
    alpha = outstate.get('control_joyful', 0.0)/100
    
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
    
    # Time delta for animation
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update sun position based on time (smoothly looping)
    time_position = (outstate['current_time'] % instate['cycle_duration']) / instate['cycle_duration']
    instate['sun_position'] = time_position
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Number of base strips for position calculation
    num_base_strips = len(instate['base_strips'])
    
    # Update cloud positions (vectorized where possible)
    for strip_id in instate['cloud_positions']:
        if strip_id in instate['cloud_speeds']:
            for i, cloud in enumerate(instate['cloud_positions'][strip_id]):
                # Move cloud position
                speed = instate['cloud_speeds'][strip_id][i]
                cloud['pos'] += speed * delta_time
                
                # Wrap around at strip end
                if strip_id in strip_manager.strips:
                    strip_length = strip_manager.get_strip(strip_id).length
                    if cloud['pos'] > strip_length + cloud['width']:
                        cloud['pos'] = -cloud['width']
    
    # Update wave offsets (vectorized)
    for strip_id in instate['wave_offsets']:
        if strip_id in instate['wave_speeds']:
            # Vectorized update of all wave offsets for this strip
            instate['wave_offsets'][strip_id] += instate['wave_speeds'][strip_id] * delta_time
            # Keep within 0-2π range (vectorized)
            instate['wave_offsets'][strip_id] %= (2 * np.pi)
    
    # For smoother transitions, use a sine wave function for sun size
    sun_cycle = 0.5 + 0.5 * np.sin(2 * np.pi * instate['sun_position'] - np.pi/2)
    
    # For color transitions, use a curve that ensures smooth transitions
    sun_height = 0.5 + 0.5 * np.sin(2 * np.pi * instate['sun_position'] - np.pi/2)
    
    # Render pattern to each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Check if this is a base strip
        if strip_id in instate['base_strips']:
            strip_length = len(buffer)
            
            # Get the vertical position of this strip (0=bottom, 1=top)
            strip_index = instate['base_strips'].index(strip_id)
            strip_vertical_position = strip_index / max(1, num_base_strips - 1)
            
            # Calculate sun center position (middle of the strip)
            sun_center = strip_length // 2
            
            # Calculate sun radius based on the smooth cycle and strip position
            base_radius = strip_length * 0.3
            strip_height_factor = 1.0 - (strip_vertical_position * 0.5)  # Larger sun at bottom
            sun_radius = max(1, int(base_radius * sun_cycle * strip_height_factor))
            
            # Create position arrays for vectorized calculations
            positions = np.arange(strip_length)
            
            # Calculate distances from sun center (vectorized)
            distances = np.abs(positions - sun_center)
            
            # Determine which pixels are inside the sun (vectorized)
            inside_sun_mask = distances <= sun_radius
            
            # Calculate normalized distances (vectorized)
            normalized_distances = np.zeros(strip_length)
            # Inside sun
            if np.any(inside_sun_mask):
                normalized_distances[inside_sun_mask] = distances[inside_sun_mask] / sun_radius
            # Outside sun
            outside_sun_mask = ~inside_sun_mask
            if np.any(outside_sun_mask):
                normalized_distances[outside_sun_mask] = np.minimum(
                    1.0, (distances[outside_sun_mask] - sun_radius) / (strip_length * 0.5)
                )
            
            # Calculate base sky colors based on sun height
            if sun_height < 0.5:
                # Sunrise/sunset phase - orange to light blue
                transition = sun_height * 2.0
                height_influence = strip_vertical_position * 0.3
                
                # Sky colors (vectorized calculation)
                sky_r = 0.9 - (0.6 * (transition + height_influence))
                sky_g = 0.5 + (0.2 * (transition + height_influence))
                sky_b = 0.2 + (0.6 * (transition + height_influence))
            else:
                # Day phase - light blue to deeper blue
                transition = (sun_height - 0.5) * 2.0
                height_influence = strip_vertical_position * 0.3
                
                # Sky colors (vectorized calculation)
                sky_r = 0.3 - (0.1 * (transition + height_influence))
                sky_g = 0.7 - (0.1 * (transition + height_influence))
                sky_b = 0.8 + (0.1 * (transition + height_influence))
            
            # Initialize color arrays
            r_values = np.zeros(strip_length)
            g_values = np.zeros(strip_length)
            b_values = np.zeros(strip_length)
            a_values = np.zeros(strip_length)
            
            # Set colors for inside sun (vectorized)
            if np.any(inside_sun_mask):
                # Calculate intensity based on normalized distance
                intensities = 1.0 - normalized_distances[inside_sun_mask] ** 2
                
                # Set colors
                r_values[inside_sun_mask] = 1.0
                g_values[inside_sun_mask] = 0.7 + (0.3 * intensities)
                b_values[inside_sun_mask] = 0.2 + (0.3 * intensities)
                a_values[inside_sun_mask] = 0.8 + (0.2 * intensities)
            
            # Set colors for outside sun (vectorized)
            if np.any(outside_sun_mask):
                # Start with sky colors
                r_values[outside_sun_mask] = sky_r
                g_values[outside_sun_mask] = sky_g
                b_values[outside_sun_mask] = sky_b
                
                # Apply sun glow effect (vectorized)
                glow_mask = (distances > sun_radius) & (distances < sun_radius * 2)
                if np.any(glow_mask):
                    glow_factors = np.maximum(0, np.minimum(0.5, 1.0 - (distances[glow_mask] - sun_radius) / sun_radius))
                    
                    # Blend with sun colors
                    r_values[glow_mask] = r_values[glow_mask] * (1 - glow_factors) + 1.0 * glow_factors
                    g_values[glow_mask] = g_values[glow_mask] * (1 - glow_factors) + 0.7 * glow_factors
                    b_values[glow_mask] = b_values[glow_mask] * (1 - glow_factors) + 0.2 * glow_factors
                
                # Set alpha
                a_values[outside_sun_mask] = 0.7 - (0.2 * normalized_distances[outside_sun_mask])
                
                # Calculate edge distances for water ripple effect (vectorized)
                left_edge_distances = positions / (strip_length * 0.25)
                right_edge_distances = (strip_length - positions - 1) / (strip_length * 0.25)
                edge_distances = np.minimum(left_edge_distances, right_edge_distances)
                
                # Create mask for edge pixels (outer 25% on each side)
                edge_mask = (edge_distances < 1.0) & outside_sun_mask
                
                if np.any(edge_mask):
                    # Ripple intensity stronger at edges and on lower strips
                    ripple_intensities = (1.0 - edge_distances[edge_mask]) * (1.0 - strip_vertical_position * 0.7)
                    
                    # Get wave offsets for this strip
                    wave_offsets = instate['wave_offsets'].get(strip_id, np.zeros(3))
                    
                    # Calculate normalized positions for ripple effect
                    norm_positions = positions[edge_mask] / strip_length
                    
                    # Calculate ripple effect (vectorized)
                    ripples = np.zeros(np.sum(edge_mask))
                    ripples += 0.5 * np.sin(norm_positions * 20 + wave_offsets[0])
                    ripples += 0.3 * np.sin(norm_positions * 35 + wave_offsets[1])
                    ripples += 0.2 * np.sin(norm_positions * 50 + wave_offsets[2])
                    
                    # Normalize to -1 to 1 range
                    ripples /= (0.5 + 0.3 + 0.2)
                    
                    # Apply ripple effect to colors
                    ripple_amount = ripples * ripple_intensities * 0.2
                    
                    # Brighten colors on wave peaks, darken in troughs
                    r_values[edge_mask] += ripple_amount
                    g_values[edge_mask] += ripple_amount
                    b_values[edge_mask] += ripple_amount * 1.2  # Slightly stronger blue modulation
                    
                    # Add slight color shift toward blue-green in wave troughs
                    trough_mask = edge_mask.copy()
                    trough_mask[edge_mask] = ripples < 0
                    if np.any(trough_mask):
                        trough_ripple_intensities = ripple_intensities[ripples < 0]
                        trough_ripple_values = ripples[ripples < 0]
                        
                        g_values[trough_mask] += np.abs(trough_ripple_values) * trough_ripple_intensities * 0.05
                        b_values[trough_mask] += np.abs(trough_ripple_values) * trough_ripple_intensities * 0.1
            
            # Apply gentle variation for a more natural look in upper sky
            if strip_vertical_position > 0.5:
                # Create a mask for upper sky (outside sun)
                upper_sky_mask = outside_sun_mask
                if np.any(upper_sky_mask):
                    # Vectorized calculation of variations
                    variations = 0.05 * np.sin(positions[upper_sky_mask] * 0.2 + outstate['current_time'])
                    
                    # Apply variations
                    r_values[upper_sky_mask] += variations
                    g_values[upper_sky_mask] += variations
                    b_values[upper_sky_mask] += variations
            
            # Ensure color values are in valid range (vectorized)
            r_values = np.clip(r_values, 0.0, 1.0)
            g_values = np.clip(g_values, 0.0, 1.0)
            b_values = np.clip(b_values, 0.0, 1.0)
            a_values = np.clip(a_values, 0.0, 1.0)
            
            # Combine into RGBA array
            rgba_values = np.stack([r_values, g_values, b_values, a_values], axis=1)
            
            # Set buffer values (vectorized)
            buffer[:] = rgba_values
            
            # Apply cloud effects (more visible on upper strips)
            if strip_vertical_position > 0.3 and strip_id in instate['cloud_positions']:
                cloud_influence = strip_vertical_position * 0.7  # More clouds visible higher up
                
                for cloud in instate['cloud_positions'][strip_id]:
                    # Calculate cloud span
                    cloud_center = int(cloud['pos'])
                    cloud_width = int(cloud['width'])
                    cloud_density = cloud['density'] * cloud_influence
                    
                    # Calculate cloud range
                    cloud_start = max(0, cloud_center - cloud_width)
                    cloud_end = min(strip_length, cloud_center + cloud_width + 1)
                    
                    if cloud_end > cloud_start:
                        # Get positions within cloud range
                        cloud_positions = np.arange(cloud_start, cloud_end)
                        
                        # Calculate distance from center (vectorized)
                        dist_from_center = np.abs(cloud_positions - cloud_center) / cloud_width
                        
                        # Cloud intensity falls off from center (vectorized)
                        cloud_intensities = cloud_density * (1.0 - dist_from_center**2)
                        
                        # Apply cloud effect only where intensity is significant
                        significant_mask = cloud_intensities > 0.01
                        if np.any(significant_mask):
                            # Filter positions and intensities
                            affect_positions = cloud_positions[significant_mask]
                            affect_intensities = cloud_intensities[significant_mask]
                            
                            # Get current pixel colors
                            curr_colors = buffer[affect_positions].copy()
                            
                            # Cloud colors
                            cloud_colors = np.array([0.9, 0.9, 0.95, 1.0])
                            
                            # Calculate alpha adjustment
                            alpha_adjust = 1.0 - 0.2 * affect_intensities
                            
                            # Reshape intensities for broadcasting
                            affect_intensities = affect_intensities[:, np.newaxis]
                            alpha_adjust = alpha_adjust[:, np.newaxis]
                            
                            # Calculate new colors (vectorized blend)
                            new_colors = curr_colors * (1 - affect_intensities) + cloud_colors[np.newaxis, :] * affect_intensities
                            
                            # Apply alpha adjustment
                            new_colors[:, 3] = curr_colors[:, 3] * alpha_adjust[:, 0]
                            
                            # Update buffer
                            buffer[affect_positions] = new_colors
        else:
            # For non-base strips, apply ambient light that matches the sunrise colors
            strip_length = len(buffer)
            
            # Get strip information if available
            strip_groups = []
            if strip_id in strip_manager.strips:
                strip = strip_manager.get_strip(strip_id)
                strip_groups = strip.groups
            
            # Determine strip type for customized effects
            is_heart = 'heart' in strip_groups
            is_spine = 'spine' in strip_groups
            is_ear = 'ear' in strip_groups
            is_head = 'head' in strip_groups
            
            # Base colors for the current sun position
            if sun_height < 0.5:
                # Sunrise/sunset colors
                transition = sun_height * 2.0
                
                # Base ambient colors
                r_base = 0.6 - (0.2 * transition)
                g_base = 0.3 + (0.3 * transition)
                b_base = 0.1 + (0.5 * transition)
                
                # Customize for different strip types
                if is_heart:
                    # Hearts get a warm glow during sunrise
                    r_base += 0.3
                    g_base -= 0.1
                elif is_spine:
                    # Spines get slight golden tint
                    r_base += 0.1
                    g_base += 0.1
                elif is_ear:
                    # Ears get slight amber tint
                    r_base += 0.2
                    g_base += 0.1
                elif is_head:
                    # Head gets a bright glow
                    r_base += 0.2
                    g_base += 0.2
                    b_base += 0.1
            else:
                # Day phase colors
                transition = (sun_height - 0.5) * 2.0
                
                # Base ambient colors
                r_base = 0.4 - (0.2 * transition)
                g_base = 0.6 + (0.1 * transition)
                b_base = 0.6 + (0.3 * transition)
                
                # Customize for different strip types
                if is_heart:
                    # Hearts get a soft glow during day
                    r_base += 0.1
                    g_base += 0.1
                    b_base += 0.2
                elif is_spine:
                    # Spines get slight green tint
                    g_base += 0.2
                elif is_ear:
                    # Ears get slight purple tint
                    r_base += 0.1
                    b_base += 0.2
                elif is_head:
                    # Head gets a bright sky blue
                    g_base += 0.1
                    b_base += 0.3
            
            # Create position array for vectorized calculations
            positions = np.arange(strip_length, dtype=float)
            normalized_positions = positions / strip_length
            
            # Apply different patterns based on strip type (vectorized)
            if is_heart:
                # Hearts get a gentle ambient glow without rhythm
                # Vectorized spatial variation calculation
                variations = 0.1 * np.cos(normalized_positions * 0.2 * strip_length)
                
                # Apply the color with subtle variation (vectorized)
                r_values = r_base + variations
                g_values = g_base + variations
                b_values = b_base + variations
                a_values = 0.7 + variations * 0.3
                
            elif is_spine:
                # Spines get a flowing wave pattern
                # Vectorized wave calculation
                wave_pos = normalized_positions - (sun_height * 2 % 1.0)
                waves = 0.5 + 0.5 * np.cos(wave_pos * 6 * np.pi)
                
                # Apply the color with wave intensity (vectorized)
                r_values = r_base * (0.7 + 0.3 * waves)
                g_values = g_base * (0.7 + 0.3 * waves)
                b_values = b_base * (0.7 + 0.3 * waves)
                a_values = 0.6 + 0.4 * waves
                
            elif is_ear or is_head:
                # Ears and head get a circular wave pattern
                # Vectorized wave calculation
                waves = 0.5 + 0.5 * np.sin(normalized_positions * 6 * np.pi + outstate['current_time'] * 2)
                
                # Apply the color with wave intensity (vectorized)
                r_values = r_base * (0.8 + 0.2 * waves)
                g_values = g_base * (0.8 + 0.2 * waves)
                b_values = b_base * (0.8 + 0.2 * waves)
                a_values = 0.7 + 0.3 * waves
                
            else:
                # Default pattern for other strips - gentle variation
                # Vectorized variation calculation
                variations = 0.1 * np.sin(normalized_positions * 0.1 * strip_length + outstate['current_time'] * 0.2)
                
                # Calculate final colors (vectorized)
                r_values = r_base + variations
                g_values = g_base + variations
                b_values = b_base + variations
                a_values = 0.6 + variations
            
            # Ensure color values are in valid range (vectorized)
            r_values = np.clip(r_values, 0.0, 1.0)
            g_values = np.clip(g_values, 0.0, 1.0)
            b_values = np.clip(b_values, 0.0, 1.0)
            a_values = np.clip(a_values, 0.0, 1.0)
            
            # Combine into RGBA array
            rgba_values = np.stack([r_values, g_values, b_values, a_values], axis=1)
            
            # Set buffer values (vectorized)
            buffer[:] = rgba_values

# ... existing code ...