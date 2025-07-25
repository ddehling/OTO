from sceneutils.imgutils import *  # noqa: F403
import numpy as np
from pathlib import Path


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

def OTO_sad_theme(instate, outstate):
    """
    Generator function that creates a sad-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_sad'] value
    2. Dark blues and grey-blues are the predominant colors 
    3. Blue raindrops on the spines with long trails that roll down and turn grey
       then wrap around to the top when they reach the bottom
    4. Similar raindrops on base strips moving in both directions with consistent respawning
    5. Asynchronous grey and blue heartbeat on heart strips with brighter, longer pulses
    6. Grey and white spots that blow away like dandelion seeds on brain and ear strips
    
    Optimized with vectorized operations for better performance.
    """
    name = 'sad_theme'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['raindrops'] = {}  # For tracking raindrops on spine strips
        instate['heart_phase_left'] = 0.0  # For left heartbeat timing
        instate['heart_phase_right'] = 0.5  # For right heartbeat timing (start offset)
        instate['dandelion_seeds'] = {}  # For tracking floating seeds
        instate['last_drop_time'] = {}  # Time tracker for raindrop creation (per strip)
        
        # Color palette (HSV values)
        instate['colors'] = {
            'deep_blue': [0.6, 0.8, 0.4],  # Deep blue
            'grey_blue': [0.6, 0.3, 0.5],  # Grey-blue
            'light_grey': [0.0, 0.0, 0.7],  # Light grey
            'dark_grey': [0.0, 0.0, 0.3],   # Dark grey
            'white': [0.0, 0.0, 0.9]        # Almost white
        }
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get sad level from outstate (default to 0)
    sad_level = outstate.get('control_sad', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = sad_level
    
    # Skip rendering if alpha is too low
    if sad_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * sad_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update heartbeat phases - slow heartbeat around 40 BPM with slight variation between sides
    beats_per_second_left = 38 / 60.0  # Slightly slower for left
    beats_per_second_right = 42 / 60.0  # Slightly faster for right
    
    instate['heart_phase_left'] = (instate['heart_phase_left'] + beats_per_second_left * delta_time) % 1.0
    instate['heart_phase_right'] = (instate['heart_phase_right'] + beats_per_second_right * delta_time) % 1.0
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Start with a dim base color - slight grey-blue tint
        base_hue, base_sat, base_val = instate['colors']['grey_blue']
        base_r, base_g, base_b = hsv_to_rgb(base_hue, base_sat * 0.5, base_val * 0.3)
        buffer[:] = [base_r, base_g, base_b, 0.2]  # Very dim base
    

        # Different effects based on strip type
        if 'spine' in strip.groups or 'base' in strip.groups:
            # Spine and base strips - raindrops effect
            
            # Initialize raindrops for this strip if not already done
            if strip_id not in instate['raindrops']:
                instate['raindrops'][strip_id] = []
                instate['last_drop_time'][strip_id] = 0.0
                
                # Pre-populate with some initial raindrops
                num_initial_drops = strip_length // 5  # About 1 drop every 5 pixels
                for _ in range(num_initial_drops):
                    # Random position along strip
                    pos = np.random.randint(0, strip_length)
                    
                    # For base strips, determine direction based on position
                    direction = 1  # Default downward for spine
                    if 'base' in strip.groups:
                        # For base: if in left half, move right; if in right half, move left
                        middle = strip_length // 2
                        direction = 1 if pos < middle else -1
                    
                    # Initial color phase based on position
                    color_phase = pos / strip_length if 'spine' in strip.groups else np.random.random()
                    
                    instate['raindrops'][strip_id].append({
                        'position': pos,
                        'speed': 8 + np.random.random() * 12,  # 8-20 pixels per second
                        'size': 8 + np.random.randint(0, 8),  # 8-15 pixels for longer trails
                        'color_phase': color_phase,  # 0.0 = blue, 1.0 = grey
                        'alpha': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 alpha
                        'direction': direction,  # 1 for down, -1 for up (base strips)
                        'life_cycles': 0  # Track how many times the drop has cycled
                    })
            
            # Get number of active drops for this strip
            active_drops = len(instate['raindrops'][strip_id])
            
            # Determine target number of drops based on strip type and length
            if 'spine' in strip.groups:
                target_drops = max(5, strip_length // 8)  # More drops on spine
            else:  # base strips
                target_drops = max(6, strip_length // 10)  # Slightly fewer on base, but ensure at least 6
            
            # Get current time
            current_time = outstate['current_time']
            
            # Check last drop time for this specific strip
            if strip_id not in instate['last_drop_time']:
                instate['last_drop_time'][strip_id] = current_time - 1.0  # Immediate creation
            
            # Calculate time since last drop for this strip
            time_since_last_drop = current_time - instate['last_drop_time'][strip_id]
            
            # Determine if we should create a new drop
            # More aggressive spawning for base strips
            if 'base' in strip.groups:
                # For base strips: create drops more aggressively if below target
                min_time_between_drops = 0.1 if active_drops < target_drops else 0.5
                should_create_drop = time_since_last_drop > min_time_between_drops
                # Force creation if severely below target
                if active_drops < target_drops * 0.5:
                    should_create_drop = time_since_last_drop > 0.05
            else:
                # For spine strips: standard creation logic
                should_create_drop = (active_drops < target_drops and time_since_last_drop > 0.1) or time_since_last_drop > 0.5
            
            # Create new drops as needed
            if should_create_drop:
                instate['last_drop_time'][strip_id] = current_time
                
                # Higher creation chance if below target
                creation_chance = 0.9 if active_drops < target_drops * 0.8 else 0.4
                
                # For base strips, even higher chance if very low on drops
                if 'base' in strip.groups and active_drops < target_drops * 0.5:
                    creation_chance = 0.95
                
                if np.random.random() < creation_chance:
                    # For spine: always start at top
                    if 'spine' in strip.groups:
                        pos = 0
                        direction = 1  # Down
                        color_phase = 0.0  # Start blue
                    # For base: start at either end with equal probability
                    elif 'base' in strip.groups:
                        start_left = np.random.random() < 0.5  # 50% chance to start from left
                        pos = 0 if start_left else strip_length - 1
                        direction = 1 if start_left else -1  # Direction based on starting position
                        color_phase = 0.0  # Start blue
                    
                    instate['raindrops'][strip_id].append({
                        'position': pos,
                        'speed': 8 + np.random.random() * 12,  # 8-20 pixels per second
                        'size': 18 + np.random.randint(0, 8),  # 8-15 pixels for longer trails
                        'color_phase': color_phase,
                        'alpha': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 alpha
                        'direction': direction,
                        'life_cycles': 0  # New drop starts with 0 cycles
                    })
            
            # Update existing raindrops
            new_raindrops = []
            
            # Process drops in batches for vectorization
            for drop in instate['raindrops'][strip_id]:
                # Update position based on direction
                drop['position'] += drop['speed'] * drop['direction'] * delta_time
                
                # Handle wrapping when drops reach the ends of the strip
                if 'spine' in strip.groups:
                    # For spine strips:
                    if drop['position'] >= strip_length:
                        # When drop reaches bottom, wrap to top with new properties
                        drop['position'] = 0
                        drop['color_phase'] = 0.0  # Reset to blue
                        drop['speed'] = 8 + np.random.random() * 12  # New random speed
                        drop['alpha'] = 0.7 + np.random.random() * 0.3  # New random alpha
                        drop['life_cycles'] += 1  # Increment cycle count
                elif 'base' in strip.groups:
                    # For base strips: handle boundary conditions
                    if drop['position'] >= strip_length:
                        # Reached right end
                        drop['position'] = 0  # Wrap to left end
                        drop['direction'] = 1  # Set direction to right
                        drop['color_phase'] = 0.0  # Reset to blue
                        drop['alpha'] = 0.7 + np.random.random() * 0.3  # New random alpha
                        drop['life_cycles'] = 0  # Reset cycle count for consistent respawning
                    elif drop['position'] < 0:
                        # Reached left end
                        drop['position'] = strip_length - 1  # Wrap to right end
                        drop['direction'] = -1  # Set direction to left
                        drop['color_phase'] = 0.0  # Reset to blue
                        drop['alpha'] = 0.7 + np.random.random() * 0.3  # New random alpha
                        drop['life_cycles'] = 0  # Reset cycle count for consistent respawning
                    
                    # Check if drop has reached the middle
                    middle = strip_length // 2
                    middle_range = strip_length * 0.1  # 10% of strip length around middle
                    
                    # If moving right (direction=1) and passed middle
                    if (drop['direction'] > 0 and 
                        middle - middle_range/2 < drop['position'] < middle + middle_range/2):
                        # Small chance to reverse direction at middle
                        if np.random.random() < 0.1:  # 10% chance per frame when in middle zone
                            drop['direction'] = -1
                            drop['alpha'] *= 0.9  # Slightly reduce alpha
                    
                    # If moving left (direction=-1) and passed middle
                    elif (drop['direction'] < 0 and 
                          middle - middle_range/2 < drop['position'] < middle + middle_range/2):
                        # Small chance to reverse direction at middle
                        if np.random.random() < 0.1:  # 10% chance per frame when in middle zone
                            drop['direction'] = 1
                            drop['alpha'] *= 0.9  # Slightly reduce alpha
                
                # Update color phase (transition from blue to grey as it moves)
                if 'spine' in strip.groups:
                    # For spine: change color based on distance from top
                    drop['color_phase'] = min(1.0, drop['position'] / strip_length)
                elif 'base' in strip.groups:
                    # For base: change color based on distance from middle
                    middle = strip_length // 2
                    distance_from_middle = abs(drop['position'] - middle) / (strip_length / 2)
                    drop['color_phase'] = distance_from_middle
                
                # Gradually reduce alpha over multiple life cycles (slower for spine, faster for base)
                if 'spine' in strip.groups:
                    max_cycles = 3  # Spine drops can cycle more times
                    cycle_alpha_reduction = 0.1  # Smaller reduction per cycle
                else:  # base strips
                    max_cycles = 3  # Base drops can now cycle more times too
                    cycle_alpha_reduction = 0.05  # Much smaller reduction for base (more persistent)
                
                # Apply alpha reduction based on cycle count
                if drop['life_cycles'] > 0:
                    drop['alpha'] *= (1.0 - (drop['life_cycles'] * cycle_alpha_reduction))
                
                # For base strips: ensure drops don't fade too much
                if 'base' in strip.groups:
                    drop['alpha'] = max(0.3, drop['alpha'])  # Maintain minimum visibility
                
                # Add drop to new list if it's still active and hasn't cycled too many times
                if drop['alpha'] > 0.1 and drop['life_cycles'] < max_cycles:
                    new_raindrops.append(drop)
                    
                    # Draw the raindrop and trail
                    pos_int = int(drop['position'])
                    
                    # Interpolate between deep blue and grey
                    blue_h, blue_s, blue_v = instate['colors']['deep_blue']
                    grey_h, grey_s, grey_v = instate['colors']['dark_grey']
                    
                    h = blue_h + (grey_h - blue_h) * drop['color_phase']
                    s = blue_s + (grey_s - blue_s) * drop['color_phase']
                    v = blue_v + (grey_v - blue_v) * drop['color_phase']
                    
                    r, g, b = hsv_to_rgb(h, s, v)
                    
                    # Draw main drop
                    if 0 <= pos_int < strip_length:
                        buffer[pos_int] = [r, g, b, drop['alpha']]
                    
                    # Prepare arrays for vectorized trail drawing
                    trail_positions = []
                    trail_colors = []
                    
                    # Calculate trail positions and colors
                    for i in range(1, drop['size']):
                        # Trail position depends on direction
                        trail_pos = pos_int - (i * drop['direction'])
                        
                        # Handle trail wrapping
                        if trail_pos >= strip_length:
                            trail_pos = trail_pos - strip_length
                        elif trail_pos < 0:
                            trail_pos = strip_length + trail_pos
                        
                        if 0 <= trail_pos < strip_length:
                            # Calculate trail alpha with gradual fade
                            trail_alpha = drop['alpha'] * (1 - (i / drop['size'])**1.5)
                            
                            # Calculate trail color (more grey as it trails)
                            trail_phase = min(1.0, drop['color_phase'] + (i / drop['size']) * 0.3)
                            trail_h = blue_h + (grey_h - blue_h) * trail_phase
                            trail_s = blue_s + (grey_s - blue_s) * trail_phase
                            trail_v = blue_v + (grey_v - blue_v) * trail_phase
                            
                            tr, tg, tb = hsv_to_rgb(trail_h, trail_s, trail_v)
                            
                            # Add to trail arrays
                            trail_positions.append(trail_pos)
                            trail_colors.append([tr, tg, tb, trail_alpha])
                    
                    # Apply all trail pixels at once if any exist
                    if trail_positions:
                        # Convert to numpy arrays for vectorized assignment
                        trail_positions = np.array(trail_positions, dtype=int)
                        trail_colors = np.array(trail_colors)
                        
                        # Set all trail pixels at once
                        for idx, pos in enumerate(trail_positions):
                            buffer[pos] = trail_colors[idx]
            
            # Replace with updated list
            instate['raindrops'][strip_id] = new_raindrops
            
        elif 'heart' in strip.groups:
            # Heart strips - asynchronous heartbeat with grey-blue color
            
            # Determine which phase to use based on left/right
            is_left = 'left' in strip.groups
            current_phase = instate['heart_phase_left'] if is_left else instate['heart_phase_right']
            
            # Create heartbeat waveform with longer linger
            if current_phase < 0.1:
                # Sharp rise (systole)
                intensity = current_phase / 0.1
            elif current_phase < 0.3:
                # Peak plateau (longer linger at peak)
                intensity = 1.0
            elif current_phase < 0.6:
                # First decline (early diastole) - more gradual
                intensity = 1.0 - 0.7 * ((current_phase - 0.3) / 0.3)
            else:
                # Second, more gradual decline (late diastole)
                intensity = 0.3 - 0.3 * ((current_phase - 0.6) / 0.4)
            
            # Stronger heartbeat but still sad
            intensity *= 0.9
            
            # Use different color based on left/right heart
            if is_left:
                # Left heart - more grey
                h, s, v = instate['colors']['dark_grey']
                # Make peak brighter
                v_peak = min(1.0, v * 2.5)
            else:
                # Right heart - more blue
                h, s, v = instate['colors']['grey_blue']
                # Make peak brighter
                v_peak = min(1.0, v * 2.0)
            
            # Adjust value based on heartbeat intensity - more dramatic range
            v_adjusted = v * 0.3 + v_peak * 0.7 * intensity
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v_adjusted)
            
            # Set uniform color for the heart strip with brighter alpha at peak - vectorized
            alpha = 0.3 + 0.7 * intensity
            buffer[:] = [r, g, b, alpha]
            
        elif 'brain' in strip.groups or 'ear' in strip.groups:
            # Brain and ear strips - dandelion seeds effect
            
            # Initialize seed particles for this strip if not already done
            if strip_id not in instate['dandelion_seeds']:
                instate['dandelion_seeds'][strip_id] = []
                
                # Create initial seeds (more for longer strips)
                num_seeds = max(3, strip_length // 10)
                for _ in range(num_seeds):
                    # Random starting position
                    pos = np.random.randint(0, strip_length)
                    
                    # Random seed properties
                    instate['dandelion_seeds'][strip_id].append({
                        'position': pos,
                        'speed': 3 + np.random.random() * 7,  # 3-10 pixels per second
                        'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                        'color': 'white' if np.random.random() < 0.3 else 'light_grey',  # Mostly grey with some white
                        'alpha': 0.5 + np.random.random() * 0.5,  # 0.5-1.0 alpha
                        'fade_rate': 0.05 + np.random.random() * 0.15  # How quickly it fades
                    })
            
            # Chance to add new seeds
            if np.random.random() < 0.05:  # 5% chance per frame
                pos = np.random.randint(0, strip_length)
                instate['dandelion_seeds'][strip_id].append({
                    'position': pos,
                    'speed': 3 + np.random.random() * 7,
                    'direction': 1 if np.random.random() < 0.5 else -1,
                    'color': 'white' if np.random.random() < 0.3 else 'light_grey',
                    'alpha': 0.5 + np.random.random() * 0.5,
                    'fade_rate': 0.05 + np.random.random() * 0.15
                })
            
            # Update existing seeds
            new_seeds = []
            for seed in instate['dandelion_seeds'][strip_id]:
                # Update position
                seed['position'] += seed['speed'] * seed['direction'] * delta_time
                
                # Handle wrapping for positions
                if seed['position'] >= strip_length:
                    seed['position'] = 0
                elif seed['position'] < 0:
                    seed['position'] = strip_length - 1
                
                # Gradually fade
                seed['alpha'] -= seed['fade_rate'] * delta_time
                
                # Keep if still visible
                if seed['alpha'] > 0.05:
                    new_seeds.append(seed)
                    
                    # Draw the seed
                    pos_int = int(seed['position'])
                    
                    # Get color
                    h, s, v = instate['colors'][seed['color']]
                    r, g, b = hsv_to_rgb(h, s, v)
                    
                    # Set pixel
                    buffer[pos_int] = [r, g, b, seed['alpha']]
                    
                    # Prepare arrays for vectorized glow drawing
                    glow_positions = []
                    glow_colors = []
                    
                    # Add small glow
                    for i in range(1, 3):
                        glow_pos1 = (pos_int + i) % strip_length
                        glow_pos2 = (pos_int - i) % strip_length
                        glow_alpha = seed['alpha'] * (1 - (i / 3))
                        
                        glow_positions.extend([glow_pos1, glow_pos2])
                        glow_colors.extend([[r, g, b, glow_alpha], [r, g, b, glow_alpha]])
                    
                    # Apply all glow pixels at once
                    if glow_positions:
                        for idx, pos in enumerate(glow_positions):
                            buffer[pos] = glow_colors[idx]
            
            # Replace with updated list
            instate['dandelion_seeds'][strip_id] = new_seeds
            
        else:
            # Other strips - subtle blue-grey pulsing - fully vectorized
            
            # Calculate pulsing effect
            pulse = 0.3 + 0.2 * np.sin(outstate['current_time'] * 0.5)
            
            # Get color - grey-blue
            h, s, v = instate['colors']['grey_blue']
            
            # Adjust value based on pulse
            v = v * pulse
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Set uniform color for the strip (vectorized)
            buffer[:] = [r, g, b, 0.3]

def OTO_angry_theme(instate, outstate):
    """
    Generator function that creates an angry-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_angry'] value
    2. Red, orange, and brown color palette to evoke fire and rage
    3. Explosive bursts on brain and ear strips that appear and dissipate rapidly
    4. Flame effects on spine strips with upward moving flames and ember particles
    5. Pulsing, intense heartbeat on heart strips with rapid, strong beats
    6. Base strips with rolling flames and occasional explosions
    
    Optimized with vectorized operations for performance.
    """
    name = 'angry_theme'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['explosions'] = {}  # Track explosion effects
        instate['flames'] = {}      # Track flame particles
        instate['embers'] = {}      # Track ember particles
        instate['heart_phase'] = 0.0  # For heartbeat timing
        instate['explosion_timer'] = {}  # Time tracker for explosion creation (per strip)
        
        # Color palette (HSV values)
        instate['colors'] = {
            'bright_red': [0.98, 0.95, 0.95],    # Intense red
            'orange': [0.05, 0.95, 0.9],         # Bright orange
            'yellow': [0.12, 0.85, 0.9],         # Fire yellow
            'deep_red': [0.98, 0.9, 0.7],        # Deep red
            'brown': [0.08, 0.8, 0.5],           # Reddish brown
            'ember': [0.05, 0.7, 0.8],           # Glowing ember orange
            'ash': [0.05, 0.2, 0.3]              # Dark ash color
        }
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get angry level from outstate (default to 0)
    angry_level = outstate.get('control_angry', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = angry_level
    
    # Skip rendering if alpha is too low
    if angry_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * angry_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Update heartbeat phase - fast heartbeat around 100-120 BPM for anger
    beats_per_second = 2.0  # 120 BPM
    instate['heart_phase'] = (instate['heart_phase'] + beats_per_second * delta_time) % 1.0
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Start with a base color - dim red glow
        base_hue, base_sat, base_val = instate['colors']['deep_red']
        base_r, base_g, base_b = hsv_to_rgb(base_hue, base_sat * 0.3, base_val * 0.2)
        buffer[:] = [base_r, base_g, base_b, 0.2]  # Dim base

        # Different effects based on strip type
        if 'spine' in strip.groups:
            # Spine strips - upward moving flames and embers
            
            # Initialize flame particles for this strip if not already done
            if strip_id not in instate['flames']:
                instate['flames'][strip_id] = []
                instate['embers'][strip_id] = []
                
                # Pre-populate with some initial flames
                num_initial_flames = strip_length // 10  # About 1 flame every 10 pixels
                for _ in range(num_initial_flames):
                    # Random position along strip
                    pos = np.random.randint(0, strip_length)
                    
                    instate['flames'][strip_id].append({
                        'position': pos,
                        'speed': 35 + np.random.random() * 20,  # 25-45 pixels per second (fast upward)
                        'size': 65 + np.random.randint(0, 10),  # 15-25 pixels for taller flames
                        'intensity': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 intensity
                        'life': 0.0,  # 0.0-1.0 life cycle
                        'duration': 3.5 + np.random.random() * 0.5  # 0.5-1.0 second lifetime
                    })
            
            # Get current time
            current_time = outstate['current_time']
            
            # Chance to create new flames
            if np.random.random() < 0.2:  # 20% chance per frame
                # Create at bottom of strip
                pos = strip_length - np.random.randint(1, 10)  # Start near bottom with slight randomness
                
                instate['flames'][strip_id].append({
                    'position': pos,
                    'speed': 35 + np.random.random() * 20,  # 25-45 pixels per second
                    'size': 15 + np.random.randint(0, 10),  # 15-25 pixels
                    'intensity': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 intensity
                    'life': 0.0,
                    'duration': 3.5 + np.random.random() * 0.5  # 0.5-1.0 second lifetime
                })
            
            # Chance to create embers
            if np.random.random() < 0.2:  # 10% chance per frame
                pos = strip_length - np.random.randint(1, 20)  # Start near bottom
                
                # Create an ember particle
                instate['embers'][strip_id].append({
                    'position': pos,
                    'speed': 40 + np.random.random() * 30,  # 40-70 pixels per second (faster than flames)
                    'wobble_amplitude': 1 + np.random.random() * 2,  # 1-3 pixels side-to-side
                    'wobble_frequency': 5 + np.random.random() * 5,  # 5-10 Hz
                    'wobble_offset': np.random.random() * 2 * np.pi,  # Random phase
                    'intensity': 0.6 + np.random.random() * 0.4,  # 0.6-1.0 intensity
                    'life': 0.0,
                    'duration': 0.7 + np.random.random() * 0.7  # 0.7-1.4 second lifetime
                })
            
            # Update existing flames - use a temporary buffer for vectorized operations
            temp_buffer = np.zeros((strip_length, 4))
            new_flames = []
            
            for flame in instate['flames'][strip_id]:
                # Update life
                flame['life'] += delta_time / flame['duration']
                
                # Update position - flames move upward (toward index 0)
                flame['position'] -= flame['speed'] * delta_time
                
                # Keep if still in strip and not at end of life
                if flame['position'] > -flame['size'] and flame['life'] < 1.0:
                    new_flames.append(flame)
                    
                    # Calculate flame intensity based on life cycle
                    if flame['life'] < 0.2:
                        # Flame growing
                        flame_alpha = flame['intensity'] * (flame['life'] / 0.2)
                    elif flame['life'] > 0.8:
                        # Flame fading
                        flame_alpha = flame['intensity'] * (1.0 - ((flame['life'] - 0.8) / 0.2))
                    else:
                        # Flame at full intensity
                        flame_alpha = flame['intensity']
                    
                    # Draw the flame
                    flame_center = int(flame['position'])
                    
                    # Calculate valid positions in the flame
                    start = max(0, flame_center)
                    end = min(strip_length, flame_center + flame['size'] + 1)
                    
                    if start < end:  # Only proceed if there are valid positions
                        # Calculate indices in the flame (0 at center, increasing downward)
                        flame_indices = np.arange(start - flame_center, end - flame_center)
                        
                        # Calculate relative positions in flame (0 at tip, 1 at base)
                        rel_positions = flame_indices / flame['size']
                        
                        # Determine color sections vectorized
                        tip_mask = rel_positions < 0.2
                        middle_mask = (rel_positions >= 0.2) & (rel_positions < 0.5)
                        base_mask = rel_positions >= 0.5
                        
                        # Initialize RGB arrays
                        r_values = np.zeros(len(flame_indices))
                        g_values = np.zeros(len(flame_indices))
                        b_values = np.zeros(len(flame_indices))
                        
                        # Apply colors based on masks
                        if np.any(tip_mask):
                            # Tip of flame - yellow/white
                            h, s, v = instate['colors']['yellow']
                            # Make tip brighter
                            v = min(1.0, v * 1.2)
                            s *= 0.8  # Less saturated (more white)
                            r, g, b = hsv_to_rgb(h, s, v)
                            r_values[tip_mask] = r
                            g_values[tip_mask] = g
                            b_values[tip_mask] = b
                        
                        if np.any(middle_mask):
                            # Middle of flame - orange
                            h, s, v = instate['colors']['orange']
                            r, g, b = hsv_to_rgb(h, s, v)
                            r_values[middle_mask] = r
                            g_values[middle_mask] = g
                            b_values[middle_mask] = b
                        
                        if np.any(base_mask):
                            # Base of flame - red
                            h, s, v = instate['colors']['deep_red']
                            r, g, b = hsv_to_rgb(h, s, v)
                            r_values[base_mask] = r
                            g_values[base_mask] = g
                            b_values[base_mask] = b
                        
                        # Calculate intensities vectorized
                        # Brightest in middle, dimmer at edges
                        intensities = 1.0 - abs((rel_positions * 2) - 1.0)
                        pixel_intensities = flame_alpha * intensities
                        
                        # Create pixel values
                        pixels = np.stack([
                            r_values, 
                            g_values, 
                            b_values, 
                            pixel_intensities
                        ], axis=1)
                        
                        # Add to temp buffer with additive blending
                        valid_positions = np.arange(start, end)
                        temp_buffer[valid_positions] = np.maximum(
                            temp_buffer[valid_positions],
                            pixels
                        )
            
            # Replace with updated list
            instate['flames'][strip_id] = new_flames
            
            # Update existing embers - vectorized where possible
            new_embers = []
            
            for ember in instate['embers'][strip_id]:
                # Update life
                ember['life'] += delta_time / ember['duration']
                
                # Update position - embers move upward (toward index 0)
                base_position = ember['position'] - ember['speed'] * delta_time
                
                # Add wobble for natural movement
                wobble = ember['wobble_amplitude'] * np.sin(
                    outstate['current_time'] * ember['wobble_frequency'] + ember['wobble_offset']
                )
                
                # Actual position with wobble
                ember['position'] = base_position
                
                # Keep if still in strip and not at end of life
                if ember['position'] > 0 and ember['life'] < 1.0:
                    new_embers.append(ember)
                    
                    # Calculate ember intensity based on life cycle
                    if ember['life'] < 0.1:
                        # Ember growing
                        ember_alpha = ember['intensity'] * (ember['life'] / 0.1)
                    elif ember['life'] > 0.7:
                        # Ember fading
                        ember_alpha = ember['intensity'] * (1.0 - ((ember['life'] - 0.7) / 0.3))
                    else:
                        # Ember at full intensity
                        ember_alpha = ember['intensity']
                    
                    # Draw the ember with wobble
                    ember_pos = int(ember['position'] + wobble)
                    
                    if 0 <= ember_pos < strip_length:
                        # Get ember color
                        h, s, v = instate['colors']['ember']
                        
                        # Adjust brightness based on life (brightest in middle of life)
                        life_factor = 1.0 - abs((ember['life'] * 2) - 1.0)
                        v = v * (0.7 + 0.3 * life_factor)
                        
                        # Convert to RGB
                        r, g, b = hsv_to_rgb(h, s, v)
                        
                        # Add to temp buffer with maximum blending
                        temp_buffer[ember_pos] = np.maximum(
                            temp_buffer[ember_pos],
                            [r, g, b, ember_alpha]
                        )
                        
                        # Add small glow around ember
                        for i in range(1, 3):
                            for offset in [-i, i]:
                                glow_pos = ember_pos + offset
                                if 0 <= glow_pos < strip_length:
                                    glow_alpha = ember_alpha * (1.0 - (i / 3.0))
                                    temp_buffer[glow_pos] = np.maximum(
                                        temp_buffer[glow_pos],
                                        [r * 0.5, g * 0.5, b * 0.5, glow_alpha * 0.5]
                                    )
            
            # Replace with updated list
            instate['embers'][strip_id] = new_embers
            
            # Apply the temp buffer to the actual buffer with additive blending
            for i in range(strip_length):
                if temp_buffer[i, 3] > 0:  # If there's any opacity
                    curr_r, curr_g, curr_b, curr_a = buffer[i]
                    r, g, b, a = temp_buffer[i]
                    
                    buffer[i] = [
                        min(1.0, curr_r + r),
                        min(1.0, curr_g + g),
                        min(1.0, curr_b + b),
                        max(curr_a, a)
                    ]
            
        elif 'heart' in strip.groups:
            # Heart strips - rapid angry heartbeat - fully vectorized
            
            # Create intense heartbeat waveform
            if instate['heart_phase'] < 0.1:
                # Very sharp rise (systole)
                intensity = instate['heart_phase'] / 0.1
            elif instate['heart_phase'] < 0.2:
                # Short peak
                intensity = 1.0
            elif instate['heart_phase'] < 0.3:
                # Sharp decline
                intensity = 1.0 - ((instate['heart_phase'] - 0.2) / 0.1)
            elif instate['heart_phase'] < 0.4:
                # Brief after-beat bump
                intensity = 0.3 * ((instate['heart_phase'] - 0.3) / 0.1)
            elif instate['heart_phase'] < 0.5:
                # Small secondary peak
                intensity = 0.3
            else:
                # Rest phase
                intensity = 0.3 * (1.0 - ((instate['heart_phase'] - 0.5) / 0.5))
            
            # Make intensity stronger for anger
            intensity = intensity * 0.7 + 0.3
            
            # Use bright red for the heart
            h, s, v = instate['colors']['bright_red']
            
            # Adjust value based on heartbeat intensity
            v_adjusted = v * intensity
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v_adjusted)
            
            # Create distance array from center
            positions = np.arange(strip_length)
            center = strip_length // 2
            dist = np.abs(positions - center) / (strip_length / 2)
            
            # Calculate pulse intensity based on distance from center
            pulse_intensities = intensity * (1.0 - dist * 0.5)
            
            # Calculate vectorized colors with pulse
            r_values = r * np.ones(strip_length)
            g_values = g * (0.7 + 0.3 * pulse_intensities)
            b_values = b * (0.7 + 0.3 * pulse_intensities)
            a_values = (0.4 + 0.6 * intensity) * (0.8 + 0.2 * pulse_intensities)
            
            # Update buffer (vectorized)
            buffer[:, 0] = r_values
            buffer[:, 1] = g_values
            buffer[:, 2] = b_values
            buffer[:, 3] = a_values
            
        elif 'brain' in strip.groups or 'ear' in strip.groups or 'head' in strip.groups:
            # Brain, ear, and head strips - explosive bursts
            
            # Initialize explosions for this strip if not already done
            if strip_id not in instate['explosions']:
                instate['explosions'][strip_id] = []
                instate['explosion_timer'][strip_id] = 0.0
                
                # Create an initial explosion
                center = np.random.randint(0, strip_length)
                instate['explosions'][strip_id].append({
                    'center': center,
                    'radius': 0.0,  # Start with zero radius
                    'max_radius': 5 + np.random.randint(0, 5),  # 5-10 pixels radius
                    'expansion_rate': 30 + np.random.random() * 20,  # 30-50 pixels per second
                    'intensity': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 intensity
                    'life': 0.0,  # 0-1 life cycle
                    'duration': 0.3 + np.random.random() * 0.3,  # 0.3-0.6 second duration (fast explosion)
                    'color_shift': np.random.random() * 0.1  # Small random hue variation
                })
            
            # Update explosion timer
            instate['explosion_timer'][strip_id] += delta_time
            
            # Check if it's time for a new explosion
            # More frequent explosions for anger
            explosion_interval = 0.2 + np.random.random() * 0.4  # 0.2-0.6 seconds between explosions
            
            if instate['explosion_timer'][strip_id] >= explosion_interval:
                instate['explosion_timer'][strip_id] = 0.0
                
                # Create a new explosion at random position
                center = np.random.randint(0, strip_length)
                
                instate['explosions'][strip_id].append({
                    'center': center,
                    'radius': 0.0,
                    'max_radius': 5 + np.random.randint(0, 5),  # 5-10 pixels radius
                    'expansion_rate': 30 + np.random.random() * 20,  # 30-50 pixels per second
                    'intensity': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 intensity
                    'life': 0.0,
                    'duration': 0.3 + np.random.random() * 0.3,  # 0.3-0.6 second duration
                    'color_shift': np.random.random() * 0.1  # Small random hue variation
                })
            
            # Create a temporary buffer for accumulating explosion effects
            temp_buffer = np.zeros((strip_length, 4))
            
            # Update existing explosions
            new_explosions = []
            for explosion in instate['explosions'][strip_id]:
                # Update life cycle
                explosion['life'] += delta_time / explosion['duration']
                
                # Update radius - grows quickly then stops
                if explosion['life'] < 0.6:  # Grow during first 60% of life
                    explosion['radius'] += explosion['expansion_rate'] * delta_time
                    explosion['radius'] = min(explosion['radius'], explosion['max_radius'])
                
                # Keep if still alive
                if explosion['life'] < 1.0:
                    new_explosions.append(explosion)
                    
                    # Calculate intensity based on life cycle
                    if explosion['life'] < 0.2:
                        # Quick ramp up
                        explosion_alpha = explosion['intensity'] * (explosion['life'] / 0.2)
                    elif explosion['life'] > 0.7:
                        # Fade out
                        explosion_alpha = explosion['intensity'] * (1.0 - ((explosion['life'] - 0.7) / 0.3))
                    else:
                        # Full intensity
                        explosion_alpha = explosion['intensity']
                    
                    # Calculate the explosion using vectorized operations
                    radius_int = int(explosion['radius'])
                    center = explosion['center']
                    
                    # Create valid pixel positions array
                    start = max(0, center - radius_int)
                    end = min(strip_length, center + radius_int + 1)
                    
                    if start < end:  # Only proceed if there are valid positions
                        # Calculate positions and distances
                        positions = np.arange(start, end)
                        distances = np.abs(positions - center) / explosion['radius'] if explosion['radius'] > 0 else np.ones(end - start)
                        
                        # Create masks for different parts of the explosion
                        center_mask = distances < 0.2
                        middle_mask = (distances >= 0.2) & (distances < 0.6)
                        edge_mask = (distances >= 0.6) & (distances <= 1.0)
                        
                        # Initialize color arrays
                        r_values = np.zeros(len(positions))
                        g_values = np.zeros(len(positions))
                        b_values = np.zeros(len(positions))
                        a_values = np.zeros(len(positions))
                        
                        # Calculate pixel intensities
                        pixel_intensities = explosion_alpha * (1.0 - distances**2)
                        
                        # Only process pixels with significant intensity
                        valid_pixels = pixel_intensities > 0.05
                        
                        if np.any(valid_pixels):
                            # Apply colors based on distance from center
                            if np.any(center_mask & valid_pixels):
                                # Center - yellow/white hot
                                h, s, v = instate['colors']['yellow']
                                # Desaturate center for white-hot look
                                s *= 0.5
                                v = min(1.0, v * 1.2)
                                r, g, b = hsv_to_rgb(h, s, v)
                                
                                r_values[center_mask & valid_pixels] = r
                                g_values[center_mask & valid_pixels] = g
                                b_values[center_mask & valid_pixels] = b
                            
                            if np.any(middle_mask & valid_pixels):
                                # Middle - orange
                                h, s, v = instate['colors']['orange']
                                h = (h + explosion['color_shift']) % 1.0  # Slight color variation
                                r, g, b = hsv_to_rgb(h, s, v)
                                
                                r_values[middle_mask & valid_pixels] = r
                                g_values[middle_mask & valid_pixels] = g
                                b_values[middle_mask & valid_pixels] = b
                            
                            if np.any(edge_mask & valid_pixels):
                                # Outer edge - red
                                h, s, v = instate['colors']['bright_red']
                                r, g, b = hsv_to_rgb(h, s, v)
                                
                                r_values[edge_mask & valid_pixels] = r
                                g_values[edge_mask & valid_pixels] = g
                                b_values[edge_mask & valid_pixels] = b
                            
                            # Set alpha values for valid pixels
                            a_values[valid_pixels] = pixel_intensities[valid_pixels]
                            
                            # Stack color components
                            rgba_values = np.stack([r_values, g_values, b_values, a_values], axis=1)
                            
                            # Update temp buffer with maximum blending
                            for i, pos in enumerate(positions):
                                if valid_pixels[i]:
                                    temp_buffer[pos] = np.maximum(temp_buffer[pos], rgba_values[i])
            
            # Replace with updated list
            instate['explosions'][strip_id] = new_explosions
            
            # Apply the temp buffer to the actual buffer with additive blending
            for i in range(strip_length):
                if temp_buffer[i, 3] > 0:  # If there's any opacity
                    curr_r, curr_g, curr_b, curr_a = buffer[i]
                    r, g, b, a = temp_buffer[i]
                    
                    buffer[i] = [
                        min(1.0, curr_r + r),
                        min(1.0, curr_g + g),
                        min(1.0, curr_b + b),
                        max(curr_a, a)
                    ]
            
        elif 'base' in strip.groups:
            # Base strips - rolling flames with occasional explosions
            
            # Initialize effects for this strip if not already done
            if strip_id not in instate['flames']:
                instate['flames'][strip_id] = []
                instate['explosions'][strip_id] = []
                instate['explosion_timer'][strip_id] = 0.0
                
                # Create initial flames along the strip
                num_initial_flames = strip_length // 15  # About 1 flame every 15 pixels
                for _ in range(num_initial_flames):
                    pos = np.random.randint(0, strip_length)
                    
                    # For base strips, determine direction based on position
                    middle = strip_length // 2
                    direction = 1 if pos < middle else -1  # Away from middle
                    
                    instate['flames'][strip_id].append({
                        'position': pos,
                        'speed': 15 + np.random.random() * 15,  # 15-30 pixels per second
                        'size': 8 + np.random.randint(0, 7),    # 8-15 pixels
                        'height': 3 + np.random.randint(0, 4),  # 3-7 pixels height
                        'intensity': 0.6 + np.random.random() * 0.4,  # 0.6-1.0 intensity
                        'direction': direction,
                        'life': 0.0,
                        'duration': 0.8 + np.random.random() * 0.7  # 0.8-1.5 second lifetime
                    })
            
            # Update explosion timer
            instate['explosion_timer'][strip_id] += delta_time
            
            # Less frequent explosions on base strips
            explosion_interval = 0.10 + np.random.random() * 0.2  # 0.1-0.3 seconds between explosions
            
            # Check if it's time for a new explosion
            if instate['explosion_timer'][strip_id] >= explosion_interval:
                instate['explosion_timer'][strip_id] = 0.0
                
                # Create a new explosion at random position
                center = np.random.randint(0, strip_length)
                
                instate['explosions'][strip_id].append({
                    'center': center,
                    'radius': 0.0,
                    'max_radius': 8 + np.random.randint(0, 7),  # 8-15 pixels radius (larger for base)
                    'expansion_rate': 40 + np.random.random() * 30,  # 40-70 pixels per second
                    'intensity': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 intensity
                    'life': 0.0,
                    'duration': 0.4 + np.random.random() * 0.4,  # 0.4-0.8 second duration
                    'color_shift': np.random.random() * 0.1  # Small random hue variation
                })
            
            # Chance to create new flames
            if np.random.random() < 0.1:  # 10% chance per frame
                # Create at a random position
                pos = np.random.randint(0, strip_length)
                
                # Determine direction (away from middle)
                middle = strip_length // 2
                direction = 1 if pos < middle else -1
                
                instate['flames'][strip_id].append({
                    'position': pos,
                    'speed': 15 + np.random.random() * 15,
                    'size': 8 + np.random.randint(0, 7),
                    'height': 3 + np.random.randint(0, 4),
                    'intensity': 0.6 + np.random.random() * 0.4,
                    'direction': direction,
                    'life': 0.0,
                    'duration': 0.8 + np.random.random() * 0.7
                })
            
            # Create a temporary buffer for accumulating flame and explosion effects
            temp_buffer = np.zeros((strip_length, 4))
            
            # Update existing flames
            new_flames = []
            for flame in instate['flames'][strip_id]:
                # Update life
                flame['life'] += delta_time / flame['duration']
                
                # Update position based on direction
                flame['position'] += flame['speed'] * flame['direction'] * delta_time
                
                # Keep if still in bounds and alive
                if ((0 <= flame['position'] < strip_length or 
                     0 <= flame['position'] + flame['size'] < strip_length) and 
                    flame['life'] < 1.0):
                    new_flames.append(flame)
                    
                    # Calculate flame intensity based on life cycle
                    if flame['life'] < 0.2:
                        # Flame growing
                        flame_alpha = flame['intensity'] * (flame['life'] / 0.2)
                    elif flame['life'] > 0.7:
                        # Flame fading
                        flame_alpha = flame['intensity'] * (1.0 - ((flame['life'] - 0.7) / 0.3))
                    else:
                        # Flame at full intensity
                        flame_alpha = flame['intensity']
                    
                    # Draw the flame - horizontal spread with height
                    flame_center = int(flame['position'])
                    
                    # Create valid horizontal positions for this flame
                    h_start = max(0, flame_center)
                    h_end = min(strip_length, flame_center + flame['size'] * abs(flame['direction']))
                    
                    if h_start < h_end:
                        # Get positions in array
                        positions = np.arange(h_start, h_end)
                        
                        # Calculate relative positions in the flame
                        if flame['direction'] > 0:
                            rel_positions = (positions - flame_center) / flame['size']
                        else:
                            rel_positions = (flame_center - positions) / flame['size']
                            rel_positions = np.flip(rel_positions)  # Correct orientation
                        
                        # Clamp relative positions to valid range
                        rel_positions = np.clip(rel_positions, 0.0, 1.0)
                        
                        # Create masks for flame sections
                        front_mask = rel_positions < 0.3  # Front of flame
                        middle_mask = (rel_positions >= 0.3) & (rel_positions < 0.7)  # Middle
                        back_mask = rel_positions >= 0.7  # Back of flame
                        
                        # Initialize color arrays
                        r_values = np.zeros(len(positions))
                        g_values = np.zeros(len(positions))
                        b_values = np.zeros(len(positions))
                        
                        # Apply colors based on section
                        if np.any(front_mask):
                            # Front - more yellow
                            h, s, v = instate['colors']['yellow']
                            r, g, b = hsv_to_rgb(h, s, v)
                            r_values[front_mask] = r
                            g_values[front_mask] = g
                            b_values[front_mask] = b
                        
                        if np.any(middle_mask):
                            # Middle - orange
                            h, s, v = instate['colors']['orange']
                            r, g, b = hsv_to_rgb(h, s, v)
                            r_values[middle_mask] = r
                            g_values[middle_mask] = g
                            b_values[middle_mask] = b
                        
                        if np.any(back_mask):
                            # Back - deeper red
                            h, s, v = instate['colors']['deep_red']
                            r, g, b = hsv_to_rgb(h, s, v)
                            r_values[back_mask] = r
                            g_values[back_mask] = g
                            b_values[back_mask] = b
                        
                        # Apply lapping effect - flames oscillate
                        time_factors = outstate['current_time'] * 8.0 + positions * 0.2
                        lap_factors = 0.2 * np.sin(time_factors)
                        
                        # Height factors give more variation
                        height_factors = np.ones(len(positions))
                        for j in range(1, flame['height'] + 1):
                            # For each "height" level, add variation
                            vert_pos = j / flame['height']
                            height_factor = 1.0 - (abs(vert_pos - 0.5) * 1.2)  # Brightest in middle height
                            
                            # Calculate pixel intensities with lapping and height factors
                            pixel_intensities = flame_alpha * height_factor * (1.0 + lap_factors)
                            pixel_intensities = np.clip(pixel_intensities, 0.0, 1.0)
                            
                            # Create RGBA values for this set of pixels
                            pixels = np.stack([r_values, g_values, b_values, pixel_intensities], axis=1)
                            
                            # Update temp buffer with maximum blending
                            temp_buffer[positions] = np.maximum(temp_buffer[positions], pixels)
            
            # Replace with updated list
            instate['flames'][strip_id] = new_flames
            
            # Update existing explosions
            new_explosions = []
            for explosion in instate['explosions'][strip_id]:
                # Update life cycle
                explosion['life'] += delta_time / explosion['duration']
                
                # Update radius - grows quickly then stops
                if explosion['life'] < 0.6:  # Grow during first 60% of life
                    explosion['radius'] += explosion['expansion_rate'] * delta_time
                    explosion['radius'] = min(explosion['radius'], explosion['max_radius'])
                
                # Keep if still alive
                if explosion['life'] < 1.0:
                    new_explosions.append(explosion)
                    
                    # Calculate intensity based on life cycle
                    if explosion['life'] < 0.2:
                        # Quick ramp up
                        explosion_alpha = explosion['intensity'] * (explosion['life'] / 0.2)
                    elif explosion['life'] > 0.7:
                        # Fade out
                        explosion_alpha = explosion['intensity'] * (1.0 - ((explosion['life'] - 0.7) / 0.3))
                    else:
                        # Full intensity
                        explosion_alpha = explosion['intensity']
                    
                    # Calculate the explosion using vectorized operations
                    radius_int = int(explosion['radius'])
                    center = explosion['center']
                    
                    # Create valid pixel positions array
                    start = max(0, center - radius_int)
                    end = min(strip_length, center + radius_int + 1)
                    
                    if start < end:  # Only proceed if there are valid positions
                        # Calculate positions and distances
                        positions = np.arange(start, end)
                        distances = np.abs(positions - center) / explosion['radius'] if explosion['radius'] > 0 else np.ones(end - start)
                        
                        # Create masks for different parts of the explosion
                        center_mask = distances < 0.2
                        middle_mask = (distances >= 0.2) & (distances < 0.6)
                        edge_mask = (distances >= 0.6) & (distances <= 1.0)
                        
                        # Initialize color arrays
                        r_values = np.zeros(len(positions))
                        g_values = np.zeros(len(positions))
                        b_values = np.zeros(len(positions))
                        a_values = np.zeros(len(positions))
                        
                        # Calculate pixel intensities
                        pixel_intensities = explosion_alpha * (1.0 - distances**2)
                        
                        # Only process pixels with significant intensity
                        valid_pixels = pixel_intensities > 0.05
                        
                        if np.any(valid_pixels):
                            # Apply colors based on distance from center
                            if np.any(center_mask & valid_pixels):
                                # Center - yellow/white hot
                                h, s, v = instate['colors']['yellow']
                                # Desaturate center for white-hot look
                                s *= 0.5
                                v = min(1.0, v * 1.2)
                                r, g, b = hsv_to_rgb(h, s, v)
                                
                                r_values[center_mask & valid_pixels] = r
                                g_values[center_mask & valid_pixels] = g
                                b_values[center_mask & valid_pixels] = b
                            
                            if np.any(middle_mask & valid_pixels):
                                # Middle - orange
                                h, s, v = instate['colors']['orange']
                                h = (h + explosion['color_shift']) % 1.0  # Slight color variation
                                r, g, b = hsv_to_rgb(h, s, v)
                                
                                r_values[middle_mask & valid_pixels] = r
                                g_values[middle_mask & valid_pixels] = g
                                b_values[middle_mask & valid_pixels] = b
                            
                            if np.any(edge_mask & valid_pixels):
                                # Outer edge - red
                                h, s, v = instate['colors']['bright_red']
                                r, g, b = hsv_to_rgb(h, s, v)
                                
                                r_values[edge_mask & valid_pixels] = r
                                g_values[edge_mask & valid_pixels] = g
                                b_values[edge_mask & valid_pixels] = b
                            
                            # Set alpha values for valid pixels
                            a_values[valid_pixels] = pixel_intensities[valid_pixels]
                            
                            # Stack color components
                            rgba_values = np.stack([r_values, g_values, b_values, a_values], axis=1)
                            
                            # Update temp buffer with maximum blending
                            for i, pos in enumerate(positions):
                                if valid_pixels[i]:
                                    temp_buffer[pos] = np.maximum(temp_buffer[pos], rgba_values[i])
            
            # Replace with updated list
            instate['explosions'][strip_id] = new_explosions
            
            # Apply the temp buffer to the actual buffer with additive blending
            for i in range(strip_length):
                if temp_buffer[i, 3] > 0:  # If there's any opacity
                    curr_r, curr_g, curr_b, curr_a = buffer[i]
                    r, g, b, a = temp_buffer[i]
                    
                    buffer[i] = [
                        min(1.0, curr_r + r),
                        min(1.0, curr_g + g),
                        min(1.0, curr_b + b),
                        max(curr_a, a)
                    ]
            
        else:
            # Other strips - fiery pulsing with heat waves - fully vectorized
            
            # Create positions array for the entire strip
            positions = np.arange(strip_length)
            
            # Create normalized positions
            norm_positions = positions / strip_length
            
            # Create heat wave pattern - multiple sine waves combined (vectorized)
            wave1 = 0.5 + 0.5 * np.sin(norm_positions * 4 * np.pi + outstate['current_time'] * 3.0)
            wave2 = 0.5 + 0.5 * np.sin(norm_positions * 7 * np.pi - outstate['current_time'] * 2.0)
            wave3 = 0.5 + 0.5 * np.sin(norm_positions * 2 * np.pi + outstate['current_time'] * 1.0)
            
            # Combine waves with different weights (vectorized)
            heat_intensities = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2)
            
            # Add pulsing effect (vectorized)
            pulse = 0.7 + 0.3 * np.sin(outstate['current_time'] * 5.0)
            heat_intensities *= pulse
            
            # Create masks for different intensity levels
            very_hot_mask = heat_intensities > 0.8
            hot_mask = (heat_intensities > 0.5) & (heat_intensities <= 0.8)
            less_hot_mask = heat_intensities <= 0.5
            
            # Initialize color arrays
            r_values = np.zeros(strip_length)
            g_values = np.zeros(strip_length)
            b_values = np.zeros(strip_length)
            a_values = np.zeros(strip_length)
            
            # Apply colors based on heat intensity (vectorized)
            if np.any(very_hot_mask):
                # Very hot - yellow
                h, s, v = instate['colors']['yellow']
                r, g, b = hsv_to_rgb(h, s, v)
                r_values[very_hot_mask] = r
                g_values[very_hot_mask] = g
                b_values[very_hot_mask] = b
            
            if np.any(hot_mask):
                # Hot - orange
                h, s, v = instate['colors']['orange']
                r, g, b = hsv_to_rgb(h, s, v)
                r_values[hot_mask] = r
                g_values[hot_mask] = g
                b_values[hot_mask] = b
            
            if np.any(less_hot_mask):
                # Less hot - deep red
                h, s, v = instate['colors']['deep_red']
                r, g, b = hsv_to_rgb(h, s, v)
                r_values[less_hot_mask] = r
                g_values[less_hot_mask] = g
                b_values[less_hot_mask] = b
            
            # Adjust brightness based on heat intensity (vectorized)
            v_adjusted = 0.6 + 0.4 * heat_intensities
            
            # Apply to RGB values
            r_values *= v_adjusted
            g_values *= v_adjusted
            b_values *= v_adjusted
            
            # Set alpha based on heat intensity
            a_values = 0.3 + 0.7 * heat_intensities
            
            # Update buffer (vectorized)
            buffer[:, 0] = r_values
            buffer[:, 1] = g_values
            buffer[:, 2] = b_values
            buffer[:, 3] = a_values
            
            # Add ember sparks with random distribution
            spark_positions = np.random.random(strip_length) < 0.005  # 0.5% chance per pixel
            if np.any(spark_positions):
                # Get ember color
                h, s, v = instate['colors']['ember']
                r, g, b = hsv_to_rgb(h, s, v)
                
                # Calculate spark intensities (vectorized)
                spark_intensities = 0.8 + np.random.random(strip_length) * 0.2
                
                # Apply sparks
                buffer[spark_positions, 0] = r
                buffer[spark_positions, 1] = g
                buffer[spark_positions, 2] = b
                buffer[spark_positions, 3] = spark_intensities[spark_positions]
        
        # Apply additional random noise texture to ALL strips (vectorized)
        # Create random intensity array
        noise_intensities = 0.02 + np.random.random(strip_length) * 0.18  # 0.02-0.2 range
        
        # Choose random colors for each pixel
        color_indices = np.random.randint(0, len(list(instate['colors'].keys())), strip_length)
        color_names = list(instate['colors'].keys())
        
        # Apply noise to each pixel
        for i in range(strip_length):
            # Get color
            h, s, v = instate['colors'][color_names[color_indices[i]]]
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Add to current pixel with additive blending
            curr_r, curr_g, curr_b, curr_a = buffer[i]
            intensity = noise_intensities[i]
            
            buffer[i] = [
                min(1.0, curr_r + r * intensity),
                min(1.0, curr_g + g * intensity),
                min(1.0, curr_b + b * intensity),
                max(curr_a, intensity)
            ]



def OTO_curious_playful(instate, outstate):
    """
    Generator function that creates a curious and playful-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_curious'] value
    2. Vibrant, saturated color palette with blues, greens, reds, oranges, purples, and whites
    3. Moving color regions that create dynamic patterns
    4. Fast movement with playful characteristics
    
    Uses HSV colorspace for color generation and blending.
    """
    name = 'curious_playful'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['color_regions'] = {}    # Track color regions per strip
        instate['color_shift'] = 0.0     # Global color shift for variation
        
        # Color palette (HSV values) - more saturated colors
        instate['colors'] = {
            'bright_blue': [0.6, 0.7, 0.95],      # Bright blue
            'vibrant_green': [0.3, 0.8, 0.9],     # Vibrant green
            'bright_red': [0.0, 0.8, 0.95],       # Bright red
            'vibrant_orange': [0.08, 0.9, 0.95],  # Vibrant orange
            'rich_purple': [0.8, 0.8, 0.9],       # Rich purple
            'hot_pink': [0.9, 0.75, 0.95],        # Hot pink
            'turquoise': [0.45, 0.8, 0.95],       # Turquoise
            'bright_yellow': [0.15, 0.8, 0.95],   # Bright yellow
            'pure_white': [0.0, 0.0, 1.0]         # Pure white
        }
        
        # Motion parameters
        instate['region_speed_multiplier'] = 1.0  # Global speed control
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get curious level from outstate (default to 0)
    curious_level = outstate.get('control_curious', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = curious_level
    
    # Skip rendering if alpha is too low
    if curious_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * curious_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update global color shift for variation - slow cycle through hues
    instate['color_shift'] = (instate['color_shift'] + 0.05 * delta_time) % 1.0
    
    # Adjust global region speed based on curious level - more curious = faster
    instate['region_speed_multiplier'] = 1.0 + curious_level  # 1.0-2.0x speed range
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Initialize color regions for this strip if not already done
        if strip_id not in instate['color_regions']:
            instate['color_regions'][strip_id] = []
            
            # Create initial color regions - divide strip into segments
            num_regions = max(2, min(6, strip_length // 30))  # 2-6 regions depending on strip length
            region_size = strip_length / num_regions
            
            for i in range(num_regions):
                # Create a color region with position centered in each segment
                center = (i + 0.5) * region_size
                
                # Get random color from palette
                color_name = np.random.choice(list(instate['colors'].keys()))
                h, s, v = instate['colors'][color_name]
                
                # Random direction for movement
                direction = 1 if np.random.random() < 0.5 else -1
                
                # Create region
                region = {
                    'center': center,
                    'size': region_size * 0.7,  # Slightly smaller than segment for initial gaps
                    'h': h,
                    's': s,
                    'v': v,
                    'speed': 5 + np.random.random() * 15,  # 5-20 pixels per second
                    'direction': direction,
                    'wobble_freq': 0.5 + np.random.random() * 1.5,  # 0.5-2.0 Hz
                    'wobble_amount': 0.2 + np.random.random() * 0.4,  # 0.2-0.6 size wobble
                    'wobble_offset': np.random.random() * 6.28,  # Random phase
                    'lifetime': 0,  # Time tracking for color changes
                    'color_change_time': 5 + np.random.random() * 10  # 5-15 seconds between color changes
                }
                
                instate['color_regions'][strip_id].append(region)
        
        # Update regions and check for collisions
        for i, region in enumerate(instate['color_regions'][strip_id]):
            # Update region position based on speed and direction
            effective_speed = region['speed'] * instate['region_speed_multiplier']
            region['center'] += effective_speed * region['direction'] * delta_time
            
            # Add wobble to size for a playful effect
            time_factor = outstate['current_time'] * region['wobble_freq']
            size_wobble = 1.0 + region['wobble_amount'] * np.sin(time_factor + region['wobble_offset'])
            region['effective_size'] = region['size'] * size_wobble  # Store for rendering
            
            # Handle wrapping around strip boundaries
            if region['center'] >= strip_length:
                region['center'] -= strip_length
            elif region['center'] < 0:
                region['center'] += strip_length
            
            # Handle collision with other regions - check if regions are too close
            for j, other_region in enumerate(instate['color_regions'][strip_id]):
                if i != j:  # Don't compare to self
                    # Calculate distance considering strip wrapping
                    direct_dist = abs(region['center'] - other_region['center'])
                    wrapped_dist = strip_length - direct_dist
                    distance = min(direct_dist, wrapped_dist)
                    
                    # Minimum allowed distance is sum of half sizes
                    min_distance = (region['effective_size'] + other_region.get('effective_size', other_region['size'])) * 0.5
                    
                    # If too close, reverse direction of both
                    if distance < min_distance * 0.8:  # 80% of minimum to create some bounce space
                        # Only reverse if moving toward each other
                        if ((region['center'] < other_region['center'] and region['direction'] > 0 and 
                             other_region['direction'] < 0) or
                            (region['center'] > other_region['center'] and region['direction'] < 0 and 
                             other_region['direction'] > 0)):
                            region['direction'] *= -1
                            other_region['direction'] *= -1
                            
                            # Add slight random speed variation on bounce
                            region['speed'] *= 0.9 + 0.2 * np.random.random()
                            other_region['speed'] *= 0.9 + 0.2 * np.random.random()
                            
                            # Keep speeds in reasonable range
                            region['speed'] = max(5, min(20, region['speed']))
                            other_region['speed'] = max(5, min(20, other_region['speed']))
            
            # Update lifetime and check for color change
            region['lifetime'] += delta_time
            if region['lifetime'] > region['color_change_time']:
                # Reset lifetime
                region['lifetime'] = 0
                
                # Choose a new color - avoid similar hue to neighbors
                available_colors = list(instate['colors'].keys())
                
                # Try to get neighboring regions (accounting for potential out-of-bounds)
                if len(instate['color_regions'][strip_id]) > 1:
                    # Find regions that are close by distance
                    neighbor_indices = []
                    for j, other in enumerate(instate['color_regions'][strip_id]):
                        if i != j:
                            direct_dist = abs(region['center'] - other['center'])
                            wrapped_dist = strip_length - direct_dist
                            distance = min(direct_dist, wrapped_dist)
                            
                            if distance < (region['size'] + other['size']) * 1.5:  # If close enough to be a neighbor
                                neighbor_indices.append(j)
                    
                    # If we have neighbors, try to avoid their colors
                    if neighbor_indices:
                        neighbor_hues = [instate['color_regions'][strip_id][j]['h'] for j in neighbor_indices]
                        
                        # Filter out colors with similar hue
                        filtered_colors = []
                        for color_name in available_colors:
                            h, s, v = instate['colors'][color_name]
                            is_similar = False
                            for n_hue in neighbor_hues:
                                # Check if hues are similar (considering wrap-around at 1.0)
                                hue_dist = min(abs(h - n_hue), 1.0 - abs(h - n_hue))
                                if hue_dist < 0.15:  # Consider similar if within 15% of hue space
                                    is_similar = True
                                    break
                            if not is_similar:
                                filtered_colors.append(color_name)
                        
                        # If we have filtered colors, use them, otherwise use all colors
                        if filtered_colors:
                            available_colors = filtered_colors
                
                # Choose a new color from available options
                new_color_name = np.random.choice(available_colors)
                h, s, v = instate['colors'][new_color_name]
                
                # Update region color
                region['h'] = h
                region['s'] = s
                region['v'] = v
                
                # Also randomize wobble parameters for variety
                region['wobble_freq'] = 0.5 + np.random.random() * 1.5
                region['wobble_amount'] = 0.2 + np.random.random() * 0.4
                region['wobble_offset'] = np.random.random() * 6.28
                
                # Set a new color change time
                region['color_change_time'] = 5 + np.random.random() * 10
        
        # -------- SIMPLIFIED RENDERING APPROACH --------
        # Initialize buffer with zeros
        buffer_hsv = np.zeros((strip_length, 4))  # [h, s, v, influence]
        
        # Render each region as a Gaussian-like distribution of influence
        pixels = np.arange(strip_length)
        
        for region in instate['color_regions'][strip_id]:
            # Calculate distance to center with wrapping
            direct_dist = np.abs(pixels - region['center'])
            wrapped_dist = strip_length - direct_dist
            distances = np.minimum(direct_dist, wrapped_dist)
            
            # Calculate influence using a Gaussian-like falloff
            sigma = region['effective_size'] / 2  # Standard deviation (half the size)
            influence = np.exp(-0.5 * (distances / sigma)**2)  # Gaussian-like falloff
            
            # Only apply where influence is significant
            mask = influence > 0.01
            
            # Add this region's contribution to the buffer
            # Additive blending for HSV values weighted by influence
            buffer_hsv[mask, 0] += region['h'] * influence[mask]  # Hue
            buffer_hsv[mask, 1] += region['s'] * influence[mask]  # Saturation
            buffer_hsv[mask, 2] += region['v'] * influence[mask]  # Value
            buffer_hsv[mask, 3] += influence[mask]  # Total influence for normalization
        
        # Normalize the HSV values by total influence
        has_influence = buffer_hsv[:, 3] > 0
        if np.any(has_influence):
            # Normalize hue, saturation, value by total influence
            buffer_hsv[has_influence, 0] /= buffer_hsv[has_influence, 3]
            buffer_hsv[has_influence, 1] /= buffer_hsv[has_influence, 3]
            buffer_hsv[has_influence, 2] /= buffer_hsv[has_influence, 3]
            
            # Wrap hue to 0-1 range
            buffer_hsv[:, 0] = buffer_hsv[:, 0] % 1.0
            
            # Clamp saturation and value to 0-1 range
            buffer_hsv[:, 1] = np.clip(buffer_hsv[:, 1], 0, 1)
            buffer_hsv[:, 2] = np.clip(buffer_hsv[:, 2], 0, 1)
            
            # Convert HSV to RGB
            rgb = np.zeros((strip_length, 3))
            r, g, b = hsv_to_rgb_vectorized(
                buffer_hsv[has_influence, 0], 
                buffer_hsv[has_influence, 1], 
                buffer_hsv[has_influence, 2]
            )
            
            # Set final RGB values
            rgb_buffer = np.zeros((strip_length, 4))  # [r, g, b, a]
            rgb_buffer[has_influence, 0] = r
            rgb_buffer[has_influence, 1] = g
            rgb_buffer[has_influence, 2] = b
            
            # Alpha based on influence (scale to reasonable range)
            rgb_buffer[has_influence, 3] = np.clip(buffer_hsv[has_influence, 3] * 0.5, 0, 1)
            
            # Add sparkles
            sparkle_chance = 0.02 * curious_level  # More sparkles when more curious
            sparkle_mask = np.random.random(strip_length) < sparkle_chance
            
            if np.any(sparkle_mask):
                # Create sparkles
                num_sparkles = np.sum(sparkle_mask)
                sparkle_h = np.random.random(num_sparkles)
                sparkle_s = np.full_like(sparkle_h, 0.2)  # Low saturation (white-ish)
                sparkle_v = np.ones_like(sparkle_h)  # Full brightness
                
                # Convert sparkles to RGB
                sr, sg, sb = hsv_to_rgb_vectorized(sparkle_h, sparkle_s, sparkle_v)
                
                # Add sparkles to the buffer
                sparkle_indices = np.where(sparkle_mask)[0]
                rgb_buffer[sparkle_indices, 0] = np.minimum(1.0, rgb_buffer[sparkle_indices, 0] + sr * 0.7)
                rgb_buffer[sparkle_indices, 1] = np.minimum(1.0, rgb_buffer[sparkle_indices, 1] + sg * 0.7)
                rgb_buffer[sparkle_indices, 2] = np.minimum(1.0, rgb_buffer[sparkle_indices, 2] + sb * 0.7)
                rgb_buffer[sparkle_indices, 3] = np.minimum(1.0, rgb_buffer[sparkle_indices, 3] + 0.3)
            
            # Copy from numpy array back to buffer
            for pixel in range(strip_length):
                if rgb_buffer[pixel, 3] > 0:
                    buffer[pixel] = [
                        rgb_buffer[pixel, 0], 
                        rgb_buffer[pixel, 1], 
                        rgb_buffer[pixel, 2], 
                        rgb_buffer[pixel, 3]
                    ]
                else:
                    buffer[pixel] = [0, 0, 0, 0]

def OTO_passionate_floral(instate, outstate):
    """
    Generator function that creates a passionate and playful floral-themed pattern.
    
    Features:
    1. Global alpha controlled by outstate['passionate_curious'] value
    2. Floral growth patterns that unfold from origination points with pink and dark blue colors
    3. Hints of green in the growth patterns to represent stems and leaves
    4. Long, moving stripes that fade away as they move
    5. Emphasis on growth and movement away from origination points
    
    Uses HSV colorspace for color generation and blending.
    """
    name = 'passionate_floral'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['bloom_points'] = {}      # Track floral bloom points per strip
        instate['moving_stripes'] = {}    # Track moving stripe patterns
        instate['growth_timer'] = 0.0     # Global timer for coordinated growth
        
        # Color palette (HSV values)
        instate['colors'] = {
            'vivid_pink': [0.9, 0.85, 0.95],      # Vibrant pink
            'soft_pink': [0.95, 0.6, 0.95],       # Softer pink
            'deep_blue': [0.65, 0.9, 0.7],        # Deep blue
            'midnight_blue': [0.7, 0.85, 0.5],    # Darker blue
            'leaf_green': [0.3, 0.8, 0.6],        # Green for leaves/stems
            'light_green': [0.35, 0.6, 0.8],      # Lighter green for new growth
            'white_glow': [0.0, 0.0, 1.0] ,        # White glow for highlights
            'bright_yellow': [0.15, 0.9, 1.0],    # Bright sunny yellow
            'golden_yellow': [0.13, 0.85, 0.9]    # Warm golden yellow
        }
        
        # Growth parameters
        instate['growth_rate'] = 1.0      # Base growth rate
        instate['max_blooms'] = 8         # Maximum blooms per strip
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get passionate_curious level from outstate (default to 0)
    passionate_level = outstate.get('control_passionate', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = passionate_level
    
    # Skip rendering if alpha is too low
    if passionate_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * passionate_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update global growth timer - controls overall growth rhythm
    instate['growth_timer'] += delta_time
    growth_phase = (np.sin(instate['growth_timer'] * 0.8) + 1.0) / 2.0  # 0.0-1.0 oscillation
    
    # Adjust growth rate based on passionate level - higher = faster growth
    effective_growth_rate = instate['growth_rate'] * (0.8 + passionate_level * 0.7)
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer based on strip type
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Start with a dark base
        buffer[:] = [0.0, 0.0, 0.05, 0.2]  # Very dim blue base
        
        # Initialize bloom points for this strip if not already done
        if strip_id not in instate['bloom_points']:
            instate['bloom_points'][strip_id] = []
            instate['moving_stripes'][strip_id] = []
            
            # Create initial bloom points
            # Fix the error by ensuring the high value is always greater than the low value
            max_blooms = max(1, min(3, strip_length // 40))
            num_initial_blooms = 1 if max_blooms <= 1 else np.random.randint(1, max_blooms)
            for _ in range(num_initial_blooms):
                # Random position along strip
                pos = np.random.randint(0, strip_length)
                
                # Create bloom with random properties
                create_new_bloom(instate, strip_id, pos, strip_length)

        
        # Determine strip type for customized effects
        is_base = 'base' in strip.groups
        is_spine = 'spine' in strip.groups
        is_heart = 'heart' in strip.groups
        is_brain = 'brain' in strip.groups
        is_ear = 'ear' in strip.groups
        
        # Chance to create new bloom points (varies by strip type)
        bloom_chance = 0.02  # Base chance
        
        # Adjust bloom chance based on strip type
        if is_base:
            bloom_chance = 0.03  # More blooms on base strips
        elif is_spine:
            bloom_chance = 0.02  # Average on spine
        elif is_heart:
            bloom_chance = 0.04  # More on heart
        elif is_brain or is_ear:
            bloom_chance = 0.025  # Slightly more on brain/ear
            
        # Adjust based on number of existing blooms (less likely if many exist)
        max_blooms = instate['max_blooms']
        existing_blooms = len(instate['bloom_points'][strip_id])
        bloom_chance *= max(0.1, 1.0 - (existing_blooms / max_blooms))
        
        # Also factor in passionate level - more passion, more blooms
        bloom_chance *= (0.5 + passionate_level * 1.0)
        
        # Try to create a new bloom
        if np.random.random() < bloom_chance * delta_time * 60:  # Scale by framerate
            # Choose a random position
            pos = np.random.randint(0, strip_length)
            
            # Check if there's already a bloom nearby
            too_close = False
            for bloom in instate['bloom_points'][strip_id]:
                # Calculate distance considering wrapping
                direct_dist = abs(pos - bloom['position'])
                wrapped_dist = strip_length - direct_dist
                distance = min(direct_dist, wrapped_dist)
                
                # Skip if too close to an existing bloom
                if distance < strip_length * 0.15:  # Minimum 15% of strip length between blooms
                    too_close = True
                    break
            
            # Create new bloom if not too close to existing ones
            if not too_close and existing_blooms < max_blooms:
                create_new_bloom(instate, strip_id, pos, strip_length)
        
        # Chance to create new moving stripes
        stripe_chance = 0.01  # Base chance
        
        # Adjust stripe chance based on strip type
        if is_base:
            stripe_chance = 0.02  # More stripes on base
        elif is_spine:
            stripe_chance = 0.03  # Most on spine
        
        # Adjust based on passionate level
        stripe_chance *= (0.5 + passionate_level * 1.0)
        
        # Try to create a new stripe
        if np.random.random() < stripe_chance * delta_time * 60:  # Scale by framerate
            # Choose a random position
            pos = np.random.randint(0, strip_length)
            
            # Random direction
            direction = 1 if np.random.random() < 0.5 else -1
            
            # Random color - primarily pinks and blues
            if np.random.random() < 0.6:  # 60% pink, 40% blue
                color_name = np.random.choice(['vivid_pink', 'soft_pink'])
            else:
                color_name = np.random.choice(['deep_blue', 'midnight_blue'])
            
            h, s, v = instate['colors'][color_name]
            
            # Create a new stripe
            instate['moving_stripes'][strip_id].append({
                'position': pos,
                'direction': direction,
                'speed': 20 + np.random.random() * 40,  # 20-60 pixels per second
                'length': 10 + np.random.randint(0, int(strip_length * 0.2)),  # 10 to 20% of strip length
                'h': h,
                's': s,
                'v': v,
                'alpha': 0.7 + np.random.random() * 0.3,  # 0.7-1.0 alpha
                'age': 0.0,
                'lifetime': 2.0 + np.random.random() * 3.0  # 2-5 seconds lifetime
            })
        
        # Update and render bloom points
        new_blooms = []
        for bloom in instate['bloom_points'][strip_id]:
            # Update growth stage
            bloom['age'] += delta_time * effective_growth_rate
            
            # Calculate growth factor (0.0-1.0)
            growth_progress = min(1.0, bloom['age'] / bloom['growth_time'])
            
            # Calculate the bloom's current radius based on growth
            # Use sigmoid-like function for natural growth curve
            growth_factor = 1.0 / (1.0 + np.exp(-10 * (growth_progress - 0.5)))
            current_radius = bloom['max_radius'] * growth_factor
            
            # Calculate pulsing effect for grown blooms
            pulse_factor = 1.0
            if growth_progress > 0.9:  # Almost fully grown
                # Add gentle pulsing
                pulse_time = outstate['current_time'] * bloom['pulse_rate']
                pulse_factor = 1.0 + 0.1 * np.sin(pulse_time) * (growth_progress - 0.9) * 10
            
            # Calculate final radius with pulse
            render_radius = current_radius * pulse_factor
            
            # Keep bloom if still within lifetime
            if bloom['age'] < bloom['lifetime']:
                new_blooms.append(bloom)
                
                # Calculate alpha based on growth and lifetime
                if bloom['age'] < bloom['growth_time']:
                    # Growing phase - fade in
                    alpha_factor = growth_progress
                else:
                    # Mature phase - maintain then fade out near end of life
                    life_remaining = 1.0 - ((bloom['age'] - bloom['growth_time']) / 
                                           (bloom['lifetime'] - bloom['growth_time']))
                    alpha_factor = max(0.0, life_remaining)
                
                # Draw the bloom
                draw_bloom(buffer, strip_length, bloom, render_radius, alpha_factor, instate['colors'])
                
                # Chance to create a leaf or tendril from mature blooms
                if growth_progress > 0.5 and np.random.random() < 0.01 * delta_time * 60:
                    # Create a moving stripe that originates from this bloom
                    direction = 1 if np.random.random() < 0.5 else -1
                    
                    # Use green for leaves/tendrils
                    color_name = np.random.choice(['leaf_green', 'light_green'])
                    h, s, v = instate['colors'][color_name]
                    
                    # Create a new stripe
                    instate['moving_stripes'][strip_id].append({
                        'position': bloom['position'],
                        'direction': direction,
                        'speed': 30 + np.random.random() * 30,  # 30-60 pixels per second
                        'length': 5 + np.random.randint(0, 10),  # 5-15 pixels
                        'h': h,
                        's': s,
                        'v': v,
                        'alpha': 0.6 + np.random.random() * 0.4,  # 0.6-1.0 alpha
                        'age': 0.0,
                        'lifetime': 1.0 + np.random.random() * 2.0  # 1-3 seconds lifetime
                    })
            
        # Update bloom list
        instate['bloom_points'][strip_id] = new_blooms
        
        # Update and render moving stripes
        new_stripes = []
        for stripe in instate['moving_stripes'][strip_id]:
            # Update position
            stripe['position'] += stripe['speed'] * stripe['direction'] * delta_time
            
            # Update age
            stripe['age'] += delta_time
            
            # Check if stripe is still visible and within lifetime
            if stripe['age'] < stripe['lifetime']:
                new_stripes.append(stripe)
                
                # Calculate alpha based on age
                if stripe['age'] < stripe['lifetime'] * 0.2:
                    # Fade in
                    alpha_factor = stripe['age'] / (stripe['lifetime'] * 0.2)
                else:
                    # Fade out gradually
                    alpha_factor = 1.0 - ((stripe['age'] - stripe['lifetime'] * 0.2) / 
                                         (stripe['lifetime'] * 0.8))
                
                # Calculate final alpha
                alpha = stripe['alpha'] * alpha_factor
                
                # Draw the stripe with fading trail
                draw_stripe(buffer, strip_length, stripe, alpha)
                
        # Update stripe list
        instate['moving_stripes'][strip_id] = new_stripes

def create_new_bloom(instate, strip_id, position, strip_length):
    """Helper function to create a new bloom point with randomized properties"""
    # Randomly choose between pink and blue for primary color
    if np.random.random() < 0.7:  # 70% pink, 30% blue
        primary_color = np.random.choice(['vivid_pink', 'soft_pink'])
    else:
        primary_color = np.random.choice(['deep_blue', 'midnight_blue'])
    
    # Random secondary color for variation
    if primary_color in ['vivid_pink', 'soft_pink']:
        # Pink primary, blue secondary
        secondary_color = np.random.choice(['deep_blue', 'midnight_blue'])
    else:
        # Blue primary, pink secondary
        secondary_color = np.random.choice(['vivid_pink', 'soft_pink'])
    
    # Random growth time and lifetime
    growth_time = 2.0 + np.random.random() * 3.0  # 2-5 seconds to fully grow
    lifetime = growth_time + 3.0 + np.random.random() * 10.0  # Additional 3-13 seconds before fading
    
    # Create the bloom
    instate['bloom_points'][strip_id].append({
        'position': position,
        'max_radius': 5 + np.random.randint(5, max(6, int(strip_length * 0.15))),  # 5-15% of strip length
        'primary_color': primary_color,
        'secondary_color': secondary_color,
        'accent_color': 'leaf_green' if np.random.random() < 0.7 else 'light_green',  # Green accent
        'layers': 2 + np.random.randint(0, 3),  # 2-4 layers
        'age': 0.0,
        'growth_time': growth_time,
        'lifetime': lifetime,
        'pulse_rate': 1.0 + np.random.random() * 2.0,  # 1-3 Hz pulse when mature
        'rotation': np.random.random() * 2 * np.pi  # Random initial rotation
    })

def draw_bloom(buffer, strip_length, bloom, radius, alpha_factor, colors):
    """Draws a floral bloom with multiple layers and colors"""
    center = int(bloom['position'])
    radius_int = int(radius)
    
    # Get colors from palette
    primary_h, primary_s, primary_v = colors[bloom['primary_color']]
    secondary_h, secondary_s, secondary_v = colors[bloom['secondary_color']]
    accent_h, accent_s, accent_v = colors[bloom['accent_color']]
    
    # Calculate layer spacing
    layer_spacing = max(1, radius_int // bloom['layers'])
    
    # Draw layers from outside in
    for layer in range(bloom['layers']):
        # Calculate layer radius
        layer_radius = radius_int - layer * layer_spacing
        if layer_radius <= 0:
            continue
            
        # Calculate layer properties
        layer_norm = layer / max(1, bloom['layers'] - 1)  # 0.0 for outer layer, 1.0 for inner
        
        # Interpolate colors between primary and secondary
        h = primary_h * (1.0 - layer_norm) + secondary_h * layer_norm
        s = primary_s * (1.0 - layer_norm) + secondary_s * layer_norm
        v = primary_v * (1.0 - layer_norm) + secondary_v * layer_norm
        
        # Inner layers are brighter
        v = min(1.0, v * (1.0 + layer_norm * 0.3))
        
        # For the innermost layer, blend with white for a glowing center
        if layer == bloom['layers'] - 1:
            s *= 0.6  # Reduce saturation
            v = min(1.0, v * 1.2)  # Increase brightness
        
        # Convert to RGB
        r, g, b = hsv_to_rgb(h, s, v)
        
        # Calculate layer alpha - inner layers more transparent for a glowing effect
        layer_alpha = (0.7 + 0.3 * layer_norm) * alpha_factor
        
        # Draw the layer with petal-like pattern
        num_petals = 5 + layer  # More petals on outer layers
        rotation = bloom['rotation'] + layer * 0.2  # Rotate each layer slightly
        
        for i in range(-layer_radius, layer_radius + 1):
            pixel_pos = (center + i) % strip_length
            
            # Calculate distance from center
            dist = abs(i) / layer_radius
            
            # Calculate angle from center
            angle = np.arccos(1.0 - dist) if dist < 1.0 else 0
            angle = (angle + rotation) % (2 * np.pi)
            
            # Create petal pattern
            petal_factor = 0.5 + 0.5 * np.cos(angle * num_petals)
            
            # Calculate pixel intensity - stronger at petal centers
            intensity = (1.0 - dist**2) * petal_factor
            
            # Skip if too dim
            if intensity < 0.05:
                continue
                
            # Apply color with intensity
            pixel_alpha = layer_alpha * intensity
            
            # Occasionally add green accent color for leaf-like details
            if layer < bloom['layers'] - 1 and np.random.random() < 0.1:
                r, g, b = hsv_to_rgb(accent_h, accent_s, accent_v)
                
            # Blend with existing pixel
            curr_r, curr_g, curr_b, curr_a = buffer[pixel_pos]
            new_r = max(curr_r, r * intensity)
            new_g = max(curr_g, g * intensity)
            new_b = max(curr_b, b * intensity)
            new_a = max(curr_a, pixel_alpha)
            
            buffer[pixel_pos] = [new_r, new_g, new_b, new_a]

def draw_stripe(buffer, strip_length, stripe, alpha):
    """Draws a moving stripe with a fading trail"""
    # Get stripe properties
    pos = int(stripe['position']) % strip_length
    length = stripe['length']
    direction = stripe['direction']
    h, s, v = stripe['h'], stripe['s'], stripe['v']
    
    # Convert base color to RGB
    base_r, base_g, base_b = hsv_to_rgb(h, s, v)
    
    # Draw the stripe with fading trail
    for i in range(length):
        # Calculate position with direction
        pixel_pos = (pos - i * direction) % strip_length
        
        # Calculate intensity based on position in stripe
        # Head of stripe is brightest, fades toward tail
        intensity = 1.0 - (i / length)**1.5
        
        # Skip if too dim
        if intensity < 0.05:
            continue
            
        # Calculate color with intensity
        r = base_r * intensity
        g = base_g * intensity
        b = base_b * intensity
        a = alpha * intensity
        
        # Blend with existing pixel
        curr_r, curr_g, curr_b, curr_a = buffer[pixel_pos]
        new_r = max(curr_r, r)
        new_g = max(curr_g, g)
        new_b = max(curr_b, b)
        new_a = max(curr_a, a)
        
        buffer[pixel_pos] = [new_r, new_g, new_b, new_a]

# ... existing code ...

def OTO_rage_lightning(instate, outstate):
    """
    Generator function that creates a passionate rage-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_rage'] value
    2. Fast blinking lightning effects in yellows, blues, and whites
    3. High-speed component causing entire strips to flash with noisy variations
    4. Rapid changes in which strips are activated to create chaotic, angry pattern
    5. Consistent low-level noise across all pixels for added intensity
    6. Intense flashing reminiscent of electrical storms and passionate rage
    """
    name = 'rage_lightning'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['flash_timer'] = 0.0       # Timer for rapid flashes
        instate['strip_change_timer'] = 0.0  # Timer for changing active strips
        instate['active_strips'] = []      # Currently active strips
        instate['flash_state'] = False     # Current flash state (on/off)
        instate['last_flash_time'] = 0.0   # Time of last flash change
        
        # Color palette (HSV values)
        instate['colors'] = {
            'electric_blue': [0.6, 0.85, 1.0],    # Intense blue
            'bright_yellow': [0.15, 0.8, 1.0],    # Bright yellow
            'white_hot': [0.0, 0.0, 1.0],         # Pure white
            'light_blue': [0.55, 0.7, 1.0],       # Light blue
            'pale_yellow': [0.13, 0.5, 1.0]       # Pale yellow
        }
        
        # Timing parameters
        instate['min_flash_time'] = 0.1    # Minimum time between flashes (seconds)
        instate['max_flash_time'] = 0.4    # Maximum time between flashes (seconds)
        instate['strip_change_time'] = 0.3  # Time between changing active strips (seconds)
        instate['active_strip_percent'] = 0.3  # Percentage of strips active at once
        
        # Noise parameters
        instate['base_noise_min'] = 0.2    # Minimum noise intensity (20%)
        instate['base_noise_max'] = 0.4    # Maximum noise intensity (40%)
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get rage level from outstate (default to 0)
    rage_level = outstate.get('control_rage', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = rage_level
    
    # Skip rendering if alpha is too low
    if rage_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * rage_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    current_time = outstate['current_time']
    
    # Update flash timer - controls rapid flash component
    instate['flash_timer'] += delta_time
    
    # Determine if it's time for a new flash
    time_since_last_flash = current_time - instate['last_flash_time']
    flash_interval = instate['min_flash_time'] + np.random.random() * (instate['max_flash_time'] - instate['min_flash_time'])
    
    # Higher rage intensifies flashing (shorter intervals)
    flash_interval = flash_interval * (1.0 - rage_level * 0.5)
    
    if time_since_last_flash >= flash_interval:
        # Time for a new flash state
        instate['last_flash_time'] = current_time
        instate['flash_state'] = not instate['flash_state']
    
    # Update strip change timer - controls which strips are active
    instate['strip_change_timer'] += delta_time
    
    # Check if it's time to change active strips
    strip_change_time = instate['strip_change_time'] * (1.0 - rage_level * 0.5)  # Faster changes with higher rage
    
    if instate['strip_change_timer'] >= strip_change_time:
        instate['strip_change_timer'] = 0.0
        
        # Select new active strips
        all_strips = list(strip_manager.strips.keys())
        if all_strips:
            # Calculate how many strips to activate
            active_percent = instate['active_strip_percent'] * (1.0 + rage_level * 0.5)  # More active strips with higher rage
            active_percent = min(0.8, active_percent)  # Cap at 80%
            
            num_to_select = max(1, int(len(all_strips) * active_percent))
            instate['active_strips'] = np.random.choice(all_strips, num_to_select, replace=False)
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip_length = len(buffer)
        
        # Determine if this is an active strip
        is_active = strip_id in instate['active_strips']
        
        # Start with a dark base - slight blue tint
        buffer[:] = [0.0, 0.0, 0.1, 0.1]  # Very dim blue base
        
        if is_active:
            # This strip is active
            
            if instate['flash_state']:
                # Flash is on - light up the entire strip with noise
                
                # Choose a base color for this strip
                color_name = np.random.choice(list(instate['colors'].keys()))
                h, s, v = instate['colors'][color_name]
                
                # Generate base RGB values
                r, g, b = hsv_to_rgb(h, s, v)
                
                # Create noise variation across the strip
                for i in range(strip_length):
                    # Add noise to color (more noise with higher rage)
                    noise_amount = 0.2 + rage_level * 0.3
                    r_noise = r * (1.0 - noise_amount + np.random.random() * noise_amount * 2)
                    g_noise = g * (1.0 - noise_amount + np.random.random() * noise_amount * 2)
                    b_noise = b * (1.0 - noise_amount + np.random.random() * noise_amount * 2)
                    
                    # Ensure values are in valid range
                    r_noise = max(0.0, min(1.0, r_noise))
                    g_noise = max(0.0, min(1.0, g_noise))
                    b_noise = max(0.0, min(1.0, b_noise))
                    
                    # Set alpha - also with some variation
                    alpha = 0.7 + np.random.random() * 0.3
                    
                    # Add to buffer
                    buffer[i] = [r_noise, g_noise, b_noise, alpha]
                
                # Add some brighter spots (more with higher rage)
                num_bright_spots = int(strip_length * (0.1 + rage_level * 0.2))
                for _ in range(num_bright_spots):
                    pos = np.random.randint(0, strip_length)
                    
                    # Brighter version of the base color
                    br, bg, bb = hsv_to_rgb(h, s * 0.7, min(1.0, v * 1.3))  # Less saturated, brighter
                    
                    buffer[pos] = [br, bg, bb, 1.0]
            else:
                # Flash is off but strip is active - add subtle ambient glow
                for i in range(strip_length):
                    # Add small ambient effect
                    flicker = 0.05 + 0.05 * np.sin(current_time * 10 + i * 0.1)
                    
                    # Use blue for ambient glow
                    h, s, v = instate['colors']['electric_blue']
                    r, g, b = hsv_to_rgb(h, s * 0.7, v * 0.4)
                    
                    # Set pixel with low intensity but higher than non-active strips
                    buffer[i] = [r, g, b, 0.2 + flicker]
        else:
            # Non-active strip - very dim ambient only
            # But occasionally (based on rage) flash briefly
            random_flash = np.random.random() < (0.01 * rage_level)
            
            if random_flash:
                # Brief random flash on non-active strip
                color_name = np.random.choice(['electric_blue', 'light_blue'])
                h, s, v = instate['colors'][color_name]
                r, g, b = hsv_to_rgb(h, s, v)
                
                # Lower intensity than active strips
                for i in range(strip_length):
                    noise = 0.7 + np.random.random() * 0.3
                    buffer[i] = [r * noise, g * noise, b * noise, 0.4]
        
        # Apply additional base noise to EVERY pixel on EVERY strip
        # This adds the consistent 20-40% intensity noise across all pixels
        for i in range(strip_length):
            # Get current pixel values
            curr_r, curr_g, curr_b, curr_a = buffer[i]
            
            # Generate random color for noise
            noise_color = np.random.choice(list(instate['colors'].keys()))
            h, s, v = instate['colors'][noise_color]
            noise_r, noise_g, noise_b = hsv_to_rgb(h, s, v)
            
            # Calculate noise intensity (20-40% range)
            noise_intensity = instate['base_noise_min'] + np.random.random() * (instate['base_noise_max'] - instate['base_noise_min'])
            
            # Increase noise with rage level
            noise_intensity *= (1.0 + rage_level * 0.5)
            noise_intensity = min(0.6, noise_intensity)  # Cap at 60% for high rage
            
            # Apply noise with additive blending
            new_r = min(1.0, curr_r + (noise_r * noise_intensity))
            new_g = min(1.0, curr_g + (noise_g * noise_intensity))
            new_b = min(1.0, curr_b + (noise_b * noise_intensity))
            new_a = max(curr_a, noise_intensity)
            
            buffer[i] = [new_r, new_g, new_b, new_a]
            
def OTO_contemplative_cosmic(instate, outstate):
    """
    Generator function that creates a contemplative cosmic-themed pattern.
    
    Features:
    1. Global alpha controlled by outstate['control_contemplative'] value
    2. Prominent, flowing nebular color fields (blues, greens, oranges) that blend and smear
    3. Slow-moving field of blinking stars (white)
    4. Green moving splotches resembling cosmic gas clouds
    5. Green/yellow blinking fireflies that move around
    
    The overall effect is a peaceful cosmic environment that evokes 
    contemplation and wonder with vivid, dynamic color flows.
    """
    name = 'contemplative_cosmic'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['stars'] = {}           # Track stars across strips
        instate['splotches'] = {}       # Track green splotches
        instate['nebulae'] = {}         # Track large nebular regions
        instate['color_flows'] = {}     # Track flowing color smears
        instate['fireflies'] = {}       # Track moving fireflies
        
        # Color palette (HSV values)
        instate['colors'] = {
            'white_star': [0.0, 0.0, 1.0],          # Pure white
            'blue_deep': [0.6, 0.8, 0.6],           # Deep blue
            'blue_light': [0.55, 0.6, 0.9],         # Light blue
            'orange_nebula': [0.08, 0.7, 0.8],      # Orange nebula
            'orange_deep': [0.05, 0.85, 0.7],       # Deep orange
            'green_cosmic': [0.35, 0.8, 0.7],       # Cosmic green
            'green_deep': [0.3, 0.9, 0.6],          # Deep green
            'yellow_glow': [0.15, 0.8, 0.9],        # Yellow glow
            'green_yellow': [0.2, 0.85, 0.8],       # Green-yellow
            'purple_cosmic': [0.75, 0.7, 0.6],      # Cosmic purple
            'teal_cosmic': [0.45, 0.7, 0.7]         # Cosmic teal
        }
        
        # Global parameters for nebular flows
        instate['flow_speed_base'] = 2.0      # Base speed for color flows
        instate['flow_update_timer'] = 0.0    # Timer for updating global flow parameters
        instate['global_flow_direction'] = 1  # Global direction that can change occasionally
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get contemplative level from outstate (default to 0)
    contemplative_level = outstate.get('control_contemplative', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = contemplative_level
    
    # Skip rendering if alpha is too low
    if contemplative_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * contemplative_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    current_time = outstate['current_time']
    
    # Update global flow parameters
    instate['flow_update_timer'] += delta_time
    if instate['flow_update_timer'] > 15.0:  # Every 15 seconds, consider changing global flow direction
        instate['flow_update_timer'] = 0.0
        if np.random.random() < 0.4:  # 40% chance to reverse global flow
            instate['global_flow_direction'] *= -1
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        
        # Start with a dark base - very deep blue
        buffer[:] = [0.0, 0.0, 0.1, 0.2]  # Very dim blue base
        
        # Initialize elements for this strip if not already done
        if strip_id not in instate['stars']:
            # Create stars
            instate['stars'][strip_id] = []
            num_stars = max(3, strip_length // 5)  # Roughly 1 star per 10 pixels
            
            for _ in range(num_stars):
                pos = np.random.randint(0, strip_length)
                instate['stars'][strip_id].append({
                    'position': pos,
                    'brightness': 0.3 + np.random.random() * 0.7,  # 0.3-1.0 brightness
                    'blink_rate': 0.2 + np.random.random() * 1.0,  # 0.2-1.2 Hz
                    'blink_offset': np.random.random() * 2 * np.pi,  # Random phase
                    'movement_speed': 0.5 + np.random.random() * 1.5,  # 0.5-2.0 pixels per second (slow)
                    'direction': 1 if np.random.random() < 0.5 else -1  # Random direction
                })
            
            # Create green splotches
            instate['splotches'][strip_id] = []
            num_splotches = max(1, strip_length // 40)  # Fewer splotches, 1 per ~40 pixels
            
            for _ in range(num_splotches):
                pos = np.random.randint(0, strip_length)
                instate['splotches'][strip_id].append({
                    'position': pos,
                    'size': 5 + np.random.randint(0, 10),  # 5-15 pixels
                    'speed': 1.0 + np.random.random() * 2.0,  # 1-3 pixels per second (slow)
                    'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                    'color': np.random.choice(['green_cosmic', 'green_deep']),  # Random green shade
                    'intensity': 0.5 + np.random.random() * 0.5,  # 0.5-1.0 intensity
                    'pulse_rate': 0.1 + np.random.random() * 0.3,  # 0.1-0.4 Hz (slow pulse)
                    'pulse_offset': np.random.random() * 2 * np.pi  # Random phase
                })
            
            # Create large nebulae - main colored regions
            instate['nebulae'][strip_id] = []
            num_nebulae = max(2, strip_length // 30)  # More nebulae: 1 per ~30 pixels
            
            for _ in range(num_nebulae):
                pos = np.random.randint(0, strip_length)
                color = np.random.choice(['blue_deep', 'blue_light', 'orange_nebula', 
                                         'green_cosmic', 'purple_cosmic', 'teal_cosmic'])
                
                instate['nebulae'][strip_id].append({
                    'position': pos,
                    'size': 20 + np.random.randint(0, 30),  # 20-50 pixels (much larger)
                    'speed': 0.8 + np.random.random() * 1.2,  # 0.8-2.0 pixels per second
                    'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                    'color': color,
                    'intensity': 0.5 + np.random.random() * 0.5,  # 0.5-1.0 intensity (much brighter)
                    'pulse_rate': 0.05 + np.random.random() * 0.15,  # 0.05-0.2 Hz (very slow pulse)
                    'pulse_offset': np.random.random() * 2 * np.pi  # Random phase
                })
            
            # Create color flows - dynamic smearing nebular effects
            instate['color_flows'][strip_id] = []
            
            # Create multiple color flow layers with different characteristics
            num_flow_layers = 4 + np.random.randint(0, 3)  # 4-6 layers (more layers)
            
            for _ in range(num_flow_layers):
                # Choose complementary colors for interesting blends
                color_pairs = [
                    ('blue_deep', 'orange_nebula'),
                    ('blue_light', 'orange_deep'),
                    ('green_cosmic', 'purple_cosmic'),
                    ('green_deep', 'blue_deep'),
                    ('yellow_glow', 'blue_light'),
                    ('teal_cosmic', 'orange_deep'),
                    ('green_yellow', 'purple_cosmic')
                ]
                color_pair = color_pairs[np.random.randint(0, len(color_pairs))]
                
                # Create a flow layer
                instate['color_flows'][strip_id].append({
                    'color1': color_pair[0],
                    'color2': color_pair[1],
                    'offset': np.random.random() * strip_length,  # Random starting position
                    'scale': 15 + np.random.random() * 45,  # 15-60 pixel scale (wavelength)
                    'speed': instate['flow_speed_base'] * (0.5 + np.random.random()),  # Varied speeds
                    'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                    'amplitude': 0.4 + np.random.random() * 0.6,  # 0.4-1.0 intensity (much stronger)
                    'phase_shift': np.random.random() * 2 * np.pi,  # Random phase
                    'drift_rate': 0.1 + np.random.random() * 0.3,  # 0.1-0.4 units per second
                    'distortion': 0.5 + np.random.random() * 1.5  # 0.5-2.0 distortion factor
                })
            
            # Create fireflies
            instate['fireflies'][strip_id] = []
            num_fireflies = max(2, strip_length // 15)  # 1 per ~30 pixels
            
            for _ in range(num_fireflies):
                pos = np.random.randint(0, strip_length)
                color = np.random.choice(['green_yellow', 'yellow_glow'])
                
                instate['fireflies'][strip_id].append({
                    'position': pos,
                    'speed': 5.0 + np.random.random() * 15.0,  # 3-8 pixels per second (faster than other elements)
                    'direction': 1 if np.random.random() < 0.5 else -1,  # Random direction
                    'color': color,
                    'blink_rate': 0.5 + np.random.random() * 1.5,  # 0.5-2.0 Hz
                    'blink_offset': np.random.random() * 2 * np.pi,  # Random phase
                    'blink_duration': 0.7 + np.random.random() * 0.8,  # 0.2-0.5 second blinks
                    'last_direction_change': current_time,
                    'direction_change_interval': 2.0 + np.random.random() * 4.0  # 2-6 seconds between direction changes
                })
        
        # Set up a new array to collect all color contributions for proper blending
        # Using numpy for efficient operations
        r_values = np.zeros(strip_length)
        g_values = np.zeros(strip_length)
        b_values = np.zeros(strip_length)
        a_values = np.zeros(strip_length)
        
        # Render large nebulae (largest, slowest color regions) first
        for nebula in instate['nebulae'][strip_id]:
            # Update position (slow movement)
            nebula['position'] += nebula['speed'] * nebula['direction'] * delta_time
            
            # Handle wrapping
            nebula['position'] %= strip_length
            
            # Calculate pulse effect (very slow and subtle)
            pulse_factor = 0.8 + 0.2 * (0.5 + 0.5 * np.sin(current_time * nebula['pulse_rate'] * 2 * np.pi + nebula['pulse_offset']))
            
            # Calculate final intensity
            intensity = nebula['intensity'] * pulse_factor
            
            # Get color
            h, s, v = instate['colors'][nebula['color']]
            
            # Convert to RGB once for efficiency
            base_r, base_g, base_b = hsv_to_rgb(h, s, v)
            
            # Draw the nebula with soft gradient
            center = int(nebula['position'])
            size = nebula['size']
            
            # Create pixel positions array with wrapping
            positions = np.arange(center - size, center + size + 1) % strip_length
            
            # Calculate distances from center
            distances = np.abs(np.arange(-size, size + 1)) / size
            
            # Create softened edge profile for organic look
            # Use a power function for a more natural falloff
            edge_profile = np.power(1.0 - distances, 2) * intensity
            
            # Add some noise for texture
            noise = np.random.random(len(edge_profile)) * 0.15 - 0.075  # ±7.5% noise
            edge_profile = np.clip(edge_profile + noise, 0, 1)
            
            # Apply to color arrays
            for i, pos in enumerate(positions):
                if edge_profile[i] > 0.01:  # Skip very low intensity
                    r_values[pos] += base_r * edge_profile[i] * 0.7  # Reduce contribution to allow blending
                    g_values[pos] += base_g * edge_profile[i] * 0.7
                    b_values[pos] += base_b * edge_profile[i] * 0.7
                    a_values[pos] += edge_profile[i] * 0.6  # More visible but still allows blending
        
        # Render dynamic color flows (nebular smearing)
        for flow in instate['color_flows'][strip_id]:
            # Update flow position
            flow_direction = flow['direction'] * instate['global_flow_direction']  # Combine with global direction
            flow['offset'] += flow['speed'] * flow_direction * delta_time
            flow['phase_shift'] += flow['drift_rate'] * delta_time
            
            # Keep offset in range
            flow['offset'] %= strip_length
            
            # Get the two colors
            h1, s1, v1 = instate['colors'][flow['color1']]
            h2, s2, v2 = instate['colors'][flow['color2']]
            
            # Create flow pattern across the strip - vectorized for efficiency
            positions = np.arange(strip_length)
            
            # Calculate normalized positions in the flow
            pos = (positions + flow['offset']) % strip_length
            pos_norm = pos / flow['scale']
            
            # Add distortion for more organic appearance
            distortion = np.sin(pos_norm * 0.5 + flow['phase_shift']) * flow['distortion']
            
            # Calculate main flow pattern
            flow_values = 0.5 + 0.5 * np.sin(pos_norm * 2 * np.pi + flow['phase_shift'] + distortion)
            
            # Apply amplitude modulation
            intensities = flow_values * flow['amplitude']
            
            # Create mask for pixels to update (where intensity is significant)
            mask = intensities > 0.1
            
            if np.any(mask):
                # Handle hue interpolation for wrapped color space
                h_diff = h2 - h1
                if abs(h_diff) > 0.5:
                    if h1 > h2:
                        h2 += 1.0
                    else:
                        h1 += 1.0
                
                # Work with the masked values for efficiency
                flow_vals_masked = flow_values[mask]
                
                # Interpolate colors (vectorized)
                h_values = (h1 * (1.0 - flow_vals_masked) + h2 * flow_vals_masked) % 1.0
                s_values = s1 * (1.0 - flow_vals_masked) + s2 * flow_vals_masked
                v_values = v1 * (1.0 - flow_vals_masked) + v2 * flow_vals_masked
                
                # Convert to RGB (need to do this per-pixel due to hsv_to_rgb limitations)
                for i, idx in enumerate(positions[mask]):
                    r, g, b = hsv_to_rgb(h_values[i], s_values[i], v_values[i])
                    
                    # Calculate intensity
                    intensity = intensities[idx]
                    
                    # Add to color arrays with stronger contribution
                    r_values[idx] += r * intensity * 0.8
                    g_values[idx] += g * intensity * 0.8
                    b_values[idx] += b * intensity * 0.8
                    a_values[idx] += intensity * 0.7  # Stronger alpha
        
        # Apply combined nebular effects to buffer
        for i in range(strip_length):
            if a_values[i] > 0:
                # Ensure values are in valid range
                r = min(1.0, r_values[i])
                g = min(1.0, g_values[i])
                b = min(1.0, b_values[i])
                a = min(0.45, a_values[i])  # Cap alpha to allow for stars and fireflies
                
                # Set buffer with nebular colors
                buffer[i] = [r, g, b, a]
        
        # Update and render stars
        for star in instate['stars'][strip_id]:
            # Update position (slow movement)
            star['position'] += star['movement_speed'] * star['direction'] * delta_time
            
            # Handle wrapping
            star['position'] %= strip_length
            
            # Calculate blink effect
            blink_factor = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(current_time * star['blink_rate'] * 2 * np.pi + star['blink_offset']))
            
            # Calculate final brightness
            brightness = star['brightness'] * blink_factor
            
            # Draw the star
            pos_int = int(star['position'])
            
            # Get base color (white)
            h, s, v = instate['colors']['white_star']
            r, g, b = hsv_to_rgb(h, s, v * brightness)
            
            # Set pixel with alpha based on brightness
            alpha = 0.3 + 0.7 * brightness
            
            # Add to buffer with additive blending
            curr_r, curr_g, curr_b, curr_a = buffer[pos_int]
            buffer[pos_int] = [
                max(curr_r, r),
                max(curr_g, g),
                max(curr_b, b),
                max(curr_a, alpha)
            ]
            
            # Add small glow around bright stars
            if brightness > 0.7:
                for offset in [-1, 1]:
                    glow_pos = (pos_int + offset) % strip_length
                    glow_alpha = alpha * 0.5  # Half the brightness
                    
                    curr_r, curr_g, curr_b, curr_a = buffer[glow_pos]
                    buffer[glow_pos] = [
                        max(curr_r, r * 0.5),
                        max(curr_g, g * 0.5),
                        max(curr_b, b * 0.5),
                        max(curr_a, glow_alpha)
                    ]
        
        # Update and render green splotches
        for splotch in instate['splotches'][strip_id]:
            # Update position
            splotch['position'] += splotch['speed'] * splotch['direction'] * delta_time
            
            # Handle wrapping
            splotch['position'] %= strip_length
            
            # Calculate pulse effect
            pulse_factor = 0.7 + 0.3 * (0.5 + 0.5 * np.sin(current_time * splotch['pulse_rate'] * 2 * np.pi + splotch['pulse_offset']))
            
            # Calculate final intensity
            intensity = splotch['intensity'] * pulse_factor
            
            # Draw the splotch as a gradient
            center = int(splotch['position'])
            size = splotch['size']
            
            # Get color
            h, s, v = instate['colors'][splotch['color']]
            
            # Draw splotch with gradient falloff
            for i in range(-size, size + 1):
                pixel_pos = (center + i) % strip_length
                
                # Calculate distance from center (normalized)
                dist = abs(i) / size
                
                # Skip if too far
                if dist > 1.0:
                    continue
                
                # Calculate intensity based on distance from center
                pixel_intensity = intensity * (1.0 - dist**2)  # Quadratic falloff
                
                # Calculate color
                r, g, b = hsv_to_rgb(h, s, v * pixel_intensity)
                alpha = pixel_intensity * 0.7  # Somewhat transparent
                
                # Add to buffer with additive blending for glow effect
                curr_r, curr_g, curr_b, curr_a = buffer[pixel_pos]
                buffer[pixel_pos] = [
                    max(curr_r, r),
                    max(curr_g, g),
                    max(curr_b, b),
                    max(curr_a, alpha)
                ]
        
        # Update and render fireflies
        new_fireflies = []
        for firefly in instate['fireflies'][strip_id]:
            # Check if it's time to change direction
            if current_time - firefly['last_direction_change'] > firefly['direction_change_interval']:
                firefly['direction'] *= -1  # Reverse direction
                firefly['last_direction_change'] = current_time
                firefly['direction_change_interval'] = 2.0 + np.random.random() * 4.0  # New interval
            
            # Update position
            firefly['position'] += firefly['speed'] * firefly['direction'] * delta_time
            
            # Handle wrapping
            firefly['position'] %= strip_length
            
            # Calculate blink effect - more distinct on/off for fireflies
            blink_phase = (current_time * firefly['blink_rate']) % 1.0
            
            # Fireflies have a brief "on" period
            is_on = blink_phase < firefly['blink_duration']
            
            # Keep all fireflies
            new_fireflies.append(firefly)
            
            if is_on:
                # Draw the firefly when it's on
                pos_int = int(firefly['position'])
                
                # Get color
                h, s, v = instate['colors'][firefly['color']]
                
                # Make color brighter for fireflies
                v = min(1.0, v * 1.2)
                
                # Convert to RGB
                r, g, b = hsv_to_rgb(h, s, v)
                
                # Set pixel with high alpha
                alpha = 0.9
                
                # Add to buffer with additive blending
                curr_r, curr_g, curr_b, curr_a = buffer[pos_int]
                buffer[pos_int] = [
                    max(curr_r, r),
                    max(curr_g, g),
                    max(curr_b, b),
                    max(curr_a, alpha)
                ]
                
                # Add small glow around firefly
                for offset in [-1, 1]:
                    glow_pos = (pos_int + offset) % strip_length
                    
                    # Glow is more green/yellow regardless of firefly color
                    glow_h, glow_s, glow_v = instate['colors']['green_yellow']
                    glow_r, glow_g, glow_b = hsv_to_rgb(glow_h, glow_s * 0.7, glow_v * 0.7)
                    
                    glow_alpha = 0.6  # Noticeable glow
                    
                    # Add to buffer with additive blending
                    curr_r, curr_g, curr_b, curr_a = buffer[glow_pos]
                    buffer[glow_pos] = [
                        max(curr_r, glow_r),
                        max(curr_g, glow_g),
                        max(curr_b, glow_b),
                        max(curr_a, glow_alpha)
                    ]
        
        # Update firefly list
        instate['fireflies'][strip_id] = new_fireflies
        
def OTO_neutral_positive(instate, outstate):
    """
    Generator function that creates a neutral/positive-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_neutral'] value
    2. Evokes sun shining through green tree foliage and sunflowers
    3. Brain strips pulse with gold and yellow colors with occasional flashes
    4. Base strips show light blue with ripples like a shallow ocean viewed from underwater
    5. Overall creates a peaceful, natural, positive atmosphere
    
    Optimized with numpy vectorization for performance.
    """
    name = 'neutral_positive'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['sun_rays'] = {}          # Track sun ray effects
        instate['foliage'] = {}           # Track foliage patterns
        instate['sunflowers'] = {}        # Track sunflower patterns
        instate['brain_pulse'] = {}       # Track brain pulse state
        instate['water_ripples'] = {}     # Track water ripple effects
        instate['flash_timer'] = 0.0      # Timer for occasional flashes
        
        # Color palette (HSV values)
        instate['colors'] = {
            'sunshine_yellow': [0.13, 0.85, 0.95],   # Bright sunshine yellow
            'golden_sun': [0.11, 0.9, 0.9],          # Golden sunlight
            'leaf_green_light': [0.3, 0.7, 0.7],     # Light green foliage
            'leaf_green_dark': [0.35, 0.8, 0.5],     # Darker green foliage
            'sky_blue': [0.6, 0.3, 0.95],            # Light sky blue
            'water_blue': [0.55, 0.6, 0.8],          # Ocean water blue
            'sunflower_center': [0.09, 0.9, 0.7],    # Sunflower center brown
            'sunflower_petal': [0.13, 1.0, 1.0]      # Bright sunflower petal
        }
        
        # Initialize brain pulse parameters
        instate['brain_pulse_rate'] = 0.5           # Pulses per second
        instate['brain_flash_chance'] = 0.02        # Chance per second for a flash
        instate['global_sun_position'] = 0.0        # Global sun position (0.0-1.0)
        instate['sun_movement_speed'] = 0.05        # Speed of sun movement
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get neutral level from outstate (default to 0)
    neutral_level = outstate.get('control_neutral', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = neutral_level
    
    # Skip rendering if alpha is too low
    if neutral_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * neutral_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    current_time = outstate['current_time']
    
    # Update global sun position (0.0-1.0 cycle)
    instate['global_sun_position'] = (instate['global_sun_position'] + 
                                      instate['sun_movement_speed'] * delta_time) % 1.0
    
    # Update flash timer for brain effects
    instate['flash_timer'] += delta_time
    should_flash = False
    if instate['flash_timer'] >= 1.0:  # Check once per second
        instate['flash_timer'] = 0.0
        if np.random.random() < instate['brain_flash_chance']:
            should_flash = True
    
    # Calculate global sun visibility factor
    sun_factor = 0.5 + 0.5 * np.sin(instate['global_sun_position'] * 2 * np.pi)
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Process each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        strip = strip_manager.get_strip(strip_id)
        strip_length = len(buffer)
        strip_groups = strip.groups
        
        # Initialize elements for this strip if not already done
        if strip_id not in instate['sun_rays']:
            # Create sun rays
            instate['sun_rays'][strip_id] = []
            num_rays = max(2, strip_length // 20)  # About 1 ray per 20 pixels
            
            for _ in range(num_rays):
                pos = np.random.randint(0, strip_length)
                width = 3 + np.random.randint(0, 7)  # 3-10 pixels wide
                
                instate['sun_rays'][strip_id].append({
                    'position': pos,
                    'width': width,
                    'intensity': 0.6 + np.random.random() * 0.4,  # 0.6-1.0 intensity
                    'movement_speed': 1.0 + np.random.random() * 2.0,  # 1-3 pixels/sec
                    'direction': 1 if np.random.random() < 0.5 else -1  # Random direction
                })
            
            # Create foliage elements
            instate['foliage'][strip_id] = []
            num_leaves = max(3, strip_length // 15)  # About 1 leaf per 15 pixels
            
            for _ in range(num_leaves):
                pos = np.random.randint(0, strip_length)
                size = 5 + np.random.randint(0, 10)  # 5-15 pixels
                
                instate['foliage'][strip_id].append({
                    'position': pos,
                    'size': size,
                    'color': 'leaf_green_light' if np.random.random() < 0.6 else 'leaf_green_dark',
                    'sway_amplitude': 1.0 + np.random.random() * 2.0,  # 1-3 pixels
                    'sway_frequency': 0.2 + np.random.random() * 0.4,  # 0.2-0.6 Hz
                    'sway_offset': np.random.random() * 2 * np.pi  # Random phase
                })
            
            # Create sunflowers
            instate['sunflowers'][strip_id] = []
            num_flowers = max(1, strip_length // 40)  # About 1 flower per 40 pixels
            
            for _ in range(num_flowers):
                pos = np.random.randint(0, strip_length)
                size = 8 + np.random.randint(0, 7)  # 8-15 pixels
                
                instate['sunflowers'][strip_id].append({
                    'position': pos,
                    'size': size,
                    'rotate_speed': 0.1 + np.random.random() * 0.2,  # 0.1-0.3 radians/sec
                    'rotation': np.random.random() * 2 * np.pi,  # Random initial rotation
                    'sway_amplitude': 0.5 + np.random.random() * 1.5,  # 0.5-2.0 pixels
                    'sway_frequency': 0.1 + np.random.random() * 0.3,  # 0.1-0.4 Hz
                    'sway_offset': np.random.random() * 2 * np.pi  # Random phase
                })
            
            # Create water ripples for base strips
            instate['water_ripples'][strip_id] = []
            num_ripples = max(2, strip_length // 25)  # About 1 ripple per 25 pixels
            
            for _ in range(num_ripples):
                pos = np.random.randint(0, strip_length)
                wavelength = 15 + np.random.randint(0, 20)  # 15-35 pixels
                
                instate['water_ripples'][strip_id].append({
                    'position': pos,
                    'wavelength': wavelength,
                    'amplitude': 0.2 + np.random.random() * 0.3,  # 0.2-0.5 intensity variation
                    'speed': 5.0 + np.random.random() * 10.0,  # 5-15 pixels/sec
                    'direction': 1 if np.random.random() < 0.5 else -1  # Random direction
                })
            
            # Initialize brain pulse state
            if 'brain' in strip_groups:
                instate['brain_pulse'][strip_id] = {
                    'phase': np.random.random(),  # Random initial phase
                    'flash_active': False,
                    'flash_intensity': 0.0,
                    'flash_decay': 5.0  # Flash decay rate
                }
        
        # Determine strip type for customized effects
        is_base = 'base' in strip_groups
        is_brain = 'brain' in strip_groups
        
        # Create position array for the whole strip once
        positions = np.arange(strip_length)
        
        # Start with a base color depending on strip type
        # Start with a base color depending on strip type
        if is_base:
            # Light blue base for water effect
            h, s, v = instate['colors']['water_blue']
            r, g, b = hsv_to_rgb(h, s, v * 0.7)  # Slightly dimmed
            buffer[:] = [r, g, b, 0.5]
        elif is_brain:
            # Darker base for brain to allow pulses to stand out
            buffer[:] = [0.1, 0.1, 0.05, 0.1]
        else:
            # Sky blue base for other strips - brighter, more saturated sky blue
            h, s, v = instate['colors']['sky_blue']
            # Increase saturation and brightness for more vibrant sky blue
            s = min(1.0, s * 1.2)  # More saturated
            v = 0.9  # Brighter
            r, g, b = hsv_to_rgb(h, s, v)
            buffer[:] = [r, g, b, 0.5]  # Increased alpha for more visibility

            
        # Apply different effects based on strip type
        if is_base:
            # Water ripples for base strips - vectorized implementation
            
            # Initialize arrays for ripple and caustic effects
            ripple_effect = np.zeros(strip_length)
            
            # Update and apply all ripples
            for ripple in instate['water_ripples'][strip_id]:
                # Update ripple position
                ripple['position'] += ripple['speed'] * ripple['direction'] * delta_time
                ripple['position'] %= strip_length
                
                # Create distance array (accounting for wrapping)
                direct_dist = np.abs(positions - ripple['position'])
                wrapped_dist = strip_length - direct_dist
                distance = np.minimum(direct_dist, wrapped_dist)
                
                # Calculate ripple effect based on sine wave (vectorized)
                ripple_phase = distance / ripple['wavelength'] * 2 * np.pi
                ripple_effect += ripple['amplitude'] * np.sin(ripple_phase + current_time * 2.0)
            
            # Add underwater caustics effect (light patterns) - vectorized
            caustic_effect = 0.15 * np.sin(positions * 0.2 + current_time * 1.5) * np.cos(positions * 0.1 + current_time)
            
            # Combine effects
            total_effect = ripple_effect + caustic_effect
            
            # Extract current buffer values
            curr_r = buffer[:, 0]
            curr_g = buffer[:, 1]
            curr_b = buffer[:, 2]
            curr_a = buffer[:, 3]
            
            # Create masks for bright and dark areas
            bright_mask = total_effect > 0
            dark_mask = ~bright_mask
            
            # Apply effects to bright areas (vectorized)
            if np.any(bright_mask):
                # Shift toward white-blue for bright spots
                new_r = curr_r.copy()
                new_g = curr_g.copy()
                new_b = curr_b.copy()
                
                new_r[bright_mask] += total_effect[bright_mask] * 0.4
                new_g[bright_mask] += total_effect[bright_mask] * 0.5
                new_b[bright_mask] += total_effect[bright_mask] * 0.6
            
            # Apply effects to dark areas (vectorized)
            if np.any(dark_mask):
                # Shift toward deeper blue for darker spots
                new_r[dark_mask] = np.maximum(0, curr_r[dark_mask] + total_effect[dark_mask] * 0.3)
                new_g[dark_mask] = np.maximum(0, curr_g[dark_mask] + total_effect[dark_mask] * 0.4)
                new_b[dark_mask] = np.maximum(0, curr_b[dark_mask] + total_effect[dark_mask] * 0.3)
            
            # Ensure all values are in valid range (vectorized)
            new_r = np.clip(new_r, 0.0, 1.0)
            new_g = np.clip(new_g, 0.0, 1.0)
            new_b = np.clip(new_b, 0.0, 1.0)
            
            # Update buffer (vectorized)
            buffer[:, 0] = new_r
            buffer[:, 1] = new_g
            buffer[:, 2] = new_b
            
        elif is_brain:
            # Brain pulse effect with gold and yellow - vectorized implementation
            if strip_id in instate['brain_pulse']:
                pulse_state = instate['brain_pulse'][strip_id]
                
                # Update pulse phase
                pulse_state['phase'] += instate['brain_pulse_rate'] * delta_time
                pulse_state['phase'] %= 1.0
                
                # Calculate pulse intensity (0.0-1.0)
                phase = pulse_state['phase']
                if phase < 0.3:
                    # Rising phase
                    pulse_intensity = phase / 0.3
                elif phase < 0.5:
                    # Peak phase
                    pulse_intensity = 1.0
                else:
                    # Falling phase
                    pulse_intensity = 1.0 - ((phase - 0.5) / 0.5)
                
                # Handle flash effect
                if should_flash and not pulse_state['flash_active']:
                    pulse_state['flash_active'] = True
                    pulse_state['flash_intensity'] = 1.0
                
                if pulse_state['flash_active']:
                    # Update flash intensity
                    pulse_state['flash_intensity'] -= pulse_state['flash_decay'] * delta_time
                    
                    if pulse_state['flash_intensity'] <= 0:
                        pulse_state['flash_active'] = False
                        pulse_state['flash_intensity'] = 0.0
                
                # Combine regular pulse with flash
                combined_intensity = max(pulse_intensity, pulse_state['flash_intensity'])
                
                # Create spatial variation (vectorized)
                position_factor = 0.8 + 0.2 * np.sin(positions * 0.2 + current_time)
                
                # Create alternating color mask
                alternate_mask = (positions % 2 == 0)
                
                # Create color arrays
                h_values = np.zeros(strip_length)
                s_values = np.zeros(strip_length)
                v_values = np.zeros(strip_length)
                
                # Set colors based on alternating pattern
                h_golden, s_golden, v_golden = instate['colors']['golden_sun']
                h_sunshine, s_sunshine, v_sunshine = instate['colors']['sunshine_yellow']
                
                # Apply alternating colors (vectorized)
                h_values[alternate_mask] = h_golden
                s_values[alternate_mask] = s_golden
                v_values[alternate_mask] = v_golden
                
                h_values[~alternate_mask] = h_sunshine
                s_values[~alternate_mask] = s_sunshine
                v_values[~alternate_mask] = v_sunshine
                
                # If flash is active, use golden color everywhere
                if pulse_state['flash_active']:
                    h_values[:] = h_golden
                    s_values[:] = s_golden
                    v_values[:] = v_golden
                
                # Adjust brightness based on pulse (vectorized)
                v_adjusted = v_values * (0.3 + 0.7 * combined_intensity * position_factor)
                
                # Convert to RGB
                # Unfortunately we need to convert each pixel individually
                r_values = np.zeros(strip_length)
                g_values = np.zeros(strip_length)
                b_values = np.zeros(strip_length)
                
                for i in range(strip_length):
                    r, g, b = hsv_to_rgb(h_values[i], s_values[i], v_adjusted[i])
                    r_values[i] = r
                    g_values[i] = g
                    b_values[i] = b
                
                # Set alpha based on intensity (vectorized)
                alpha_values = 0.2 + 0.8 * combined_intensity * position_factor
                
                # Update buffer (vectorized)
                buffer[:, 0] = r_values
                buffer[:, 1] = g_values
                buffer[:, 2] = b_values
                buffer[:, 3] = alpha_values
            
        else:
            # Default effect: Sunlight through foliage - vectorized implementation
            
            # Initialize buffer to capture foliage
            foliage_r = np.zeros(strip_length)
            foliage_g = np.zeros(strip_length)
            foliage_b = np.zeros(strip_length)
            foliage_a = np.zeros(strip_length)
            
            # Draw foliage first as background
            for leaf in instate['foliage'][strip_id]:
                # Calculate swaying motion
                sway = leaf['sway_amplitude'] * np.sin(current_time * leaf['sway_frequency'] * 2 * np.pi + 
                                                      leaf['sway_offset'])
                
                # Calculate actual position with sway
                actual_pos = (leaf['position'] + sway) % strip_length
                center = int(actual_pos)
                size = leaf['size']
                
                # Create range of positions for this leaf
                leaf_range = np.arange(center - size, center + size + 1) % strip_length
                
                # Calculate distances from center (normalized)
                rel_positions = np.arange(-size, size + 1)
                distances = np.abs(rel_positions) / size
                
                # Calculate leaf shape factor - more oval than circular (vectorized)
                valid_mask = distances <= 1.0
                shape_factors = np.zeros_like(distances)
                shape_factors[valid_mask] = 1.0 - distances[valid_mask]**1.5
                
                # Calculate final intensities
                intensities = shape_factors * 0.7  # Leaves are partially transparent
                
                # Add some texture/variation to the leaf (vectorized)
                texture = 0.15 * np.sin(rel_positions * 0.8 + leaf['position'] * 0.3)
                
                # Get leaf color
                h, s, v = instate['colors'][leaf['color']]
                
                # Apply texture to color
                v_adjusted = np.clip(v * (1.0 + texture), 0.0, 1.0)
                
                # Convert to RGB (need to do each pixel individually due to hsv_to_rgb limitations)
                for i, pos in enumerate(leaf_range):
                    if intensities[i] > 0.1:  # Skip very low intensity pixels
                        r, g, b = hsv_to_rgb(h, s, v_adjusted[i])
                        
                        # Blend with existing foliage buffer using maximum
                        if intensities[i] > foliage_a[pos]:
                            foliage_r[pos] = r
                            foliage_g[pos] = g
                            foliage_b[pos] = b
                            foliage_a[pos] = intensities[i]
            
            # Apply foliage to buffer where alpha is significant
            foliage_mask = foliage_a > 0.1
            if np.any(foliage_mask):
                idx = np.where(foliage_mask)[0]
                for i in idx:
                    # Blend with existing buffer
                    intensity = foliage_a[i]
                    curr_r, curr_g, curr_b, curr_a = buffer[i]
                    
                    buffer[i] = [
                        foliage_r[i] * intensity + curr_r * (1 - intensity),
                        foliage_g[i] * intensity + curr_g * (1 - intensity),
                        foliage_b[i] * intensity + curr_b * (1 - intensity),
                        max(curr_a, intensity)
                    ]
            
            # Draw sunflowers
            for flower in instate['sunflowers'][strip_id]:
                # Update rotation
                flower['rotation'] += flower['rotate_speed'] * delta_time
                
                # Calculate swaying motion
                sway = flower['sway_amplitude'] * np.sin(current_time * flower['sway_frequency'] * 2 * np.pi + 
                                                        flower['sway_offset'])
                
                # Calculate actual position with sway
                actual_pos = (flower['position'] + sway) % strip_length
                center = int(actual_pos)
                
                # Draw the sunflower petals
                num_petals = 8
                
                # Get petal color
                h, s, v = instate['colors']['sunflower_petal']
                r_petal, g_petal, b_petal = hsv_to_rgb(h, s, v)
                
                for petal in range(num_petals):
                    # Calculate petal direction
                    angle = flower['rotation'] + petal * (2 * np.pi / num_petals)
                    
                    # Calculate petal positions (extending from center)
                    petal_length = int(flower['size'] * 0.8)
                    petal_dir_x = np.cos(angle)
                    
                    # Calculate all positions for this petal at once
                    lengths = np.arange(petal_length)
                    petal_positions = (center + np.round(petal_dir_x * lengths)).astype(int) % strip_length
                    
                    # Set all petal pixels at once
                    buffer[petal_positions] = [r_petal, g_petal, b_petal, 0.8]
                
                # Draw center
                center_size = max(1, int(flower['size'] * 0.3))
                
                # Create range of positions for center
                center_range = np.arange(center - center_size, center + center_size + 1) % strip_length
                
                # Calculate distances from center (normalized)
                rel_positions = np.arange(-center_size, center_size + 1)
                distances = np.abs(rel_positions) / center_size
                
                # Calculate intensity based on distance (vectorized)
                valid_mask = distances <= 1.0
                intensities = np.zeros_like(distances)
                intensities[valid_mask] = 1.0 - distances[valid_mask]**2
                
                # Add texture to center (vectorized)
                texture = 0.1 * np.sin(rel_positions * 2.0 + current_time)
                
                # Get center color
                h, s, v = instate['colors']['sunflower_center']
                
                # Apply texture to color (vectorized)
                v_adjusted = np.clip(v * (1.0 + texture), 0.0, 1.0)
                
                # Convert to RGB and apply to buffer
                for i, pos in enumerate(center_range):
                    if valid_mask[i]:
                        r, g, b = hsv_to_rgb(h, s, v_adjusted[i])
                        buffer[pos] = [r, g, b, 0.9]
            
            # Draw sun rays last (on top) - vectorized where possible
            for ray in instate['sun_rays'][strip_id]:
                # Update position
                ray['position'] += ray['movement_speed'] * ray['direction'] * delta_time
                ray['position'] %= strip_length
                
                # Draw the ray
                center = int(ray['position'])
                width = ray['width']
                
                # Create range of positions for this ray
                ray_range = np.arange(center - width, center + width + 1) % strip_length
                
                # Calculate distances from center (normalized)
                rel_positions = np.arange(-width, width + 1)
                distances = np.abs(rel_positions) / width
                
                # Calculate soft edge profile (vectorized)
                valid_mask = distances <= 1.0
                edge_profiles = np.zeros_like(distances)
                edge_profiles[valid_mask] = (1.0 - distances[valid_mask]**2) * ray['intensity'] * sun_factor
                
                # Add shimmer effect (vectorized)
                shimmer = 0.2 * np.sin(current_time * 5.0 + ray_range * 0.3)
                
                # Get sunshine color
                h, s, v = instate['colors']['sunshine_yellow']
                
                # Apply to buffer
                for i, pos in enumerate(ray_range):
                    if edge_profiles[i] > 0.05:  # Skip very low intensity pixels
                        # Less saturated for sun rays
                        r, g, b = hsv_to_rgb(h, s * 0.7, v * (1.0 + shimmer[i]))
                        
                        # Get current pixel values
                        curr_r, curr_g, curr_b, curr_a = buffer[pos]
                        
                        # Blend additively for light rays
                        edge_profile = edge_profiles[i]
                        buffer[pos] = [
                            min(1.0, curr_r + r * edge_profile * 0.7),
                            min(1.0, curr_g + g * edge_profile * 0.7),
                            min(1.0, curr_b + b * edge_profile * 0.7),
                            max(curr_a, edge_profile * 0.7)
                        ]
                        
def OTO_danger_pulse(instate, outstate):
    """
    Generator function that creates a pulsing effect indicating danger.
    
    Features:
    1. Global alpha controlled by outstate['control_danger'] value
    2. All pixels slowly cycle from 50% to 100% brightness
    3. Uses red color to indicate danger
    
    Simple but effective for indicating danger state.
    """
    name = 'danger_pulse'
    buffers = outstate['buffers']
    strip_manager = buffers.strip_manager

    if instate['count'] == 0:
        # Register our generator on first run
        buffers.register_generator(name)
        
        # Initialize parameters
        instate['pulse_phase'] = 0.0      # Current phase of the pulse (0.0-1.0)
        instate['pulse_speed'] = 0.5      # Pulses per second (adjust for faster/slower)
        
        return

    if instate['count'] == -1:
        # Cleanup when pattern is ending
        buffers.generator_alphas[name] = 0
        return

    # Get danger level from outstate (default to 0)
    danger_level = outstate.get('control_danger', 0.0)/100
    
    # Apply alpha level to the generator
    buffers.generator_alphas[name] = danger_level
    
    # Skip rendering if alpha is too low
    if danger_level < 0.01:
        return
    
    # Apply fade-out if the generator is ending
    remaining_time = instate['duration'] - instate['elapsed_time']
    if remaining_time < 10.0:
        fade_alpha = remaining_time / 10.0
        fade_alpha = max(0.0, fade_alpha)
        buffers.generator_alphas[name] = fade_alpha * danger_level
    
    # Get delta time for animation calculations
    delta_time = outstate['current_time'] - outstate['last_time']
    
    # Update pulse phase
    instate['pulse_phase'] += instate['pulse_speed'] * delta_time
    instate['pulse_phase'] %= 1.0  # Keep within 0-1 range
    
    # Calculate brightness factor using sine wave for smooth pulsing
    # Map sine wave from -1 to 1 → 0.5 to 1.0 brightness
    brightness = 0.5 + 0.25 * (1.0 + np.sin(instate['pulse_phase'] * 2 * np.pi))
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Set danger color (bright red)
    r, g, b = 1.0, 1.0, 1.0
    
    # Process each buffer
    for strip_id, buffer in pattern_buffers.items():
        # Skip if strip doesn't exist in manager
        if strip_id not in strip_manager.strips:
            continue
            
        # Fill buffer with pulsing red at calculated brightness
        buffer[:] = [r * brightness, g * brightness, b * brightness, 0.8]