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
    
    # Get all buffers for this generator
    pattern_buffers = buffers.get_all_buffers(name)
    
    # Update heartbeat phases - slow heartbeat around 40 BPM with slight variation between sides
    beats_per_second_left = 38 / 60.0  # Slightly slower for left
    beats_per_second_right = 42 / 60.0  # Slightly faster for right
    
    instate['heart_phase_left'] = (instate['heart_phase_left'] + beats_per_second_left * delta_time) % 1.0
    instate['heart_phase_right'] = (instate['heart_phase_right'] + beats_per_second_right * delta_time) % 1.0
    
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
                    
                    # Draw the raindrop
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
                    
                    # Draw drop trail (longer with gradual fade)
                    for i in range(1, drop['size']):
                        # Trail position depends on direction
                        trail_pos = pos_int - (i * drop['direction'])
                        
                        # Handle trail wrapping
                        if trail_pos >= strip_length:
                            trail_pos = trail_pos - strip_length
                        elif trail_pos < 0:
                            trail_pos = strip_length + trail_pos
                        
                        if 0 <= trail_pos < strip_length:
                            # Fade alpha for trail - more gradual fade
                            trail_alpha = drop['alpha'] * (1 - (i / drop['size'])**1.5)
                            
                            # Slight color adjustment for trail (more grey as it trails)
                            trail_phase = min(1.0, drop['color_phase'] + (i / drop['size']) * 0.3)
                            trail_h = blue_h + (grey_h - blue_h) * trail_phase
                            trail_s = blue_s + (grey_s - blue_s) * trail_phase
                            trail_v = blue_v + (grey_v - blue_v) * trail_phase
                            
                            tr, tg, tb = hsv_to_rgb(trail_h, trail_s, trail_v)
                            buffer[trail_pos] = [tr, tg, tb, trail_alpha]
            
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
            
            # Set uniform color for the heart strip with brighter alpha at peak
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
                    
                    # Add small glow
                    for i in range(1, 3):
                        glow_pos = (pos_int + i) % strip_length
                        glow_pos2 = (pos_int - i) % strip_length
                        glow_alpha = seed['alpha'] * (1 - (i / 3))
                        
                        buffer[glow_pos] = [r, g, b, glow_alpha]
                        buffer[glow_pos2] = [r, g, b, glow_alpha]
            
            # Replace with updated list
            instate['dandelion_seeds'][strip_id] = new_seeds
            
        else:
            # Other strips - subtle blue-grey pulsing
            
            # Calculate pulsing effect
            pulse = 0.3 + 0.2 * np.sin(outstate['current_time'] * 0.5)
            
            # Get color - grey-blue
            h, s, v = instate['colors']['grey_blue']
            
            # Adjust value based on pulse
            v = v * pulse
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Set uniform color for the strip
            buffer[:] = [r, g, b, 0.3]


# ... existing code ...

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
    
    Uses vectorized operations for performance where possible.
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
            
            # Update existing flames
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
                    
                    # Draw main body of flame
                    for i in range(flame['size']):
                        pixel_pos = flame_center + i
                        
                        if 0 <= pixel_pos < strip_length:
                            # Calculate position in flame (0 at tip, 1 at base)
                            rel_pos = i / flame['size']
                            
                            # Select color based on position in flame
                            if rel_pos < 0.2:
                                # Tip of flame - yellow/white
                                h, s, v = instate['colors']['yellow']
                                # Make tip brighter
                                v = min(1.0, v * 1.2)
                                s *= 0.8  # Less saturated (more white)
                            elif rel_pos < 0.5:
                                # Middle of flame - orange
                                h, s, v = instate['colors']['orange']
                            else:
                                # Base of flame - red
                                h, s, v = instate['colors']['deep_red']
                            
                            # Calculate pixel intensity based on position in flame
                            # Brightest in middle, dimmer at edges
                            intensity_factor = 1.0 - abs((rel_pos * 2) - 1.0)
                            pixel_intensity = flame_alpha * intensity_factor
                            
                            # Convert to RGB
                            r, g, b = hsv_to_rgb(h, s, v)
                            
                            # Blend with existing pixel (additive for fire effect)
                            curr_r, curr_g, curr_b, curr_a = buffer[pixel_pos]
                            new_r = min(1.0, curr_r + r * pixel_intensity)
                            new_g = min(1.0, curr_g + g * pixel_intensity)
                            new_b = min(1.0, curr_b + b * pixel_intensity)
                            new_a = max(curr_a, pixel_intensity)
                            
                            buffer[pixel_pos] = [new_r, new_g, new_b, new_a]
            
            # Replace with updated list
            instate['flames'][strip_id] = new_flames
            
            # Update existing embers
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
                        
                        # Set pixel with additive blending
                        curr_r, curr_g, curr_b, curr_a = buffer[ember_pos]
                        new_r = min(1.0, curr_r + r * ember_alpha)
                        new_g = min(1.0, curr_g + g * ember_alpha)
                        new_b = min(1.0, curr_b + b * ember_alpha)
                        new_a = max(curr_a, ember_alpha)
                        
                        buffer[ember_pos] = [new_r, new_g, new_b, new_a]
                        
                        # Add small glow around ember
                        for i in range(1, 3):
                            for offset in [-i, i]:
                                glow_pos = ember_pos + offset
                                if 0 <= glow_pos < strip_length:
                                    glow_alpha = ember_alpha * (1.0 - (i / 3.0))
                                    
                                    # Blend with existing pixel
                                    curr_r, curr_g, curr_b, curr_a = buffer[glow_pos]
                                    new_r = min(1.0, curr_r + r * glow_alpha * 0.5)
                                    new_g = min(1.0, curr_g + g * glow_alpha * 0.5)
                                    new_b = min(1.0, curr_b + b * glow_alpha * 0.5)
                                    new_a = max(curr_a, glow_alpha * 0.5)
                                    
                                    buffer[glow_pos] = [new_r, new_g, new_b, new_a]
            
            # Replace with updated list
            instate['embers'][strip_id] = new_embers
            
        elif 'heart' in strip.groups:
            # Heart strips - rapid angry heartbeat
            
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
            
            # Set color for the heart strip with variable alpha
            alpha = 0.4 + 0.6 * intensity
            buffer[:] = [r, g, b, alpha]
            
            # Add pulsing effect
            for i in range(strip_length):
                # Distance from center
                center = strip_length // 2
                dist = abs(i - center) / (strip_length / 2)
                
                # Pulse stronger toward center
                pulse_intensity = intensity * (1.0 - dist * 0.5)
                
                # Calculate color with pulse
                pulse_r, pulse_g, pulse_b = r, g * (0.7 + 0.3 * pulse_intensity), b * (0.7 + 0.3 * pulse_intensity)
                pulse_a = alpha * (0.8 + 0.2 * pulse_intensity)
                
                buffer[i] = [pulse_r, pulse_g, pulse_b, pulse_a]
            
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
                    
                    # Draw the explosion
                    radius_int = int(explosion['radius'])
                    for i in range(-radius_int, radius_int + 1):
                        pos = explosion['center'] + i
                        
                        if 0 <= pos < strip_length:
                            # Calculate distance from center
                            dist = abs(i) / explosion['radius'] if explosion['radius'] > 0 else 0
                            
                            # Intensity falls off from center
                            pixel_intensity = explosion_alpha * (1.0 - dist**2)
                            
                            # Skip if too dim
                            if pixel_intensity < 0.05:
                                continue
                            
                            # Select color based on distance from center
                            if dist < 0.2:
                                # Center - yellow/white hot
                                h, s, v = instate['colors']['yellow']
                                # Desaturate center for white-hot look
                                s *= 0.5
                                v = min(1.0, v * 1.2)
                            elif dist < 0.6:
                                # Middle - orange
                                h, s, v = instate['colors']['orange']
                                h = (h + explosion['color_shift']) % 1.0  # Slight color variation
                            else:
                                # Outer edge - red
                                h, s, v = instate['colors']['bright_red']
                            
                            # Convert to RGB
                            r, g, b = hsv_to_rgb(h, s, v)
                            
                            # Blend with existing pixel (additive for explosion effect)
                            curr_r, curr_g, curr_b, curr_a = buffer[pos]
                            new_r = min(1.0, curr_r + r * pixel_intensity)
                            new_g = min(1.0, curr_g + g * pixel_intensity)
                            new_b = min(1.0, curr_b + b * pixel_intensity)
                            new_a = max(curr_a, pixel_intensity)
                            
                            buffer[pos] = [new_r, new_g, new_b, new_a]
            
            # Replace with updated list
            instate['explosions'][strip_id] = new_explosions
            
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
            explosion_interval = 0.10 + np.random.random() * 0.2  # 1-2 seconds between explosions
            
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
            
            # Update existing flames
            new_flames = []
            for flame in instate['flames'][strip_id]:
                # Update life
                flame['life'] += delta_time / flame['duration']
                
                # Update position based on direction
                flame['position'] += flame['speed'] * flame['direction'] * delta_time
                
                # Keep if still in bounds and alive
                if (0 <= flame['position'] < strip_length or 
                    0 <= flame['position'] + flame['size'] < strip_length) and flame['life'] < 1.0:
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
                    
                    # For each horizontal position of the flame
                    for i in range(flame['size']):
                        h_pos = flame_center + (i * flame['direction'])
                        
                        if 0 <= h_pos < strip_length:
                            # Get base flame color
                            rel_pos = i / flame['size']  # Relative position in flame
                            
                            if rel_pos < 0.3:
                                # Front of flame - more yellow
                                h, s, v = instate['colors']['yellow']
                            elif rel_pos < 0.7:
                                # Middle of flame - orange
                                h, s, v = instate['colors']['orange']
                            else:
                                # Back of flame - deeper red
                                h, s, v = instate['colors']['deep_red']
                            
                            # Draw flame with height and lapping effect
                            for j in range(1, flame['height'] + 1):
                                # Calculate vertical position in flame (simulated via intensity)
                                vert_pos = j / flame['height']
                                
                                # Adjust color based on height
                                if j == flame['height']:
                                    # Top of flame - more yellow/white
                                    h_adj = h
                                    s_adj = s * 0.7  # Less saturated
                                    v_adj = min(1.0, v * 1.1)  # Brighter
                                else:
                                    h_adj = h
                                    s_adj = s
                                    v_adj = v * (0.7 + 0.3 * (j / flame['height']))  # Brighter toward top
                                
                                # Convert to RGB
                                r, g, b = hsv_to_rgb(h_adj, s_adj, v_adj)
                                
                                # Calculate lapping effect - flames oscillate
                                time_factor = outstate['current_time'] * 8.0 + h_pos * 0.2
                                lap_factor = 0.2 * np.sin(time_factor)
                                
                                # Apply intensity with lapping and height factors
                                height_factor = 1.0 - (abs(vert_pos - 0.5) * 1.2)  # Brightest in middle height
                                pixel_intensity = flame_alpha * height_factor * (1.0 + lap_factor)
                                pixel_intensity = max(0.0, min(1.0, pixel_intensity))
                                
                                # Blend with existing pixel
                                curr_r, curr_g, curr_b, curr_a = buffer[h_pos]
                                new_r = min(1.0, curr_r + r * pixel_intensity)
                                new_g = min(1.0, curr_g + g * pixel_intensity)
                                new_b = min(1.0, curr_b + b * pixel_intensity)
                                new_a = max(curr_a, pixel_intensity)
                                
                                buffer[h_pos] = [new_r, new_g, new_b, new_a]
            
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
                    
                    # Draw the explosion
                    radius_int = int(explosion['radius'])
                    for i in range(-radius_int, radius_int + 1):
                        pos = explosion['center'] + i
                        
                        if 0 <= pos < strip_length:
                            # Calculate distance from center
                            dist = abs(i) / explosion['radius'] if explosion['radius'] > 0 else 0
                            
                            # Intensity falls off from center
                            pixel_intensity = explosion_alpha * (1.0 - dist**2)
                            
                            # Skip if too dim
                            if pixel_intensity < 0.05:
                                continue
                            
                            # Select color based on distance from center
                            if dist < 0.2:
                                # Center - yellow/white hot
                                h, s, v = instate['colors']['yellow']
                                # Desaturate center for white-hot look
                                s *= 0.5
                                v = min(1.0, v * 1.2)
                            elif dist < 0.6:
                                # Middle - orange
                                h, s, v = instate['colors']['orange']
                                h = (h + explosion['color_shift']) % 1.0  # Slight color variation
                            else:
                                # Outer edge - red
                                h, s, v = instate['colors']['bright_red']
                            
                            # Convert to RGB
                            r, g, b = hsv_to_rgb(h, s, v)
                            
                            # Blend with existing pixel (additive for explosion effect)
                            curr_r, curr_g, curr_b, curr_a = buffer[pos]
                            new_r = min(1.0, curr_r + r * pixel_intensity)
                            new_g = min(1.0, curr_g + g * pixel_intensity)
                            new_b = min(1.0, curr_b + b * pixel_intensity)
                            new_a = max(curr_a, pixel_intensity)
                            
                            buffer[pos] = [new_r, new_g, new_b, new_a]
            
            # Replace with updated list
            instate['explosions'][strip_id] = new_explosions
            
        else:
            # Other strips - fiery pulsing with heat waves
            
            # Create a base heat wave pattern across the strip
            for i in range(strip_length):
                # Calculate normalized position
                norm_pos = i / strip_length
                
                # Create a heat wave pattern - multiple sine waves combined
                wave1 = 0.5 + 0.5 * np.sin(norm_pos * 4 * np.pi + outstate['current_time'] * 3.0)
                wave2 = 0.5 + 0.5 * np.sin(norm_pos * 7 * np.pi - outstate['current_time'] * 2.0)
                wave3 = 0.5 + 0.5 * np.sin(norm_pos * 2 * np.pi + outstate['current_time'] * 1.0)
                
                # Combine waves with different weights
                heat_intensity = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2)
                
                # Add pulsing effect
                pulse = 0.7 + 0.3 * np.sin(outstate['current_time'] * 5.0)
                heat_intensity *= pulse
                
                # Select color based on heat intensity
                if heat_intensity > 0.8:
                    # Very hot - yellow
                    h, s, v = instate['colors']['yellow']
                elif heat_intensity > 0.5:
                    # Hot - orange
                    h, s, v = instate['colors']['orange']
                else:
                    # Less hot - deep red
                    h, s, v = instate['colors']['deep_red']
                
                # Adjust brightness based on heat intensity
                v = v * (0.6 + 0.4 * heat_intensity)
                
                # Convert to RGB
                r, g, b = hsv_to_rgb(h, s, v)
                
                # Set pixel with alpha based on heat intensity
                alpha = 0.3 + 0.7 * heat_intensity
                buffer[i] = [r, g, b, alpha]
                
                # Occasionally add ember sparks
                if np.random.random() < 0.005:  # 0.5% chance per pixel per frame
                    spark_intensity = 0.8 + np.random.random() * 0.2
                    h, s, v = instate['colors']['ember']
                    r, g, b = hsv_to_rgb(h, s, v)
                    buffer[i] = [r, g, b, spark_intensity]

        for i in range(strip_length):
            # Random choice from color palette
            color_name = np.random.choice(list(instate['colors'].keys()))
            h, s, v = instate['colors'][color_name]
            
            # Random intensity between 0.02 and 0.2
            noise_intensity = 0.2 + np.random.random() * 0.4
            
            # Convert to RGB
            r, g, b = hsv_to_rgb(h, s, v)
            
            # Blend with existing pixel (additive blending)
            curr_r, curr_g, curr_b, curr_a = buffer[i]
            new_r = min(1.0, curr_r + r * noise_intensity)
            new_g = min(1.0, curr_g + g * noise_intensity)
            new_b = min(1.0, curr_b + b * noise_intensity)
            new_a = max(curr_a, noise_intensity)
            
            buffer[i] = [new_r, new_g, new_b, new_a]   

# ... existing code ...
def OTO_curious_playful(instate, outstate):
    """
    Generator function that creates a curious and playful-themed pattern across all strips.
    
    Features:
    1. Global alpha controlled by outstate['control_curious'] value
    2. Vibrant, saturated color palette with blues, greens, reds, oranges, purples, and whites
    3. Color regions that collide and blend at edges but don't pass through each other
    4. Paint-like effect where colors blend at boundaries creating dynamic interfaces
    5. Fast movement with playful characteristics
    
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
            'vibrant_orange': [0.08, 0.9, 0.95], # Vibrant orange
            'rich_purple': [0.8, 0.8, 0.9],       # Rich purple
            'hot_pink': [0.9, 0.75, 0.95],         # Hot pink
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
                    'blend_width': 10 + np.random.random() * 15,  # 10-25 pixels blend width
                    'lifetime': 0,  # Time tracking for color changes
                    'color_change_time': 5 + np.random.random() * 10  # 5-15 seconds between color changes
                }
                
                instate['color_regions'][strip_id].append(region)
        
        # Initialize pixel ownership array - track which region controls each pixel
        pixel_owners = np.full(strip_length, -1)  # -1 means no owner
        pixel_strengths = np.zeros(strip_length)  # Strength of ownership (for blending)
        
        # First pass - determine pixel ownership
        for i, region in enumerate(instate['color_regions'][strip_id]):
            # Update region position based on speed and direction
            effective_speed = region['speed'] * instate['region_speed_multiplier']
            region['center'] += effective_speed * region['direction'] * delta_time
            
            # Add wobble to size for a playful effect
            time_factor = outstate['current_time'] * region['wobble_freq']
            size_wobble = 1.0 + region['wobble_amount'] * np.sin(time_factor + region['wobble_offset'])
            effective_size = region['size'] * size_wobble
            
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
                    min_distance = (effective_size + other_region['size']) * 0.5
                    
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
            
            # Calculate influence on each pixel
            for pixel in range(strip_length):
                # Calculate distance to region center, handling wrapping
                direct_dist = abs(pixel - region['center'])
                wrapped_dist = strip_length - direct_dist
                distance = min(direct_dist, wrapped_dist)
                
                # Calculate strength of influence - strong in center, falls off toward edges
                if distance < effective_size:
                    # Normalized distance (0 at center, 1 at edge)
                    normalized_dist = distance / effective_size
                    
                    # Strength falls off with square of distance
                    strength = 1.0 - normalized_dist**2
                    
                    # Apply strength if stronger than current owner
                    if strength > pixel_strengths[pixel]:
                        pixel_owners[pixel] = i
                        pixel_strengths[pixel] = strength
            
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
        
        # Initialize buffer with black (transparent)
        buffer[:] = [0, 0, 0, 0]
        
        # Render pixels based on ownership and blend across boundaries
        for pixel in range(strip_length):
            # If pixel has an owner, use that region's color
            if pixel_owners[pixel] >= 0:
                region = instate['color_regions'][strip_id][pixel_owners[pixel]]
                
                # Get base color from region
                base_h, base_s, base_v = region['h'], region['s'], region['v']
                
                # Calculate how close we are to the edge of this region's influence
                edge_factor = pixel_strengths[pixel]
                
                # Check neighboring pixels for blending (look ahead and behind)
                blend_colors = []
                blend_strengths = []
                
                # Get blend width for this region
                blend_width = region['blend_width']
                
                # Look at pixels ahead and behind for different owners
                for offset in range(1, int(blend_width) + 1):
                    for direction in [-1, 1]:
                        # Get wrapped pixel position
                        check_pixel = (pixel + direction * offset) % strip_length
                        
                        # If this pixel has a different owner, consider blending
                        if pixel_owners[check_pixel] >= 0 and pixel_owners[check_pixel] != pixel_owners[pixel]:
                            # Get the other region
                            other_region = instate['color_regions'][strip_id][pixel_owners[check_pixel]]
                            
                            # Calculate blend strength based on distance
                            blend_strength = max(0, 1.0 - (offset / blend_width))
                            
                            # Add to blend colors and strengths
                            blend_colors.append((other_region['h'], other_region['s'], other_region['v']))
                            blend_strengths.append(blend_strength)
                
                # Calculate final color by blending
                if blend_colors:
                    # Normalize blend strengths
                    total_blend = sum(blend_strengths)
                    normalized_strengths = [s / (total_blend + 1.0) for s in blend_strengths]
                    
                    # Edge factor influences how much the base color is present
                    base_influence = 0.5 + 0.5 * edge_factor
                    
                    # Calculate the blended color
                    h_sum = base_h * base_influence
                    s_sum = base_s * base_influence
                    v_sum = base_v * base_influence
                    
                    for (h, s, v), strength in zip(blend_colors, normalized_strengths):
                        # Add contribution from this blend color
                        blend_contribution = strength * (1.0 - base_influence)
                        
                        # Special handling for hue to handle the circular nature
                        if abs(h - base_h) > 0.5:
                            # Wrap around 0-1 boundary
                            if h > base_h:
                                h_sum += (h - 1.0) * blend_contribution
                            else:
                                h_sum += (h + 1.0) * blend_contribution
                        else:
                            h_sum += h * blend_contribution
                        
                        s_sum += s * blend_contribution
                        v_sum += v * blend_contribution
                    
                    # Wrap hue back to 0-1 range
                    final_h = h_sum % 1.0
                    final_s = min(1.0, max(0.0, s_sum))
                    final_v = min(1.0, max(0.0, v_sum))
                else:
                    # No blending needed
                    final_h = base_h
                    final_s = base_s
                    final_v = base_v
                
                # Convert to RGB
                r, g, b = hsv_to_rgb(final_h, final_s, final_v)
                
                # Alpha is based on edge factor - more transparent at edges
                alpha = 0.7 + 0.3 * edge_factor
                
                # Set pixel color
                buffer[pixel] = [r, g, b, alpha]
            else:
                # No owner - leave black/transparent
                pass
                
        # Add sparkles for curiosity (small bright points that appear briefly)
        sparkle_chance = 0.02 * curious_level  # More sparkles when more curious
        for pixel in range(strip_length):
            if np.random.random() < sparkle_chance:
                # Create a sparkle
                sparkle_h = np.random.random()  # Random hue
                sparkle_s = 0.2  # Low saturation (white-ish)
                sparkle_v = 1.0  # Full brightness
                
                # Convert to RGB
                sr, sg, sb = hsv_to_rgb(sparkle_h, sparkle_s, sparkle_v)
                
                # Set pixel with additive blending
                r, g, b, a = buffer[pixel]
                buffer[pixel] = [
                    min(1.0, r + sr * 0.7),
                    min(1.0, g + sg * 0.7),
                    min(1.0, b + sb * 0.7),
                    min(1.0, a + 0.3)
                ]


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