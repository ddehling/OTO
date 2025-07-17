from sceneutils.imgutils import *  # noqa: F403
import numpy as np
from pathlib import Path


def OTO_pattern_cycle(instate, outstate):
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
        
        return

    if instate['count'] == -1:
        # Cleanup all pattern buffers
        buffers.generator_alphas[name] = 0
        for pattern_name in instate['pattern_names']:
            buffers.generator_alphas[pattern_name] = 0
        return

    # Set main generator alpha to full
    buffers.generator_alphas[name] = 1.0
    
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
        buffers.generator_alphas[pattern_name] = pattern_alpha
        
        # Get buffers for this pattern
        pattern_buffers = buffers.get_all_buffers(pattern_name)
        #print(pattern_name)
        # Render this pattern (placeholder for pattern-specific code)
        if i == 0:
            # Pattern 0 code would go here
            # Example: Fill with solid color
            pass
        
        elif i == 1:
            # Pattern 1 code would go here
            # Example: Create chasing lights
            pass
        
        elif i == 2:
            # Pattern 2 code would go here
            # Example: Generate pulsing effect
            pass
        
        elif i == 3:
            # Pattern 3 code would go here
            # Example: Random sparkles
            pass
        
        elif i == 4:
            # Pattern 4 code would go here
            # Example: Rainbow pattern
            pass