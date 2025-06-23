# LED-Sign System Patterns

## System Architecture

The LED-Sign system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Stories.py (Main)                    │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 Environmental System                    │
└───────┬───────────────────┬───────────────┬─────────────┘
        │                   │               │
        ▼                   ▼               ▼
┌───────────────┐  ┌────────────────┐  ┌───────────────┐
│ Event System  │  │ Rendering      │  │ Audio System  │
└───────┬───────┘  │ Pipeline       │  └───────┬───────┘
        │          └────────┬───────┘          │
        │                   │                  │
        ▼                   ▼                  ▼
┌───────────────┐  ┌────────────────┐  ┌───────────────┐
│ Scene Utils   │  │ DMX/sACN       │  │ Sound Files   │
│ (Effects)     │  │ Output         │  │ & Processing  │
└───────────────┘  └────────────────┘  └───────────────┘
```

### Key Components

1. **Main Controller (Stories.py)**
   - Entry point and main loop
   - Initializes all subsystems
   - Manages the overall application flow

2. **Environmental System**
   - Central coordinator for the simulation
   - Manages weather states and transitions
   - Controls celestial bodies and environmental parameters
   - Schedules random events based on current state

3. **Event System**
   - Handles scheduling and execution of timed events
   - Manages active events and their lifecycle
   - Provides a state dictionary shared across components

4. **Rendering Pipeline**
   - OpenGL-based rendering engine
   - Manages framebuffers for different displays
   - Handles image planes, textures, and visual effects

5. **Audio System**
   - Manages ambient sound playback
   - Handles event-triggered sound effects
   - Processes microphone input for reactive effects

6. **Scene Utilities**
   - Collection of specialized visual effects
   - Themed scene generators (forest, mountain, etc.)
   - Weather and environmental effect implementations

7. **DMX/sACN Output**
   - Converts rendered frames to DMX data
   - Manages network communication with LED controllers
   - Handles addressing and mapping of pixels

## Design Patterns

### 1. Event-Based Architecture

The system uses an event-based architecture to coordinate visual and audio effects:

- **Event Scheduler**: Central component that manages timed events
- **Event Queue**: Priority queue of upcoming events sorted by start time
- **Active Events**: Collection of currently running events
- **Event Callbacks**: Functions called on event start, update, and end

```python
# Example event scheduling pattern
scheduler.schedule_event(delay, duration, action, *args, **kwargs)
```

### 2. State Management

A shared state dictionary is used to communicate between components:

- **Central State**: Maintained by the EventScheduler
- **Component-Specific State**: Each component can store its own state
- **State Propagation**: Environmental parameters are pushed to the state dictionary

```python
# Example state access pattern
outstate['wind'] = smooth_wind(self.current_time, 30, 20) * self.weather_params["wind_speed"]
```

### 3. Layered Rendering

The rendering system uses a layered approach:

- **Image Planes**: Individual layers with position, scale, and transparency
- **Depth Sorting**: Planes are rendered from back to front
- **Texture Updates**: Dynamic content through texture updates
- **Framebuffers**: Separate framebuffers for different displays

### 4. Factory Methods

Scene utilities use factory methods to create specific effects:

- **Effect Generators**: Functions that create specific visual effects
- **Parameterized Creation**: Effects are customized through parameters
- **Reusable Components**: Common patterns are abstracted into reusable functions

### 5. Observer Pattern

The system implements a loose observer pattern:

- **Environmental Changes**: Weather state changes notify observers
- **Audio Reactivity**: Sound input triggers visual responses
- **OSC Messages**: External control messages update system state

## Component Relationships

### Environmental System ↔ Event System

- Environmental system schedules events through the event system
- Event system executes callbacks that modify environmental parameters
- Shared state dictionary communicates changes between systems

### Environmental System ↔ Rendering Pipeline

- Environmental parameters control visual appearance
- Rendering pipeline visualizes the current environmental state
- Image planes are created and updated based on environmental conditions

### Event System ↔ Audio System

- Events trigger sound playback
- Sound engine reports playback status to event system
- Audio analysis feeds back into the event system state

### Rendering Pipeline ↔ DMX Output

- Rendered frames are converted to DMX data
- Pixel mapping translates image coordinates to LED addresses
- Multiple receivers handle different sections of the display

## Critical Implementation Paths

### Rendering Loop

1. Update environmental system
   - Process weather transitions
   - Update celestial bodies
   - Generate random events
   - Update shared state

2. Process active events
   - Execute event callbacks
   - Update event state
   - Remove completed events
   - Add new events from queue

3. Render frames
   - Clear framebuffers
   - Render image planes in depth order
   - Apply post-processing effects
   - Read back pixel data

4. Send to LED controllers
   - Convert frames to DMX data
   - Map pixels to LED addresses
   - Send data via sACN protocol
   - Handle network errors

### Weather Transition Path

1. Trigger weather state change
   - Select new weather state
   - Set transition duration
   - Store target parameters

2. Interpolate parameters
   - Calculate transition progress
   - Blend between start and target values
   - Update current parameters

3. Apply visual effects
   - Update fog, lighting, and color parameters
   - Schedule weather-specific events
   - Transition ambient sound

### Audio Reactivity Path

1. Capture audio input
   - Process microphone data
   - Analyze frequency spectrum
   - Detect sound intensity

2. Update visual parameters
   - Modify effect intensity based on sound
   - Trigger events on sound thresholds
   - Adjust animation speeds and scales

## Technical Decisions

### OpenGL for Rendering

- **Decision**: Use OpenGL (via ModernGL) for rendering instead of pure NumPy/PIL
- **Rationale**: 
  - Hardware acceleration for better performance
  - Built-in support for effects like blending and fog
  - Easier handling of layered content
  - Better scaling to higher resolutions

### sACN Protocol

- **Decision**: Use sACN (E1.31) protocol for LED control
- **Rationale**:
  - Industry standard for LED lighting control
  - Support for multiple universes
  - Network-based for flexible installation
  - Compatible with many LED controllers

### Event-Based Architecture

- **Decision**: Use an event-based system rather than fixed animation loops
- **Rationale**:
  - More flexible timing of effects
  - Easier coordination between audio and visual elements
  - Better support for random and triggered events
  - Cleaner separation of concerns

### Shared State Dictionary

- **Decision**: Use a central state dictionary rather than direct component coupling
- **Rationale**:
  - Reduces tight coupling between components
  - Easier to add new components that respond to state
  - Simplifies debugging by centralizing state
  - More flexible for future extensions

### Python Implementation

- **Decision**: Implement in Python rather than C++ or other languages
- **Rationale**:
  - Faster development and iteration
  - Rich ecosystem for graphics, audio, and networking
  - Easier to modify and extend
  - Good performance with hardware-accelerated libraries
