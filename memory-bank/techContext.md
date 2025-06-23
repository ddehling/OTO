# LED-Sign Technical Context

## Technologies Used

### Core Technologies

1. **Python**
   - Primary programming language
   - Version: 3.x
   - Used for all application logic, rendering, and control

2. **OpenGL / ModernGL**
   - Hardware-accelerated graphics rendering
   - Used for creating visual effects and compositing layers
   - Provides shader-based rendering pipeline

3. **NumPy**
   - Numerical computing library
   - Used for efficient array operations and mathematical calculations
   - Core component for image processing and data manipulation

4. **sACN (E1.31)**
   - Streaming ACN protocol for DMX over Ethernet
   - Industry standard for controlling lighting fixtures
   - Used to send pixel data to LED controllers

5. **OpenCV**
   - Computer vision library
   - Used for image processing and visualization
   - Provides window management for simulation view

### Audio Technologies

1. **Audio Processing**
   - Real-time audio analysis
   - Frequency spectrum analysis
   - Volume detection and reactivity

2. **Audio Playback**
   - Threaded audio engine for concurrent sound playback
   - Support for various audio formats (WAV, MP3, FLAC)
   - Volume control and fading capabilities

3. **OSC (Open Sound Control)**
   - Network protocol for communication between multimedia devices
   - Used for external control and integration
   - Supports real-time parameter adjustment

### Graphics and Visualization

1. **PIL/Pillow**
   - Python Imaging Library
   - Used for image loading and basic processing
   - Texture creation and manipulation

2. **scikit-image**
   - Advanced image processing
   - Color space conversions (HSV/RGB)
   - Used for visual effect generation

3. **Shader Programming**
   - GLSL shaders for visual effects
   - Fragment shaders for post-processing
   - Vertex shaders for geometry manipulation

## Development Environment

### Hardware Requirements

1. **Development System**
   - Modern CPU with multiple cores
   - GPU with OpenGL support
   - Sufficient RAM for image processing (8GB+ recommended)
   - Network interface for LED control

2. **LED Hardware**
   - LED controllers compatible with sACN/E1.31 protocol
   - Addressable RGB/RGBA LED strips or panels
   - Network infrastructure for communication

3. **Audio Hardware**
   - Microphone for audio input (optional)
   - Audio output device for monitoring
   - Audio interface for higher quality (optional)

### Software Dependencies

1. **Python Packages**
   - moderngl: OpenGL context and rendering
   - numpy: Numerical computing
   - opencv-python: Image processing and visualization
   - pillow: Image handling
   - scikit-image: Advanced image processing
   - sacn: sACN/E1.31 protocol implementation
   - python-osc: OSC protocol implementation
   - heapq (standard library): Priority queue for event scheduling

2. **System Libraries**
   - OpenGL drivers
   - Audio drivers
   - Network stack with UDP support

3. **Development Tools**
   - Python IDE or code editor
   - Git for version control
   - Virtual environment management (venv, conda, etc.)

## Technical Constraints

### Performance Constraints

1. **Frame Rate**
   - Target frame rate: 20-30 FPS
   - Consistent timing for smooth animations
   - Efficient rendering to maintain performance

2. **Network Bandwidth**
   - sACN protocol overhead
   - Multiple universes for larger installations
   - Reliable delivery of pixel data

3. **Processing Limitations**
   - Real-time rendering requirements
   - Audio processing overhead
   - Event scheduling and management

### Hardware Constraints

1. **LED Controller Limitations**
   - Maximum number of pixels per controller
   - Universe size limitations (typically 170 RGB pixels per universe)
   - Refresh rate capabilities

2. **Display Resolution**
   - Primary display: 120x60 pixels
   - Secondary display: 300x32 pixels
   - Pixel mapping to physical layout

3. **Network Reliability**
   - Handling network interruptions
   - Recovering from connection failures
   - Maintaining synchronization across controllers

### Software Constraints

1. **Python GIL (Global Interpreter Lock)**
   - Limitations on true parallel processing
   - Need for threaded approach for concurrent operations
   - Careful management of shared resources

2. **OpenGL Context Management**
   - Single context per thread
   - Resource cleanup and management
   - Cross-platform compatibility considerations

3. **Audio Processing Latency**
   - Real-time audio analysis challenges
   - Balancing responsiveness with stability
   - Synchronizing audio and visual elements

## Dependencies and External Libraries

### Core Dependencies

1. **moderngl**
   - Purpose: Modern OpenGL wrapper for Python
   - Usage: Creating OpenGL context, managing shaders, rendering
   - Key features: Framebuffers, textures, vertex arrays

2. **numpy**
   - Purpose: Numerical computing library
   - Usage: Array operations, mathematical calculations
   - Key features: Vectorized operations, efficient memory usage

3. **opencv-python**
   - Purpose: Computer vision and image processing
   - Usage: Image manipulation, visualization
   - Key features: Window management, image conversion

4. **sacn**
   - Purpose: sACN/E1.31 protocol implementation
   - Usage: Sending DMX data to LED controllers
   - Key features: Universe management, multicast support

### Audio Dependencies

1. **python-osc**
   - Purpose: OSC protocol implementation
   - Usage: External control and integration
   - Key features: UDP-based messaging, pattern matching

2. **Audio Libraries**
   - Various libraries for audio playback and analysis
   - Threaded audio engine implementation
   - Support for different audio formats

### Visualization Dependencies

1. **pillow**
   - Purpose: Image processing library
   - Usage: Loading images, basic manipulations
   - Key features: Format conversion, image creation

2. **scikit-image**
   - Purpose: Advanced image processing
   - Usage: Color space conversion, effects
   - Key features: Filters, transformations, color manipulation

## Tool Usage Patterns

### Development Workflow

1. **Code Organization**
   - Modular structure with clear separation of concerns
   - Core functionality in dedicated modules
   - Scene utilities for specific effects

2. **Testing Approach**
   - Visual testing through simulation window
   - Parameter adjustment during runtime
   - Incremental development of effects

3. **Deployment Process**
   - Configuration files for different installations
   - Hardware-specific settings in separate files
   - Runtime parameter adjustment

### Configuration Management

1. **Hardware Configuration**
   - LED controller IP addresses and universe assignments
   - Pixel mapping in configuration files
   - Display dimensions and layout

2. **Content Configuration**
   - Weather state parameters and transitions
   - Effect probabilities and characteristics
   - Audio-visual mapping settings

3. **Runtime Configuration**
   - OSC control for parameter adjustment
   - State transitions through external triggers
   - Dynamic reconfiguration capabilities

### Debugging and Monitoring

1. **Visualization Tools**
   - OpenCV windows for real-time preview
   - Simulated LED display for development
   - Visual feedback for state changes

2. **Logging and Diagnostics**
   - Console output for status and errors
   - Performance monitoring
   - Event tracking and timing

3. **Network Diagnostics**
   - sACN packet monitoring
   - Connection status tracking
   - Error handling for network issues

## File Structure and Organization

```
LED-Sign/
├── Stories.py                  # Main entry point
├── corefunctions/              # Core system components
│   ├── Events.py               # Event scheduling system
│   ├── ImageToDMX.py           # DMX/sACN output
│   ├── newrender.py            # OpenGL rendering
│   ├── soundinput.py           # Audio input processing
│   └── soundtestthreaded.py    # Audio playback engine
├── sceneutils/                 # Visual effect implementations
│   ├── environmental.py        # Base environmental effects
│   ├── celestial_bodies.py     # Celestial object definitions
│   ├── weather_params.py       # Weather state parameters
│   ├── forest.py               # Forest-themed effects
│   ├── mountain.py             # Mountain-themed effects
│   └── [other effect modules]  # Various specialized effects
├── imgutils/                   # Image processing utilities
├── otherutils/                 # Miscellaneous utilities
├── DMXconfig/                  # LED controller configurations
│   └── [Unit*.txt]             # Pixel mapping files
├── media/                      # Media assets
│   ├── images/                 # Image assets
│   └── sounds/                 # Audio files
└── Setup/                      # Setup and configuration
    └── requirements.txt        # Python dependencies
```

## Technical Roadmap and Evolution

### Current Technical State

The system currently implements:
- Full environmental simulation with weather states
- Celestial body rendering and animation
- Various themed scenes and effects
- Audio playback and basic reactivity
- sACN output to multiple LED controllers
- OSC input for external control

### Short-term Technical Goals

1. **Performance Optimization**
   - Improve rendering efficiency
   - Optimize network communication
   - Reduce CPU/GPU usage

2. **Enhanced Audio Reactivity**
   - More sophisticated audio analysis
   - Better mapping of audio features to visual effects
   - Improved synchronization

3. **Additional Visual Effects**
   - New themed scenes
   - More dynamic and interactive effects
   - Enhanced transitions between states

### Long-term Technical Vision

1. **Advanced Control Systems**
   - Web-based control interface
   - Remote monitoring and configuration
   - Scheduled state changes and sequences

2. **Machine Learning Integration**
   - Pattern recognition for audio
   - Adaptive effect generation
   - Learning from audience reactions

3. **Expanded Hardware Support**
   - Additional LED controller types
   - Support for non-LED display technologies
   - Integration with other lighting systems

4. **Interactivity Through Motion Tracking**
    - Human pose and face detection detection using MediaPipe
    - Extract useful features from MediaPipe data
    - Use features to interact with scenes
