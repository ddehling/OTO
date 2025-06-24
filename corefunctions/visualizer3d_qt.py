import sys
import numpy as np
import math
import queue
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QMatrix4x4, QVector3D, QColor, QCursor, QPainter, QFont
from PyQt5.QtCore import Qt, QTimer, QSize, QPoint, QRect

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("PyOpenGL not available. Install with: pip install PyOpenGL")

class LED3DVisualizer(QMainWindow):
    def __init__(self, strip_manager, width=1200, height=800):
        # Ensure there's a QApplication instance
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        
        super().__init__()
        
        # Store strip manager
        self.strip_manager = strip_manager
        
        # Initialize UI
        self.setWindowTitle("3D LED Strip Visualizer")
        self.setGeometry(100, 100, width, height)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create OpenGL widget
        self.gl_widget = LED3DOpenGLWidget(self.strip_manager)
        main_layout.addWidget(self.gl_widget)
        
        # Add help label
        self.help_label = QLabel()
        self.help_label.setStyleSheet("color: white; background-color: rgba(0,0,0,100);")
        self.help_label.setText(
            "Mouse Controls: Left-drag: Rotate | Right-drag: Pan | Scroll: Zoom | Spacebar: Toggle rotation | R: Reset view"
        )
        main_layout.addWidget(self.help_label)
        
        # Set up message queue for thread communication
        self.message_queue = queue.Queue()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_queue)
        self.timer.start(16)  # ~60fps
        
        # Show the window
        self.show()
        print("3D Visualizer window shown")
        
        # Process some events to ensure window is displayed
        self.app.processEvents()
    
    def process_queue(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                if message[0] == 'update_colors':
                    self.gl_widget.update_colors(message[1])
                self.message_queue.task_done()
        except queue.Empty:
            pass
    
    def update(self, output_buffers):
        """Update from main thread"""
        # Copy the buffer data
        colors_copy = {}
        for strip_id, buffer in output_buffers.items():
            colors_copy[strip_id] = buffer.copy()
        
        # Send to UI thread
        self.message_queue.put(('update_colors', colors_copy))
        
        # Process events to keep UI responsive
        if QApplication.instance():
            QApplication.instance().processEvents()
    
    def close(self):
        """Override close to clean up"""
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
        super().close()

class LED3DOpenGLWidget(QOpenGLWidget):
    def __init__(self, strip_manager):
        super().__init__()
        
        self.strip_manager = strip_manager
        self.strip_colors = {}
        self.strip_coordinates = {}
        
        # Camera settings
        self.camera_distance = 15.0
        self.camera_elevation = 30.0  # degrees
        self.camera_azimuth = 0.0     # degrees
        self.camera_target = [0.0, 0.0, 0.0]  # Look at center
        
        # Mouse interaction variables
        self.last_mouse_pos = QPoint()
        self.mouse_pressed = False
        self.right_mouse_pressed = False
        self.auto_rotate = True
        
        # Label cache for 3D text
        self.label_cache = {}
        
        # Set focus to receive key events
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
        # Set minimum size
        self.setMinimumSize(QSize(800, 600))
        
        # Create strip coordinates
        self._create_strip_coordinates()
        
        # Set up a timer for continuous rotation
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        self.animation_timer.start(30)  # ~33fps
    
    def _create_strip_coordinates(self):
        """Create 3D coordinates for each LED strip"""
        strip_count = len(self.strip_manager.strips)
        if strip_count == 0:
            print("Warning: No strips found in strip_manager")
            return
            
        print(f"Creating 3D coordinates for {strip_count} strips")
        
        # Create a circular arrangement of strips
        radius = 5.0
        height_offset = -2.0
        
        for i, (strip_id, strip) in enumerate(self.strip_manager.strips.items()):
            # Calculate angle for positioning on a circle
            angle = (i / strip_count) * 2 * math.pi
            center_x = radius * math.cos(angle)
            center_z = radius * math.sin(angle)
            
            # Generate positions for each LED in the strip
            coords = np.zeros((strip.length, 3), dtype=float)
            
            # Calculate strip direction vector - slightly tilted outward
            dx = center_x * 0.1  # Tilt outward slightly
            dy = 1.0             # Main direction is up
            dz = center_z * 0.1  # Tilt outward slightly
            
            # Normalize direction vector
            mag = math.sqrt(dx*dx + dy*dy + dz*dz)
            dx, dy, dz = dx/mag, dy/mag, dz/mag
            
            # Scale for strip length in 3D space
            strip_length = 4.0
            dx *= strip_length
            dy *= strip_length
            dz *= strip_length
            
            # Position each LED along this line
            for j in range(strip.length):
                t = j / (strip.length - 1) if strip.length > 1 else 0.5
                coords[j, 0] = center_x + dx * t
                coords[j, 1] = height_offset + dy * t
                coords[j, 2] = center_z + dz * t
            
            self.strip_coordinates[strip_id] = coords
            print(f"Created coordinates for strip {strip_id} with {strip.length} LEDs")
            
            # Increment height offset for next strip
            height_offset += 0.5  # Space between strips
    
    def update_colors(self, colors):
        """Update strip colors"""
        self.strip_colors = colors
        self.update()  # Trigger a redraw
    
    def animate(self):
        """Animate the view by rotating slightly"""
        if self.auto_rotate:
            self.camera_azimuth += 0.5  # Rotate by 0.5 degrees each frame
            if self.camera_azimuth >= 360.0:
                self.camera_azimuth -= 360.0
            self.update()
    
    def initializeGL(self):
        """Initialize OpenGL settings"""
        # Set clear color (background)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Position the light
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        
        # Enable color material mode
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    def resizeGL(self, width, height):
        """Handle window resize events"""
        # Set the viewport
        glViewport(0, 0, width, height)
        
        # Set up the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100.0)
        
        # Return to model view matrix
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Draw the scene"""
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up the model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Convert spherical to Cartesian coordinates for camera position
        azimuth_rad = math.radians(self.camera_azimuth)
        elevation_rad = math.radians(self.camera_elevation)
        
        # Calculate camera position in spherical coordinates
        x = self.camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        y = self.camera_distance * math.sin(elevation_rad)
        z = self.camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        
        # Position the camera
        gluLookAt(
            x + self.camera_target[0], y + self.camera_target[1], z + self.camera_target[2],  # Eye position
            self.camera_target[0], self.camera_target[1], self.camera_target[2],  # Look at position
            0.0, 1.0, 0.0   # Up vector
        )
        
        # Draw the coordinate axes
        self._draw_axes()
        
        # Draw a grid on the ground plane
        self._draw_grid()
        
        # Draw the LED strips
        self._draw_strips()
    
    def _draw_axes(self):
        """Draw XYZ axes for reference"""
        glDisable(GL_LIGHTING)  # Disable lighting for axes
        
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(5.0, 0.0, 0.0)
        
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 5.0, 0.0)
        
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 5.0)
        glEnd()
        glLineWidth(1.0)
        
        glEnable(GL_LIGHTING)  # Re-enable lighting
    
    def _draw_grid(self):
        """Draw a reference grid on the ground plane"""
        glDisable(GL_LIGHTING)  # Disable lighting for grid
        
        grid_size = 10
        grid_step = 1.0
        
        glColor3f(0.3, 0.3, 0.3)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        
        # Draw grid lines along X axis
        for i in range(-grid_size, grid_size + 1):
            glVertex3f(-grid_size * grid_step, -2.0, i * grid_step)
            glVertex3f(grid_size * grid_step, -2.0, i * grid_step)
        
        # Draw grid lines along Z axis
        for i in range(-grid_size, grid_size + 1):
            glVertex3f(i * grid_step, -2.0, -grid_size * grid_step)
            glVertex3f(i * grid_step, -2.0, grid_size * grid_step)
        
        glEnd()
        
        glEnable(GL_LIGHTING)  # Re-enable lighting
    
    def _draw_strips(self):
        """Draw all LED strips with their colors"""
        if not self.strip_coordinates:
            return
        
        # Draw each strip
        for strip_id, coords in self.strip_coordinates.items():
            colors = self.strip_colors.get(strip_id)
            
            # Draw the strip as a line
            glDisable(GL_LIGHTING)  # Disable lighting for the line
            glLineWidth(2.0)
            glBegin(GL_LINE_STRIP)
            glColor3f(0.4, 0.4, 0.4)  # Gray for the strip line
            for pos in coords:
                glVertex3f(pos[0], pos[1], pos[2])
            glEnd()
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)  # Re-enable lighting
            
            # Draw LED spheres if colors are available
            if colors is not None:
                for i, pos in enumerate(coords):
                    if i < len(colors):
                        color = colors[i]
                        self._draw_led_sphere(pos, 0.15, color)
            
            # Draw strip ID using billboard technique (facing camera)
            if len(coords) > 0:
                pos = coords[0]  # First LED position
                self._draw_billboard_text(strip_id, pos[0], pos[1] + 0.3, pos[2])
    
    def _draw_led_sphere(self, position, radius, color):
        """Draw a sphere at the given position with the given color"""
        x, y, z = position
        r, g, b, a = color
        
        # Set the color (with alpha)
        glColor4f(r, g, b, a)
        
        # Push the current matrix
        glPushMatrix()
        
        # Move to the sphere position
        glTranslatef(x, y, z)
        
        # Create a sphere
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluSphere(quad, radius, 10, 10)  # Create the sphere
        gluDeleteQuadric(quad)
        
        # Restore the matrix
        glPopMatrix()
    
    def _draw_billboard_text(self, text, x, y, z):
        """Draw text that always faces the camera (billboard technique)"""
        # We'll use simple colored cubes or planes instead of text for simplicity
        # and to avoid Qt painter issues
        
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.0)  # Yellow for visibility
        
        # Calculate the size of the billboard based on distance
        size = 0.2  # Base size
        
        glPushMatrix()
        glTranslatef(x, y, z)
        
        # Draw a simple marker
        glBegin(GL_QUADS)
        glVertex3f(-size/2, -size/2, 0)
        glVertex3f(size/2, -size/2, 0)
        glVertex3f(size/2, size/2, 0)
        glVertex3f(-size/2, size/2, 0)
        glEnd()
        
        glPopMatrix()
        glEnable(GL_LIGHTING)
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        self.last_mouse_pos = event.pos()
        
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            self.auto_rotate = False  # Stop auto-rotation when user interacts
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            self.right_mouse_pressed = True
            self.setCursor(Qt.SizeAllCursor)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = False
        elif event.button() == Qt.RightButton:
            self.right_mouse_pressed = False
            
        if not (self.mouse_pressed or self.right_mouse_pressed):
            self.setCursor(Qt.ArrowCursor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for rotation and panning"""
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        
        if self.mouse_pressed:  # Left button - rotate
            # Rotate camera position around target
            self.camera_azimuth += dx * 0.5
            self.camera_elevation += dy * 0.5
            
            # Clamp elevation to avoid gimbal lock
            self.camera_elevation = max(-89.0, min(89.0, self.camera_elevation))
            
            self.update()
        elif self.right_mouse_pressed:  # Right button - pan
            # Calculate pan amount in world space
            # This is simplified and doesn't account for camera orientation
            pan_speed = 0.02 * self.camera_distance
            
            # Calculate right vector (cross product of view direction and up vector)
            azimuth_rad = math.radians(self.camera_azimuth)
            right_x = math.cos(azimuth_rad)
            right_z = -math.sin(azimuth_rad)
            
            # Pan left/right
            self.camera_target[0] -= right_x * dx * pan_speed
            self.camera_target[2] -= right_z * dx * pan_speed
            
            # Pan up/down (simplified)
            self.camera_target[1] += dy * pan_speed
            
            self.update()
            
        self.last_mouse_pos = event.pos()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        zoom_factor = 1.1
        
        # Get the number of degrees scrolled
        delta = event.angleDelta().y()
        
        # Calculate zoom factor
        if delta > 0:
            # Zoom in
            self.camera_distance /= zoom_factor
        else:
            # Zoom out
            self.camera_distance *= zoom_factor
        
        # Clamp distance to reasonable values
        self.camera_distance = max(2.0, min(50.0, self.camera_distance))
        
        self.update()
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Space:
            # Toggle auto-rotation
            self.auto_rotate = not self.auto_rotate
            
        elif event.key() == Qt.Key_R:
            # Reset view
            self.camera_distance = 15.0
            self.camera_elevation = 30.0
            self.camera_azimuth = 0.0
            self.camera_target = [0.0, 0.0, 0.0]
            
        self.update()

def create_strip_visualizer(strip_manager):
    """Creates a 3D visualizer for LED strips using PyQt and OpenGL"""
    if not OPENGL_AVAILABLE:
        print("PyOpenGL is required for 3D visualization.")
        print("Install with: pip install PyOpenGL")
        return None
    
    try:
        # Check if PyQt5 is installed
        import PyQt5
        visualizer = LED3DVisualizer(strip_manager)
        return visualizer
    except ImportError as e:
        print(f"PyQt5 is not installed. Error: {e}")
        print("To install PyQt5, run: pip install PyQt5")
        return None
    except Exception as e:
        print(f"Error creating 3D visualizer: {e}")
        return None