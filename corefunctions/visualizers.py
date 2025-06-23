import cv2
import numpy as np

def create_strip_visualizer(strip_manager):
    """Creates an OpenCV-based visualizer for LED strips"""
    
    class StripVisualizer:
        def __init__(self, strip_manager):
            self.strip_manager = strip_manager
            self.window_name = "LED Strip Visualization"
            self.led_size = 10  # Size of each LED in pixels
            self.spacing = 5    # Spacing between strips
            self.max_length = max(strip.length for strip in strip_manager.strips.values())
            
            # Calculate canvas size
            self.width = self.max_length * (self.led_size + 2) + 200  # Extra space for labels
            self.height = len(strip_manager.strips) * (self.led_size + self.spacing) + 50
            
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
        
        def update(self, output_buffers):
            # Create blank canvas
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw each strip
            y_pos = 30
            for i, (strip_id, buffer) in enumerate(output_buffers.items()):
                # Draw strip label
                cv2.putText(canvas, strip_id, (10, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw each LED
                for j in range(len(buffer)):
                    x = j * (self.led_size + 2) + 150  # Offset for labels
                    
                    # Get color (BGR for OpenCV)
                    color = buffer[j, :3] * 255
                    color_bgr = (int(color[2]), int(color[1]), int(color[0]))
                    
                    # Draw LED
                    cv2.rectangle(canvas, 
                                 (x, y_pos), 
                                 (x + self.led_size, y_pos + self.led_size),
                                 color_bgr, -1)
                
                y_pos += self.led_size + self.spacing
            
            # Show the result
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)  # Small delay
        
        def close(self):
            cv2.destroyWindow(self.window_name)
    
    return StripVisualizer(strip_manager)