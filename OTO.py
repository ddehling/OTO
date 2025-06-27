import numpy as np
import time
from corefunctions.Events import EventScheduler
from sceneutils.OTO_dev import *  # noqa: F403
from corefunctions.input_panel import InputPanel 
 # noqa: F403
#
class EnvironmentalSystem:
    def __init__(self, scheduler):
        self.scheduler = scheduler

        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0
        self.input_panel = InputPanel(title="OTO Control Panel")
        self.input_panel.values_changed.connect(self.on_input_values_changed)
        
        # Store the latest input values
        self.input_values = self.input_panel.get_values()

                # Initialize smoothed values dictionary to track and smooth values
        self.smoothed_values = self.input_values.copy()
        
        # Set smoothing factor (0-1, lower values mean more smoothing)
        self.smoothing_factor = 0.01
        
        # Track last update time for consistent smoothing
        self.last_update_time = time.time()

    def on_input_values_changed(self, values):
        """Handle updates from the input panel"""
        self.input_values = values
        
        # You can do immediate processing of values here if needed
        # For example, apply intensity to global settings
        
        # Pass input values to scheduler state

        

    def get_control_values(self):
        """Return the current control values from the input panel"""
        return self.input_panel.get_values()

    def transition_update(self):
        yo=1
        # self.progress = 1.0
        # if self.current_weather != self.target_weather:
        #     self.progress = min(
        #         1.0, (self.current_time - self.transition_start) / self.transition_time
        #     )

        #     start_params = self.get_weather_params(self.current_weather)
        #     target_params = self.get_weather_params(self.target_weather)

        #     # Interpolate parameters
        #     for param in target_params:
        #         if isinstance(target_params[param], (int, float, np.ndarray)):
        #             self.weather_params[param] = (
        #                 target_params[param] - start_params[param]
        #             ) * self.progress + start_params[param]
        #         else:
        #             # For non-numeric parameters, just use the target value
        #             self.weather_params[param] = target_params[param]

        #     if self.progress >= 1.0:
        #         self.current_weather = self.target_weather
        #         self.weather_params = target_params.copy()

    def send_variables(self):
        """Send variables to scheduler with exponential smoothing"""
        self.scheduler.state["time"] = self.current_time
        
        # Get the latest control values
        raw_values = self.input_panel.get_values()
        
        # Calculate elapsed time since last update for time-based smoothing
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # Adjust smoothing factor based on time elapsed to ensure consistent behavior
        # regardless of frame rate
        frame_smoothing = min(1.0, self.smoothing_factor * dt * 60)  # Normalize to 60fps
        
        # Apply exponential smoothing and send to scheduler
        for key, raw_value in raw_values.items():
            if key in ['selected_mode', 'selected_effect']:
                # For discrete values like selections, we don't smooth
                smooth_value = raw_value
                self.smoothed_values[key] = smooth_value
            else:
                # For continuous values, apply exponential smoothing
                if key in self.smoothed_values:
                    current_smooth = self.smoothed_values[key]
                    smooth_value = current_smooth + frame_smoothing * (raw_value - current_smooth)
                    self.smoothed_values[key] = smooth_value
                else:
                    # Initialize if not exists
                    smooth_value = raw_value
                    self.smoothed_values[key] = smooth_value
            
            # Store smoothed value in scheduler state
            if isinstance(smooth_value, (int, float)) and not isinstance(smooth_value, bool):
                self.scheduler.state[f"control_{key}"] = round(smooth_value, 2)
            else:
                self.scheduler.state[f"control_{key}"] = smooth_value

            
    def random_events(self):
        randcheck = np.random.random()

    def get_control_values(self):
        """Return the current raw control values from the input panel"""
        return self.input_panel.get_values()
    
    def get_smoothed_values(self):
        """Return the current smoothed values"""
        return self.smoothed_values.copy()
    
    def set_smoothing_factor(self, factor):
        """Set the smoothing factor (0-1, lower means more smoothing)"""
        self.smoothing_factor = max(0.0, min(1.0, factor))                


    def update(self):
        """Update the environmental system - should be called each frame"""

        self.current_time = time.time()
       
        # OSC handling
        messages = self.scheduler.get_osc_messages()
        if messages != []:
            print(messages)  # Eventually want to pass these to the scheduler
            
        # Handle transitions
        self.transition_update()

        # Update celestial bodies

            
        # Apply current parameters to scheduler state
        self.send_variables()
        
        # Random events
        self.random_events()

        
        # Update the scheduler
        self.scheduler.update()


# Main execution
if __name__ == "__main__":
    scheduler = EventScheduler()
    env_system = EnvironmentalSystem(scheduler)
    scheduler.setup_visualizer(True) 
    # Start with summer bloom weather
   
    #env_system.scheduler.schedule_event(0, 240, OTO_heartbeat)# noqa: F405
    #env_system.scheduler.schedule_event(00, 80, OTO_blink)  # noqa: F405
    env_system.scheduler.schedule_event(00, 80, OTO_point_traveler)
    lasttime = time.time()
    FRAME_TIME = 1 / 40
    first_time = time.perf_counter()
    try:
        while True:
            # Update environmental system
            env_system.update()

            current_time = time.perf_counter()
            
            elapsed = current_time - lasttime
            sleep_time = max(0, FRAME_TIME - elapsed)
            #time.sleep(sleep_time)

            # Print stats if needed
            #print(["%.2f" % (1/(time.perf_counter()-lasttime)), "%.2f" % len(scheduler.active_events), len(scheduler.event_queue),"%.3f" %((lasttime-first_time)/3600)])
            lasttime = time.perf_counter()

    except KeyboardInterrupt:
        print("Done!")