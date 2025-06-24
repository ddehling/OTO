import numpy as np
import time
from corefunctions.Events import EventScheduler
from sceneutils.OTO_dev import *  # noqa: F403
 # noqa: F403
#
class EnvironmentalSystem:
    def __init__(self, scheduler):
        self.scheduler = scheduler

        self.transition_time = 0
        self.transition_start = 0
        self.progress = 0

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

        self.scheduler.state["time"] = self.current_time


    def random_events(self):
        randcheck = np.random.random()

                


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
   
    env_system.scheduler.schedule_event(0, 20, test)
    env_system.scheduler.schedule_event(10, 40, OTO_blink)  # noqa: F405
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