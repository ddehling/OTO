import numpy as np
from sacn import sACNsender
import math
class SACNPixelSender:
    def __init__(self, receivers, start_universe=1):
        """
        Initialize the SACNPixelSender with receiver configurations.
        :param receivers: List of dicts, each with 'ip', 'pixel_count', and 'addressing_array' keys.
        """
        self.receivers = receivers
        self.sender = sACNsender()
        self.sender.start()

        # Set up universes for each receiver
        self.receiver_universes = []
        universe_counter = start_universe
        
        # First process RGB strips
        rgb_receivers = [r for r in receivers if r.get('type', 'RGB') == 'RGB']
        if rgb_receivers:
            print(f"Setting up RGB universes starting at {universe_counter}")
            for receiver in rgb_receivers:
                # Each RGB universe can hold 170 pixels (510 bytes)
                universe_count = math.ceil(receiver['pixel_count'] / 170)
                receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                self.receiver_universes.append(receiver_universes)
                universe_counter += universe_count

                # Activate universes for this receiver
                for universe in receiver_universes:
                    self.sender.activate_output(universe)
                    self.sender[universe].destination = receiver['ip']
        
        # Calculate the start for RGBW universes
        rgbw_start_universe = universe_counter
        
        # Then process RGBW strips
        rgbw_receivers = [r for r in receivers if r.get('type', 'RGBW') == 'RGBW']
        if rgbw_receivers:
            print(f"Setting up RGBW universes starting at {rgbw_start_universe}")
            for receiver in rgbw_receivers:
                # Each RGBW universe can hold 128 pixels (512 bytes)
                universe_count = math.ceil(receiver['pixel_count'] / 128)
                receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                self.receiver_universes.append(receiver_universes)
                universe_counter += universe_count

                # Activate universes for this receiver
                for universe in receiver_universes:
                    self.sender.activate_output(universe)
                    self.sender[universe].destination = receiver['ip']

        # Calculate the start for DMX universes
        dmx_start_universe = universe_counter
        
        # Finally process DMX strips
        dmx_receivers = [r for r in receivers if r.get('type', 'DMX') == 'DMX']
        if dmx_receivers:
            print(f"Setting up DMX universes starting at {dmx_start_universe}")
            for receiver in dmx_receivers:
                # Each DMX universe can hold 73 pixels (511 bytes, leaving 1 byte unused)
                universe_count = math.ceil(receiver['pixel_count'] / 73)
                receiver_universes = list(range(universe_counter, universe_counter + universe_count))
                self.receiver_universes.append(receiver_universes)
                universe_counter += universe_count

                # Activate universes for this receiver
                for universe in receiver_universes:
                    self.sender.activate_output(universe)
                    self.sender[universe].destination = receiver['ip']

    def create_mask(self, height, width):
        """
        Creates a binary mask showing which pixels are mapped by receivers.
        
        :param height: Height of the source image
        :param width: Width of the source image
        :return: numpy array of shape (height, width) with 1s where pixels are mapped
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for receiver in self.receivers:
            # Clip coordinates to valid image boundaries
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, height - 1)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, width - 1)
            
            # Set mapped pixels to 1
            mask[x_coords, y_coords] = 1
            
        return mask

    def send(self, source_array):
        """
        Send pixel data to all configured receivers based on their addressing arrays.
        :param source_array: numpy array of shape (strips, pixels, 3) containing source pixel data.
        """
        for receiver, universes in zip(self.receivers, self.receiver_universes):
            # Vectorized extraction of pixel data
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, source_array.shape[0] - 1)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, source_array.shape[1] - 1)
            
            # Extract the RGB values for each pixel
            receiver_data = source_array[x_coords, y_coords]

            # Send data in 170-pixel chunks
            for i, universe in enumerate(universes):
                start = i * 170
                end = min(start + 170, receiver['pixel_count'])
                universe_data = receiver_data[start:end].flatten()
                # Pad the last universe if necessary
                if universe_data.size < 510:
                    universe_data = np.pad(universe_data, (0, 510 - universe_data.size), 'constant')
                self.sender[universe].dmx_data = universe_data.tobytes()

    def send_from_buffers(self, output_buffers, strip_info):
        """
        Send pixel data directly from output buffers.
        
        :param output_buffers: Dictionary mapping strip_id to buffer arrays
        :param strip_info: List of tuples (strip_id, length, direction, type)
        """
        gamma = 2.2
        for receiver, universes in zip(self.receivers, self.receiver_universes):
            # Concatenate all strip data for this receiver
            all_pixel_data = []
            
            # Get the receiver type (RGB, RGBW, or DMX)
            receiver_type = receiver.get('type', 'RGB')
            
            for strip_id, length, direction, strip_type in strip_info:
                # Skip if strip type doesn't match receiver type
                if strip_type != receiver_type:
                    continue
                    
                # Get the buffer for this strip
                buffer = output_buffers[strip_id]
                
                # Convert from float (0-1) to uint8 (0-255)
                rgb_data = (np.power(buffer[:, :3], gamma) * 255).astype(np.uint8)
                
                # Reverse the strip if direction is -1
                if direction == -1:
                    rgb_data = rgb_data[::-1]
                    
                # Add to the concatenated data
                all_pixel_data.append(rgb_data)
                
            if not all_pixel_data:
                continue  # Skip if no strips match this receiver
                
            # Concatenate all strips
            receiver_data = np.concatenate(all_pixel_data)
            
            # Calculate pixels per universe based on type
            if receiver_type == 'RGB':
                pixels_per_universe = 170  # 170 RGB pixels per universe
            elif receiver_type == 'RGBW':
                pixels_per_universe = 128  # 128 RGBW pixels per universe
            else:  # DMX
                pixels_per_universe = 73   # 73 DMX pixels per universe
            
            # Send data in chunks
            for i, universe in enumerate(universes):
                start = i * pixels_per_universe
                end = min(start + pixels_per_universe, len(receiver_data))
                
                # Check if we have enough data
                if start < len(receiver_data):
                    # Get the data for this universe
                    universe_data = receiver_data[start:end]
                    
                    if receiver_type == 'RGB':
                        # Flatten the RGB data
                        universe_data = universe_data.flatten()
                        
                        # Pad the last universe if necessary
                        if universe_data.size < 510:  # 510 = 170 pixels * 3 bytes
                            universe_data = np.pad(universe_data, (0, 510 - universe_data.size), 'constant')
                            
                    elif receiver_type == 'RGBW':
                        # Create RGBW data with W set to 0
                        rgbw_data = np.zeros((universe_data.shape[0], 4), dtype=np.uint8)
                        rgbw_data[:, :3] = universe_data  # Set RGB components
                        
                        # Flatten the RGBW data
                        universe_data = rgbw_data.flatten()
                        
                        # Pad if necessary
                        if universe_data.size < 512:  # 512 = 128 pixels * 4 bytes
                            universe_data = np.pad(universe_data, (0, 512 - universe_data.size), 'constant')
                            
                    else:  # DMX
                        # Create DMX data: [255, R, G, B, 0, 0, 0] for each pixel
                        dmx_data = np.zeros((universe_data.shape[0], 7), dtype=np.uint8)
                        dmx_data[:, 0] = 255  # First byte is always 255
                        dmx_data[:, 1:4] = universe_data  # Set RGB components
                        # Bytes 4-6 are already 0
                        
                        # Flatten the DMX data
                        universe_data = dmx_data.flatten()
                        
                        # Pad if necessary (ensure we have a full universe of 512 bytes)
                        if universe_data.size < 512:
                            universe_data = np.pad(universe_data, (0, 512 - universe_data.size), 'constant')
                    
                    # Send the data
                    self.sender[universe].dmx_data = universe_data.tobytes()

    def close(self):
        """
        Properly close the sACN sender
        """
        self.sender.stop()

    def analyze_row_groups(self, max_pixels_per_group=170):
        """
        Analyze and group pixels in rows that belong to the same receiver.
        
        :param max_pixels_per_group: Maximum number of pixels per group (default 170 for sACN universe limit)
        :return: Dictionary mapping receiver indices to lists of pixel groups
        """
        receiver_groups = {}
        
        for idx, receiver in enumerate(self.receivers):
            # Get coordinates from addressing array
            coordinates = receiver['addressing_array']
            #find the unique rows
            rows = np.unique(coordinates[:, 0])
            #find the number of pixels in each row
            row_counts = {row: np.sum(coordinates[:, 0] == row) for row in rows}         
            #initialize the group list
            groups = []
            #initialize the current group
            current_group = []
            #initialize the current row
            current_row = None
            #iterate through the rows
            group_pixel_count = 0
            pixels_in_group = []
            for row in rows:
                if current_row is None:
                    current_row = row
                #if the row is different from the current row
                row_count = row_counts[row]
                if (group_pixel_count + row_count) <= max_pixels_per_group:
                    #store the current group
                    current_group.append(row)
                    #reset the group count
                    group_pixel_count += row_count
                    #reset the current group
                else:
                    groups.append(current_group)
                    current_group = [row]
                    pixels_in_group.append(group_pixel_count)
                    group_pixel_count = row_count
            #handle the last group
            groups.append(current_group)
            pixels_in_group.append(group_pixel_count)
            print(groups, pixels_in_group,receiver['ip'])

# The rest of the code (generate_frame_data and main function) remains the same

def generate_frame_data():
    """
    Generate random RGB pixel data for each frame.
    In a real application, this would pull data from your actual source.
    """
    width, height = 100, 100  # Example dimensions
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

def make_indicesHS(filename):
    in_list=np.loadtxt(filename, delimiter=',').tolist()
    indices = []
    for sublist in in_list:       
        if sublist[2]>0:
            for m in range(int(sublist[2])):
                indices.append([sublist[0], m+sublist[1]])
        else:
            for m in range(int(-sublist[2])):
                indices.append([sublist[0], sublist[1]-sublist[2]-1-m])   
    return np.array(indices).astype(int)

def main():
    receivers = [
        {
            'ip': '192.168.68.121',
            'pixel_count': 3500,
            #'addressing_array':make_indicesH()
            'addressing_array':make_indicesHS(r"data.txt")
        }
    ]

    sender = SACNPixelSender(receivers)
    sender.analyze_row_groups(255)
    sender.close()

if __name__ == "__main__":
    main()