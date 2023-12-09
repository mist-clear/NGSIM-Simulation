import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load CSV file
trajectories = pd.read_csv('trajectories-0750am-0805am.csv').to_numpy()

# Section limits
section_limits = [500, 900]

# Limits of x-y axis
x_limit = [section_limits[0] - 5, section_limits[1] + 5]
y_limit = [-0, 80]

frames = np.unique(trajectories[:, 1])
plt.figure(figsize=(20, 4))
for i in range(len(frames)):
    frame_data = trajectories[(trajectories[:, 1] == frames[i]) &
                              (trajectories[:, 5] >= section_limits[0]) &
                              (trajectories[:, 5] <= section_limits[1])]

    if frame_data.size == 0:
        continue

    # Get needed fields
    lateral_pos = frame_data[:, 4]
    longitude_pos = frame_data[:, 5]
    vehicle_id = frame_data[:, 0]
    length = frame_data[:, 8]
    width = frame_data[:, 9]
    vehicle_class = frame_data[:, 10]

    # Construct vehicle bounding boxes
    bounding_box_arr = np.column_stack((longitude_pos - length, lateral_pos - width / 2, length, width))

    # Set title
    plt.title(f'NGSIM US-101 trajectories - frame: {int(frames[i])}')

    # Plot road boundaries
    plt.plot(x_limit, [0, 0], color='black', linestyle='--')
    plt.plot(x_limit, [12.3, 12.3], color='blue', linestyle='-.')
    plt.plot(x_limit, [24.6, 24.6], color='blue', linestyle='-.')
    plt.plot(x_limit, [36.9, 36.9], color='blue', linestyle='-.')
    plt.plot(x_limit, [49.2, 49.2], color='blue', linestyle='-.')
    plt.plot(x_limit, [61.5, 61.5], color='blue', linestyle='-.')
    plt.plot(x_limit, [73.8, 73.8], color='blue', linestyle='--')

    # Plot vehicle bounding boxes according to vehicle class
    # Red -> Motorcycle, Yellow -> Auto, Green -> Truck
    for j in range(len(bounding_box_arr)):
        if vehicle_class[j] == 1:
            plt.gca().add_patch(patches.Rectangle((bounding_box_arr[j, 0], bounding_box_arr[j, 1]),
                                                  bounding_box_arr[j, 2], bounding_box_arr[j, 3],
                                                  facecolor='yellow'))
        elif vehicle_class[j] == 2:
            plt.gca().add_patch(patches.Rectangle((bounding_box_arr[j, 0], bounding_box_arr[j, 1]),
                                                  bounding_box_arr[j, 2], bounding_box_arr[j, 3],
                                                  facecolor='red'))
        else:
            plt.gca().add_patch(patches.Rectangle((bounding_box_arr[j, 0], bounding_box_arr[j, 1]),
                                                  bounding_box_arr[j, 2], bounding_box_arr[j, 3],
                                                  facecolor='green'))

    # Add vehicle id to each vehicle
    for j in range(len(vehicle_id)):
        plt.text(longitude_pos[j] - 2 * length[j] / 3, lateral_pos[j], str(int(vehicle_id[j])),
                 color='blue', fontsize=8, ha='left', va='bottom')

    # Create custom legend for vehicle classes
    red_patch = patches.Patch(color='yellow', label='Motorcycle')
    yellow_patch = patches.Patch(color='red', label='Auto')
    green_patch = patches.Patch(color='green', label='Truck')
    plt.legend(handles=[red_patch, yellow_patch, green_patch], loc='upper right', fontsize=14)
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.pause(0.001)
    plt.clf()
