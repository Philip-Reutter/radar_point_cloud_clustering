import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

from functions import load_json, getdata_all_sensors, get_radar_data, get_odometry_data
from constants import COLOR_CLASS_DICT, HDF5_FILE, JSON_FILE, IMAGES_PATH 

ego_in_center = True
if ego_in_center:
    x_min, x_max = -100, 100
    y_min, y_max = -50, 150
else:
    x_min, x_max = -1000, 1000
    y_min, y_max = -1000, 1000

# Update the plot with new data.
def update_plot(frame):
    global all_data, radar_x_scene, radar_y_scene, ego_x, ego_y, img, ax

    # Extract current frame data
    data = all_data[frame]

    odometry = get_odometry_data(HDF5_FILE)

    odometry_index = data[0][0]
    image = data[0][1]
    ranges = [item[2] for item in data]
    merged_range = [ranges[0][0], ranges[-1][1]]
    radar_indices = merged_range

    if ego_in_center:
        ego_x = 0
        ego_y = 0
    else:
        ego_x = odometry[odometry_index]["x_seq"]
        ego_y = odometry[odometry_index]["y_seq"]
    
    # Extract radar data for these indices
    radar_x_scene, radar_y_scene, v, label_id = get_radar_data(HDF5_FILE, radar_indices, ego_in_center)
    label = [COLOR_CLASS_DICT[label] for label in label_id]
    radar_scene = [[radar_x_scene[i], radar_y_scene[i], label[i]] for i in range(len(radar_x_scene))]

    image_path = f"{IMAGES_PATH}/{image}"
    img = mpimg.imread(image_path)

    # Clear previous data in axes
    ax[0].clear()
    ax[1].clear()
    
    # Left subplot: Update radar detections and ego vehicle position
    ax[0].scatter(radar_x_scene, radar_y_scene, color=[sublist[1] for sublist in label], s=3, label="Radar Detections")
    ax[0].scatter(ego_x, ego_y, color="red", s=100, label="Ego Vehicle")
    ax[0].set_xlabel("X Position (m)")
    ax[0].set_ylabel("Y Position (m)")
    ax[0].set_title("Radar Detections and Ego Vehicle")
    ax[0].legend()
    ax[0].grid()

    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)
    ax[0].set_aspect("equal", adjustable="box")

    # Right subplot: Update image
    ax[1].imshow(img)
    ax[1].axis("off")
    ax[1].set_title(f"Image: {image}")

    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05, wspace=0.02)

def main():
    global all_data, fig, ax

    scenes_data = load_json(JSON_FILE)
    all_data = []
    
    current_timestamp = str(scenes_data["first_timestamp"])
    scenes = scenes_data["scenes"]

    # Collect all data
    while current_timestamp:
        data, current_timestamp = getdata_all_sensors(current_timestamp, scenes)
        all_data.append(data)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ani = FuncAnimation(fig, update_plot, frames=len(all_data), interval=100, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
