import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
import numpy as np
from pynput import keyboard

from clustering_DBSCAN import cluster_radar_points_dbscan
from clustering_Meanshift import cluster_radar_points_meanshift
from functions import load_json, get_radar_data, getdata_all_sensors
from constants import HDF5_FILE, JSON_FILE, IMAGES_PATH

paused = False # global pause state
ego_in_center = True
if ego_in_center:
    #x_min, x_max = -100, 100
    #y_min, y_max = -50, 150
    x_min, x_max = -50, 50
    y_min, y_max = -10, 80
else:
    x_min, x_max = -1000, 1000
    y_min, y_max = -1000, 1000

def toggle_pause():
    """Toggle the pause state of the animation and code execution."""
    global paused
    paused = not paused

# Keyboard listener to toggle pause on Space key press
def on_press(key):
    try:
        if key == keyboard.Key.space:
            toggle_pause()
    except AttributeError:
        pass

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()

def wait_while_paused():
    """Pause code execution if the paused flag is set."""
    while paused:
        plt.pause(0.1) # Check pause state every 100ms

def update_plot(frame):
    """Update the plot with new data."""
    global all_data, radar_x_scene, radar_y_scene, radar_vr_compensated_scene, ego_x, ego_y, img, ax

    wait_while_paused() # Pause if the global paused state is active

    data = all_data[frame]

    with h5py.File(HDF5_FILE, "r") as f:
        odometry = f["odometry"][:]

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

        radar_data = get_radar_data(HDF5_FILE, radar_indices)
        radar_x_scene, radar_y_scene, radar_vr_compensated_scene = radar_data[:3]

        cluster_labels_dbscan = cluster_radar_points_dbscan(radar_x_scene, radar_y_scene, radar_vr_compensated_scene)
        cluster_labels_meanshift = cluster_radar_points_meanshift(radar_x_scene, radar_y_scene, radar_vr_compensated_scene)

    img_path = IMAGES_PATH + f"/{image}"
    img = mpimg.imread(img_path)

    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    unique_labels_dbscan = set(cluster_labels_dbscan)
    for label in unique_labels_dbscan:
        if label == -1:
            color = "gray"
        else:
            color = plt.cm.tab10(label % 10)
        ax[0].scatter(
            np.array(radar_x_scene)[cluster_labels_dbscan == label],
            np.array(radar_y_scene)[cluster_labels_dbscan == label],
            color=color,
            #s=3,
            s=10,
            label=f"Cluster {label}" if label != -1 else "Noise"
        )

    unique_labels_meanshift = set(cluster_labels_meanshift)
    for label in unique_labels_meanshift:
        if label == -1:
            color = "gray"
        else:
            color = plt.cm.tab10(label % 10)
        ax[2].scatter(
            np.array(radar_x_scene)[cluster_labels_meanshift == label],
            np.array(radar_y_scene)[cluster_labels_meanshift == label],
            color=color,
            s=3,
            label=f"Cluster {label}" if label != -1 else "Noise"
        )

    #ax[0].scatter(ego_x, ego_y, color="red", s=100, label="Ego Vehicle")
    ax[0].scatter(ego_x, ego_y, color="black", s=100, label="Ego Vehicle")
    ax[0].set_xlabel("X Position (m)", fontsize=12)
    ax[0].set_ylabel("Y Position (m)", fontsize=12)
    ax[0].set_title("Radar Detections with DBSCAN Clustering", fontsize=16)
    ax[0].legend(loc="upper right")
    ax[0].grid()
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)
    ax[0].set_aspect("equal", adjustable="box")

    ax[2].scatter(ego_x, ego_y, color="red", s=100, label="Ego Vehicle")
    ax[2].set_xlabel("X Position (m)", fontsize=12)
    ax[2].set_ylabel("Y Position (m)", fontsize=12)
    ax[2].set_title("Radar Detections with MeanShift Clustering", fontsize=16)
    ax[2].legend(loc="upper right")
    ax[2].grid()
    ax[2].set_xlim(x_min, x_max)
    ax[2].set_ylim(y_min, y_max)
    ax[2].set_aspect("equal", adjustable="box")

    ax[1].imshow(img)
    ax[1].axis("off")
    ax[1].set_title(f"Image: {image}", fontsize=16)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.15)

def main():
    global all_data, fig, ax

    scenes_data = load_json(JSON_FILE)
    all_data = []

    current_timestamp = str(scenes_data["first_timestamp"])
    scenes = scenes_data["scenes"]

    while current_timestamp:
        wait_while_paused() # Pause the data processing if needed
        data, current_timestamp = getdata_all_sensors(current_timestamp, scenes)
        all_data.append(data)

    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    ani = FuncAnimation(fig, update_plot, frames=len(all_data), interval=50, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
