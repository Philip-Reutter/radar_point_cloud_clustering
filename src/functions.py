import json
import h5py

def load_json(file_path):
    """Load the scenes data from the JSON file."""
    with open(file_path, 'r') as file:
        scenes_data = json.load(file)
    return scenes_data

# gets newest data from exactly 4 sensors
def getdata_all_sensors(current_timestamp, scenes):
    data = []
    latest_sensor_data = {}
    while current_timestamp and len(latest_sensor_data)<4:
        scene = scenes.get(current_timestamp)
        if not scene:
            current_timestamp = None
            break
        latest_sensor_data[scene["sensor_id"]] = (
            scene["odometry_index"],
            scene["image_name"],
            scene["radar_indices"]
        )
        current_timestamp = str(scene["next_timestamp"])
    data = list(latest_sensor_data.values())
    if len(data)>3:
        return data, current_timestamp
    else:
        return None, None

# retrieve radar data based on given indices.
def get_radar_data(hdf5_file, radar_indices, ego_in_center=True, rcs_requested=False):
    with h5py.File(hdf5_file, "r") as f:
        radar_data = f["radar_data"][:]
        if ego_in_center:
            radar_x_scene = -radar_data[radar_indices[0]:radar_indices[1]]["y_cc"]
            radar_y_scene = radar_data[radar_indices[0]:radar_indices[1]]["x_cc"]
        else:
            radar_x_scene = radar_data[radar_indices[0]:radar_indices[1]]["x_seq"]
            radar_y_scene = radar_data[radar_indices[0]:radar_indices[1]]["y_seq"]
        label_id = radar_data[radar_indices[0]:radar_indices[1]]["label_id"]
        vr_compensated = radar_data[radar_indices[0]:radar_indices[1]]["vr_compensated"]
        v = [value if abs(value) > 0.1 else 0 for value in vr_compensated]
        if rcs_requested:
            rcs = radar_data[radar_indices[0]:radar_indices[1]]["rcs"]
            return radar_x_scene, radar_y_scene, v, label_id, rcs
        return radar_x_scene, radar_y_scene, v, label_id

def get_odometry_data(hdf5_file):
    with h5py.File(hdf5_file, "r") as f:
        odometry = f["odometry"][:]
    return odometry
