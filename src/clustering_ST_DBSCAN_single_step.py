import numpy as np
import matplotlib.pyplot as plt

from functions import load_json, getdata_all_sensors, get_radar_data
from constants import SCENE, HDF5_FILE, JSON_FILE

class ST_DBSCAN:
    def __init__(self, eps, eps_t, min_pts, min_pts_new_cluster, alpha):
        """
        Initialize the ST-DBSCAN algorithm.

        Parameters:
        - eps: Spatial distance threshold.
        - eps_t: Temporal distance threshold.
        - min_pts: Minimum number of points to form a cluster.
        - alpha: Weight of velocity in distance calculation
        """
        self.eps = eps
        self.eps_t = eps_t
        self.min_pts = min_pts
        self.min_pts_new_cluster = min_pts_new_cluster
        self.alpha = alpha
    
    def fit(self, data):
        """
        Apply ST-DBSCAN clustering to the data.

        Parameters:
        - data: numpy array of shape (n_samples, 4), where each row is [x, y, v, timestamp].

        Returns:
        - labels: numpy array of cluster labels for each point (-1 indicates noise).
        """
        n_samples = np.shape(data)[0]
        timestamps = np.array(data[:, 3])
        self.labels = np.array(data[:, 5]) # get the current labels, at start it is -1 for every point
        current_timestamp = timestamps[-1]
        used_clusters = np.unique(self.labels)
        self.cluster_ids = [i for i in range(50) if i not in used_clusters] # allocate 50 new cluster ids
        visited = np.zeros(n_samples, dtype=bool)  # Track visited points

        labeled_indices = [i for i, value in enumerate(self.labels) if value != -1] # get all previously labeled points
        new_points = [i for i, value in enumerate(data) if value[3] == current_timestamp] # get all new points
        
        # get indices of labled points by cluster size
        unique, counts = np.unique(data[labeled_indices, 5], return_counts=True)
        value_count_pairs = list(zip(unique, counts))
        sorted_by_count = sorted(value_count_pairs, key=lambda x: x[1], reverse=True)
        filtered_indices = []
        if labeled_indices:
            for value, _ in sorted_by_count:
                value_indices = np.where(data[:, 5] == value)[0]
                filtered_indices.extend(value_indices)
            filtered_indices = np.array(filtered_indices)

        for i in filtered_indices: # iterate over all previously labeled points, starting with the largest cluster
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._get_neighbors(i, data) # get all new neigbors of previous clusters

            if len(neighbors) >= self.min_pts:
                self._expand_cluster(i, neighbors, visited, data, self.min_pts)

        for i in new_points: # iterate over all new points
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._get_neighbors(i, data)

            if len(neighbors) < self.min_pts_new_cluster:
                self.labels[i] = -1 # Mark as noise
            else:
                self._expand_cluster(i, neighbors, visited, data, self.min_pts_new_cluster)

        data[:,5] = self.labels
        return self.labels, data

    def _get_neighbors(self, idx, data):
        """
        Find neighbors of a point within spatial and temporal thresholds.

        Parameters:
        - idx: Index of the point.
        - data: numpy array of shape (n_samples, 4).

        Returns:
        - neighbors: List of indices of neighboring points.
        """
        neighbors = []
        x, y, v, t , _, _ = data[idx]
        for j in range(data.shape[0]):
            if j == idx:
                continue

            x2, y2, v2, t2, _, cluster_id2 = data[j]
            spatial_dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            temporal_dist = abs(t - t2)
            velocity_diff = abs(v - v2)
            total_dist = spatial_dist + self.alpha * velocity_diff
            if temporal_dist <= self.eps_t and total_dist <= self.eps:
                neighbors.append(j)

        return neighbors

    def _expand_cluster(self, idx, neighbors, visited, data, min_pts):
        """
        Expand the cluster from a seed point.

        Parameters:
        - idx: Index of the seed point.
        - neighbors: List of indices of neighboring points.
        - visited: Array indicating visited points.
        - data: numpy array of shape (n_samples, 4).
        """
        if self.labels[idx] == -1:
            cluster_id = self.cluster_ids.pop(0)
            self.labels[idx] = cluster_id
        else:
            cluster_id = self.labels[idx]

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._get_neighbors(neighbor_idx, data)
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)

            if self.labels[neighbor_idx] == -1:  # If it was labeled as noise, now it's part of the cluster
                self.labels[neighbor_idx] = cluster_id
            i += 1


def get_data_from_one_sequence():
    scenes_data = load_json(JSON_FILE)
    all_data = []
    current_timestamp = str(scenes_data["first_timestamp"])
    scenes = scenes_data["scenes"]

    # collect all timestamp/index/radar_range data 
    j = 0 # number of timestamp
    while current_timestamp:
        j += 1
        data, current_timestamp = getdata_all_sensors(current_timestamp, scenes)
        if data:
            all_data.append(data + [j])

    # iterate over data for every measurement-group to get radar data
    radar_x = []
    radar_y = []
    v = []
    label_ids = []
    for i in range(len(all_data)//2):
        data = all_data[i]
        ranges = [item[2] for item in data[:len(data)-1]] # exclude number of timestamp j
        merged_range = [ranges[0][0], ranges[-1][1]] # range from start of first item to end of last item
        radar_x_scene, radar_y_scene, v_scene, label_id = get_radar_data(HDF5_FILE, merged_range)
        radar_x.append(radar_x_scene)
        radar_y.append(radar_y_scene)
        v.append(v_scene)
        label_ids.append(label_id)

    timestamp = np.array([timestamp[-1] for timestamp in all_data[:len(all_data)//2]])
    combined_data = []

    for i in range(len(timestamp)):
        x_vals = radar_x[i]
        y_vals = radar_y[i]
        v_vals = v[i]
        ts = timestamp[i]
        gt_vals = label_ids[i]
        for j in range(len(x_vals)):
            combined_data.append((x_vals[j], y_vals[j], v_vals[j], ts, gt_vals[j]))

    # Convert the combined_data list to a numpy array
    combined_data = np.array(combined_data)
    np.save(f"data{SCENE}.npy", combined_data) # save the data to a file

    return combined_data

def cluster(data):
    # Filter data
    x_min, x_max = -50, 50
    y_min, y_max = -10, 80
    v_min, v_max = -60, 60
    condition_v = (data[:, 2] <= v_max) & (data[:, 2] >= v_min)  # velocity <= 60 m/s and velocity >= -60 m/s
    condition_x = (data[:, 0] <= x_max) & (data[:, 0] >= x_min)  # |x| <= 100 and x >= -20
    condition_y = (data[:, 1] <= y_max) & (data[:, 1] >= y_min)  # y <= 50 and y >= -50
    condition = condition_v & condition_x & condition_y
    data = data[condition]

    length_of_column = data.shape[0]
    new_column = -np.ones(length_of_column)
    data = np.append(data, new_column[:, None], axis=1) # add label id = -1 column to the data array

    scan = ST_DBSCAN(eps=3, eps_t=2, min_pts=8, min_pts_new_cluster=8, alpha=0.3)
    for i in range(3, len(np.unique(data[:, 3]))): # iterate over all timestamps, starting from the third
        labels, new_data = scan.fit(data[np.isin(data[:, 3], [i, i-1, i-2])]) # fit the data for the current timestamp and the two previous ones
        data[np.isin(data[:, 3], [i, i-1, i-2])] = new_data
        
        plt.figure(figsize=(12, 8))
        plot_data = data[data[:, 3] == i]
        #plot_all_data = data[np.isin(data[:, 3], [i, i-1, i-2])]
        plot_labels = labels[-len(plot_data):]
        #plot_all_labels = labels[-len(plot_all_data):]

        unique_labels = np.unique(labels)
        colors = [
        'red', 'blue', 'lime', 'cyan', 'darkviolet', 'yellow', 
        'orange', 'deeppink', 'aqua', 'pink'
        ]
        for label in unique_labels:
            label_indices = np.where(plot_labels == label)[0]
            #label_indices = np.where(plot_all_labels == label)[0]
            #print("LA:", label_indices)
            plt.scatter(plot_data[label_indices, 0], plot_data[label_indices, 1],
                        color='grey' if label == -1 else colors[int(label%len(colors))],
                        label=f'Cluster {int(label)}' if label != -1 else 'Noise',
                        s=10)
            #plt.scatter(plot_all_data[label_indices, 0], plot_all_data[label_indices, 1],
            #           color='grey' if label == -1 else colors[int(label%len(colors))],
            #           #label=f'Cluster {int(label)}' if label != -1 else 'Noise',
            #           s=10)
        plt.scatter(0, 0, c='black', s=100)
        plt.xlabel("X Position [m]", fontsize=12)
        plt.ylabel("Y Position [m]", fontsize=12)
        plt.xlim(-50, 50)
        plt.ylim(-10, 80)
        plt.title("Radar Detections with ST-DBSCAN Clustering", fontsize=16)
        plt.grid(True)
        plt.legend(loc = 'upper right')
        plt.gca().set_aspect('equal', adjustable='box')
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()
        plt.close('all')

if __name__ == "__main__":
    try:
        data = np.load(f"data{SCENE}.npy", allow_pickle=True)
        print("Data loaded from file.")
    except:
        data = get_data_from_one_sequence()
    print("Data shape: ", np.shape(data))
    cluster(data)
