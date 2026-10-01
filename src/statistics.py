from clustering_DBSCAN import cluster_radar_points_dbscan
from clustering_Meanshift import cluster_radar_points_meanshift
from functions import load_json, get_radar_data, getdata_all_sensors
from constants import HDF5_FILE, JSON_FILE

def main():
    total_clusters_dbscan = 0
    total_clusters_meanshift = 0
    total_time_steps = 0

    scenes_data = load_json(JSON_FILE)
    all_data = []

    current_timestamp = str(scenes_data["first_timestamp"])
    scenes = scenes_data["scenes"]

    # collect all timestamp/index/radar_range data 
    while current_timestamp:
        data, current_timestamp = getdata_all_sensors(current_timestamp, scenes)
        if data:
            all_data.append(data)

    for scene in range(len(all_data)):
        data = all_data[scene]

        ranges = [item[2] for item in data]
        merged_range = [ranges[0][0], ranges[-1][1]]

        radar_x_scene, radar_y_scene, radar_vr_compensated_scene, _ = get_radar_data(HDF5_FILE, merged_range)

        cluster_labels_dbscan = cluster_radar_points_dbscan(radar_x_scene, radar_y_scene, radar_vr_compensated_scene)
        cluster_labels_meanshift = cluster_radar_points_meanshift(radar_x_scene, radar_y_scene, radar_vr_compensated_scene)

        # Count clusters
        num_clusters_dbscan = len(set(cluster_labels_dbscan)) - (1 if -1 in cluster_labels_dbscan else 0)
        num_clusters_meanshift = len(set(cluster_labels_meanshift)) - (1 if -1 in cluster_labels_meanshift else 0)

        # Add to total clusters
        total_clusters_dbscan += num_clusters_dbscan
        total_clusters_meanshift += num_clusters_meanshift
        total_time_steps += 1
        if total_time_steps % 10 == 0:
            print(f"Processed {total_time_steps}/{len(all_data)} time steps")

    print("Total clusters (DBSCAN):", total_clusters_dbscan)
    print("Total clusters (MEANSHIFT):",total_clusters_meanshift)

    # Calculate and print average number of clusters
    avg_clusters_dbscan = total_clusters_dbscan / total_time_steps
    avg_clusters_meanshift = total_clusters_meanshift / total_time_steps
    print(f"Average number of clusters (DBSCAN): {avg_clusters_dbscan}")
    print(f"Average number of clusters (MeanShift): {avg_clusters_meanshift}")

if __name__ == "__main__":
    main()
