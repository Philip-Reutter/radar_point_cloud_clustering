import numpy as np
from sklearn.cluster import DBSCAN

def cluster_radar_points_dbscan(x_scene, y_scene, vr_compensated):
    """
    Perform DBSCAN clustering on radar points.
    :param x_scene: List or array of X-coordinates.
    :param y_scene: List or array of Y-coordinates.
    :param vr_compensated: List or array of compensated velocities.
    :return: Cluster labels for each point.
    """
    points = np.column_stack((x_scene, y_scene, vr_compensated))
    dbscan = DBSCAN(eps=5, min_samples=3)
    labels = dbscan.fit_predict(points)
    return labels
