import numpy as np
from sklearn.cluster import MeanShift

def cluster_radar_points_meanshift(x_scene, y_scene, vr_compensated, bandwidth=10):
    """
    Perform MeanShift clustering on radar points.
    :param x_scene: List or array of X-coordinates.
    :param y_scene: List or array of Y-coordinates.
    :param vr_compensated: List or array of compensated velocities.
    :param bandwidth: Bandwidth parameter for the MeanShift algorithm.
                      If None, it will be estimated automatically.
    :return: Cluster labels for each point.
    """
    points = np.column_stack((x_scene, y_scene, vr_compensated))
    meanshift = MeanShift(bandwidth=bandwidth)
    labels = meanshift.fit_predict(points)
    return labels
