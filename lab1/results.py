import folium
import pandas as pd
import numpy as np
import random
from pandas import Timestamp
from sklearn.cluster import DBSCAN

from detection import ANOMALY_CLUSTER_RADIUS


def spatio_temporal_distance(batch1, batch2):
    """
    Custom distance function that considers both spatial and temporal distance between batches

    Args:
        batch1, batch2: tuples of (start_time, end_time, middle_point)
    """
    _, _, middle_point1 = batch1
    _, _, middle_point2 = batch2

    spatial_dist = calculate_distance(middle_point1, middle_point2)
    temp_dist = time_distance(batch1, batch2)

    return spatial_dist if temp_dist == 0 else float("inf")


def find_point_clusters(batches, cluster_radius=ANOMALY_CLUSTER_RADIUS):
    """
    Find clusters of anomaly batches using DBSCAN with combined space-time distance.

    Args:
        batches: List of (start_time, end_time, middle_point) tuples

    Returns:
        List of cluster labels for each input batch (-1 indicates noise points)
    """
    if len(batches) < 2:
        return [-1] * len(batches)

    # Create a distance matrix using the custom distance function
    n = len(batches)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = spatio_temporal_distance(batches[i], batches[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    # Run DBSCAN with precomputed distance matrix
    clustering = DBSCAN(eps=cluster_radius, min_samples=2, metric="precomputed").fit(
        distance_matrix
    )

    return clustering.labels_.tolist()


def calculate_center_point(coords):
    """
    Calculate the center point of a list of coordinates.
    """
    return np.mean(coords, axis=0)


def visualize_anomaly_clusters(anomaly_data, cluster_labels):
    """
    Visualize anomaly clusters on a map with points, vessel IDs, and timestamps.

    Args:
        anomaly_data: DataFrame with columns 'mmsi', 'start_time', 'end_time', 'middle_point'
        cluster_labels: List of cluster assignments
    """
    m = folium.Map(location=[54.687157, 25.279652], zoom_start=6)

    # Generate random colors for each cluster
    max_cluster_id = max(cluster_labels) if cluster_labels else -1
    colors = [
        "#%06x" % random.randint(0, 0xFFFFFF) for _ in range(max_cluster_id + 2)
    ]  # +2 for -1 and safety

    # Create feature groups
    clusters_layer = folium.FeatureGroup(name="Anomaly Clusters")
    points_layer = folium.FeatureGroup(name="Anomaly Points")

    # Group points by cluster
    cluster_points = {}
    for i, (mmsi, start_time, end_time, middle_point, label) in enumerate(
        zip(
            anomaly_data["mmsi"],
            anomaly_data["start_time"],
            anomaly_data["end_time"],
            anomaly_data["middle_point"],
            cluster_labels,
        )
    ):
        if label not in cluster_points:
            cluster_points[label] = []

        cluster_points[label].append((mmsi, start_time, end_time, middle_point))

    # Plot each cluster
    for cluster_id, points in cluster_points.items():
        if cluster_id >= 0:  # Skip noise points marked as -1
            # Get center point of cluster
            cluster_coords = [p[3] for p in points]
            center = calculate_center_point(cluster_coords)

            # Add cluster circle
            folium.Circle(
                location=center,
                radius=ANOMALY_CLUSTER_RADIUS * 1000,  # Convert km to meters
                color=colors[cluster_id],
                fill=True,
                fill_opacity=0.2,
                popup=f"Cluster {cluster_id}",
            ).add_to(clusters_layer)

            # Add individual points
            for mmsi, start_time, end_time, (lat, lon) in points:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color=colors[cluster_id],
                    fill=True,
                    fill_opacity=0.7,
                    popup=f'MMSI: {mmsi}<br>Start: {start_time.strftime("%Y-%m-%d %H:%M:%S")}<br>End: {end_time.strftime("%Y-%m-%d %H:%M:%S")}',
                ).add_to(points_layer)

    # Add the layers to the map
    clusters_layer.add_to(m)
    points_layer.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    m.save("output/anomaly_clusters.html")


if __name__ == "__main__":

    results_df = pd.read_csv("output/results.csv")

    # Filter anomalous vessels
    anomaly_vessels = results_df[results_df["is_anomaly"] == True].copy()

    # Prepare data for clustering
    anomaly_data = []
    for _, row in anomaly_vessels.iterrows():
        anomaly_batches = eval(row["anomaly_batches"])
        for start_time, end_time, middle_point in anomaly_batches:
            anomaly_data.append(
                {
                    "mmsi": row["MMSI"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "middle_point": middle_point,
                }
            )

    anomaly_df = pd.DataFrame(anomaly_data)

    if len(anomaly_df) > 0:
        # Prepare batches for clustering
        batches = [
            (row["start_time"], row["end_time"], row["middle_point"])
            for _, row in anomaly_df.iterrows()
        ]

        # Cluster anomalies using spatio-temporal distance
        anomaly_clusters = find_point_clusters(batches)

        # Add cluster assignments back to anomaly data
        anomaly_df["cluster"] = anomaly_clusters

        # Visualize anomalies with time component
        visualize_anomaly_clusters(anomaly_df, anomaly_clusters)

        # Print results
        print(
            f"\nFound {max(anomaly_clusters) + 1 if anomaly_clusters and max(anomaly_clusters) >= 0 else 0} clusters of anomalies"
        )
        print(f"Found {len(anomaly_vessels)} vessels with potential GPS spoofing")
        print(anomaly_vessels[["MMSI", "point_count", "max_speed"]])
    else:
        print("No anomalies detected")
