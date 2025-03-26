import folium
import pandas as pd
import numpy as np
import random
from pandas import Timestamp
from sklearn.cluster import DBSCAN

from detection import ANOMALY_CLUSTER_RADIUS, TIME_THRESHOLD, EARTH_RADIUS



def calculate_distance(pos1, pos2):  # -> kilometers
    """Calculate the great circle distance between two points on Earth."""
    lat1, lon1 = pos1
    lat2, lon2 = pos2
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS * c


def time_distance(batch1, batch2):
    """
    Check if two batches overlap within TIME_THRESHOLD minutes

    Args:
        batch1, batch2: tuples of (start_time, end_time, middle_point)

    Returns:
        0 if batches overlap within threshold, large value otherwise
    """
    start1, end1, _ = batch1
    start2, end2, _ = batch2

    # Check if one batch starts within TIME_THRESHOLD minutes of the other batch ending
    if (start1 - end2).total_seconds() / 60 <= TIME_THRESHOLD or (
        start2 - end1
    ).total_seconds() / 60 <= TIME_THRESHOLD:
        return 0
    else:
        return 1000  # Large value to separate in clustering


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


def visualize_anomaly_clusters(anomaly_data):
    """
    Visualize anomaly clusters on a map with vessel tracks colored by anomaly status.

    Args:
        anomaly_data: DataFrame with columns 'mmsi', 'start_time', 'end_time', 'middle_point', 'all_positions', 'cluster'
    """
    m = folium.Map(location=[54.687157, 25.279652], zoom_start=6)

    # Generate random colors for each cluster
    max_cluster_id = max(anomaly_data["cluster"]) if len(anomaly_data) > 0 and max(anomaly_data["cluster"]) >= 0 else -1
    cluster_colors = [
        "#%06x" % random.randint(0, 0xFFFFFF) for _ in range(max_cluster_id + 2)
    ]  # +2 for -1 and safety

    # Create feature groups
    clusters_layer = folium.FeatureGroup(name="Anomaly Clusters")
    tracks_layer = folium.FeatureGroup(name="Vessel Tracks")
    track_tooltips_layer = folium.FeatureGroup(name="Track Tooltips")
    points_layer = folium.FeatureGroup(name="Points with Speed")

    # First, plot clusters
    # Group by cluster to calculate centers
    cluster_centers = {}
    for _, row in anomaly_data.iterrows():
        label = row["cluster"]
        middle_point = row["middle_point"]
        if label >= 0:  # Skip noise points marked as -1
            if label not in cluster_centers:
                cluster_centers[label] = []
            cluster_centers[label].append(middle_point)
    
    # Plot cluster circles
    for cluster_id, points in cluster_centers.items():
        center = calculate_center_point(points)
        folium.Circle(
            location=center,
            radius=ANOMALY_CLUSTER_RADIUS * 1000,  # Convert km to meters
            color=cluster_colors[cluster_id],
            fill=True,
            fill_opacity=0.2,
            popup=f"Cluster {cluster_id}",
        ).add_to(clusters_layer)

    # Plot tracks for each anomaly batch
    for _, row in anomaly_df.iterrows():
        mmsi = row['mmsi']
        ship_type = row['ship_type']
        start_time = row['start_time']
        end_time = row['end_time']
        all_positions = row['all_positions']
        label = row['cluster']
        
        # Extract positions and anomaly flags
        positions = []
        color_values = []
        for (lat, lon), is_anomaly, speed in all_positions:
            positions.append([lat, lon])
            color_values.append(1.0 if is_anomaly else 0.0)
            
            # Add a marker for each point with speed information
            color = 'red' if is_anomaly else 'green'
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                tooltip=f"MMSI: {mmsi}<br>Speed: {speed:.2f} km/h<br>{'Anomalous' if is_anomaly else 'Normal'}"
            ).add_to(points_layer)
        
        tooltip = f'''
        MMSI: {mmsi}<br>
        SHIP_TYPE: {ship_type}<br>
        Start: {start_time.strftime("%Y-%m-%d %H:%M:%S")}<br>
        End: {end_time.strftime("%Y-%m-%d %H:%M:%S")}<br>
        Cluster: {label if label >= 0 else "None"}
        '''

        # Add tooltip polyline with hover effects using folium.Element
        folium.PolyLine(
            locations=positions,
            weight=10,
            opacity=0.1,
            tooltip=tooltip,
        ).add_to(track_tooltips_layer)

        # Shift the color values by 1
        color_values = color_values[1:] + [0]
        
        # Add the color line for anomaly visualization
        folium.ColorLine(
            positions=positions,
            colors=color_values,
            colormap=["green", "red"],  # green for normal, red for anomalous
            weight=4,
            opacity=0.8,
        ).add_to(tracks_layer)

    # Add the layers to the map
    clusters_layer.add_to(m)
    tracks_layer.add_to(m)
    track_tooltips_layer.add_to(m)
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
        for start_time, end_time, middle_point, all_positions, total_points, total_anomaly_points in anomaly_batches:
            anomaly_data.append(
                {
                    "mmsi": row["MMSI"],
                    "ship_type": row["ship_type"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "middle_point": middle_point,
                    "all_positions": all_positions,
                    "point_count": total_points,
                    "total_anomaly_points": total_anomaly_points,
                }
            )

    anomaly_df = pd.DataFrame(anomaly_data)

    # Prepare batches for clustering
    batches = [
        (row["start_time"], row["end_time"], row["middle_point"])
        for _, row in anomaly_df.iterrows()
    ]
    # Cluster anomalies using spatio-temporal distance
    anomaly_clusters = find_point_clusters(batches)
    # Add cluster assignments back to anomaly data
    anomaly_df["cluster"] = anomaly_clusters


    # Print results
    print(
        f"\nFound {max(anomaly_clusters) + 1 if anomaly_clusters and max(anomaly_clusters) >= 0 else 0} clusters of anomalies"
    )
    print(f"Found {len(anomaly_vessels)} vessels with potential GPS spoofing")
    print(anomaly_vessels[["MMSI", "point_count", "max_speed"]])

    print("Visualizing anomaly clusters...")
    # visualize_anomaly_clusters(anomaly_df)

    # Prepare the results
    #  (1. how much spoofed ships, how long and how many anomalies)

    def prepare_vessel_results(group):
        mmsi = group["mmsi"].iloc[0]
        ship_type = group["ship_type"].iloc[0]
        total_points = group["point_count"].sum()
        total_anomaly_points = group["total_anomaly_points"].sum()
        total_batches = len(group)
        total_time = (group["end_time"].max() - group["start_time"].min()).total_seconds() / 3600
        different_clusters = len(group[group["cluster"] >= 0]["cluster"].unique())
        return pd.Series(
            {
                "Ship Type": ship_type,
                "Total Points": total_points,
                "Total Anomaly Points": total_anomaly_points,
                "Total Batches": total_batches,
                "Total Time (h)": total_time,
                "Different Clusters": different_clusters,
            }
        )

    print(anomaly_df.groupby("mmsi").apply(prepare_vessel_results))

    # (2. How many clusters and their sizes and time spans)

    def prepare_cluster_results(group):
        cluster_id = group["cluster"].iloc[0]
        different_vessels = len(group["mmsi"].unique())
        total_points = group["point_count"].sum()
        total_anomaly_points = group["total_anomaly_points"].sum()
        total_batches = len(group)
        total_time = (group["end_time"].max() - group["start_time"].min()).total_seconds() / 60
        return pd.Series(
            {
                "Different Vessels": different_vessels,
                "Total Points": total_points,
                "Total Anomaly Points": total_anomaly_points,
                "Total Batches": total_batches,
                "Total Time (minutes)": total_time,
            }
        )
    
    cluster_results = anomaly_df[anomaly_df["cluster"] >= 0].groupby("cluster").apply(prepare_cluster_results)
    cluster_results= cluster_results.astype(int)
    print(cluster_results)
    print("Done!")


    ################################################
    # Extra
    ################################################

    # Plot 2579999
    # file_path = 'data/aisdk-test.csv'
    # df = pd.read_csv(file_path)
    # print(df[df["MMSI"] == 219018851].iloc[0])

    # print(df["Ship type"].value_counts())

    # print(df[(~df["Ship type"].isna()) & (df["Ship type"] != "Undefined")].iloc[100])

    
    # print(anomaly_df.mmsi.value_counts())

    # print(anomaly_df.sort_values("point_count", ascending=False))