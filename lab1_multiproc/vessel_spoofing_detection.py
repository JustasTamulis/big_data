import timeit_wrapper as tw
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import DBSCAN
import folium
from folium import plugins
import multiprocessing as mp
import random

# Constants
SPEED_THRESHOLD = 600 * 1.5  # Record speed in km/h with 50% margin
ALLOWED_1_ANOMALY_PER_N_POINTS = 100
ANOMALY_CLUSTER_RADIUS = 20  # Radius in km to check for vessel clusters
EARTH_RADIUS = 6371  # Earth's radius in kilometers

# JT - checked
def calculate_distance(pos1, pos2): # -> kilometers
    """Calculate the great circle distance between two points on Earth."""
    lat1, lon1 = pos1
    lat2, lon2 = pos2
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return EARTH_RADIUS * c

# JT - checked
def find_point_clusters(coords, cluster_radius=ANOMALY_CLUSTER_RADIUS):
    """
    Find clusters of points using DBSCAN with haversine distance.
    
    Args:
        coords: List of [latitude, longitude] coordinates
        
    Returns:
        List of cluster labels for each input coordinate (-1 indicates noise points)
    """
    if len(coords) < 2:
        return [-1] * len(coords)
    
    coords = np.array(coords)
    clustering = DBSCAN(
        eps=cluster_radius,
        min_samples=2,
        metric=calculate_distance
    ).fit(coords)
    return clustering.labels_.tolist()

def calculate_center_point(coords):
    """
    Calculate the center point of a list of coordinates.
    """
    return np.mean(coords, axis=0)

def detect_vessel_anomalies(vessel_data):
    """
    Detect potential GPS spoofing for a single vessel
    
    Args:
        vessel_data: DataFrame containing vessel data
        
    Returns:
        tuple: (MMSI, list of anomaly timestamps, average speed)
    """
    mmsi = vessel_data.iloc[0]['MMSI']
    point_count = len(vessel_data)
    if point_count < 2:
        return mmsi, point_count, 0, 0, False, 0, None
    
    vessel_data = vessel_data.sort_values('Timestamp')
    travel_time = (vessel_data.iloc[-1]['Timestamp'] - vessel_data.iloc[0]['Timestamp']).total_seconds()
    anomalies = []
    max_speed = 0
    
    for i in range(1, len(vessel_data)):
        prev_row = vessel_data.iloc[i-1]
        curr_row = vessel_data.iloc[i]
        
        # Calculate time difference
        time_diff = (curr_row['Timestamp'] - prev_row['Timestamp']).total_seconds() / 3600  # in hours

        if time_diff > 0:
            # Calculate distance and speed
            distance = calculate_distance(
                (prev_row['Latitude'], prev_row['Longitude']),
                (curr_row['Latitude'], curr_row['Longitude'])
            )
            speed = distance / time_diff  # km/h
            
            max_speed = max(max_speed, speed)
            
            # Check for anomalous speed
            if speed > SPEED_THRESHOLD:
                anomalies.append(i)
    
    # For each n points, allow 1 anomaly
    if len(anomalies) < point_count / ALLOWED_1_ANOMALY_PER_N_POINTS:
        return mmsi, point_count, travel_time, max_speed, False, len(anomalies), None

    # center point of points from not the anomalies
    middle_point = calculate_center_point(
        vessel_data.iloc[~np.array(anomalies)][['Latitude', 'Longitude']]
    )

    return mmsi, point_count, travel_time, max_speed, True, len(anomalies), middle_point.tolist()

def visualize_anomaly_clusters(middle_points, anomaly_clusters):
    """
    Visualize anomaly clusters on a map.
    """
    m = folium.Map(location=[54.687157, 25.279652], zoom_start=6)
    # Generate random colors for each cluster
    colors = ['#%06x' % random.randint(0, 0xFFFFFF) for _ in range(max(anomaly_clusters) + 1)]
    
    # Create a feature group for clusters
    cluster_layer = folium.FeatureGroup(name='Anomaly Clusters')
    
    # Plot each middle point with its cluster color
    for point, cluster_id in zip(middle_points, anomaly_clusters):
        if cluster_id >= 0:  # Skip noise points marked as -1
            folium.Circle(
                location=point,
                radius=ANOMALY_CLUSTER_RADIUS * 1000,  # Convert km to meters
                color=colors[cluster_id],
                fill=True,
                fill_opacity=0.2,
                popup=f'Cluster {cluster_id}'
            ).add_to(cluster_layer)
    
    # Add the cluster layer to the map
    cluster_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    m.save('output/anomaly_clusters.html')

def process_parallel(df, num_processes, chunk_size=10):
    """Process vessel data in parallel"""
    vessel_groups = (group for _, group in df.groupby('MMSI'))
    with mp.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(detect_vessel_anomalies, vessel_groups, chunksize=chunk_size)
        return list(results)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['# Timestamp'], format='%d/%m/%Y %H:%M:%S')
    df['MMSI'] = df['MMSI'].astype(str)
    df = df[['MMSI', 'Timestamp', 'Latitude', 'Longitude']]
    return df

if __name__ == '__main__':
    # print("Distance from Vilnius to Kaunas: ", calculate_distance(54.687157, 25.279652, 54.898521, 23.903597))

    # Load and prepare data
    file_path = 'data/aisdk-small.csv'
    print("\nLoading data...")
    df = load_data(file_path)
    
    # Get total number of vessels for progress reporting
    total_vessels = df['MMSI'].nunique()
    print("Total vessels: ", total_vessels)
    print("Total points: ", len(df))
    
    # Run parallel processing with dynamic distribution
    print("\nRunning vessel spoofing detection...")
    results = process_parallel(df, num_processes=mp.cpu_count(), chunk_size=10)
    
    # Print the results
    results_df = pd.DataFrame(results, columns=['MMSI', 'point_count', 'travel_time', 'max_speed', 'is_anomaly', 'anomaly_count', 'middle_point'])
    anomaly_vessels = results_df[results_df['is_anomaly'] == True]
    print(results_df.sort_values('point_count', ascending=False)[['MMSI', 'point_count', 'travel_time', 'max_speed', 'anomaly_count']].head(10))

    # Cluster anomalies
    middle_points = anomaly_vessels['middle_point'].tolist()
    anomaly_clusters = find_point_clusters(middle_points)

    # Join anomaly_vessels with anomaly_clusters and save to csv
    anomaly_vessels = anomaly_vessels.join(pd.DataFrame(anomaly_clusters, columns=['cluster_id']))
    anomaly_vessels.to_csv('output/anomaly_vessels.csv', index=False)

    # Display anomalies on map
    visualize_anomaly_clusters(middle_points, anomaly_clusters)

    # Print results
    print(f"\nFound {max(anomaly_clusters)} clusters of anomalies:")
    print(f"\nFound {len(anomaly_vessels)} vessels with potential GPS spoofing:")
    print(anomaly_vessels.sort_values('anomaly_count', ascending=False))
