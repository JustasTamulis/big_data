# outdated
def visualize_on_map_outdated(
    df, anomaly_vessels, anomaly_clusters, output_file="vessel_map.html"
):
    """
    Create an interactive map visualization of vessels, anomalies, and clusters.

    Args:
        df: Original DataFrame with vessel data
        anomaly_vessels: List of vessels with anomalies
        anomaly_clusters: List of anomaly clusters
        output_file: Name of the output HTML file
    """
    # Calculate map center
    center_lat = df["Latitude"].mean()
    center_lon = df["Longitude"].mean()

    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Add vessel positions (sample of normal vessels)
    normal_vessels = df[~df["MMSI"].isin([v["MMSI"] for v in anomaly_vessels])]
    sample_size = min(
        1000, normal_vessels.MMSI.nunique() - 1
    )  # Limit to 100 normal vessels for clarity
    sampled_vessels = (
        normal_vessels.groupby("MMSI").last().sample(n=sample_size, random_state=42)
    )

    # Create vessel position layer
    vessel_layer = folium.FeatureGroup(name="Normal Vessels")
    for _, vessel in normal_vessels.groupby("MMSI").last().iterrows():
        folium.CircleMarker(
            location=[vessel["Latitude"], vessel["Longitude"]],
            radius=3,
            color="blue",
            fill=True,
            popup=f"MMSI: {vessel.name}",
        ).add_to(vessel_layer)
    vessel_layer.add_to(m)

    # Create anomaly layer
    anomaly_layer = folium.FeatureGroup(name="Anomalies")
    for vessel in anomaly_vessels:
        mmsi = vessel["MMSI"]
        vessel_anomalies = [
            a for cluster in anomaly_clusters for a in cluster if a["mmsi"] == mmsi
        ]

        if vessel_anomalies:
            # Create a line connecting anomalous positions
            locations = [[a["latitude"], a["longitude"]] for a in vessel_anomalies]
            folium.PolyLine(
                locations=locations,
                color="red",
                weight=2,
                opacity=0.8,
                popup=f"MMSI: {mmsi}, Anomalies: {vessel['anomaly_count']}",
            ).add_to(anomaly_layer)

            # Add markers for each anomalous position
            for anomaly in vessel_anomalies:
                folium.CircleMarker(
                    location=[anomaly["latitude"], anomaly["longitude"]],
                    radius=5,
                    color="red",
                    fill=True,
                    popup=f"MMSI: {mmsi}<br>Speed: {int(anomaly['speed'])} km/h<br>Time: {anomaly['timestamp']}",
                ).add_to(anomaly_layer)
    anomaly_layer.add_to(m)

    # Create cluster layer
    cluster_layer = folium.FeatureGroup(name="Anomaly Clusters")
    colors = [
        "#%06x" % random.randint(0, 0xFFFFFF) for _ in range(len(anomaly_clusters))
    ]

    for i, cluster in enumerate(anomaly_clusters):
        # Create a polygon for each cluster
        locations = [[point["latitude"], point["longitude"]] for point in cluster]
        if len(locations) > 2:  # Need at least 3 points for a polygon
            folium.Polygon(
                locations=locations,
                color=colors[i],
                fill=True,
                weight=1,
                opacity=0.4,
                popup=f"Cluster {i+1}: {len(cluster)} anomalies",
            ).add_to(cluster_layer)
    cluster_layer.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add fullscreen option
    plugins.Fullscreen().add_to(m)

    # Save the map
    m.save(output_file)
    print(f"\nMap has been saved to {output_file}")
