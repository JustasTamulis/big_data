# vessel spoofing detection

import timeit_wrapper as tw
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


# Constants
SPEED_THRESHOLD = 600 * 1.5  # Record speed in km/h with 50% margin
TIME_THRESHOLD = 10  # minutes - batch window for anomalies
ANOMALY_CLUSTER_RADIUS = 20  # Radius in km to check for vessel clusters
EARTH_RADIUS = 6371  # Earth's radius in kilometers
EXCLUDED_STATUSES = [
    "moored",
    "at anchor",
    "Constrained by her draught",
    "Restricted maneuverability",
]


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


def detect_vessel_anomalies(vessel_data):
    """
    Detect potential GPS spoofing for a single vessel with time-based analysis

    Args:
        vessel_data: DataFrame containing vessel data

    Returns:
        tuple: (MMSI, point_count, max_speed, is_anomaly, anomaly_batches)
        where:
            - MMSI: The unique identifier of the vessel.
            - point_count: The number of data points for the vessel.
            - max_speed: The maximum speed recorded for the vessel (in km/h).
            - is_anomaly: A boolean indicating if the vessel has anomalies.
            - anomaly_batches: A list of tuples, where each tuple contains:
                - start_time (datetime): The start time of the anomaly batch.
                - end_time (datetime): The end time of the anomaly batch.
                - middle_point (tuple): The geographical center point (latitude, longitude) of the anomaly batch.
    """
    mmsi = vessel_data.iloc[0]["MMSI"]
    point_count = len(vessel_data)
    if point_count < 2:
        return mmsi, point_count, 0, False, []

    vessel_data = vessel_data.sort_values("Timestamp")
    anomalies = []
    max_speed = 0

    for i in range(1, len(vessel_data)):
        prev_row = vessel_data.iloc[i - 1]
        curr_row = vessel_data.iloc[i]

        # Calculate time difference
        time_diff = (
            curr_row["Timestamp"] - prev_row["Timestamp"]
        ).total_seconds() / 3600  # in hours

        if time_diff > 0:
            # Calculate distance and speed
            distance = calculate_distance(
                (prev_row["Latitude"], prev_row["Longitude"]),
                (curr_row["Latitude"], curr_row["Longitude"]),
            )
            speed = distance / time_diff  # km/h

            max_speed = max(max_speed, speed)

            # Check for anomalous speed
            if speed > SPEED_THRESHOLD:
                anomalies.append(
                    (
                        i,
                        curr_row["Timestamp"],
                        curr_row["Latitude"],
                        curr_row["Longitude"],
                        speed,
                    )
                )

    # Group anomalies into time batches
    anomaly_batches = []
    if anomalies:
        current_batch = [anomalies[0]]
        for i in range(1, len(anomalies)):
            prev_time = current_batch[-1][1]
            curr_time = anomalies[i][1]
            time_diff = (curr_time - prev_time).total_seconds() / 60  # in minutes

            if time_diff <= TIME_THRESHOLD:
                # Add to current batch
                current_batch.append(anomalies[i])
            else:
                # Start a new batch if the previous one has more than 1 anomaly
                if (
                    len(current_batch) > 1
                ):  # Only consider batches with multiple anomalies
                    start_time = current_batch[0][1]
                    end_time = current_batch[-1][1]

                    # Find normal points during this time window
                    normal_points = vessel_data[
                        (vessel_data["Timestamp"] >= start_time)
                        & (vessel_data["Timestamp"] <= end_time)
                    ]

                    # Exclude the anomalous points
                    anomaly_indices = [a[0] for a in current_batch]
                    normal_points = normal_points.iloc[
                        ~normal_points.index.isin(anomaly_indices)
                    ]

                    # Calculate middle point from normal points (or anomalies if no normal points)
                    if len(normal_points) > 0:
                        middle_lat = normal_points["Latitude"].median()
                        middle_lon = normal_points["Longitude"].median()
                    else:
                        # If no normal points in window, use median of anomalies
                        middle_lat = np.median([a[2] for a in current_batch])
                        middle_lon = np.median([a[3] for a in current_batch])

                    middle_point = (middle_lat, middle_lon)
                    anomaly_batches.append((start_time, end_time, middle_point))

                current_batch = [anomalies[i]]

        # Add the last batch if it has more than 1 anomaly
        if len(current_batch) > 1:
            start_time = current_batch[0][1]
            end_time = current_batch[-1][1]

            # Find normal points during this time window
            normal_points = vessel_data[
                (vessel_data["Timestamp"] >= start_time)
                & (vessel_data["Timestamp"] <= end_time)
            ]

            # Exclude the anomalous points
            anomaly_indices = [a[0] for a in current_batch]
            normal_points = normal_points.iloc[
                ~normal_points.index.isin(anomaly_indices)
            ]

            # Calculate middle point from normal points
            if len(normal_points) > 0:
                middle_lat = normal_points["Latitude"].median()
                middle_lon = normal_points["Longitude"].median()
            else:
                # If no normal points in window, use median of anomalies
                middle_lat = np.median([a[2] for a in current_batch])
                middle_lon = np.median([a[3] for a in current_batch])

            middle_point = (middle_lat, middle_lon)
            anomaly_batches.append((start_time, end_time, middle_point))

    # Ship is anomalous if it has at least one batch
    is_anomaly = len(anomaly_batches) > 0

    return mmsi, point_count, max_speed, is_anomaly, anomaly_batches


def process_chunk(chunk):
    """Process a chunk of vessel data"""
    # Prepare the chunk
    chunk["Timestamp"] = pd.to_datetime(
        chunk["# Timestamp"], format="%d/%m/%Y %H:%M:%S"
    )
    chunk["MMSI"] = chunk["MMSI"].astype(str)

    # Filter out excluded navigational statuses
    chunk = chunk[~chunk["Navigational status"].isin(EXCLUDED_STATUSES)]
    chunk = chunk[["MMSI", "Timestamp", "Latitude", "Longitude"]]

    results = []
    for mmsi, group in chunk.groupby("MMSI"):
        results.append(detect_vessel_anomalies(group))
    return results


@tw.timeit
def process_file_in_chunks(file_path, chunk_size=10000, num_processes=None):
    """Process the CSV file in chunks using multiple processes"""
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"Processing with {num_processes} workers, chunk size: {chunk_size}")

    # Create a generator of chunks
    chunks = pd.read_csv(file_path, chunksize=chunk_size)

    all_results = []

    # Process each chunk
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                chunk_results = future.result()
                all_results.extend(chunk_results)
                print(f"Processed chunk {chunk_index+1}")
            except Exception as e:
                print(f"Error processing chunk {chunk_index+1}: {e}")

    return all_results


if __name__ == "__main__":
    # Load and prepare data
    file_path = "data/aisdk-test.csv"
    # file_path = 'data/aisdk-2025-02-09.csv'
    print("\nProcessing data in chunks...")

    # Process the file in chunks
    results = process_file_in_chunks(file_path, chunk_size=100000, num_processes=16)

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results,
        columns=["MMSI", "point_count", "max_speed", "is_anomaly", "anomaly_batches"],
    )

    results_df.to_csv("output/results.csv", index=False)

    anomalies = int(results_df["is_anomaly"].sum())
    if anomalies > 0:
        print("Found", anomalies, "vessels with potential GPS spoofing")
    else:
        print("No anomalies detected")
