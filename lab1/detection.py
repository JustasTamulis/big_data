# vessel spoofing detection

import timeit_wrapper as tw
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


# Constants
SPEED_THRESHOLD = 400  # Record speed in km/h with 50% margin
TIME_THRESHOLD = 10  # minutes - batch window for anomalies
ANOMALY_CLUSTER_RADIUS = 20  # Radius in km to check for vessel clusters
ANGLE_DIFF_ANOMALY = 90  # Angle difference in degrees for heading anomalies
EARTH_RADIUS = 6371  # Earth's radius in kilometers
MINIMUM_ANOMALIES_PER_BATCH = (
    3  # Number of anomalies per time per vessel to be anomalous
)
EXCLUDED_STATUSES = [
    "moored",
    "at anchor",
    "Constrained by her draught",
    "Restricted maneuverability",
]
EXCLUDED_MMSI = ["2579999"]  # 2579999 is a test MMSI that should be excluded

# Spoofing detection


def process_anomaly_batch(current_batch, vessel_data, speeds):
    """
    Process a batch of anomalies to calculate its time window and middle point.

    Args:
        current_batch: List of anomaly tuples (index, timestamp, lat, lon, speed)
        vessel_data: DataFrame containing vessel data

    Returns:
        tuple: (start_time, end_time, middle_point) or None if batch is invalid
    """
    # Only process batches with more than constant anomalies
    if len(current_batch) <= MINIMUM_ANOMALIES_PER_BATCH - 1:
        return None

    start_time = current_batch[0][1]
    end_time = current_batch[-1][1]

    anomaly_indices = [a[0] for a in current_batch]

    # Find normal points during this time window
    batch_points = vessel_data[
        (vessel_data["Timestamp"] >= start_time)
        & (vessel_data["Timestamp"] <= end_time)
    ]

    batch_speeds = speeds[
        (vessel_data["Timestamp"] >= start_time)
        & (vessel_data["Timestamp"] <= end_time)
    ]

    # Calculate middle point from normal points (or anomalies if no normal points)

    normal_points = batch_points.iloc[~batch_points.index.isin(anomaly_indices)]
    if len(normal_points) > 0:
        middle_lat = normal_points["Latitude"].median()
        middle_lon = normal_points["Longitude"].median()
    else:
        # If no normal points in window, use median of anomalies
        middle_lat = np.median([a[2] for a in current_batch])
        middle_lon = np.median([a[3] for a in current_batch])

    middle_point = (middle_lat, middle_lon)

    # Save all positions with anomaly flags
    all_positions = list(
        zip(
            zip(batch_points["Latitude"], batch_points["Longitude"]),
            batch_points.index.isin(anomaly_indices).astype(bool),
            batch_speeds,
        )
    )

    total_points = len(batch_points)
    total_anomaly_points = len(current_batch)

    return (
        start_time,
        end_time,
        middle_point,
        all_positions,
        total_points,
        total_anomaly_points,
    )


def calculate_distance_matrix(lat1, lon1, lat2, lon2):
    """
    Vectorized calculation of great circle distances between sets of coordinates

    Args:
        lat1, lon1: arrays of latitude and longitude for first points
        lat2, lon2: arrays of latitude and longitude for second points

    Returns:
        Array of distances in kilometers
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS * c


def calculate_heading_matrix(lat1, lon1, lat2, lon2):
    """
    Calculate the angle between two points in the range [0, 360]

    Args:
        lat1, lon1: Latitude and longitude of the first point
        lat2, lon2: Latitude and longitude of the second point

    Returns:
        The bearing in degrees in the range [0, 360]
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    angle = np.degrees(np.arctan2(y, x))
    return (angle + 360) % 360


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
    ship_type = vessel_data.iloc[0]["Ship type"]
    point_count = len(vessel_data)
    if point_count < 2:
        return None

    # Sort by timestamp and drop duplicates
    vessel_data = (
        vessel_data.sort_values("Timestamp")
        .drop_duplicates("Timestamp")
        .reset_index(drop=True)
    )

    # Extract data to numpy arrays for faster processing
    timestamps = vessel_data["Timestamp"].values
    lats = vessel_data["Latitude"].values
    lons = vessel_data["Longitude"].values

    # Create shifted arrays (exclude first row as it won't have a previous value)
    prev_lats = np.roll(lats, 1)
    prev_lons = np.roll(lons, 1)
    prev_timestamps = np.roll(timestamps, 1)

    # Set first element to NaN to avoid calculating with wrapped values
    prev_lats[0] = np.nan
    prev_lons[0] = np.nan
    prev_timestamps[0] = np.datetime64("NaT")

    ##############################
    # Speed anomaly detection

    # Calculate time differences in hours (vectorized)
    time_diffs = (timestamps - prev_timestamps).astype("timedelta64[s]").astype(
        float
    ) / 3600
    # Calculate distances (vectorized)
    distances = calculate_distance_matrix(prev_lats, prev_lons, lats, lons)
    # Calculate speeds
    speeds = np.divide(
        distances, time_diffs, out=np.zeros_like(distances), where=time_diffs > 0
    )
    # Find maximum speed (ignoring NaN values)
    max_speed = np.nanmax(speeds) if not np.isnan(speeds).all() else 0
    # Find indices of anomalous speeds
    anomaly_indices = np.where(speeds > SPEED_THRESHOLD)[0]

    ##############################
    # Heading anomaly detection
    # Ignored, because the gps coordinates rounding fluctuation is too high

    # calculated_headings = calculate_heading_matrix(lats, lons, prev_lats, prev_lons)

    # # Reported headings are in the range [0, 360]
    # reported_headings = vessel_data["Heading"].values

    # # Calculate the difference between calculated and reported headings
    # # Make sure to handle the case, that 0 is close to 360

    # heading_diffs = np.abs(calculated_headings - reported_headings)
    # heading_diffs = np.minimum(heading_diffs, 360 - heading_diffs)

    # # Find indices of anomalous headings
    # heading_anomaly_indices = np.where(heading_diffs > ANGLE_DIFF_ANOMALY)[0]

    # # Combine speed and heading anomalies with OR
    # anomaly_indices = np.union1d(anomaly_indices, heading_anomaly_indices)
    ##############################

    if len(anomaly_indices) == 0:
        return None

    # Create anomalies list
    anomalies = [
        (int(i), timestamps[i], lats[i], lons[i], speeds[i]) for i in anomaly_indices
    ]

    # Group anomalies into time batches - this part is harder to vectorize
    # due to its sequential nature
    anomaly_batches = []
    if anomalies:
        current_batch = [anomalies[0]]
        for i in range(1, len(anomalies)):
            prev_time = current_batch[-1][1]
            curr_time = anomalies[i][1]
            time_diff = (curr_time - prev_time).astype("timedelta64[s]").astype(
                float
            ) / 60  # in minutes

            if time_diff <= TIME_THRESHOLD:
                # Add to current batch
                current_batch.append(anomalies[i])
            else:
                # Process current batch and start a new one
                batch_result = process_anomaly_batch(current_batch, vessel_data, speeds)
                if batch_result:
                    anomaly_batches.append(batch_result)
                current_batch = [anomalies[i]]

        # Process the last batch
        batch_result = process_anomaly_batch(current_batch, vessel_data, speeds)
        if batch_result:
            anomaly_batches.append(batch_result)

    # Ship is anomalous if it has at least one batch
    is_anomaly = len(anomaly_batches) > 0

    return mmsi, ship_type, point_count, max_speed, is_anomaly, anomaly_batches


def process_chunk(chunk):
    """Process a chunk of vessel data"""
    # Prepare the chunk
    chunk["Timestamp"] = pd.to_datetime(
        chunk["# Timestamp"], format="%d/%m/%Y %H:%M:%S"
    )
    chunk["MMSI"] = chunk["MMSI"].astype(str)
    chunk["Ship type"] = chunk["Ship type"].astype(str)

    # Filter out excluded navigational statuses
    chunk = chunk[~chunk["Navigational status"].isin(EXCLUDED_STATUSES)]
    chunk = chunk[~chunk["MMSI"].isin(EXCLUDED_MMSI)]
    chunk = chunk[~chunk["Heading"].isna()]
    chunk = chunk[
        ["MMSI", "Ship type", "Timestamp", "Latitude", "Longitude", "Heading"]
    ]

    results = []
    for mmsi, group in chunk.groupby("MMSI"):
        r = detect_vessel_anomalies(group)
        if r:
            results.append(r)
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


@tw.timeit
def process_file_in_chunks_with_pooling(
    file_path, chunk_size=10000, num_processes=None, verbose=False
):
    """Process the CSV file in chunks using multiprocessing.Pool's imap_unordered"""
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"Processing with {num_processes} workers, chunk size: {chunk_size}")

    # Create a generator of chunks
    chunks = pd.read_csv(file_path, chunksize=chunk_size)

    all_results = []

    # Process each chunk using a Pool with imap_unordered
    with mp.Pool(processes=num_processes) as pool:
        # imap_unordered returns results as they become available
        results_iterator = pool.imap_unordered(process_chunk, chunks)

        # Collect results as they arrive
        for i, chunk_results in enumerate(results_iterator):
            all_results.extend(chunk_results)
            if verbose:
                print(f"Processed chunk {i+1}")

    return all_results


if __name__ == "__main__":
    # Load and prepare data
    # file_path = 'data/aisdk-test.csv'
    file_path = "data/aisdk-2025-02-09.csv"
    print("\nProcessing data in chunks...")

    # Process the file in chunks
    results = process_file_in_chunks(file_path, chunk_size=1000000, num_processes=16)

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results,
        columns=[
            "MMSI",
            "ship_type",
            "point_count",
            "max_speed",
            "is_anomaly",
            "anomaly_batches",
        ],
    )

    results_df.to_csv("output/results.csv", index=False)

    anomalies = results_df[results_df["is_anomaly"]].copy()
    if len(anomalies) > 0:
        print("Found", len(anomalies), "vessels with potential GPS spoofing")
    else:
        print("No anomalies detected")
