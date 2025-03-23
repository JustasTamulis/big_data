import timeit_wrapper as tw
import numpy as np
import pandas as pd
from multiprocessing import Pool
from datetime import datetime

DISTANCE_THRESHOLD = 0.25

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_vessel_distance(vessel_data):
    """
    Calculate total distance traveled for a vessel given its position data
    
    Args:
        vessel_data: DataFrame containing Latitude, Longitude and Timestamp columns for a single vessel
        
    Returns:
        tuple: (vessel MMSI, total distance traveled)
    """
    if len(vessel_data) < 2:
        return vessel_data.iloc[0]['MMSI'], 0

    # Sort by timestamp to ensure chronological order
    vessel_data = vessel_data.sort_values('Timestamp')
    
    distances = []
    coords = vessel_data[['Latitude', 'Longitude']].values
    
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1]
        lat2, lon2 = coords[i]
        distances.append(calculate_distance(lat1, lon1, lat2, lon2))

    return vessel_data.iloc[0]['MMSI'], sum(distances)

@tw.timeit
def process_sequential(df):
    """Process vessel distances sequentially"""
    results = []
    for mmsi, group in df.groupby('MMSI'):
        results.append(calculate_vessel_distance(group))
    return pd.DataFrame(results, columns=['MMSI', 'Total Distance'])

@tw.timeit
def process_parallel(df, num_processes=4):
    """Process vessel distances in parallel"""
    # Pre-group the data to avoid redundant grouping operations
    vessel_groups = [group for _, group in df.groupby('MMSI')]
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(calculate_vessel_distance, vessel_groups)
    
    return pd.DataFrame(results, columns=['MMSI', 'Total Distance'])

if __name__ == '__main__':
    file_path = 'aisdk-small.csv'
    df = pd.read_csv(file_path)
    print(df.head())
    # Convert Timestamp column to datetime
    df['Timestamp'] = pd.to_datetime(df['# Timestamp'], format='%d/%m/%Y %H:%M:%S')
    
    # Sequential processing
    result_sequential = process_sequential(df)
    print("Sequential Results:")
    print(result_sequential[result_sequential['Total Distance'] > DISTANCE_THRESHOLD])

    # Parallel processing
    result_parallel = process_parallel(df)
    print("\nParallel Results:")
    print(result_parallel[result_parallel['Total Distance'] > DISTANCE_THRESHOLD])
