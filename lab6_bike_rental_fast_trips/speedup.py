# This file is used to calculate the speedup of a parallel program.

# all functionality is copied over from multiwork.ipynb

import pandas as pd
import folium
import os
import networkx as nx
import osmnx as ox
import numpy as np

MEAN_LAT = 41.90234770710154
MEAN_LNG = -87.64440345010253
BIKE_SPEED = 20 # km/h
MIN_STATION_COUNT = 100
MIN_PAIR_COUNT = 20

# The 5th and 95th percentiles of latitude and longitude from all data
MIN_LNG, MIN_LAT, MAX_LNG, MAX_LAT = (-87.7, 41.8, -87.6, 41.96909)

DATA_DIR = 'bike_rent'
FILES = os.listdir(DATA_DIR)

GRAPH_FILE = "all_public_percentile_bbox.graphml"

# None -> graph
def load_graph():
    G = ox.load_graphml(GRAPH_FILE)
    return G

G = load_graph()
# Load the graph once in the main process
# The other processses will copy


##########################################################
# Functions
###########################################################

# start_lat, start_lng, end_lat, end_lng -> float
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6378
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(
        dlon / 2
    ) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# start_lat, start_lng, end_lat, end_lng, duration -> bool
def this_is_fast_enough(start_lat, start_lng, end_lat, end_lng, duration):
    # Get the manhattan distance in km
    dist1 = calculate_distance(start_lat, start_lng, end_lat, start_lng)
    dist2 = calculate_distance(end_lat, start_lng, end_lat, end_lng)
    manhattan_distance = dist1 + dist2
    # Calculate how much it takes with the bike speed
    maximum_fast_ride_duration = (manhattan_distance / BIKE_SPEED) * 3600  # in seconds
    return maximum_fast_ride_duration * 1.5 > duration  # less than the maximum fast ride duration


# DataFrame -> DataFrame
def filter_and_prepare_data(df):
    # Remove rows with NA in start_station_id or end_station_id
    df = df.dropna(subset=['start_station_id', 'end_station_id']).reset_index(drop=True)
    
    # Convert the 'started_at' and 'ended_at' columns to datetime
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    
    # Calculate duration in seconds
    df['duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds()
    
    # Filter by duration, keeping trips from 1 minute to 1 hour
    df = df[(df['duration'] >= 60) & (df['duration'] <= 3600)].reset_index(drop=True)

    if len(df) == 0:
        return df
     # apply this is fast enough
    df['is_fast_enough'] = df.apply(
        lambda row: this_is_fast_enough(
            row['start_lat'], row['start_lng'], 
            row['end_lat'], row['end_lng'], 
            row['duration']
        ), axis=1
    )
    df = df[df.is_fast_enough].reset_index(drop=True)

    station_counts = pd.concat([df.start_station_id, df.end_station_id]).value_counts()
    station_counts = station_counts[station_counts > MIN_STATION_COUNT]
    df = df[df.start_station_id.isin(station_counts.index) & df.end_station_id.isin(station_counts.index)]
    df=df.reset_index(drop=True)
    
    return df

# DataFrame -> dict
def get_stats(df):
    if len(df) == 0:
        return {
            'bike_counts': {},
            'member_counts': {},
            'daily_counts': {}
        }
    bike_counts = df.rideable_type.value_counts().to_dict()
    member_counts = df.member_casual.value_counts().to_dict()
    daily_counts = df['started_at'].dt.date.value_counts().sort_index().to_dict()

    stats = {
        'bike_counts': bike_counts,
        'member_counts': member_counts,
        'daily_counts': daily_counts
    }

    return stats

# list[dict] -> dict
def join_stats(stats):
    """
    Join multiple stats dictionaries into one.
    """
    joined_stats = {
        'bike_counts': {},
        'member_counts': {},
        'daily_counts': {}
    }

    for stat in stats:
        for key in joined_stats.keys():
            for k, v in stat[key].items():
                if k in joined_stats[key]:
                    joined_stats[key][k] += v
                else:
                    joined_stats[key][k] = v

    return joined_stats

# DataFrame, station_dict -> station_dict
def get_station_coordinates(df, station_coords):

    # Extract start and end station coordinates to one DataFrame
    start_coords = df[['start_station_id', 'start_lat', 'start_lng']]
    end_coords = df[['end_station_id', 'end_lat', 'end_lng']]
    start_coords = start_coords.rename(columns={
        'start_station_id': 'station_id',
        'start_lat': 'latitude',
        'start_lng': 'longitude'
    })
    end_coords = end_coords.rename(columns={
        'end_station_id': 'station_id',
        'end_lat': 'latitude',
        'end_lng': 'longitude'
    })
    all_coords = pd.concat([start_coords, end_coords])

    # Remove existing stations from the DataFrame
    all_coords = all_coords[~ all_coords.station_id.isin(station_coords.keys())]
    
    # Calculate the median coordinates for each station
    all_coords = all_coords.groupby('station_id').median().reset_index()
    new_station_coords = all_coords.set_index('station_id').T.to_dict('list')
    
    # Concate dictionaries
    station_coords.update(new_station_coords)
    return station_coords

# DataFrame -> list[tuple]
def get_trip_pairs(df):
    def sort_pair(row):
        return tuple(sorted([row['start_station_id'], row['end_station_id']]))
    df['pair'] = df.apply(sort_pair, axis=1)

    pair_counts = df["pair"].value_counts()
    pair_counts = pair_counts[pair_counts > MIN_PAIR_COUNT]
    return pair_counts.index

# Graph, start_lat, start_lng, end_lat, end_lng -> float
def calculate_fastest_travel_time(G, start_lat, start_lng, end_lat, end_lng):
    # In the graph, get the nodes closest to the points
    origin_node = ox.nearest_nodes(G, Y=start_lat, X=start_lng)
    destination_node = ox.nearest_nodes(G, Y=end_lat, X=end_lng)

    # compute travel time in seconds
    try:
        travel_time_in_seconds = nx.shortest_path_length(G, origin_node, destination_node, weight='travel_time')
    except nx.NetworkXNoPath:
        # print(f"No path found from ({start_lat}, {start_lng}) to ({end_lat}, {end_lng})")
        return -1
    
    return travel_time_in_seconds

"""
‣ One reader process:
    - reads CSV files,
    - filters and prepares data,
    - adds new “station” coordinates,
    - emits <start,end> pairs.

‣ N worker processes:
    - pull pairs, look up / cache a distance, calculate the fastest travel time,
    - store results in a shared list.

Shared counters give the dashboard:
    Files 1/3 | Queue 14 | Waiting 2/3 | Processed 26
"""

import multiprocessing as mp
import time
from queue import Empty          # non-blocking Queue ops

# ───────────────────────── helpers ──────────────────────────────
def inc(val, n=1):               # atomic += n for a mp.Value
    with val.get_lock():
        val.value += n

# ──────────────────────── producer ──────────────────────────────
def data_reader(queue,
                station_coords, lock,
                files, data_dir,
                items_in_q, files_done):
    for file in files:
        df = pd.read_csv(f'{data_dir}/{file}').head(100000)
        df = filter_and_prepare_data(df)
        
        # Get new station coordinates
        with lock:
            station_coords = get_station_coordinates(df, station_coords)

        # Get trip pairs
        trip_pairs = get_trip_pairs(df)

        # emit trip pairs
        for a, b in trip_pairs:
            queue.put((a, b))
            inc(items_in_q)

        inc(files_done)              # one file done

    queue.put((None, None))          # poison pill

# ──────────────────────── consumer ─────────────────────────────
def distance_worker(wid, queue,
                    G_main,
                    station_coords, distance_cache, results, lock,
                    items_in_q, items_done, workers_waiting):
    processed = calculated = 0

    # No need to load the graph in each worker, just pass it
    G = G_main.copy()
    
    while True:
        try:
            inc(workers_waiting)     # going to block
            a, b = queue.get(timeout=1)
            inc(workers_waiting, -1)
        except Empty:                # nothing to do
            inc(workers_waiting, -1)
            continue

        if a is None:                # shutdown signal
            queue.put((None, None))
            break

        key = (a, b)
        cached = True
        inc(workers_waiting)
        with lock:                   # cache lookup / insert
            inc(workers_waiting, -1)
            if key not in distance_cache:
                start_lan, start_lon = station_coords[a]
                end_lan, end_lon = station_coords[b]
                cached = False

        if not cached:
            fastest_time = calculate_fastest_travel_time(G, start_lan, start_lon, end_lan, end_lon)
            with lock:               # cache insert
                distance_cache[key] = fastest_time
            calculated += 1

        processed += 1
        inc(items_in_q, -1)
        inc(items_done)

    with lock:
        results.append(dict(worker=wid,
                            processed=processed,
                            calculated=calculated))

# ────────────────── tiny dashboard in parent ───────────────────
def monitor(total_files, nworkers,
            files_done, items_in_q, workers_waiting, items_done,
            procs):
    start_time = time.perf_counter()
    while any(p.is_alive() for p in procs):
        ctime = time.perf_counter()
        elapsed = ctime - start_time
        print(f"\rFiles {files_done.value}/{total_files} | "
              f"Queue {items_in_q.value} | "
              f"Waiting {workers_waiting.value}/{nworkers} | "
              f"Processed {items_done.value} | "
              f"Elapsed {elapsed:.1f}s",
              end="", flush=True)
        time.sleep(0.2)
    print()                          # newline when finished


# Parallel processing function

def run_the_process(nworkers):
    # ──────────────────────── bootstrap ────────────────────────────
    queue = mp.Queue()
    manager = mp.Manager()
    station_coords = manager.dict()    # station_id → coord
    distance_cache = manager.dict()    # (a,b)     → dist
    results        = manager.list()
    lock           = manager.Lock()

    # shared dashboard counters
    files_done      = mp.Value('i', 0)
    items_in_q      = mp.Value('i', 0)
    items_done      = mp.Value('i', 0)
    workers_waiting = mp.Value('i', 0)

    reader = mp.Process(target=data_reader,
                    args=(queue, station_coords, lock,
                            FILES, DATA_DIR,
                            items_in_q, files_done))

    workers = [mp.Process(target=distance_worker,
                            args=(wid, queue,
                                G, station_coords, distance_cache, 
                                results, lock,
                                items_in_q, items_done, workers_waiting))
                for wid in range(nworkers)]

    procs = [reader, *workers]
    for p in procs: p.start()

    monitor(len(FILES), nworkers,
        files_done, items_in_q, workers_waiting, items_done,
        procs)

    for p in procs: p.join()


import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.2f}s")
        return total_time

    return timeit_wrapper

# function to run paralel and return the time
@timeit
def run_parallel(nworkers):
    return run_the_process(nworkers)

if __name__ == "__main__":
    nworkers = [1, 2, 4, 8, 16, 32]
    results = []

    FILES = FILES[:4]
    for n in nworkers:
        print(f"Running with {n} workers...")
        result = run_parallel(n)
        results.append(result)
    print("All processes completed.")
    print("Results:", results)

    # Save the results to a json
    import json
    with open('speedup_results.json', 'w') as f:
        json.dump(results, f)


