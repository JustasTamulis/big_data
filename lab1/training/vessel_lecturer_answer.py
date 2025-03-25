import pandas as pd
import numpy as np
import multiprocessing as mp
import timeit_wrapper as tw
from tqdm import tqdm


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(
        dlon / 2
    ) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


@tw.timeit
def single_vessel_distance(mmsi, chunk):
    vessel_data = chunk[chunk["MMSI"] == mmsi].copy()

    if len(vessel_data) < 2:
        return mmsi, 0

    vessel_data["distance"] = 0.0
    for i in range(1, len(vessel_data)):
        lat1, lon1 = (
            vessel_data.iloc[i - 1]["Latitude"],
            vessel_data.iloc[i - 1]["Longitude"],
        )
        lat2, lon2 = vessel_data.iloc[i]["Latitude"], vessel_data.iloc[i]["Longitude"]
        vessel_data.loc[vessel_data.index[i], "distance"] = calculate_distance(
            lat1, lon1, lat2, lon2
        )

    total_distance = vessel_data["distance"].sum()
    return mmsi, total_distance


def process_large_file(file_path, chunk_size=10000):
    results = []
    pool = mp.Pool(mp.cpu_count() - 1)
    print(f"Using {mp.cpu_count() - 1} processes")

    total_rows = sum(1 for row in open(file_path)) - 1
    total_chunks = total_rows // chunk_size + 1

    with tqdm(total=total_chunks, unit="chunk", desc="Processing chuncks") as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            mmsis = chunk["MMSI"].unique()
            results.extend(
                pool.starmap(single_vessel_distance, [(mmsi, chunk) for mmsi in mmsis])
            )
            pbar.update(1)

    pool.close()
    pool.join()
    result_df = pd.DataFrame(results, columns=["MMSI", "Total Distance"])
    return result_df


if __name__ == "__main__":
    print("x")
    # file_path = 'aisdk-2025-02-09.csv'
    file_path = "aisdk-small.csv"
    result_df = process_large_file(file_path)
    print(result_df)
