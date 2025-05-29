Task:

Data

The provided dataset contains Automatic Identification System (AIS) data for vessels, including details such as MMSI (Maritime Mobile Service Identity), timestamp, latitude, and longitude. Students will need to calculate the distance traveled by each vessel throughout the day and determine which vessel has the longest route.
Tasks

    Data Retrieval
        Download the dataset from the given URL and unzip it to access the .csv or similar format file contained within.

    Data Preparation
        Load the data into a PySpark DataFrame.
        Ensure that the data types for latitude, longitude, and timestamp are appropriate for calculations and sorting.

    Data Processing with PySpark
        Calculate the distance between consecutive positions for each vessel using a suitable geospatial library or custom function that can integrate with PySpark.
        Aggregate these distances by MMSI to get the total distance traveled by each vessel on that day.

    Identifying the Longest Route
        Sort or use an aggregation function to determine which vessel traveled the longest distance.

    Output
        The final output should be the MMSI of the vessel that traveled the longest distance, along with the computed distance.

    Code Documentation and Comments
        Ensure the code is well-documented, explaining key PySpark transformations and actions used in the process.

    Deliverables
        A PySpark script that completes the task from loading to calculating and outputting the longest route.
        A brief report or set of comments within the code that discusses the findings and any interesting insights about the data or the computation process.

Evaluation Criteria

    Correct implementation of data loading and preprocessing.
    Accuracy of the distance calculation.
    Efficiency of PySpark transformations and actions.
    Clarity and completeness of documentation and code comment