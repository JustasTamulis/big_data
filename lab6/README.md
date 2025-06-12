# Bike rent - fast trips

## Objectives
- Detection of fast point-to-point bike trips by comparing actual and possible shortest route durations.
- Comparisson of the bike type and membership distributions in fast and other trips.
- Showcase what part of all trips does the fast trips take throughout the year.
- Visualization of common poin-to-point routes.
 

## Plan:
1. filter
    - remove NA
    - remove long durations 
    - leave only from known station to station
2. check stats
    - distribution per type, membership
    - count each day total trips
3. preprocess:
    - for each station, find the center
    - change all coords for trips to centers
    - calculate the actual duration (time end - time start)
4. calculate the shortest routes' durations
    - go through all trips
    - save the duration (from start_id, to end_id), reuse it (just 2d matrix)
    - append the shortest durations 
5. filter 
    - remove routes which took much longer than the shortest possible duration (20% longer)
6. check stats
    - distribution per type, membership
    - count each day total trips
    - compare to original stats
7. plot the routes
    - group by start and end station
    - the more frequent the route is, the bigger the line



## parallelization:

exact details will depend on sequential tests.

1. count time take of each step.
2. count how much it would take for 1 worker in sequential.
3. Different workers for different jobs.
4. Some overview of the load and data movemvemt, stack sizes?

workers:
- loaders
- distance calculator: with loaded G from osmnx

shared objects:
- station_id -> coords
- matrix with distances
- queue (stack for routes to be calculated)

stacks:
- loaded-filtered


1. worker 1
    - loads
    - filters
    - gets distribution
    - finds unknown station centers (saves [station_id, coords])
    - pushes all routes (unique [station_start, station_end]) to stack
2. worker 2
    - reads stack 1 item
    - checks if its in matrix
        - gets value if it is
        - calculates the duration if it is not
            - saves to matrix
    - 