# Bike Rent - Fast Trips Analysis

## Overview
This project analyzes bike sharing data to identify fast point-to-point trips by comparing actual trip durations with the shortest possible route durations using OpenStreetMap data. The analysis includes parallel processing for efficient computation and visualization of results.

**Main notebook:** `multiwork.ipynb` - Contains the complete analysis pipeline with parallel processing implementation.

## Objectives
- Detection of fast point-to-point bike trips by comparing actual and possible shortest route durations
- Comparison of bike type and membership distributions between fast and regular trips
- Analysis of temporal patterns in fast trips throughout the year
- Visualization of common point-to-point routes on an interactive map

## Implementation Approach

### Data Processing Pipeline
1. **Data Filtering**
   - Remove trips with missing station information
   - Filter trips by duration (1 minute to 1 hour)
   - Keep only trips between well-used stations (>100 trips per station)

2. **Route Analysis**
   - Calculate shortest possible route durations using OSMnx and NetworkX
   - Use Manhattan distance approximation for initial filtering
   - Cache route calculations for efficiency

3. **Fast Trip Detection**
   - Identify trips that are within 110% of the optimal route duration
   - Compare distributions between all trips and fast trips

4. **Visualization**
   - Generate interactive maps showing popular fast routes
   - Create statistical comparisons across different dimensions

### Parallel Processing Architecture

The implementation uses a producer-consumer pattern with multiple worker processes:

- **Reader Process**: Loads CSV files, filters data, extracts station coordinates, and queues route pairs
- **Worker Processes**: Calculate shortest path durations using cached lookups and OSMnx routing
- **Shared Resources**: Station coordinates dictionary and distance cache for efficient reuse

**Performance Analysis:** Parallel execution times are measured using `speedup.py` across different worker counts.

## Key Findings

### Route Patterns
- Most fast trips occur between city center stations and more distant locations
- Popular routes show clear commuting patterns

### Temporal Patterns
- **Weekdays vs Weekends**: Higher percentage of fast trips during weekdays, indicating commuter behavior
- **Seasonal Trends**: More fast trips percentage-wise during winter months

### User Behavior
- **Membership**: Members perform fast trips more frequently than casual users
- **Bike Types**: Electric bikes are more commonly used for fast trips, likely due to their inherent speed advantage
