import numpy as np
import pandas as pd


def daylength(dayOfYear: pd.Series, lat: pd.Series | float) -> pd.Series:
    """Computes the length of the day (the time between sunrise and
    sunset) given a pandas Series of day of the year and latitude.

    Function uses the Brock model for the computations.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45 * np.sin(np.deg2rad(360.0 * (283.0 + dayOfYear) / 365.0))

    # Calculate the core trigonometric argument matrix
    val = -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))

    # Clip the values strictly between -1.0 and 1.0.
    # This automatically converts completely polar days to arccos(-1) -> 24 hours
    # and polar nights to arccos(1) -> 0 hours safely without throwing math errors.
    val_clipped = np.clip(val, -1.0, 1.0)

    hourAngle = np.rad2deg(np.arccos(val_clipped))
    return 2.0 * hourAngle / 15.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on the Earth's surface.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point in degrees.
        lat2, lon2: Latitude and longitude of the second point in degrees.
        
    Returns:
        float: Distance in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def pair_cities_by_proximity(city_dict: dict) -> dict:
    """
    Dynamically pairs up cities based on their geographic proximity using a 
    greedy Haversine matching algorithm.

    Args:
        city_dict (dict): Dictionary of cities with their 'lat' and 'lon' coordinates.

    Returns:
        dict: A dictionary mapping a group_id (int) to a list of city names (usually pairs).
    """
    # Sort keys to ensure the greedy pairing is deterministic and reproducible
    unpaired = sorted(list(city_dict.keys()))
    groups = {}
    group_id = 1
    
    while len(unpaired) >= 2:
        city_a = unpaired.pop(0)
        # Find the closest geographic neighbor to city_a
        best_match = min(
            unpaired, 
            key=lambda c: haversine_distance(
                city_dict[city_a]["lat"], city_dict[city_a]["lon"],
                city_dict[c]["lat"], city_dict[c]["lon"]
            )
        )
        unpaired.remove(best_match)
        groups[group_id] = [city_a, best_match]
        group_id += 1
        
    # Handle the odd city out, if an odd number of cities was provided
    if unpaired:
        groups[group_id] = [unpaired[0]]

    return groups