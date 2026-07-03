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
