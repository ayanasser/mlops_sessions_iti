"""Minimal ride-duration model used by the FastAPI example."""


class RideDurationModel:
    """A tiny placeholder model that estimates trip duration in minutes.

    Replace `predict` with a real trained model when available. The current
    heuristic assumes an average speed and adds a small per-passenger overhead.
    """

    #: Assumed average speed in km per minute (~30 km/h).
    AVG_SPEED_KM_PER_MIN = 0.5
    #: Extra minutes added per passenger (boarding overhead).
    PASSENGER_OVERHEAD_MIN = 0.5

    def predict(self, features: list[float]) -> float:
        """Predict ride duration in minutes.

        Args:
            features: ``[distance_km, passengers]``.

        Returns:
            Estimated duration in minutes, rounded to 2 decimals.
        """
        distance, passengers = features[0], features[1]
        duration = distance / self.AVG_SPEED_KM_PER_MIN
        duration += passengers * self.PASSENGER_OVERHEAD_MIN
        return round(duration, 2)
