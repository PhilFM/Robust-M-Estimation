import numpy as np

def circular_median_radians(angles):
    """
    Calculates the circular median for a list or array of angles in radians.
    """
    # Ensure angles are wrapped within [0, 2*pi)
    angles = np.asarray(angles) % (2 * np.pi)

    best_median = None
    min_dist_sum = float('inf')
                
    # Evaluate each angle as a candidate median point
    for candidate in angles:
        # Compute the absolute differences
        diff = np.abs(angles - candidate)
        # Find the shortest arc distance on the circle
        distances = np.minimum(diff, 2 * np.pi - diff)
        total_distance = np.sum(distances)
                    
        # Keep track of the angle that minimizes the total distance
        if total_distance < min_dist_sum:
            min_dist_sum = total_distance
            best_median = candidate
                    
    return best_median            

