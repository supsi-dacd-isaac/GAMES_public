import numpy as np
from sklearn.neighbors import KernelDensity
import math
from scipy.stats import entropy


def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 6371 * c  # Radius of Earth in kilometers
    return distance


def kl_divergence_between_sets(set1, set2, bandwidth=0.5, grid_resolution=100):
    """
    Calculate the Kullback-Leibler (KL) divergence between two sets of observations using KDE.
    Parameters:
    - set1: NumPy array, first set of observations (n x 2).
    - set2: NumPy array, second set of observations (m x 2).
    - bandwidth: Bandwidth parameter for the kernel density estimators (default: 0.5).
    - grid_resolution: Number of grid points for PDF evaluation (default: 100).
    Returns:
    - kl_divergence: KL divergence between the two sets.
    """
    # Create kernel density estimators for both datasets
    kde1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    # Fit the kernel density estimators to the data
    kde1.fit(set1)
    kde2.fit(set2)
    # Generate a range of values for which to evaluate the PDFs
    x_values = np.linspace(set1[:, 0].min(), set1[:, 0].max(), grid_resolution)
    y_values = np.linspace(set1[:, 1].min(), set1[:, 1].max(), grid_resolution)
    X, Y = np.meshgrid(x_values, y_values)
    xy_values = np.column_stack([X.ravel(), Y.ravel()])
    # Evaluate the PDFs for the grid
    pdf1 = np.exp(kde1.score_samples(xy_values)).reshape(X.shape)
    pdf2 = np.exp(kde2.score_samples(xy_values)).reshape(X.shape)
    # Calculate the KL divergence from set1 to set2
    kl_divergence = entropy(pdf1.ravel(), pdf2.ravel())
    return kl_divergence



def bhattacharyya_distance(set1, set2):
    # Convert the input sets to numpy arrays
    set1 = np.array(set1)
    set2 = np.array(set2)

    # Calculate the means and covariances of each set
    mean1 = np.mean(set1, axis=0)
    mean2 = np.mean(set2, axis=0)
    cov1 = np.cov(set1, rowvar=False)
    cov2 = np.cov(set2, rowvar=False)

    # Calculate the Bhattacharyya coefficient
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)

    diff_mean = mean1 - mean2

    term1 = 0.25 * np.dot(np.dot(diff_mean, 0.5 * (inv_cov1 + inv_cov2)), diff_mean.T)
    term2 = 0.5 * np.log(np.linalg.det(0.5 * (cov1 + cov2)) / np.sqrt(det_cov1 * det_cov2))

    # Calculate Bhattacharyya distance
    b_distance = term1 + term2

    # Normalize the Bhattacharyya distance by the square root of the product of sample sizes
    normalization_factor = np.sqrt(set1.shape[0] * set2.shape[0])
    normalized_distance = b_distance / normalization_factor

    return normalized_distance




