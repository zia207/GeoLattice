import os

# Allocate only the device memory this script uses. The default JAX pool
# asks for most of the GPU, which fails when another process already holds it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class VariogramCalculator:
    """
    Calculate variogram (semivariogram) for spatial data using JAX.
    The variogram shows the spatial correlation of a regionalized variable.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_variogram(
        points: jnp.ndarray,
        values: jnp.ndarray,
        max_distance: float,
        num_bins: int = 50
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate the variogram for given spatial points and values.
        
        Args:
            points: (N, 2) array of spatial coordinates
            values: (N,) array of measured values at points
            max_distance: maximum distance for variogram calculation
            num_bins: number of distance bins
            
        Returns:
            (distances, variogram_values) tuple
        """
        points = jnp.asarray(points)
        values = jnp.asarray(values)
        n = points.shape[0]

        # One entry per unordered pair. The diagonal is a zero lag and is excluded.
        iu, ju = jnp.triu_indices(n, k=1)
        pair_dist = jnp.linalg.norm(points[iu] - points[ju], axis=-1)
        # γ(h) = (1 / (2 N(h))) * Σ [Z(xi) - Z(xi+h)]²
        pair_gamma = 0.5 * (values[iu] - values[ju]) ** 2

        bin_edges = jnp.linspace(0, max_distance, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Bins are [lower, upper). Distances outside that range are dropped.
        bin_idx = jnp.searchsorted(bin_edges, pair_dist, side="right") - 1
        in_range = (bin_idx >= 0) & (bin_idx < num_bins)
        safe_idx = jnp.clip(bin_idx, 0, num_bins - 1)

        numer = jnp.zeros(num_bins).at[safe_idx].add(
            jnp.where(in_range, pair_gamma, 0.0)
        )
        counts = jnp.zeros(num_bins).at[safe_idx].add(
            jnp.where(in_range, 1.0, 0.0)
        )
        variogram_values = jnp.where(counts > 0, numer / counts, 0.0)

        return bin_centers, variogram_values
    
    @staticmethod
    def calculate_directional_variogram(
        points: jnp.ndarray,
        values: jnp.ndarray,
        direction: jnp.ndarray,
        max_distance: float,
        num_bins: int = 50
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate directional variogram in a specific direction.
        
        Args:
            points: (N, 2) array of spatial coordinates
            values: (N,) array of measured values
            direction: (2,) direction vector
            max_distance: maximum distance
            num_bins: number of distance bins
            
        Returns:
            (distances, directional_variogram_values)
        """
        # Normalize direction
        direction = direction / jnp.linalg.norm(direction)
        
        # Project points onto direction
        projections = points @ direction
        
        # Sort by projection
        sorted_indices = jnp.argsort(projections)
        sorted_projections = projections[sorted_indices]
        sorted_values = values[sorted_indices]
        
        # Calculate variogram along direction
        bin_edges = jnp.linspace(0, max_distance, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        def calculate_directional_bin(bin_idx):
            h = bin_centers[bin_idx]
            
            # Find pairs separated by distance h in this direction
            n = sorted_projections.shape[0]
            
            def calculate_pair_sum(i):
                # Find j such that projection[j] - projection[i] ≈ h
                target = sorted_projections[i] + h
                
                # Find nearest index
                diffs = jnp.abs(sorted_projections - target)
                j = jnp.argmin(diffs)
                
                close = jnp.abs(sorted_projections[j] - target) < h * 0.1
                diff = sorted_values[i] - sorted_values[j]
                return jnp.where(close, diff ** 2, 0.0)
            
            # Sum over all i
            pair_sums = jax.vmap(calculate_pair_sum)(jnp.arange(n))
            total_sum = jnp.sum(pair_sums)
            
            # Count valid pairs
            valid_pairs = jnp.sum(pair_sums > 0)
            return jnp.where(
                valid_pairs > 0,
                total_sum / (2 * jnp.maximum(valid_pairs, 1)),
                0.0,
            )
        
        variogram_values = jax.vmap(calculate_directional_bin)(
            jnp.arange(num_bins)
        )
        
        return bin_centers, variogram_values


class KrigingInterpolator:
    """
    Kriging (Kriging) interpolation using JAX.
    Implements Ordinary Kriging with variogram-based weights.
    """
    
    def __init__(self):
        self.variogram_calculator = VariogramCalculator()
    
    @staticmethod
    def calculate_variogram_model(
        distances: jnp.ndarray,
        variogram_values: jnp.ndarray,
        model_type: str = 'spherical'
    ) -> callable:
        """
        Fit a variogram model to the calculated variogram.
        
        Args:
            distances: array of distances
            variogram_values: array of variogram values
            model_type: type of variogram model
            
        Returns:
            function that calculates variogram for any distance
        """
        if model_type == 'spherical':
            # Spherical model: γ(h) = C * (1.5 * h/a - 0.5 * (h/a)³) for h <= a
            # Fit parameters using least squares
            def spherical_model(h, C, a):
                h_norm = h / a
                return jnp.where(
                    h <= a,
                    C * (1.5 * h_norm - 0.5 * h_norm**3),
                    C
                )
            
            # Simple fitting: use maximum variogram value as C
            C = jnp.max(variogram_values)
            
            # Find range parameter a (distance where variogram stabilizes)
            threshold = 0.95 * C
            a = distances[jnp.argmin(jnp.abs(variogram_values - threshold))]
            
            return lambda h: spherical_model(h, C, a)
            
        elif model_type == 'exponential':
            # Exponential model: γ(h) = C * (1 - exp(-3h/a))
            def exponential_model(h, C, a):
                return C * (1 - jnp.exp(-3 * h / a))
            
            C = jnp.max(variogram_values)
            
            # Find a such that γ(a) ≈ 0.95 * C
            threshold = 0.95 * C
            a = distances[jnp.argmin(jnp.abs(variogram_values - threshold))]
            
            return lambda h: exponential_model(h, C, a)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def kriging_interpolate(
        known_points: jnp.ndarray,
        known_values: jnp.ndarray,
        unknown_points: jnp.ndarray,
        variogram_func: callable,
        nugget: float = 1.0,
        max_neighbors: int = 10
    ) -> jnp.ndarray:
        """
        Perform kriging interpolation at unknown points.
        
        Args:
            known_points: (N, 2) array of known spatial coordinates
            known_values: (N,) array of known values
            unknown_points: (M, 2) array of points to interpolate
            variogram_func: function that returns variogram for given distance
            nugget: nugget effect parameter (0 < nugget <= 1)
            max_neighbors: maximum number of neighbors to use
            
        Returns:
            (M,) array of interpolated values
        """
        known_points = jnp.asarray(known_points)
        known_values = jnp.asarray(known_values)
        unknown_points = jnp.asarray(unknown_points)
        k = min(int(max_neighbors), int(known_points.shape[0]))

        def interpolate_point(point):
            # Nearest neighbors in pure JAX so vmap can trace the search.
            dists = jnp.linalg.norm(known_points - point[None, :], axis=1)
            neighbor_indices = jnp.argsort(dists)[:k]
            neighbor_dists = dists[neighbor_indices]
            neighbor_points = known_points[neighbor_indices]
            neighbor_values = known_values[neighbor_indices]

            # Ordinary kriging system:
            # [γ(h_ij)  1] [λ]   [γ(h_i0)]
            # [1 ... 1  0] [μ] = [1      ]
            pair_dist = jnp.linalg.norm(
                neighbor_points[:, None, :] - neighbor_points[None, :, :],
                axis=-1,
            )
            gamma = variogram_func(pair_dist)
            gamma_0 = variogram_func(neighbor_dists)

            matrix = jnp.zeros((k + 1, k + 1)).at[:k, :k].set(gamma)
            matrix = matrix.at[:k, k].set(1.0).at[k, :k].set(1.0)
            rhs = jnp.concatenate([gamma_0, jnp.ones(1)])

            # Add small regularization for numerical stability
            matrix = matrix + nugget * jnp.eye(k + 1) * 1e-10
            weights = jnp.linalg.solve(matrix, rhs)
            return jnp.dot(weights[:k], neighbor_values)

        return jax.vmap(interpolate_point)(unknown_points)
    
    @staticmethod
    def simple_kriging(
        known_points: jnp.ndarray,
        known_values: jnp.ndarray,
        unknown_points: jnp.ndarray,
        range_param: float,
        sill: float,
        nugget: float = 1.0
    ) -> jnp.ndarray:
        """
        Simplified kriging with spherical variogram model.
        
        Args:
            known_points: (N, 2) array of known points
            known_values: (N,) array of known values
            unknown_points: (M, 2) array of points to interpolate
            range_param: range parameter 'a' in variogram model
            sill: sill value 'C' in variogram model
            nugget: nugget effect parameter
            
        Returns:
            (M,) array of interpolated values
        """
        # Define spherical variogram
        def spherical_variogram(h):
            h_norm = h / range_param
            return jnp.where(
                h <= range_param,
                sill * (1.5 * h_norm - 0.5 * h_norm**3),
                sill
            )
        
        # Use the general kriging method
        return KrigingInterpolator.kriging_interpolate(
            known_points,
            known_values,
            unknown_points,
            spherical_variogram,
            nugget
        )


# Example usage and demonstration
def example_usage():
    """
    Example demonstrating variogram calculation and kriging interpolation.
    """
    # Generate sample data
    key = random.PRNGKey(42)
    
    # Create a grid of known points
    x = jnp.linspace(0, 10, 10)
    y = jnp.linspace(0, 10, 10)
    X, Y = jnp.meshgrid(x, y)
    known_points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    
    # Generate known values (simulated spatial field)
    def spatial_field(points):
        x, y = points[:, 0], points[:, 1]
        return jnp.sin(x * 0.5) * jnp.cos(y * 0.3) + 0.1 * x + 0.05 * y
    
    known_values = spatial_field(known_points)
    
    # Calculate variogram
    variogram_calc = VariogramCalculator()
    distances, variogram_values = variogram_calc.calculate_variogram(
        known_points,
        known_values,
        max_distance=5.0,
        num_bins=30
    )
    
    print("Variogram Calculation Results:")
    print(f"Distances: {distances[:5]}...")
    print(f"Variogram values: {variogram_values[:5]}...")
    
    # Fit variogram model
    kriging = KrigingInterpolator()
    variogram_func = kriging.calculate_variogram_model(
        distances,
        variogram_values,
        model_type='spherical'
    )
    
    # Generate unknown points for interpolation
    unknown_points = jnp.array([
        [2.5, 3.5],
        [7.5, 2.5],
        [5.0, 5.0],
        [1.5, 8.5]
    ])
    
    # Perform kriging interpolation
    interpolated_values = kriging.kriging_interpolate(
        known_points,
        known_values,
        unknown_points,
        variogram_func,
        nugget=1.0,
        max_neighbors=5
    )
    
    print("\nKriging Interpolation Results:")
    for i, point in enumerate(unknown_points):
        true_value = spatial_field(jnp.array([point]))[0]
        print(f"Point {point}: interpolated={interpolated_values[i]:.4f}, "
              f"true={true_value:.4f}, error={abs(interpolated_values[i] - true_value):.4f}")
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Plot variogram
    plt.subplot(1, 2, 1)
    plt.plot(distances, variogram_values, 'b-', label='Variogram')
    plt.xlabel('Distance (h)')
    plt.ylabel('γ(h)')
    plt.title('Variogram')
    plt.grid(True)
    plt.legend()
    
    # Plot interpolation results
    plt.subplot(1, 2, 2)
    plt.scatter(known_points[:, 0], known_points[:, 1], 
                c=known_values, cmap='viridis', label='Known points')
    plt.scatter(unknown_points[:, 0], unknown_points[:, 1],
                c=interpolated_values, cmap='viridis', marker='x', 
                s=100, label='Interpolated')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Kriging Interpolation')
    plt.colorbar()
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return distances, variogram_values, interpolated_values


# Additional utility functions
def calculate_cross_variogram(
    points1: jnp.ndarray,
    values1: jnp.ndarray,
    points2: jnp.ndarray,
    values2: jnp.ndarray,
    max_distance: float,
    num_bins: int = 50
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate cross-variogram between two spatial fields.
    """
    points1 = jnp.asarray(points1)
    values1 = jnp.asarray(values1)
    points2 = jnp.asarray(points2)
    values2 = jnp.asarray(values2)

    pair_dist = jnp.linalg.norm(
        points1[:, None, :] - points2[None, :, :], axis=-1
    ).ravel()
    pair_gamma = (0.5 * (values1[:, None] - values2[None, :]) ** 2).ravel()

    bin_edges = jnp.linspace(0, max_distance, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_idx = jnp.searchsorted(bin_edges, pair_dist, side="right") - 1
    in_range = (bin_idx >= 0) & (bin_idx < num_bins)
    safe_idx = jnp.clip(bin_idx, 0, num_bins - 1)

    numer = jnp.zeros(num_bins).at[safe_idx].add(
        jnp.where(in_range, pair_gamma, 0.0)
    )
    counts = jnp.zeros(num_bins).at[safe_idx].add(
        jnp.where(in_range, 1.0, 0.0)
    )
    cross_variogram = jnp.where(counts > 0, numer / counts, 0.0)

    return bin_centers, cross_variogram


# Run example if script is executed
if __name__ == "__main__":
    distances, variogram_values, interpolated_values = example_usage()