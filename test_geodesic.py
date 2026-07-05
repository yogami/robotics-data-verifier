import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def compute_geodesic_distance(R1, R2):
    # Vectorized relative rotation matrices: R_rel = R1^T @ R2
    R_rel = np.matmul(R1.transpose(0, 2, 1), R2)
    I_batch = np.tile(np.eye(3), (R_rel.shape[0], 1, 1))

    # Mathematically stable SO(3) distance via Frobenius norm
    diff = R_rel - I_batch
    fro_norm = np.linalg.norm(diff, axis=(1, 2))
    theta_rad = 2.0 * np.arcsin(np.clip(fro_norm / (2.0 * np.sqrt(2.0)), 0.0, 1.0))
    return theta_rad

def test_geodesic_distance():
    # 1. Test Small Angle
    rot_small = R_scipy.from_euler('x', 5.0, degrees=True).as_matrix()
    R1 = np.eye(3)[np.newaxis, ...]
    R2 = rot_small[np.newaxis, ...]
    dist = compute_geodesic_distance(R1, R2)
    assert np.isclose(np.degrees(dist)[0], 5.0)
    
    # 2. Test 90 degrees
    rot_90 = R_scipy.from_euler('y', 90.0, degrees=True).as_matrix()
    R2_90 = rot_90[np.newaxis, ...]
    dist = compute_geodesic_distance(R1, R2_90)
    assert np.isclose(np.degrees(dist)[0], 90.0)
    
    # 3. Test 180 degrees (edge case)
    rot_180 = R_scipy.from_euler('z', 180.0, degrees=True).as_matrix()
    R2_180 = rot_180[np.newaxis, ...]
    dist = compute_geodesic_distance(R1, R2_180)
    assert np.isclose(np.degrees(dist)[0], 180.0)
    
    # 4. Test Scipy cross-validation for random rotations
    for _ in range(100):
        # Generate two random rotations
        r1_scipy = R_scipy.random()
        r2_scipy = R_scipy.random()
        
        # Scipy's geodesic distance is the magnitude of the rotation vector of the difference
        r_diff = r1_scipy.inv() * r2_scipy
        true_dist = r_diff.magnitude()
        
        R1_rand = r1_scipy.as_matrix()[np.newaxis, ...]
        R2_rand = r2_scipy.as_matrix()[np.newaxis, ...]
        
        dist_rad = compute_geodesic_distance(R1_rand, R2_rand)[0]
        assert np.isclose(dist_rad, true_dist), f"Failed for {dist_rad} != {true_dist}"

if __name__ == "__main__":
    test_geodesic_distance()
    print("Geodesic test passed!")
