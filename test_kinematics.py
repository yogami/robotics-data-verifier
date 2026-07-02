"""
FK Validation Test Suite for ViperX 300 Kinematics.
Tests against known URDF poses to catch joint-mapping and offset bugs.
"""
import numpy as np
import pytest
from kinematics import BimanualForwardKinematics


@pytest.fixture
def fk():
    return BimanualForwardKinematics()


class TestZeroPose:
    """Validate FK output at the zero configuration against URDF expectations."""

    def test_zero_pose_tcp_x(self, fk):
        """At q=0, TCP x should be sum of all X-axis link offsets."""
        pos, _ = fk.solve_arm_fk(np.zeros(7))
        expected_x = fk.L2_x + fk.L3_x + fk.L4_x + fk.L5_x + fk.L6_x  # 0.615
        assert abs(pos[0] - expected_x) < 1e-6, f"TCP x={pos[0]:.6f}, expected {expected_x}"

    def test_zero_pose_tcp_y(self, fk):
        """At q=0, TCP y should be zero (arm in XZ plane)."""
        pos, _ = fk.solve_arm_fk(np.zeros(7))
        assert abs(pos[1]) < 1e-6, f"TCP y={pos[1]:.6f}, expected 0.0"

    def test_zero_pose_tcp_z(self, fk):
        """At q=0, TCP z should be L1_z + L2_z (base + shoulder height)."""
        pos, _ = fk.solve_arm_fk(np.zeros(7))
        expected_z = fk.L1_z + fk.L2_z  # 0.4385
        assert abs(pos[2] - expected_z) < 1e-6, f"TCP z={pos[2]:.6f}, expected {expected_z}"

    def test_zero_pose_rotation_is_identity(self, fk):
        """At q=0, rotation matrix should be identity (no rotation)."""
        _, R = fk.solve_arm_fk(np.zeros(7))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


class TestKnownPoses:
    """Validate FK at specific joint configurations with analytically known results."""

    def test_shoulder_minus_90(self, fk):
        """q2=-π/2: shoulder rotates arm down (URDF Y-axis convention)."""
        joints = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])
        pos, _ = fk.solve_arm_fk(joints)
        # In the Interbotix URDF, q2=-π/2 rotates the shoulder so the arm
        # points downward. TCP z should be significantly below zero-pose z.
        zero_pos, _ = fk.solve_arm_fk(np.zeros(7))
        assert pos[2] < zero_pos[2] - 0.3, f"Shoulder -π/2 should drop TCP well below zero-pose, got z={pos[2]:.4f}"

    def test_waist_90(self, fk):
        """q1=π/2: waist rotates 90° — TCP should be in the Y direction."""
        joints = np.array([np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pos, _ = fk.solve_arm_fk(joints)
        # After 90° waist rotation, x→y, y→-x
        expected_x = 0.0
        expected_y = fk.L2_x + fk.L3_x + fk.L4_x + fk.L5_x + fk.L6_x  # 0.615
        assert abs(pos[0] - expected_x) < 1e-6, f"TCP x={pos[0]:.4f}, expected ~0"
        assert abs(pos[1] - expected_y) < 1e-6, f"TCP y={pos[1]:.4f}, expected {expected_y}"

    def test_elbow_90(self, fk):
        """q3=π/2: elbow bends 90° — arm folds, TCP z increases, TCP x decreases."""
        joints = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0])
        pos, _ = fk.solve_arm_fk(joints)
        # With elbow at 90°, the forearm points upward
        # TCP x should be less than zero-pose x
        zero_pos, _ = fk.solve_arm_fk(np.zeros(7))
        assert pos[0] < zero_pos[0], "Elbow 90° should reduce TCP x"
        assert pos[2] > zero_pos[2], "Elbow 90° should increase TCP z"


class TestBatchConsistency:
    """Verify vectorized batch FK matches single-frame FK."""

    def test_batch_matches_single(self, fk):
        """Batch of 5 random configs should match individual single-frame calls."""
        np.random.seed(42)
        configs = np.random.uniform(-1.0, 1.0, (5, 7))
        batch_pos, batch_R = fk.solve_arm_fk(configs)

        for i in range(5):
            single_pos, single_R = fk.solve_arm_fk(configs[i])
            np.testing.assert_allclose(
                batch_pos[i], single_pos, atol=1e-10,
                err_msg=f"Position mismatch at config {i}"
            )
            np.testing.assert_allclose(
                batch_R[i], single_R, atol=1e-10,
                err_msg=f"Rotation mismatch at config {i}"
            )

    def test_bimanual_batch(self, fk):
        """Bimanual 14-D batch should work correctly."""
        np.random.seed(123)
        configs_14d = np.random.uniform(-0.5, 0.5, (3, 14))
        (l_pos, l_R), (r_pos, r_R) = fk.solve_bimanual_fk(configs_14d)
        assert l_pos.shape == (3, 3)
        assert r_pos.shape == (3, 3)
        assert l_R.shape == (3, 3, 3)
        assert r_R.shape == (3, 3, 3)


class TestDimensionGuards:
    """Verify dimension checking prevents silent garbage computation."""

    def test_16d_input_handled(self, fk):
        """16-D input (Mobile ALOHA with base velocity) should slice correctly."""
        joints_16 = np.zeros(16)
        # Set base velocity dims to garbage
        joints_16[0] = 999.0
        joints_16[1] = 999.0
        # Set arm joints to zero
        (l_pos, _), (r_pos, _) = fk.solve_bimanual_fk(joints_16)
        # Should get the zero-pose result (base velocity ignored)
        expected_x = fk.L2_x + fk.L3_x + fk.L4_x + fk.L5_x + fk.L6_x
        assert abs(l_pos[0] - expected_x) < 1e-6

    def test_16d_batch_handled(self, fk):
        """Batch 16-D input should also work."""
        joints_16 = np.zeros((3, 16))
        joints_16[:, 0] = 999.0  # garbage base vel
        joints_16[:, 1] = 999.0  # garbage base vel
        (l_pos, _), (r_pos, _) = fk.solve_bimanual_fk(joints_16)
        assert l_pos.shape == (3, 3)

    def test_bad_dims_raise_error(self, fk):
        """10-D input should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 14 or 16"):
            fk.solve_bimanual_fk(np.zeros(10))

    def test_bad_batch_dims_raise_error(self, fk):
        """Batch with 10 columns should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 14 or 16"):
            fk.solve_bimanual_fk(np.zeros((5, 10)))


class TestAngleWrapping:
    """Regression tests for joint angle boundary handling."""

    def test_waist_near_pi_no_phantom_drift(self, fk):
        """Leader at +π and follower at -π should show near-zero Cartesian drift."""
        leader = np.array([np.pi - 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        follower = np.array([-np.pi + 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Angle-wrapped difference
        d = leader - follower
        d = (d + np.pi) % (2 * np.pi) - np.pi
        mapped_leader = follower + d

        l_pos, _ = fk.solve_arm_fk(mapped_leader)
        f_pos, _ = fk.solve_arm_fk(follower)
        drift = np.linalg.norm(l_pos - f_pos)

        # The actual angular difference is 0.02 rad → small Cartesian drift
        assert drift < 0.015, f"Expected small drift for 0.02 rad waist diff, got {drift:.4f}m"

    def test_without_wrapping_joint_diff_is_huge(self, fk):
        """Without angle wrapping, raw joint-space difference is ~2π (phantom)."""
        leader = np.array([np.pi - 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        follower = np.array([-np.pi + 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # WITHOUT wrapping — raw joint-space difference is ~2π
        raw_diff = np.abs(leader[0] - follower[0])
        assert raw_diff > 6.0, f"Expected ~2π raw joint diff, got {raw_diff:.4f}"

        # WITH wrapping — actual angular difference is only ~0.02 rad
        d = leader - follower
        d = (d + np.pi) % (2 * np.pi) - np.pi
        wrapped_diff = np.abs(d[0])
        assert wrapped_diff < 0.03, f"Expected ~0.02 wrapped diff, got {wrapped_diff:.4f}"
