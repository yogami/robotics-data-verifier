import numpy as np

class BimanualForwardKinematics:
    """
    Solves analytical Forward Kinematics for a bimanual ALOHA setup.
    Models the Interbotix ViperX 300 Follower Arm (800mm total reach) using
    exact URDF translations and axis conventions to map joint states 
    to Cartesian Tool Center Point (TCP) coordinates and 3D rotation matrices.
    Supports both single-frame (1D) and vectorized batch (2D) inputs.
    """
    def __init__(self):
        # Official Interbotix ViperX 300 URDF Joint Offsets (meters)
        self.L1_z = 0.1385  # Waist to Shoulder (Z-axis offset)
        self.L2_x = 0.050   # Shoulder to Elbow X-offset
        self.L2_z = 0.300   # Shoulder to Elbow Z-offset
        self.L3_x = 0.300   # Elbow to Forearm Roll
        self.L4_x = 0.065   # Forearm Roll to Wrist Pitch
        self.L5_x = 0.100   # Wrist Pitch to Wrist Roll
        self.L6_x = 0.100   # Wrist Roll to Gripper TCP

    def solve_arm_fk(self, joints):
        """
        Solves analytical FK for single frame (1D array of shape (7,)) 
        or vectorized batch of frames (2D array of shape (N, 7)).
        Expected joint order: 
            0: waist (Z-rotation)
            1: shoulder (Y-pitch)
            2: elbow (Y-pitch)
            3: forearm_roll (X-roll)
            4: wrist_pitch (Y-pitch)
            5: wrist_roll (X-roll)
            6: gripper (ignored for arm kinematics)
        
        Returns:
            position: [x, y, z] of the TCP (shape (3,) or (N, 3)).
            R: 3x3 orientation rotation matrix (shape (3, 3) or (N, 3, 3)).

        Note: At TCP positions near the base Z-axis (r≈0 in cylindrical coords),
        the waist angle becomes unobservable from position alone. Position-drift
        attribution to joint 0 (waist) is unreliable in this configuration.
        """
        joints = np.asarray(joints)
        is_batch = joints.ndim == 2
        
        if is_batch:
            N = joints.shape[0]
            if joints.shape[1] < 6:
                return np.zeros((N, 3)), np.tile(np.eye(3), (N, 1, 1))
            q1, q2, q3, q4, q5, q6 = [joints[:, i] for i in range(6)]
        else:
            if len(joints) < 6:
                return np.array([0.0, 0.0, 0.0]), np.eye(3)
            q1, q2, q3, q4, q5, q6 = joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]

        # 1. Base to Waist: Z-rotation
        c1, s1 = np.cos(q1), np.sin(q1)
        if is_batch:
            T01 = np.zeros((N, 4, 4))
            T01[:, 0, 0] = c1; T01[:, 0, 1] = -s1
            T01[:, 1, 0] = s1; T01[:, 1, 1] = c1
            T01[:, 2, 2] = 1.0; T01[:, 2, 3] = self.L1_z
            T01[:, 3, 3] = 1.0
        else:
            T01 = np.array([
                [c1, -s1, 0, 0],
                [s1,  c1, 0, 0],
                [0,   0,  1, self.L1_z],
                [0,   0,  0, 1]
            ])
        
        # 2. Waist to Shoulder: Y-rotation and offsets
        c2, s2 = np.cos(q2), np.sin(q2)
        if is_batch:
            R_y2 = np.zeros((N, 4, 4))
            R_y2[:, 0, 0] = c2; R_y2[:, 0, 2] = -s2
            R_y2[:, 1, 1] = 1.0
            R_y2[:, 2, 0] = s2; R_y2[:, 2, 2] = c2
            R_y2[:, 3, 3] = 1.0
            
            offset_2 = np.tile(np.eye(4), (N, 1, 1))
            offset_2[:, 0, 3] = self.L2_x
            offset_2[:, 2, 3] = self.L2_z
            T12 = R_y2 @ offset_2
        else:
            T12 = np.array([
                [c2,  0, -s2, 0],
                [0,   1, 0,   0],
                [s2,  0, c2,  0],
                [0,   0, 0,   1]
            ]) @ np.array([
                [1, 0, 0, self.L2_x],
                [0, 1, 0, 0],
                [0, 0, 1, self.L2_z],
                [0, 0, 0, 1]
            ])
        
        # 3. Shoulder to Elbow: Y-rotation and offsets
        c3, s3 = np.cos(q3), np.sin(q3)
        if is_batch:
            R_y3 = np.zeros((N, 4, 4))
            R_y3[:, 0, 0] = c3; R_y3[:, 0, 2] = -s3
            R_y3[:, 1, 1] = 1.0
            R_y3[:, 2, 0] = s3; R_y3[:, 2, 2] = c3
            R_y3[:, 3, 3] = 1.0
            
            offset_3 = np.tile(np.eye(4), (N, 1, 1))
            offset_3[:, 0, 3] = self.L3_x
            T23 = R_y3 @ offset_3
        else:
            T23 = np.array([
                [c3,  0, -s3, 0],
                [0,   1, 0,   0],
                [s3,  0, c3,  0],
                [0,   0, 0,   1]
            ]) @ np.array([
                [1, 0, 0, self.L3_x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # 4. Elbow to Forearm Roll: X-rotation and offsets
        c4, s4 = np.cos(q4), np.sin(q4)
        if is_batch:
            R_x4 = np.zeros((N, 4, 4))
            R_x4[:, 0, 0] = 1.0
            R_x4[:, 1, 1] = c4; R_x4[:, 1, 2] = -s4
            R_x4[:, 2, 1] = s4; R_x4[:, 2, 2] = c4
            R_x4[:, 3, 3] = 1.0
            
            offset_4 = np.tile(np.eye(4), (N, 1, 1))
            offset_4[:, 0, 3] = self.L4_x
            T34 = R_x4 @ offset_4
        else:
            T34 = np.array([
                [1, 0,   0,   0],
                [0, c4, -s4,  0],
                [0, s4,  c4,  0],
                [0, 0,   0,   1]
            ]) @ np.array([
                [1, 0, 0, self.L4_x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # 5. Forearm Roll to Wrist Pitch: Y-rotation and offsets
        c5, s5 = np.cos(q5), np.sin(q5)
        if is_batch:
            R_y5 = np.zeros((N, 4, 4))
            R_y5[:, 0, 0] = c5; R_y5[:, 0, 2] = -s5
            R_y5[:, 1, 1] = 1.0
            R_y5[:, 2, 0] = s5; R_y5[:, 2, 2] = c5
            R_y5[:, 3, 3] = 1.0
            
            offset_5 = np.tile(np.eye(4), (N, 1, 1))
            offset_5[:, 0, 3] = self.L5_x
            T45 = R_y5 @ offset_5
        else:
            T45 = np.array([
                [c5,  0, -s5, 0],
                [0,   1, 0,   0],
                [s5,  0, c5,  0],
                [0,   0, 0,   1]
            ]) @ np.array([
                [1, 0, 0, self.L5_x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # 6. Wrist Pitch to Wrist Roll: X-rotation and TCP offset
        c6, s6 = np.cos(q6), np.sin(q6)
        if is_batch:
            R_x6 = np.zeros((N, 4, 4))
            R_x6[:, 0, 0] = 1.0
            R_x6[:, 1, 1] = c6; R_x6[:, 1, 2] = -s6
            R_x6[:, 2, 1] = s6; R_x6[:, 2, 2] = c6
            R_x6[:, 3, 3] = 1.0
            
            offset_6 = np.tile(np.eye(4), (N, 1, 1))
            offset_6[:, 0, 3] = self.L6_x
            T56 = R_x6 @ offset_6
        else:
            T56 = np.array([
                [1, 0,   0,  0],
                [0, c6, -s6, 0],
                [0, s6,  c6, 0],
                [0, 0,   0,  1]
            ]) @ np.array([
                [1, 0, 0, self.L6_x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # Chain product of homogenous transformations
        T = T01 @ T12 @ T23 @ T34 @ T45 @ T56
        
        if is_batch:
            pos = T[:, :3, 3]
            R = T[:, :3, :3]
        else:
            pos = T[:3, 3]
            R = T[:3, :3]
            
        return pos, R

    def solve_bimanual_fk(self, joints_14d):
        """
        Solves FK for bimanual 14-D joint array.
        Supports shape (14,) and batch shape (N, 14).
        Also accepts 16-D inputs (base velocity prepended); the first 2
        columns/elements are stripped automatically.
        """
        joints_14d = np.asarray(joints_14d)
        if joints_14d.ndim == 2:
            d = joints_14d.shape[1]
            if d == 16:
                joints_14d = joints_14d[:, 2:16]
            elif d != 14:
                raise ValueError(f'Expected 14 or 16 joint dims, got {d}')
            left_pos, left_R = self.solve_arm_fk(joints_14d[:, :7])
            right_pos, right_R = self.solve_arm_fk(joints_14d[:, 7:14])
        else:
            d = len(joints_14d)
            if d == 16:
                joints_14d = joints_14d[2:16]
            elif d != 14:
                raise ValueError(f'Expected 14 or 16 joint dims, got {d}')
            left_pos, left_R = self.solve_arm_fk(joints_14d[:7])
            right_pos, right_R = self.solve_arm_fk(joints_14d[7:14])
        return (left_pos, left_R), (right_pos, right_R)

    def validate_zero_pose(self):
        """
        Validates FK output at q=[0,0,0,0,0,0,0] against known URDF
        zero-pose geometry.  Raises ValueError on mismatch.
        Returns True if all checks pass.
        """
        q_zero = np.zeros(7)
        pos, _ = self.solve_arm_fk(q_zero)

        expected_x = self.L2_x + self.L3_x + self.L4_x + self.L5_x + self.L6_x  # 0.615
        expected_z = self.L1_z + self.L2_z  # 0.4385
        expected_y = 0.0
        tol = 1e-6

        errors = []
        if abs(pos[0] - expected_x) > tol:
            errors.append(f'TCP x: expected {expected_x}, got {pos[0]}')
        if abs(pos[1] - expected_y) > tol:
            errors.append(f'TCP y: expected {expected_y}, got {pos[1]}')
        if abs(pos[2] - expected_z) > tol:
            errors.append(f'TCP z: expected {expected_z}, got {pos[2]}')

        if errors:
            raise ValueError(
                'Zero-pose FK validation failed:\n' + '\n'.join(errors)
            )
        return True


# Self-test: validate FK matches URDF zero-pose on import
_fk_validator = BimanualForwardKinematics()
_fk_validator.validate_zero_pose()
