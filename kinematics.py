import numpy as np

class BimanualForwardKinematics:
    """
    Solves analytical Forward Kinematics for a bimanual ALOHA setup.
    Models the ViperX 300 Follower Arm (750mm nominal reach, 800mm link sum)
    to calculate Cartesian Tool Center Point (TCP) coordinates and 
    orientation rotation matrices in the robot's physical workspace.
    Supports both single-frame (1D) and vectorized batch (2D) inputs for high-performance edge compute.
    """
    def __init__(self):
        # ViperX 300 Link Lengths (meters)
        self.L1 = 0.1385  # Base to shoulder (Z-axis offset)
        self.L2 = 0.300   # Shoulder to elbow (X-axis offset in home pose)
        self.L3 = 0.300   # Elbow to forearm roll (X-axis offset)
        self.L4_a = 0.100 # Forearm roll to wrist pitch (X-axis offset)
        self.L4_b = 0.100 # Wrist pitch to gripper TCP (X-axis offset)

    def solve_arm_fk(self, joints):
        """
        Solves analytical FK for single frame (1D array of shape (7,)) 
        or vectorized batch of frames (2D array of shape (N, 7)).
        Expected joints order: 
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

        # 1. Base to Shoulder: Z-axis translation and Z-rotation
        c1, s1 = np.cos(q1), np.sin(q1)
        if is_batch:
            T01 = np.zeros((N, 4, 4))
            T01[:, 0, 0] = c1; T01[:, 0, 1] = -s1
            T01[:, 1, 0] = s1; T01[:, 1, 1] = c1
            T01[:, 2, 2] = 1.0; T01[:, 2, 3] = self.L1
            T01[:, 3, 3] = 1.0
        else:
            T01 = np.array([
                [c1, -s1, 0, 0],
                [s1,  c1, 0, 0],
                [0,   0,  1, self.L1],
                [0,   0,  0, 1]
            ])
        
        # 2. Shoulder to Elbow: Y-rotation and X-translation
        c2, s2 = np.cos(q2), np.sin(q2)
        if is_batch:
            R_y2 = np.zeros((N, 4, 4))
            R_y2[:, 0, 0] = c2; R_y2[:, 0, 2] = -s2
            R_y2[:, 1, 1] = 1.0
            R_y2[:, 2, 0] = s2; R_y2[:, 2, 2] = c2
            R_y2[:, 3, 3] = 1.0
            
            offset_2 = np.tile(np.eye(4), (N, 1, 1))
            offset_2[:, 0, 3] = self.L2
            T12 = R_y2 @ offset_2
        else:
            T12 = np.array([
                [c2,  0, -s2, 0],
                [0,   1, 0,   0],
                [s2,  0, c2,  0],
                [0,   0, 0,   1]
            ]) @ np.array([
                [1, 0, 0, self.L2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # 3. Elbow to Forearm Roll: Y-rotation and X-translation
        c3, s3 = np.cos(q3), np.sin(q3)
        if is_batch:
            R_y3 = np.zeros((N, 4, 4))
            R_y3[:, 0, 0] = c3; R_y3[:, 0, 2] = -s3
            R_y3[:, 1, 1] = 1.0
            R_y3[:, 2, 0] = s3; R_y3[:, 2, 2] = c3
            R_y3[:, 3, 3] = 1.0
            
            offset_3 = np.tile(np.eye(4), (N, 1, 1))
            offset_3[:, 0, 3] = self.L3
            T23 = R_y3 @ offset_3
        else:
            T23 = np.array([
                [c3,  0, -s3, 0],
                [0,   1, 0,   0],
                [s3,  0, c3,  0],
                [0,   0, 0,   1]
            ]) @ np.array([
                [1, 0, 0, self.L3],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # 4. Forearm Roll to Wrist Pitch: X-rotation and translation L4_a
        c4, s4 = np.cos(q4), np.sin(q4)
        if is_batch:
            R_x4 = np.zeros((N, 4, 4))
            R_x4[:, 0, 0] = 1.0
            R_x4[:, 1, 1] = c4; R_x4[:, 1, 2] = -s4
            R_x4[:, 2, 1] = s4; R_x4[:, 2, 2] = c4
            R_x4[:, 3, 3] = 1.0
            
            offset_4 = np.tile(np.eye(4), (N, 1, 1))
            offset_4[:, 0, 3] = self.L4_a
            T34 = R_x4 @ offset_4
        else:
            T34 = np.array([
                [1, 0,   0,   0],
                [0, c4, -s4,  0],
                [0, s4,  c4,  0],
                [0, 0,   0,   1]
            ]) @ np.array([
                [1, 0, 0, self.L4_a],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # 5. Wrist Pitch to Wrist Roll: Y-rotation and translation L4_b
        c5, s5 = np.cos(q5), np.sin(q5)
        if is_batch:
            R_y5 = np.zeros((N, 4, 4))
            R_y5[:, 0, 0] = c5; R_y5[:, 0, 2] = -s5
            R_y5[:, 1, 1] = 1.0
            R_y5[:, 2, 0] = s5; R_y5[:, 2, 2] = c5
            R_y5[:, 3, 3] = 1.0
            
            offset_5 = np.tile(np.eye(4), (N, 1, 1))
            offset_5[:, 0, 3] = self.L4_b
            T45 = R_y5 @ offset_5
        else:
            T45 = np.array([
                [c5,  0, -s5, 0],
                [0,   1, 0,   0],
                [s5,  0, c5,  0],
                [0,   0, 0,   1]
            ]) @ np.array([
                [1, 0, 0, self.L4_b],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        
        # 6. Wrist Roll to end effector TCP: X-rotation
        c6, s6 = np.cos(q6), np.sin(q6)
        if is_batch:
            T56 = np.zeros((N, 4, 4))
            T56[:, 0, 0] = 1.0
            T56[:, 1, 1] = c6; T56[:, 1, 2] = -s6
            T56[:, 2, 1] = s6; T56[:, 2, 2] = c6
            T56[:, 3, 3] = 1.0
        else:
            T56 = np.array([
                [1, 0,   0,  0],
                [0, c6, -s6, 0],
                [0, s6,  c6, 0],
                [0, 0,   0,  1]
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
        """
        joints_14d = np.asarray(joints_14d)
        if joints_14d.ndim == 2:
            left_pos, left_R = self.solve_arm_fk(joints_14d[:, :7])
            right_pos, right_R = self.solve_arm_fk(joints_14d[:, 7:14])
        else:
            left_pos, left_R = self.solve_arm_fk(joints_14d[:7])
            right_pos, right_R = self.solve_arm_fk(joints_14d[7:14])
        return (left_pos, left_R), (right_pos, right_R)
