import numpy as np

class BimanualForwardKinematics:
    """
    Solves analytical Forward Kinematics for a bimanual ALOHA setup.
    Models the ViperX 300 Follower Arm (750mm nominal reach, 800mm link sum)
    to calculate Cartesian Tool Center Point (TCP) coordinates and 
    orientation rotation matrices in the robot's physical workspace.
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
        Solves analytical FK for a single 7-D arm joint array.
        Expected joints order: 
            0: waist (Z-rotation)
            1: shoulder (Y-pitch)
            2: elbow (Y-pitch)
            3: forearm_roll (X-roll)
            4: wrist_pitch (Y-pitch)
            5: wrist_roll (X-roll)
            6: gripper (ignored for arm kinematics)
        
        Returns:
            position: [x, y, z] of the TCP in meters.
            R: 3x3 orientation rotation matrix.
        """
        if len(joints) < 6:
            return np.array([0.0, 0.0, 0.0]), np.eye(3)
            
        q1 = joints[0]  # waist Z
        q2 = joints[1]  # shoulder Y (pitch)
        q3 = joints[2]  # elbow Y (pitch)
        q4 = joints[3]  # forearm roll X (roll)
        q5 = joints[4]  # wrist pitch Y (pitch)
        q6 = joints[5]  # wrist roll X (roll)
        
        # 1. Base to Shoulder: Z-axis translation and Z-rotation
        c1, s1 = np.cos(q1), np.sin(q1)
        T01 = np.array([
            [c1, -s1, 0, 0],
            [s1,  c1, 0, 0],
            [0,   0,  1, self.L1],
            [0,   0,  0, 1]
        ])
        
        # 2. Shoulder to Elbow: Y-rotation and X-translation
        c2, s2 = np.cos(q2), np.sin(q2)
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
        T56 = np.array([
            [1, 0,   0,  0],
            [0, c6, -s6, 0],
            [0, s6,  c6, 0],
            [0, 0,   0,  1]
        ])
        
        # Chain product of homogenous transformations
        T = T01 @ T12 @ T23 @ T34 @ T45 @ T56
        
        pos = T[:3, 3]
        R = T[:3, :3]
        return pos, R

    def solve_bimanual_fk(self, joints_14d):
        """
        Solves FK for bimanual 14-D joint array.
        """
        left_pos, left_R = self.solve_arm_fk(joints_14d[:7])
        right_pos, right_R = self.solve_arm_fk(joints_14d[7:14])
        return (left_pos, left_R), (right_pos, right_R)
