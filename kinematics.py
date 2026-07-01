import numpy as np

class BimanualForwardKinematics:
    """
    Solves analytical Forward Kinematics for a bimanual ALOHA setup.
    Models the ViperX 300 Follower Arm (750mm reach) to correctly calculate
    Cartesian end-effector coordinates in the robot's physical workspace.
    """
    def __init__(self):
        # ViperX 300 Link Lengths (meters)
        self.L1 = 0.1385  # Base to shoulder
        self.L2 = 0.300   # Shoulder to elbow
        self.L3 = 0.300   # Elbow to wrist
        self.L4 = 0.200   # Wrist to Tool Center Point (TCP)

    def solve_arm_fk(self, joints):
        """
        Solves FK for a single 7-D arm joint array.
        Expected joints order: [waist, shoulder, elbow, wrist_pitch, wrist_roll, gripper, ...]
        Returns:
            position: [x, y, z] of the gripper in meters.
            R: 3x3 rotation matrix representing orientation.
        """
        if len(joints) < 5:
            return np.array([0.0, 0.0, 0.0]), np.eye(3)
            
        q1 = joints[0] # waist rotation (around Z)
        q2 = joints[1] # shoulder pitch (around Y)
        q3 = joints[2] # elbow pitch (around Y)
        q4 = joints[3] # wrist pitch (around Y)
        q5 = joints[4] # wrist roll (around X)
        
        # 1. Calculate X, Y, Z Position (planar projection)
        r = (self.L2 * np.cos(q2) + 
             self.L3 * np.cos(q2 + q3) + 
             self.L4 * np.cos(q2 + q3 + q4))
             
        z = (self.L1 + 
             self.L2 * np.sin(q2) + 
             self.L3 * np.sin(q2 + q3) + 
             self.L4 * np.sin(q2 + q3 + q4))
             
        x = r * np.cos(q1)
        y = r * np.sin(q1)
        pos = np.array([x, y, z])
        
        # 2. Calculate 3D Rotation Matrix R = R_z(q1) @ R_y(q2 + q3 + q4) @ R_x(q5)
        # R_z(waist)
        c1, s1 = np.cos(q1), np.sin(q1)
        R_z = np.array([
            [c1, -s1, 0],
            [s1,  c1, 0],
            [0,   0,  1]
        ])
        
        # R_y(shoulder + elbow + wrist_pitch)
        q234 = q2 + q3 + q4
        c234, s234 = np.cos(q234), np.sin(q234)
        R_y = np.array([
            [c234,  0, s234],
            [0,     1, 0   ],
            [-s234, 0, c234]
        ])
        
        # R_x(wrist_roll)
        c5, s5 = np.cos(q5), np.sin(q5)
        R_x = np.array([
            [1, 0,   0  ],
            [0, c5, -s5 ],
            [0, s5,  c5 ]
        ])
        
        # Combine transformations
        R = R_z @ R_y @ R_x
        
        return pos, R

    def solve_bimanual_fk(self, joints_14d):
        """
        Solves FK for bimanual 14-D joint array.
        Returns:
            left_pos, left_R
            right_pos, right_R
        """
        left_joints = joints_14d[:7]
        right_joints = joints_14d[7:14]
        
        left_pos, left_R = self.solve_arm_fk(left_joints)
        right_pos, right_R = self.solve_arm_fk(right_joints)
        
        return (left_pos, left_R), (right_pos, right_R)
