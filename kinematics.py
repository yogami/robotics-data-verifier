import numpy as np

class BimanualForwardKinematics:
    """
    Solves analytical Forward Kinematics for a bimanual ALOHA setup.
    Models the WidowX 250 6-DOF robot arm.
    """
    def __init__(self):
        # WidowX 250 link lengths (meters)
        self.L1 = 0.1195  # Base to shoulder
        self.L2 = 0.250   # Shoulder to elbow
        self.L3 = 0.250   # Elbow to wrist
        self.L4 = 0.180   # Wrist to Tool Center Point (TCP)

    def solve_arm_fk(self, joints):
        """
        Solves FK for a single 7-D arm joint array.
        Expected joints order: [waist, shoulder, elbow, wrist_pitch, wrist_roll, gripper, ...]
        Returns [x, y, z] position of the tool center point (TCP).
        """
        # Ensure we have at least 4 joints for the planar position
        if len(joints) < 4:
            return np.array([0.0, 0.0, 0.0])
            
        q1 = joints[0] # waist rotation (around Z)
        q2 = joints[1] # shoulder pitch (around Y)
        q3 = joints[2] # elbow pitch (around Y)
        q4 = joints[3] # wrist pitch (around Y)
        
        # Calculate radial distance in the XY plane
        r = (self.L2 * np.cos(q2) + 
             self.L3 * np.cos(q2 + q3) + 
             self.L4 * np.cos(q2 + q3 + q4))
             
        # Calculate height along the Z axis
        z = (self.L1 + 
             self.L2 * np.sin(q2) + 
             self.L3 * np.sin(q2 + q3) + 
             self.L4 * np.sin(q2 + q3 + q4))
             
        # Project radial distance to X and Y axes
        x = r * np.cos(q1)
        y = r * np.sin(q1)
        
        return np.array([x, y, z])

    def solve_bimanual_fk(self, joints_14d):
        """
        Solves FK for bimanual 14-D joint array.
        Returns:
            left_tcp: [x, y, z]
            right_tcp: [x, y, z]
        """
        # Left arm is first 7 joints, Right arm is second 7 joints
        left_joints = joints_14d[:7]
        right_joints = joints_14d[7:14]
        
        left_tcp = self.solve_arm_fk(left_joints)
        right_tcp = self.solve_arm_fk(right_joints)
        
        return left_tcp, right_tcp
