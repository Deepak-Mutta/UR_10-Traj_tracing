import sympy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=8, suppress=True)
# Define symbolic joint variables
t1, t2, t3, t4, t5, t6 = sp.symbols('t1 t2 t3 t4 t5 t6', real=True)

# Denavit-Hartenberg parameters for UR5: [theta_offset, d, a, alpha]
DH_params = [
    (t1 + sp.pi, 0.1273,     0.0,     sp.pi/2),
    (t2,         0.0,       -0.612,   0.0     ),
    (t3,         0.0,       -0.5723,  0.0     ),
    (t4,         0.163941,   0.0,     sp.pi/2),
    (t5,         0.1157,     0.0,    -sp.pi/2),
    (t6,         0.0922,     0.0,     0.0     )
]

d1 = DH_params[0][1]
d4 = DH_params[3][1]    
d5 = DH_params[4][1]
d6 = DH_params[5][1]
a2 = DH_params[1][2]
a3 = DH_params[2][2]

# Symbolic elementary transformation matrix function
def Te(theta, d, a, alpha):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d            ],
        [0,              0,                            0,                           1            ]
    ])

# Compute full symbolic transformation
T = sp.eye(4)
T_individual=[]
for params in DH_params:
    ind_matrix = Te(*params)
    T = T * ind_matrix
    T_individual.append(ind_matrix)

def inverse_kinematics(target_pos, target_quat):
    """
    Compute UR5 inverse kinematics: returns joint angles [t1..t6] in degrees.

    Parameters
    ----------
    target_pos : array-like of length 3
        Desired end-effector position [x, y, z].
    target_quat : array-like of length 4
        Desired end-effector orientation as quaternion [qx, qy, qz, qw].

    Returns
    -------
    numpy.ndarray
        Joint angles [t1..t6] in degrees.
    """
    # Convert quaternion to rotation matrix
    w, x, y, z = target_quat
    rot = R.from_quat([x, y, z, w])
    R_mat = rot.as_matrix()
    h06 = np.eye(4)
    h06[:3, :3] = R_mat
    h06[:3,  3] = target_pos
    # print(h06)

    # Compute wrist center position
    wrist_center = np.array(target_pos) - d6 * R_mat[:, 2]

    # Placeholder for IK solution (to be implemented)
    angles_rad = np.zeros(6)  

    # Solving for Theta1
    theta1 = np.arctan2(wrist_center[1], wrist_center[0]) + np.arccos(d4/np.linalg.norm(wrist_center[:2])) - np.pi/2
    angles_rad[0] = theta1

    # Solving for theta5 using theta 1
    T1 = T_individual[0].subs(t1, theta1)
    T16 = T1.inv() *  h06
    t16z = T16[2, 3]
    theta5 = -np.arccos(float((t16z - d4)/d6))
    angles_rad[4] = theta5
    
    # Solving for theta6 using theta5
    T61 = T16.inv()
    zx =  float(T61[0, 2])
    zy =  float(T61[1, 2])
    theta6 = np.arctan2(-zy/np.sin(theta5), zx/np.sin(theta5))
    angles_rad[5] = theta6

    # Solving for theta 3 using theta1, theta5, theta6
    T1 = T_individual[0].subs(t1, theta1)
    T16 = T1.inv() *  h06
    T15 = T16 * (T_individual[5].subs(t6, theta6).inv())
    T14 = T15 * (T_individual[4].subs(t5, theta5).inv())

    p13 = T14 * sp.Matrix([0, -d4, 0, 1]) - sp.Matrix([0, 0, 0, 1])

    p13 = np.array(p13).astype(np.float64)

    distance = np.linalg.norm(p13[:3])
    theta3 = np.arccos((distance**2 - a2**2 - a3**2) / (2 * a2 * a3))
    angles_rad[2] = theta3

    # Solving for theta2
    theta2 = -np.arctan2(p13[1,0],-p13[0,0]) + np.arcsin(a3 * np.sin(theta3) / distance)
    angles_rad[1]= theta2

    # solving for theta4
    T14 = T15 * (T_individual[4].subs(t5, theta5).inv())

    T2_inv = (T_individual[1].subs(t2, theta2)).inv()
    T3_inv = (T_individual[2].subs(t3, theta3)).inv()

    t34 = T3_inv * T2_inv * T14
    xx = float(t34[0,0])
    yx = float(t34[1,0])
    theta4 = np.arctan2(yx, xx)
    angles_rad[3] = theta4

    # Convert to degrees
    # angles_deg = np.rad2deg(angles_rad)
    
    return angles_rad

if __name__ == "__main__":
    # Example usage
    # target_pos = [0.780039 , 0.642892 , 0.293993]
    # target_quat = [0.31538, -0.213682, -0.698451, -0.60584]
    
    target_pos = [0.2,  0.2, 0.25]
    target_quat = [0, -0.7071, 0.7071, 0.0]
    angles = np.rad2deg(inverse_kinematics(target_pos, target_quat))
    print("Computed joint angles:", angles)