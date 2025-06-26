import sympy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

# Define symbolic joint variables
np.set_printoptions(precision=10, suppress=True)
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
Ai = []
for params in DH_params:
    T = T * Te(*params)
    Ai.append(T)



# Extract symbolic end-effector position
pos_sym = T[:3, 3]
# Lambdify for position and full transform
_fk_pos = sp.lambdify((t1, t2, t3, t4, t5, t6), pos_sym, 'numpy')
_fk_T   = sp.lambdify((t1, t2, t3, t4, t5, t6), T, 'numpy')
_fk_1   = sp.lambdify((t1), Ai[0], 'numpy')
_fk_2   = sp.lambdify((t1,t2), Ai[1], 'numpy')
_fk_3   = sp.lambdify((t1,t2,t3), Ai[2], 'numpy')
_fk_4   = sp.lambdify((t1,t2,t3,t4), Ai[3], 'numpy')
_fk_5   = sp.lambdify((t1,t2,t3,t4,t5), Ai[4], 'numpy')
_fk_6   = sp.lambdify((t1,t2,t3,t4,t5,t6), Ai[5], 'numpy')


def forward_kinematics(angles_deg):
    """
    Compute UR5 forward kinematics: returns a 7-element vector [x, y, z, qx, qy, qz, qw].

    Parameters
    ----------
    angles_deg : array-like of length 6
        Joint angles [t1..t6] in degrees.

    Returns
    -------
    numpy.ndarray
        Concatenated position and quaternion [x, y, z, qx, qy, qz, qw].
    """
    # Convert to radians
    angles_rad = np.deg2rad(angles_deg)
    # Position: flatten to 1D
    pos = np.array(_fk_pos(*angles_rad), dtype=float).flatten()
    # Full transform
    T_full = np.array(_fk_T(*angles_rad), dtype=float)
    print(T_full)
    R_mat = T_full[:3, :3]
    rot = R.from_matrix(R_mat) 
    qx, qy, qz, qw = rot.as_quat()
    # optional: force qw ≥ 0
    quat_wxyz = np.array([qw, qx, qy, qz])

    # optionally, enforce qw ≥ 0 for a unique hemisphere:
    if quat_wxyz[0] < 0:
        quat_wxyz = -quat_wxyz
    pos = np.array(_fk_pos(*np.deg2rad(angles_deg)), dtype=float).flatten()
    return np.concatenate((pos, quat_wxyz))

def joint_pos(angles_deg,joint_id):
    angles_rad = np.deg2rad(angles_deg)
    if joint_id==1:
        T = np.array(_fk_1(*angles_rad[:1]), dtype=float)
    elif joint_id==2:
        T = np.array(_fk_2(*angles_rad[:2]), dtype=float)
    elif joint_id==3:
        T = np.array(_fk_3(*angles_rad[:3]), dtype=float)
    elif joint_id==4:
        T = np.array(_fk_4(*angles_rad[:4]), dtype=float)
    elif joint_id==5:
        T = np.array(_fk_5(*angles_rad[:5]), dtype=float)
    elif joint_id==6:
        T = np.array(_fk_6(*angles_rad[:6]), dtype=float)
    else:
        raise ValueError(f"Invalid joint_id {joint_id}")
    
    if T.ndim == 1:
        T = T.reshape(4, 4)

    pos   = T[:3, 3]
    R_mat = T[:3, :3]
    rot   = R.from_matrix(R_mat)
    qx, qy, qz, qw = rot.as_quat()
    quat_wxyz = np.array([qw, qx, qy, qz])
    if quat_wxyz[0] < 0:
        quat_wxyz = -quat_wxyz

    return np.concatenate((pos, quat_wxyz))

# Example usage
if __name__ == "__main__":
    angles = [9.57625294, -130.99350741,  156.5652118,  -115.57170439,  -90.0, 9.57625294]
    print("Input joint angles")
    print(angles)
    result = forward_kinematics(angles)
    print(f"end_effector position: {result}")

