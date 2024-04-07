import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh import Bvh

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    with open(bvh_file_path) as f:
        mocap = Bvh(f.read())
    joint_name = mocap.get_joints_names()
    joint_parent = []
    joint_offset = []
    for jn in joint_name:
        joint_parent.append(mocap.joint_parent_index(jn))
        joint_offset.append(np.array(mocap.joint_offset(jn)))
    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    motion_channels_data = motion_data[frame_id]
    root_position = np.array(motion_channels_data[0:3])
    joint_local_rotation = []
    for i in range(len(joint_name)):
        joint_local_rotation.append(motion_channels_data[3*i+3:3*i+6])
    
    joint_positions = []
    joint_orientations = []
    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            joint_orientation = R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
            joint_position = root_position.reshape(-1,)
        else:
            parent_global_orientation = R.from_quat(joint_orientations[joint_parent[i]])
            joint_orientation = parent_global_orientation * R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
            parent_global_position = joint_positions[joint_parent[i]].reshape((-1,1))
            joint_position = parent_global_position+parent_global_orientation.as_matrix()@joint_offset[i].reshape(-1,1)
        joint_orientations.append(joint_orientation.as_quat())
        joint_positions.append(joint_position.reshape(-1,))
    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    with open(T_pose_bvh_path) as f:
        mocap_T = Bvh(f.read())
    with open(A_pose_bvh_path) as f:
        mocap_A = Bvh(f.read())

    A_motion_data = load_motion_data(A_pose_bvh_path)[:1000,:]
    motion_data = []

    T_joint_names = mocap_T.get_joints_names()

    # 矩阵并行加速（改进）
    for i in range(np.shape(A_motion_data)[0]):
        data_in_line = []
        for name_T in T_joint_names:
            idx_A = mocap_A.get_joint_index(name=name_T)
            if name_T == "RootJoint":
                data_in_line += list(A_motion_data[i][0:6])
            elif name_T == 'lShoulder':
                Rot = (R.from_euler('XYZ', list(A_motion_data[i][idx_A * 3 + 3: idx_A*3 + 6]), degrees=True).as_matrix())@(
                       R.from_euler('XYZ', [0., 0., -45.], degrees=True).as_matrix())
                data_in_line += list(R.from_matrix(Rot).as_euler('XYZ',degrees=True))
            elif name_T == 'rShoulder':
                Rot = (R.from_euler('XYZ', list(A_motion_data[i][idx_A * 3 + 3: idx_A*3 + 6]), degrees=True).as_matrix())@( 
                       R.from_euler('XYZ', [0., 0., 45.], degrees=True).as_matrix())
                data_in_line += list(R.from_matrix(Rot).as_euler('XYZ',True))
            else:
                data_in_line += list(A_motion_data[i][idx_A*3+3:idx_A*3+6])
        motion_data.append(np.array(data_in_line).reshape(-1,))
    motion_data = np.array(motion_data)
    return motion_data
