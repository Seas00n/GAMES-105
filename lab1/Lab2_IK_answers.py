import numpy as np
from scipy.spatial.transform import Rotation as R
import pytorch_kinematics as pk
from pytorch_kinematics import frame, chain
from treelib import Tree, Node
import torch


class MetaData:
    def __init__(self, joint_name, joint_parent, joint_initial_position, root_joint, end_joint):
        """
        一些固定信息，其中joint_initial_position是T-pose下的关节位置，可以用于计算关节相互的offset
        root_joint是固定节点的索引，并不是RootJoint节点
        """
        self.joint_name = joint_name
        self.joint_parent = joint_parent
        self.joint_initial_position = joint_initial_position
        self.root_joint = root_joint
        self.end_joint = end_joint
        self.create_tree()
        self.path_ik,_,_,_ = self.get_path_from_root_to_end()
        
    def create_tree(self):
        """
        添加一个树简化程序
        """
        self.tree = Tree()
        path,path_name,_,_ = self.get_path_from_root_to_end()
        for i in range(len(self.joint_name)):
            name = self.joint_name[i]
            identifier = i
            if self.joint_parent[i] == -1:
                self.tree.create_node(name, identifier)
            else:
                self.tree.create_node(name, identifier, 
                                      parent=self.joint_parent[i])
        

        if path_name[0] != self.joint_name[0] and path.__contains__(0):
            new_tree = None
            for i in range(len(path)-1):
                if path[i] == 0:
                    break
                else:
                    sub_tree_new_root = self.tree.subtree(path[i])
                    # print(sub_tree_new_root.show(stdout=False))
                    sub_tree_parent_rest = self.tree.subtree(path[i+1])
                    # print(sub_tree_parent_rest.show(stdout=False))
                    # for node in sub_tree_parent_rest.all_nodes_itr():
                    #     print(node.identifier, node.tag)
                    sub_tree_parent_rest.remove_node(path[i])
                    if new_tree is None:
                        sub_tree_new_root.paste(path[i], sub_tree_parent_rest)
                        new_tree = sub_tree_new_root
                    else:
                        new_tree.paste(path[i], sub_tree_parent_rest)
                    self.tree.remove_node(path[i])
            self.tree = new_tree
        elif not path.__contains__(0):
            self.tree = self.tree.subtree(path[0])

        print(self.tree.show(stdout=False))

    def get_path_from_root_to_end(self):
        """
        辅助函数，返回从root节点到end节点的路径
        
        输出：
            path: 各个关节的索引
            path_name: 各个关节的名字
        Note: 
            如果root_joint在脚，而end_joint在手，那么此路径会路过RootJoint节点。
            在这个例子下path2返回从脚到根节点的路径，path1返回从根节点到手的路径。
            你可能会需要这两个输出。
        """
        
        # 从end节点开始，一直往上找，直到找到腰部节点
        path1 = [self.joint_name.index(self.end_joint)]
        while self.joint_parent[path1[-1]] != -1:
            path1.append(self.joint_parent[path1[-1]])
        

        # 从root节点开始，一直往上找，直到找到腰部节点
        path2 = [self.joint_name.index(self.root_joint)]
        while self.joint_parent[path2[-1]] != -1:
            path2.append(self.joint_parent[path2[-1]])
        

        # 合并路径，消去重复的节点
        while path1 and path2 and path2[-1] == path1[-1]:
            path1.pop()
            a = path2.pop()
            
        path2.append(a)
        path = path2 + list(reversed(path1))
        path_name = [self.joint_name[i] for i in path]
        print("From Start to End:")
        for i in range(len(path)-1):
            print(path_name[i]+":{}->".format(path[i]),end='')
        print(path_name[-1]+":{}".format(path[-1]))
        return path, path_name, path1, path2

def part1_inverse_kinematics(meta_data:MetaData, joint_positions, joint_orientations,
                             target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path_ik = meta_data.path_ik
    joint_orientations = R.from_quat(joint_orientations).as_matrix()
    
    num_iter = 100

    strench_range = np.linalg.norm(
        (meta_data.joint_initial_position[path_ik[1:]]-meta_data.joint_initial_position[path_ik[0:-1]]),
        axis=1).sum()
    
    def rodrigus_formula(vec1, vec2):
        """
        旋转向量转旋转矩阵
        """
        if np.allclose(vec1, vec2, atol=1e-3, rtol=1e-3):
            return np.eye(3), vec1
        else:
            axis = np.cross(vec1, vec2)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(
                np.dot(vec1, vec2) /
                (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return R.from_rotvec(axis * angle).as_matrix(), axis

    def ccd(joint_positions, joint_orientations, out_of_strench):
        root_inverse = False
        for i in range(len(path_ik)-1, 0, -1):
            curr_index = path_ik[i]
            parent_index = path_ik[i-1]
            if parent_index == 0:
                root_inverse = True
            vec1 = joint_positions[path_ik[-1]]-joint_positions[parent_index]
            vec1 = vec1/np.linalg.norm(vec1)
            vec2 = target_pose-joint_positions[parent_index]
            vec2 = vec2/np.linalg.norm(vec2)
            rot, axis = rodrigus_formula(vec1, vec2)
            # if meta_data.joint_name[parent_index] == "lKnee" or meta_data.joint_name[parent_index]=="rKnee":
            #     rot_back,_ = rodrigus_formula(np.array([0,0,-1]), axis)
            #     rot = rot_back@rot 
            all_child_joint_node = meta_data.tree.subtree(parent_index).all_nodes()
            all_child_joint = [n.identifier for n in all_child_joint_node]
            all_joint_name = [meta_data.joint_name[i] for i in all_child_joint]
            if root_inverse:
                all_rot_joint = all_child_joint[1:]
            else:
                all_rot_joint = all_child_joint
            joint_orientations[all_rot_joint] = rot@joint_orientations[all_rot_joint]
            position_update = rot@(joint_positions[all_child_joint,:]-joint_positions[parent_index,:]).T
            joint_positions[all_child_joint] = position_update.T[:]+joint_positions[parent_index,:]
            distance_to_target = np.linalg.norm(joint_positions[path_ik[-1]]-target_pose)
        
        return joint_positions, joint_orientations

    iter_ = 0
    for _ in range(num_iter):
        distance = np.linalg.norm(target_pose-joint_positions[path_ik[0]])
        out_of_strench = distance > strench_range
        threshold = 0.01 if not out_of_strench else distance-strench_range+0.01*distance/strench_range
        
        if not np.linalg.norm(joint_positions[path_ik[-1]]-target_pose)<threshold:
            joint_positions, joint_orientations = ccd(joint_positions, joint_orientations,out_of_strench)
            iter_ += 1
        else:
            break
    print("Iteration Time:", iter_)
    joint_orientations = R.from_matrix(joint_orientations).as_quat()
    return joint_positions, joint_orientations



def part2_inverse_kinematics(meta_data:MetaData, joint_positions, joint_orientations,
                             relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path_ik = meta_data.path_ik
    joint_orientations = R.from_quat(joint_orientations).as_matrix()

    num_iter = 100
    idx_root = path_ik[0]
    idx_end = path_ik[-1]
    target_pose = joint_positions[idx_root]+np.array([relative_x,0,relative_z]).reshape((3,))
    target_pose[1] = target_height
    
    strench_range = np.linalg.norm(
        (meta_data.joint_initial_position[path_ik[1:]]-meta_data.joint_initial_position[path_ik[0:-1]]),
        axis=1).sum()
    
    def rodrigus_formula(vec1, vec2):
        """
        旋转向量转旋转矩阵
        """
        if np.allclose(vec1, vec2, atol=1e-3, rtol=1e-3):
            return np.eye(3)
        else:
            axis = np.cross(vec1, vec2)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(
                np.dot(vec1, vec2) /
                (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return R.from_rotvec(axis * angle).as_matrix(), axis

    def ccd(joint_positions, joint_orientations, out_of_strench):
        root_inverse = False
        for i in range(len(path_ik)-1, 0, -1):
            curr_index = path_ik[i]
            parent_index = path_ik[i-1]
            if parent_index == 0:
                root_inverse = True
            vec1 = joint_positions[path_ik[-1]]-joint_positions[parent_index]
            vec1 = vec1/np.linalg.norm(vec1)
            vec2 = target_pose-joint_positions[parent_index]
            vec2 = vec2/np.linalg.norm(vec2)
            rot, axis = rodrigus_formula(vec1, vec2)    
            all_child_joint_node = meta_data.tree.subtree(parent_index).all_nodes()
            all_child_joint = [n.identifier for n in all_child_joint_node]
            all_joint_name = [meta_data.joint_name[i] for i in all_child_joint]
            if root_inverse:
                all_rot_joint = all_child_joint[1:]
            else:
                all_rot_joint = all_child_joint
            joint_orientations[all_rot_joint] = rot@joint_orientations[all_rot_joint]
            position_update = rot@(joint_positions[all_child_joint,:]-joint_positions[parent_index,:]).T
            joint_positions[all_child_joint] = position_update.T[:]+joint_positions[parent_index,:]
            distance_to_target = np.linalg.norm(joint_positions[path_ik[-1]]-target_pose)
           
        return joint_positions, joint_orientations


    iter_ = 0
    for _ in range(num_iter):
        distance = np.linalg.norm(target_pose-joint_positions[idx_root])
        out_of_strench = distance > strench_range
        threshold = 0.01 if not out_of_strench else distance-strench_range+0.01*distance/strench_range
        if not np.linalg.norm(joint_positions[idx_end]-target_pose)<threshold:
            joint_positions, joint_orientations = ccd(joint_positions, joint_orientations, out_of_strench)
            iter_ += 1
        else:
            break
    print("Iter Time:", iter_)
    joint_orientations = R.from_matrix(joint_orientations).as_quat()
    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                             left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    path_ik_left = meta_data.path_ik
    path_ik_right = [meta_data.joint_name.index('rWrist_end')]
    while meta_data.joint_parent[path_ik_right[-1]] != 2:
        path_ik_right.append(meta_data.joint_parent[path_ik_right[-1]])
    path_ik_right = list(reversed(path_ik_right))
    #显然 path_ik_right 和path_ik_left 同时循环到Path共有节点
    joint_orientations = R.from_quat(joint_orientations).as_matrix()
    
    num_iter = 100

    strench_range_left = np.linalg.norm(
        (meta_data.joint_initial_position[path_ik_left[1:]]-
         meta_data.joint_initial_position[path_ik_left[0:-1]]),
        axis=1).sum()
    strench_range_right = np.linalg.norm(
        (meta_data.joint_initial_position[path_ik_right[1:]]-
         meta_data.joint_initial_position[path_ik_right[0:-1]]),
        axis=1).sum()

    def rodrigus_formula(vec1, vec2):
        """
        旋转向量转旋转矩阵
        """
        if np.allclose(vec1, vec2, atol=1e-3, rtol=1e-3):
            return np.eye(3)
        else:
            axis = np.cross(vec1, vec2)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(
                np.dot(vec1, vec2) /
                (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return R.from_rotvec(axis * angle).as_matrix()

    def ccd(joint_positions, joint_orientations, out_of_strench=False):
        root_inverse = False
        double_ccd = True
        for i in range(len(path_ik_left)-1, 0, -1):
            curr_index_left = path_ik_left[i]
            parent_index_left = path_ik_left[i-1]
            if parent_index_left == 2:
                double_ccd = False
            if parent_index_left == 0:
                root_inverse = True
            if double_ccd:
                curr_index_right = path_ik_right[-len(path_ik_left)+i]
                parent_index_right = path_ik_right[-len(path_ik_left)+i-1]
                print("----------------------------------")
                print("right decent joint: "+meta_data.joint_name[parent_index_right]+"->"+meta_data.joint_name[curr_index_right])
                print("left decent joint: "+meta_data.joint_name[parent_index_left]+"->"+meta_data.joint_name[curr_index_left])
                vec1_left = joint_positions[path_ik_left[-1]]-joint_positions[parent_index_left]
                vec1_left = vec1_left/np.linalg.norm(vec1_left)
                vec2_left = left_target_pose-joint_positions[parent_index_left]
                vec2_left = vec2_left/np.linalg.norm(vec2_left)
                rot_left = rodrigus_formula(vec1_left, vec2_left)

                vec1_right = joint_positions[path_ik_right[-1]]-joint_positions[parent_index_right]
                vec1_right = vec1_right/np.linalg.norm(vec1_right)
                vec2_right = right_target_pose-joint_positions[parent_index_right]
                vec2_right = vec2_right/np.linalg.norm(vec2_right)
                rot_right = rodrigus_formula(vec1_right, vec2_right)
                all_child_joint_left = path_ik_left[i-1:]
                all_child_joint_left_name = [meta_data.joint_name[n] for n in all_child_joint_left]
                print(all_child_joint_left_name)
                all_child_joint_right = path_ik_right[-len(path_ik_left)+i-1:]
                all_child_joint_right_name = [meta_data.joint_name[n] for n in all_child_joint_right]
                print(all_child_joint_right_name)
                
                joint_orientations[all_child_joint_left] = rot_left@joint_orientations[all_child_joint_left]
                joint_orientations[all_child_joint_right] = rot_right@joint_orientations[all_child_joint_right]
                position_update_left = rot_left@(joint_positions[all_child_joint_left,:]-joint_positions[parent_index_left,:]).T
                joint_positions[all_child_joint_left] = position_update_left.T[:]+joint_positions[parent_index_left,:]
                position_update_right =  rot_right@(joint_positions[all_child_joint_right,:]-joint_positions[parent_index_right,:]).T
                joint_positions[all_child_joint_right] = position_update_right.T[:]+joint_positions[parent_index_right,:]
            else:
                print("----------------------------------")
                print("all decent joint: "+meta_data.joint_name[parent_index_left]+"->"+meta_data.joint_name[curr_index_left])
                vec1 = joint_positions[path_ik_left[-1]]-joint_positions[parent_index_left]
                vec1 += joint_positions[path_ik_right[-1]]-joint_positions[parent_index_right]
                vec1 = vec1/np.linalg.norm(vec1)
                vec2 = left_target_pose-joint_positions[parent_index_left]
                vec2 += right_target_pose-joint_positions[parent_index_right]
                vec2 = vec2/np.linalg.norm(vec2)
                rot = rodrigus_formula(vec1, vec2)
                all_child_joint_node = meta_data.tree.subtree(parent_index_left).all_nodes()
                all_child_joint = [n.identifier for n in all_child_joint_node]
                all_joint_name = [meta_data.joint_name[i] for i in all_child_joint]
                if root_inverse:
                    all_rot_joint = all_child_joint[1:]
                else:
                    all_rot_joint = all_child_joint
                joint_orientations[all_rot_joint] = rot@joint_orientations[all_rot_joint]
                position_update = rot@(joint_positions[all_child_joint,:]-joint_positions[parent_index_left,:]).T
                joint_positions[all_child_joint] = position_update.T[:]+joint_positions[parent_index_left,:]
                continue
        return joint_positions, joint_orientations

    iter_ = 0
    for _ in range(num_iter):
        distance = np.linalg.norm(left_target_pose-joint_positions[path_ik_left[0]])
        out_of_strench_left = distance > strench_range_left
        distance = np.linalg.norm(left_target_pose-joint_positions[path_ik_right[0]])
        out_of_strench_right = distance > strench_range_right
        
        threshold = 0.01
        left_reach = np.linalg.norm(joint_positions[path_ik_left[-1]]-left_target_pose)<threshold
        right_reach = np.linalg.norm(joint_positions[path_ik_right[-1]]-right_target_pose)<threshold
        if (not left_reach) or (not right_reach):
            joint_positions, joint_orientations = ccd(joint_positions, joint_orientations)
            iter_ += 1
        else:
            break
    print("Iteration Time:", iter_)
    joint_orientations = R.from_matrix(joint_orientations).as_quat()
    return joint_positions, joint_orientations


def part1_inverse_kinematics_example(meta_data, joint_positions, joint_orientations,
                             target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    def norm_vec(vec):
        return vec / np.linalg.norm(vec)

    def vec_length(vec):
        return np.linalg.norm(vec)

    def rodrigus_formula(vec1, vec2):
        """
        旋转向量转旋转矩阵
        """
        vec1 = norm_vec(vec1)
        vec2 = norm_vec(vec2)
        if np.allclose(vec1, vec2, atol=1e-3, rtol=1e-3):
            return np.eye(3)
        else:
            axis = np.cross(vec1, vec2)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(
                np.dot(vec1, vec2) /
                (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return R.from_rotvec(axis * angle).as_matrix()

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_orientations = R.from_quat(joint_orientations).as_matrix()

    num_iter = 100
    origin_pos = joint_positions.copy()
    def get_all_child_joint(joint_parents, parent_index):
        all_child_joint = []
        for idx in range(len(joint_orientations)):
            curr_path = [idx]
            while joint_parents[curr_path[-1]] != -1:
                curr_path.append(joint_parents[curr_path[-1]])
            curr_path.remove(idx)
            if parent_index in curr_path:
                all_child_joint.append(idx)
        return all_child_joint

    def cyclic_coordinate_descent_ik(joint_positions, joint_orientations,
                                     joint_parent, path, target_pose, out_of_stretch):
        """
        递归函数，计算逆运动学
        输入:
            joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
            joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
            kinematic_tree: 一个字典，key为关节名字，value为其父节点的名字
        输出:
            经过IK后的姿态
            joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
            joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        """
        kinematic_tree = joint_parent.copy()
        old_path = path.copy()
        path = path.copy()

        for i in range(len(path) - 1):
            kinematic_tree[path[i + 1]] = path[i]
        if 0 in path:
            path.remove(0)

        if 0 in old_path:
            for i, p_id in enumerate(kinematic_tree):
                if p_id == 0:
                    p_id = old_path[old_path.index(0) - 1]
                    kinematic_tree[i] = p_id
        kinematic_tree[path[0]] = -1

        if out_of_stretch:
            start = len(path) - 3
        else:
            start = 0
        for i in range(start, len(path) - 2):
            
            curr_index = path[len(path) - 1 - i]
            parent_index = meta_data.joint_parent[curr_index]
            vec1 = joint_positions[path[-1]] - joint_positions[parent_index]
            vec2 = target_pose - joint_positions[parent_index]
            rot = rodrigus_formula(vec1, vec2)
            all_child_joint = get_all_child_joint(kinematic_tree, parent_index)

            if (kinematic_tree[curr_index] != joint_parent[curr_index]):
                rot_joints = all_child_joint
                # print(curr_index, 'inverse')
            else:
                rot_joints = all_child_joint + [parent_index]
            joint_positions[all_child_joint] = (
                rot[None] @ (joint_positions[all_child_joint][:, :, None] -
                            joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0]
            
            all_joint_name = [meta_data.joint_name[i] for i in rot_joints]
            print(all_joint_name)
            joint_orientations[
                rot_joints] = rot[None] @ joint_orientations[rot_joints]

        return joint_positions, joint_orientations

    def fabrik(joint_positions, joint_orientations, joint_parent, path,
               target_pose, out_of_stretch, origin_pos, joint_name):
        """
        递归函数，计算逆运动学
        输入:
            joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
            joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
            kinematic_tree: 一个字典，key为关节名字，value为其父节点的名字
        输出:
            经过IK后的姿态
            joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
            joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        """
        kinematic_tree = joint_parent.copy()
        old_path = path.copy()
        path = path.copy()
   
        for i in range(len(path) - 1):
            # path[i+1] Child
            # path[i] Parent
            kinematic_tree[path[i + 1]] = path[i]
        if 0 in path:
            path.remove(0)

        protected_ids = []
        if 0 in old_path:
            for i, p_id in enumerate(kinematic_tree):
                if p_id == 0:
                    new_p_id = old_path[old_path.index(0) - 1]
                    kinematic_tree[i] = new_p_id
                    protected_ids.append(i)
        kinematic_tree[path[0]] = -1
            
        if kinematic_tree[0] in protected_ids:
            protected_ids.remove(kinematic_tree[0])

        root = origin_pos[path[0]].copy()

        last_position_state = joint_positions.copy()
        if out_of_stretch:
        
            vec1 = joint_positions[path[-1]] - joint_positions[path[0]]
            vec2 = target_pose - joint_positions[path[0]]
            rot = rodrigus_formula(vec1, vec2)
            joint_positions = (
                rot[None] @ (joint_positions[:, :, None] -
                            joint_positions[path[0]][None, :, None]) +
                joint_positions[path[0]][None, :, None])[..., 0]

            joint_orientations = rot[None] @ joint_orientations
            return joint_positions, joint_orientations
        # stage1
        for i in range(len(path)-1):
        
            curr_index = path[len(path) - 1 - i]

            if i == 0:
                joint_positions[curr_index] = target_pose

            last_end = joint_positions[curr_index]
            parent_index = path[len(path) - 2 - i]
            vec1 = last_position_state[curr_index] - joint_positions[parent_index]
            vec2 = last_end - joint_positions[parent_index]
            rot = rodrigus_formula(vec1, vec2)
            all_child_joint_start = get_all_child_joint(kinematic_tree, parent_index)
            all_child_joint_end = get_all_child_joint(kinematic_tree, curr_index)
            curr_bone_child = list(set(all_child_joint_start) - set(all_child_joint_end) - set([curr_index]))

            if (kinematic_tree[curr_index] != joint_parent[curr_index]):
                rot_joints = curr_bone_child + [curr_index]
                # print(curr_index, 'inverse')
            else:
                rot_joints = curr_bone_child + [parent_index]

            if 0 in old_path:
                if curr_index in protected_ids and curr_index in path and len(path) - 1 - i <= old_path.index(0):
                    rot_joints = curr_bone_child
            pos_joints = curr_bone_child + [parent_index]
            # rot_joints = curr_bone_child + [parent_index]
            # from IPython import embed
            # embed()
            curr_position_rotated = (
                rot[None] @ (last_position_state[curr_index][None, :, None] -
                             joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0]
            bone_length = vec_length(origin_pos[curr_index] - origin_pos[parent_index])
            joint_positions[pos_joints] = (
                rot[None] @ (joint_positions[pos_joints][:, :, None] -
                             joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0] + norm_vec(vec2) * (vec_length(vec2) - bone_length)

            joint_orientations[
                rot_joints] = rot[None] @ joint_orientations[rot_joints]
            

        last_position_state = joint_positions.copy()

        for i in range(len(path)-1):

            curr_index = path[i]

            if i == 0:
                joint_positions[curr_index] = root

            last_end = joint_positions[curr_index]
            parent_index = path[i+1]
            vec1 = last_position_state[curr_index] - joint_positions[parent_index]
            vec2 = last_end - joint_positions[parent_index]
            rot = rodrigus_formula(vec1, vec2)
            all_child_joint_start = get_all_child_joint(kinematic_tree, curr_index)
            all_child_joint_end = get_all_child_joint(kinematic_tree, parent_index)
            curr_bone_child = list(set(all_child_joint_start) - set(all_child_joint_end) - set([parent_index]))

            # from IPython import embed
            # embed()
            if (kinematic_tree[curr_index] != joint_parent[curr_index]):
                rot_joints = curr_bone_child + [parent_index]
                # print(curr_index, 'inverse')
            else:
                rot_joints = curr_bone_child + [curr_index]

            if 0 in old_path:
                if curr_index in protected_ids and curr_index in path and i <= old_path.index(0):
                    rot_joints = curr_bone_child # [parent_index]
            pos_joints = curr_bone_child + [parent_index]
 
            curr_position_rotated = (
                rot[None] @ (last_position_state[curr_index][None, :, None] -
                             joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0]
            bone_length = vec_length(origin_pos[curr_index] - origin_pos[parent_index])
            joint_positions[pos_joints] = (
                rot[None] @ (joint_positions[pos_joints][:, :, None] -
                             joint_positions[parent_index][None, :, None]) + 
                joint_positions[parent_index][None, :, None])[..., 0] +  norm_vec(vec2) * (vec_length(vec2) - bone_length)

            # all_joint_name = [meta_data.joint_name[i] for i in rot_joints]
            # print(all_joint_name)
            joint_orientations[
                rot_joints] = rot[None] @ joint_orientations[rot_joints]
            
        return joint_positions, joint_orientations

    whole_length = 0
    for i in range(len(path) - 1):
        whole_length += vec_length(meta_data.joint_initial_position[path[i + 1]] -
                                    meta_data.joint_initial_position[path[i]])
    
    iter_ = 0
    for iter_idx in range(num_iter):
        thershold = 0.01
        out_of_stretch = False
        target_length = vec_length(target_pose - origin_pos[path[0]])
        if target_length > whole_length:
            out_of_stretch = True
            thershold = target_length - whole_length + 0.01 * target_length / whole_length
        iter_ += 1
        if not vec_length(joint_positions[path[-1]] - target_pose) < thershold:
            joint_positions, joint_orientations = cyclic_coordinate_descent_ik(
                joint_positions, joint_orientations, meta_data.joint_parent,
                path, target_pose, out_of_stretch)
            # joint_positions, joint_orientations = fabrik(
            #     joint_positions, joint_orientations, meta_data.joint_parent,
            #     path, target_pose, out_of_stretch, meta_data.joint_initial_position, meta_data.joint_name)
            # print(joint_positions[path[-1]] - target_pose, iter_idx)
        else:
            break
    # print("Iteration Time", iter_)
    joint_orientations = R.from_matrix(joint_orientations).as_quat()
    return joint_positions, joint_orientations

