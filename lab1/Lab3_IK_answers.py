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
                    print(sub_tree_new_root.show(stdout=False))
                    sub_tree_parent_rest = self.tree.subtree(path[i+1])
                    print(sub_tree_parent_rest.show(stdout=False))
                    for node in sub_tree_parent_rest.all_nodes_itr():
                        print(node.identifier, node.tag)
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
        
        # print("Path1 from End:{} to RootJoint".format(self.end_joint))

        # 从root节点开始，一直往上找，直到找到腰部节点
        path2 = [self.joint_name.index(self.root_joint)]
        while self.joint_parent[path2[-1]] != -1:
            path2.append(self.joint_parent[path2[-1]])
        
        # print("Path2 from Start:{} to RootJoint".format(self.root_joint))

        # 合并路径，消去重复的节点
        while path1 and path2 and path2[-1] == path1[-1]:
            path1.pop()
            a = path2.pop()
            
        path2.append(a)
        path = path2 + list(reversed(path1))
        path_name = [self.joint_name[i] for i in path]
        print("From Start to End:")
        for i in range(len(path)-1):
            print(path_name[i]+":{}->".format(i),end='')
        print(path_name[-1]+":{}".format(len(path)-1))
        return path, path_name, path1, path2

def single_step_ccd(meta_data:MetaData, joint_positions, joint_orientations,
                             target_pose, ccd_idx):
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
    print("-----------------------------------------")
    path_ik,_,_,_ = meta_data.get_path_from_root_to_end()
    joint_orientations = R.from_quat(joint_orientations).as_matrix()
    
    num_iter = 100
    origin_pos = joint_positions.copy()

    strench_range = np.linalg.norm(
        (meta_data.joint_initial_position[path_ik[1:]]-meta_data.joint_initial_position[path_ik[0:-1]]),
        axis=1).sum()
        

    def ccd(joint_positions, joint_orientations, out_of_strench):
        i = ccd_idx
        next_ccd_idx = ccd_idx-1
        if next_ccd_idx == 0:
            next_ccd_idx = len(path_ik)-1
        
        print("ccd_idx:", ccd_idx)
        curr_index = path_ik[i]
        parent_index = path_ik[i-1]
        print("decent joint: "+meta_data.joint_name[parent_index]+"->"+meta_data.joint_name[curr_index])
       
        vec1 = joint_positions[path_ik[-1]]-joint_positions[parent_index]
        # print("current parent_to_end", vec1)
        vec1 = vec1/np.linalg.norm(vec1)
        vec2 = target_pose-joint_positions[parent_index]
        # print("desired parent_to_end", vec2)
        vec2 = vec2/np.linalg.norm(vec2)

        print(meta_data.tree.subtree(parent_index).show(stdout=False))
        all_child_joint_node = meta_data.tree.subtree(parent_index).all_nodes()
        all_child_joint = [n.identifier for n in all_child_joint_node]
        all_joint_name = [meta_data.joint_name[i] for i in all_child_joint]
        if all_child_joint.__contains__(0):
            all_rot_joint = all_child_joint[1:]
        else:
            all_rot_joint = all_child_joint
        print("joint need to rotate", all_joint_name)
        
        if np.allclose(vec1, vec2, atol=1e-3,rtol=1e-3) or meta_data.joint_name[parent_index]=="lKnee" or meta_data.joint_name[parent_index]=='rKnee':
            rot = np.eye(3)
        else:
            axis = np.cross(vec1, vec2)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(vec1, vec2) /
                (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            rot = R.from_rotvec(axis*angle).as_matrix()
        print("rotate vec1 to vec2:")
        print(rot)
        
        joint_orientations[all_rot_joint] = rot@joint_orientations[all_rot_joint]
        position_update = rot@(joint_positions[all_child_joint,:]-joint_positions[parent_index,:]).T
        np.save("./debug/joint_position{}.npy".format(ccd_idx), joint_positions)
        np.save("./debug/joint_orientation{}.npy".format(ccd_idx), joint_orientations)
        joint_positions[all_child_joint] = position_update.T[:]+joint_positions[parent_index,:]
        distance_to_target = np.linalg.norm(joint_positions[path_ik[-1]]-target_pose)
        print("end after decent:", distance_to_target)
        # print("-----------------------------------------")
    
        return joint_positions, joint_orientations, next_ccd_idx

    
    distance = np.linalg.norm(target_pose-origin_pos[path_ik[0]])
    out_of_strench = distance>strench_range
    threshold = 0.01 if not out_of_strench else distance-strench_range+0.01*distance/strench_range
    rotate_over = False
    if not np.linalg.norm(joint_positions[path_ik[-1]]-target_pose)<threshold:
        joint_positions, joint_orientations,next_ccd = ccd(joint_positions, joint_orientations,out_of_strench)            
    
    joint_orientations = R.from_matrix(joint_orientations).as_quat()
    rotate_over = np.linalg.norm(joint_positions[path_ik[-1]]-target_pose)<threshold
    
    return joint_positions, joint_orientations, next_ccd, rotate_over


