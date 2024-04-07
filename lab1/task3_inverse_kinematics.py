from viewer import SimpleViewer
from scipy.spatial.transform import Rotation as R
from Lab3_IK_answers import *

    
def part1_hard(viewer, target_pos):
    """
    完成part1_inverse_kinematics，我们将根节点设在**左脚部**，末端节点设在左手
    """
    marker = viewer.create_marker(target_pos, [1, 0, 0, 1])
    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()
    path_ik,_,_,_ = meta_data.get_path_from_root_to_end()
    ccd_num = len(path_ik)-1
    
    class UpdateHandle:
        def __init__(self, marker, joint_position, joint_orientation, ccd_num):
            self.marker = marker
            self.joint_position = joint_position
            self.joint_orientation = joint_orientation
            self.ccd_num = ccd_num
            self.rotate_over = False

        def update_func(self, viewer):
            self.marker.setPos(target_pos[0], target_pos[1], target_pos[2])
            if not self.rotate_over:
                self.joint_position, self.joint_orientation, self.ccd_num, self.rotate_over= single_step_ccd(meta_data, self.joint_position, 
                                                                                        self.joint_orientation, target_pos,
                                                                                        self.ccd_num)
            viewer.show_pose(joint_name, self.joint_position, self.joint_orientation)
    handle = UpdateHandle(marker, joint_position, joint_orientation, ccd_num)
    handle.update_func(viewer)
    viewer.update_marker_func = handle.update_func
    viewer.run()
    pass

def main():
    viewer = SimpleViewer()
    
    # part1
    # part1_simple(viewer, np.array([0.5, 0.75, 0.5]))
    part1_hard(viewer, np.array([0.5, 0.5, 0.5]))
    # part1_animation(viewer, np.array([0.5, 0.5, 0.5]))
    
    # part2
    # part2(viewer, 'data/walk60.bvh')
    
    # bonus(viewer, np.array([0.5, 0.5, 0.5]), np.array([0, 0.5, 0.5]))

if __name__ == "__main__":
    main()