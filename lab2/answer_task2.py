# 以下部分均为可更改部分

from answer_task1 import *
from smooth_utils import *
State = {
    "idle":int(0),
    "walking":int(1),
    "running":int(2)
}

class CharacterController():
    def __init__(self, controller) -> None:
        # motion materials
        self.motions = []
        self.motions.append(BVHMotion('motion_material/idle.bvh'))
        self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))
        self.motions.append(BVHMotion('motion_material/run_forward.bvh'))
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0
        self.cur_state = State['idle']
        # loop motion
        self.loop_motions = []
        idle_state_motion = build_loop_motion(self.motions[0])
        walk_state_motion = build_loop_motion(self.motions[1])
        run_state_motion = build_loop_motion(self.motions[2])
        self.loop_motions.append(idle_state_motion)
        self.loop_motions.append(walk_state_motion)
        self.loop_motions.append(run_state_motion)
        # self.idle2move_motion = concatenate_two_motions(self.motions[1], self.motions[0], mix_frame1=60, mix_time=30)
        # self.move2idle_motion = concatenate_two_motions(self.motions[0], self.motions[1], mix_frame1=60, mix_time=30)
        pass
    



    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        # 一个简单的例子，输出第i帧的状态
        # joint_name = self.loop_motions[0].joint_name
        # joint_translation, joint_orientation = self.loop_motions[2].batch_forward_kinematics()
        # joint_translation = joint_translation[self.cur_frame]
        # joint_orientation = joint_orientation[self.cur_frame]
        
        # self.cur_root_pos = joint_translation[0]
        # self.cur_root_rot = joint_orientation[0]
        # self.cur_frame = (self.cur_frame + 1) % self.loop_motions[2].motion_length
        
        joint_name = self.loop_motions[self.cur_state].joint_name
        joint_translation, joint_orientation = self.state_machine(
            desired_pos_list, 
            desired_rot_list,
            desired_vel_list,
            desired_avel_list
        )

        return joint_name, joint_translation, joint_orientation
    
    
    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)
        
        return character_state

    # 你的其他代码,state matchine, motion matching, learning, etc.
    def state_change_motion_adjust(self,last_state:int):
        cur_motion:BVHMotion = self.loop_motions[self.cur_state]
        if self.cur_state != last_state:
            # 改变现在的idle_loop_motion的朝向到原有朝向产生新的idle_loop_motion
            last_motion:BVHMotion = self.loop_motions[last_state]
            last_facing_direction = R.from_quat(
                last_motion.joint_rotation[self.cur_frame,0,:]
            ).as_matrix()[[0,2],2]
            last_target_translation = last_motion.joint_position[self.cur_frame, 0, [0,2]]
            cur_motion = cur_motion.translation_and_rotation(
                frame_num = 0,
                target_translation_xz=last_target_translation,
                target_facing_direction_xz=last_facing_direction
            )
            self.loop_motions[self.cur_state] = cur_motion
            self.cur_frame = 0

    def velocity_change_motion_adjust(self, desired_pos_list, desired_rot_list,
                                            desired_vel_list, desired_avel_list):
        cur_motion:BVHMotion = self.loop_motions[self.cur_state]
        # 找到关键帧在motions里对应的帧
        key_frame = [(self.cur_frame + 20 * i) % self.motions[self.cur_state].motion_length for i in range(6)]
        # 对root的pos和rot进行damping平滑 
        pos_in_key_frame = cur_motion.joint_position[key_frame,0,:]
        diff_pos_in_key_frame = desired_pos_list-pos_in_key_frame
        diff_pos_in_key_frame[:,1] = 0
        rot_in_key_frame = R.from_quat(cur_motion.joint_rotation[key_frame,0,:])
        diff_rot_in_key_frame = (R.from_quat(desired_rot_list[0:-1])*rot_in_key_frame.inv()).as_rotvec()
        # 帧速度*60 = 秒速度
        # desired_vel和desired_avel都是秒速度
        vel_in_key_frame = (cur_motion.joint_position[key_frame, 0, :] - cur_motion.joint_position[[(frame - 1) for frame in key_frame], 0, :])/20#此处为帧速度
        diff_vel_in_key_frame = (desired_vel_list/60-vel_in_key_frame) #帧速度减帧速度
        avel_in_key_frame = quat_to_avel(cur_motion.joint_rotation[:,0,:], 1)
        diff_root_avel_in_key_frame = desired_avel_list[0:-1]/60 - avel_in_key_frame[[(frame-1) for frame in key_frame]] #帧速度
        
        for i in range(self.cur_frame, self.cur_frame+self.motions[self.cur_state].motion_length//2):
            half_time = 0.2
            index = (i - self.cur_frame) // 20 #第几段key_frame之间
            dt = (i-self.cur_frame) % 20 #damping插多少帧
            if index >= 6:
                print(index)
                break
            off_pos, _ = decay_spring_implicit_damping_pos(diff_pos_in_key_frame[index], diff_vel_in_key_frame[index], half_time, dt/60)
            off_rot, _ = decay_spring_implicit_damping_rot(diff_rot_in_key_frame[index], diff_root_avel_in_key_frame[index], half_time, dt/60)

            cur_motion.joint_position[ i % self.motions[self.cur_state].motion_length, 0, :] += off_pos
            cur_motion.joint_rotation[ i % self.motions[self.cur_state].motion_length, 0, :] = (R.from_rotvec(off_rot) * R.from_quat(cur_motion.joint_rotation[ i % self.motions[self.cur_state].motion_length, 0, :])).as_quat()
        self.loop_motions[self.cur_state] = cur_motion

    def state_machine(self, desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list):
        last_state = self.cur_state
        cur_vel_xz = np.linalg.norm(np.array(desired_vel_list[0][[0,2]]))
        if cur_vel_xz < 0.2:
            self.cur_state = State['idle']
        elif 0.2 <= cur_vel_xz < 1.5:
            self.cur_state = State['walking']
        else:
            self.cur_state = State['running']

        if self.cur_state == State["idle"]:
            if self.cur_state != last_state:
                # 改变现在的idle_loop_motion的朝向到原有朝向产生新的idle_loop_motion
                self.state_change_motion_adjust(last_state)    
            joint_translation, joint_orientation = self.loop_motions[self.cur_state].batch_forward_kinematics()
            joint_translation = joint_translation[self.cur_frame]
            joint_orientation = joint_orientation[self.cur_frame]
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]
            self.cur_frame = (self.cur_frame + 1) % self.motions[self.cur_state].motion_length
        elif self.cur_state == State["walking"] or self.cur_state==State['running']:
            if self.cur_state != last_state:
                self.state_change_motion_adjust(last_state)
            self.velocity_change_motion_adjust(
                desired_pos_list, desired_rot_list,
                desired_vel_list, desired_avel_list
            )
            joint_translation, joint_orientation = self.loop_motions[self.cur_state].batch_forward_kinematics()
            joint_translation = joint_translation[self.cur_frame]
            joint_orientation = joint_orientation[self.cur_frame]
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]
            self.cur_frame = (self.cur_frame + 1) % self.motions[self.cur_state].motion_length

            

        return joint_translation, joint_orientation
    