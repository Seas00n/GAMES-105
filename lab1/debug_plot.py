import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from viewer import SimpleViewer
from Lab2_IK_answers import MetaData

viewer = SimpleViewer()
joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([-0.9,0.9])
ax.set_ylim([-0.9, 0.9])
ax.set_zlim([0,1.8])
plt.ion()
plt.show(block=False)

# meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'lWrist_end')


def add_text(joint_positions):
    t_list = []
    global ax
    for i in range(np.shape(joint_positions)[0]):
        t = ax.text(joint_positions[i,2], joint_positions[i,0], joint_positions[i,1],
                    joint_name[i][0:3], size=8)
        t_list.append(t)
        if joint_parent[i] == -1:
            continue
        else:
            link = plt.plot([joint_positions[i,2], joint_positions[joint_parent[i],2]],
                           [joint_positions[i,0], joint_positions[joint_parent[i],0]],
                           [joint_positions[i,1], joint_positions[joint_parent[i],1]],
                           color='black')
    return t_list

def clear_text(t_list):
    for t in t_list:
        t.remove()

def clear_line():
    for l in ax.get_lines():
        l.remove()

for i in range(12,0,-1):
    joint_positions = np.load("./debug/joint_position{}.npy".format(i))
    p = ax.scatter(joint_positions[:,2], joint_positions[:,0], joint_positions[:,1])
    t_list = add_text(joint_positions)
    plt.show(block=False)
    plt.pause(0.01)
    p.remove()
    clear_text(t_list)
    clear_line()