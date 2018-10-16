import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def visualize_3d(data, fig=None, subplot_idx=None, subplot_dims=(1,1), ax=None, show_axis=False, show_ticks=True, axis_range=1.5):
    data = np.array(data)

    xs = data[0]
    ys = data[1]
    zs = data[2]

    # left arm data
    left_arm_xs = [xs[15],xs[14],xs[14],xs[13]]
    left_arm_ys = [ys[15],ys[14],ys[14],ys[13]]
    left_arm_zs = [zs[15],zs[14],zs[14],zs[13]]

    # right arm data
    right_arm_xs = [xs[10], xs[11], xs[11], xs[12]]
    right_arm_ys = [ys[10], ys[11], ys[11], ys[12]]
    right_arm_zs = [zs[10], zs[11], zs[11], zs[12]]

    # left leg data
    left_leg_xs = [xs[5],xs[4],xs[4],xs[3]]
    left_leg_ys = [ys[5],ys[4],ys[4],ys[3]]
    left_leg_zs = [zs[5],zs[4],zs[4],zs[3]]

    # right leg data
    right_leg_xs = [xs[0],xs[1],xs[1],xs[2]]
    right_leg_ys = [ys[0],ys[1],ys[1],ys[2]]
    right_leg_zs = [zs[0],zs[1],zs[1],zs[2]]

    # torso data
    hip_xs = [xs[3], xs[2]]
    hip_ys = [ys[3], ys[2]]
    hip_zs = [zs[3], zs[2]]

    shoulder_xs = [xs[13], xs[12]]
    shoulder_ys = [ys[13], ys[12]]
    shoulder_zs = [zs[13], zs[12]]

    # spine data
    spine_xs = [xs[6], xs[7]]
    spine_ys = [ys[6], ys[7]]
    spine_zs = [zs[6], zs[7]]

    if not fig==None and not ax==None:
        pass
    else:
        if fig == None:
            fig = plt.figure()
            ax = fig.add_subplot(subplot_dims[0], subplot_dims[1], 1, projection='3d')
        else:
            if subplot_idx == None:
                ax = fig.add_subplot(subplot_dims[0], subplot_dims[1], 1, projection='3d')
            else:
                ax = fig.add_subplot(subplot_dims[0], subplot_dims[1], subplot_idx, projection='3d')

    ax.view_init(azim=-90, elev=-65)
    ax.set_xlim([-axis_range, axis_range])
    ax.set_ylim([-axis_range, axis_range])
    ax.set_zlim([-axis_range, axis_range])

    #left arm
    ax.plot(left_arm_xs, left_arm_ys, left_arm_zs)

    # right arm
    ax.plot(right_arm_xs, right_arm_ys, right_arm_zs)

    # left leg
    ax.plot(left_leg_xs, left_leg_ys, left_leg_zs)

    # right leg
    ax.plot(right_leg_xs, right_leg_ys, right_leg_zs)

    # torso
    ax.plot(hip_xs, hip_ys, hip_zs)
    ax.plot(shoulder_xs, shoulder_ys, shoulder_zs)

    # spine
    ax.plot(spine_xs, spine_ys, spine_zs,c='purple')
    # ax.plot(shoulder_xs, shoulder_ys, shoulder_zs)

    # ax.scatter(xs[[0,1,2,3,4,5,9,10,11,12,13,14,15]], ys[[0,1,2,3,4,5,9,10,11,12,13,14,15]], zs[[0,1,2,3,4,5,9,10,11,12,13,14,15]])
    # ax.scatter(xs[[0,1,2,3,4,5,8,9,10,11,12,13,14,15]], ys[[0,1,2,3,4,5,8,9,10,11,12,13,14,15]], zs[[0,1,2,3,4,5,8,9,10,11,12,13,14,15]])
    # ax.scatter(xs[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], ys[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], zs[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    ax.scatter(xs[[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15]], ys[[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15]], zs[[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15]])

    ax.set_aspect('equal')

    if show_axis == False:
        ax.set_axis_off()

    if show_ticks == False:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    return fig, ax