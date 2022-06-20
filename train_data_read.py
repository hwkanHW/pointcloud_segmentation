#! /usr/bin/env python
import numpy as np
import open3d as o3d
import sys
import os
import matplotlib.pyplot as plt
import utils.show3d_balls as show3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,"data")


if __name__ == "__main__":
    # read data from pcd
    pcd = o3d.io.read_point_cloud(os.path.join(DATA_DIR,"test_label.pcd"))
    NUM_POINTS = len(np.asarray(pcd.colors))
    # load data into numpy
    points = np.asarray(pcd.points)
    points[:,0] = -points[:,0]
    points[:,2] = points[:,2]
    points[:,1] = -points[:,1]
    colors = np.asarray(pcd.colors)
    pcd.points = o3d.utility.Vector3dVector()
    # read label
    labels = []
    with open(os.path.join(DATA_DIR,"test_label.pcd"), 'r') as f:
        i=0
        k=0
        for line in f:
            ls = line.strip().split()
            i=i+1
            if i>10:
                labels.append(int(ls[-2]))
    labels = np.array(labels).astype(np.int32)
    # visualization
    cmap = plt.cm.get_cmap("hsv", 4)
    cmap = np.array([cmap(i) for i in range(10)])[:,:3]
    gt = cmap[labels, :]
    show3d.showpoints(points,gt,ballradius=1)
    # print(pcd.get_center())
    # print(pcd.has_colors())
    # print(pcd.has_points())
    # print(np.asarray(pcd.colors).shape)
    # visualizaion
    # o3d.visualization.draw_geometries([pcd],
    #                                   lookat=[ -0.00071885912563767005, 0.036710214427356704, 0.098602843157838918 ],
    #                                   up = [ 0.036150451334005476, 0.064092308544618698, 0.99728898562742052 ],
    #                                   front = [ -0.99912036372943447, 0.023539986228374905, 0.034704003076456005 ],
    #                                   zoom = 0.57999999999999985)
    