#! /usr/bin/env python
import numpy as np
import open3d as o3d
import argparse
import os
import matplotlib.pyplot as plt
import utils.show3d_balls as show3d
import utils.provider as provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data/label")

parser = argparse.ArgumentParser()
parser.add_argument('--if_vis', type=bool, default=False, help='if visualize point cloud')
FLAGS = parser.parse_args()


def read_label(file_path):
    labels = []
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            ls = line.strip().split()
            i = i + 1
            if i > 10:
                labels.append(int(ls[-2])+1)
    labels = np.array(labels).astype(np.int32)
    return labels


def read_points(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    NUM_POINTS = len(np.asarray(pcd.colors))
    print("%s points are loaded" % NUM_POINTS)
    # load data into numpy
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


def simple_segmentation(colors):
    seed = []
    for i in range(len(colors)):
        if colors[i, 0] >= 0.6 and colors[i, 1] <= 0.35 and colors[i, 2] <= 0.35:
            seed.append(i)
    return np.asarray(seed)


if __name__ == "__main__":
    # file name
    filename = "0001.pcd"
    filepath = os.path.join(DATA_DIR, filename)
    # read data from pcd
    points, colors = read_points(filepath)
    seed = simple_segmentation(colors)
    print("%s seed point is choosen" % len(seed))
    # read label
    labels = read_label(filepath)
    #
    pcd = o3d.geometry.PointCloud()
    seed_neighbour = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(20):
        index = int(len(seed)*np.random.random([1,1]))
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[seed[i],:], 8000)
        seed_neighbour.points = o3d.utility.Vector3dVector(points[idx, :])
        seed_neighbour.colors = o3d.utility.Vector3dVector(colors[idx, :])
        o3d.visualization.draw_geometries([seed_neighbour],
                                          lookat=[1.4235, -0.58550000499999999, 0.2029999945],
                                          up=[0.21272584752548176, 0.97218161465942343, -0.09803377944750484],
                                          front=[-0.97709155214739762, 0.21100017921228287, -0.027767302616098732],
                                          zoom=1.12)
    # visualization
    if FLAGS.if_vis:
        points[:, 0] = -points[:, 0]
        points[:, 2] = points[:, 2]
        points[:, 1] = -points[:, 1]
        cmap = plt.cm.get_cmap("hsv", 4)
        cmap = np.array([cmap(i) for i in range(30)])[:, :3]
        gt = cmap[labels, :]
        show3d.showpoints(points, gt, ballradius=1)
