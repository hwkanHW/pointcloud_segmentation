#! /usr/bin/env python
import numpy as np
import open3d as o3d
import argparse
import os
import matplotlib.pyplot as plt
import utils.show3d_balls as show3d
import utils.provider as provider
from utils.dbscan import dbscan

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

def find_region(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.get_max_bound(),pcd.get_min_bound()


if __name__ == "__main__":
    # file name
    filename = "0001.pcd"
    filepath = os.path.join(DATA_DIR, filename)
    # color map
    cmap = plt.cm.get_cmap("hsv", 4)
    cmap = np.array([cmap(i) for i in range(30)])[:, :3]
    # read data from pcd
    points, colors = read_points(filepath)
    seed = simple_segmentation(colors)
    print("%s seed points are choosen" % len(seed))
    # read label
    labels = read_label(filepath)
    # #
    pcd = o3d.geometry.PointCloud()
    seed_neighbour = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    seed_points = points[seed]
    for i in range(200):
        index = int(len(seed) * np.random.random([1, 1]))
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[index, :], 8000)
        train_pts = points[idx, :]
        train_lab = labels[idx]
        num_apple_points = len(np.where(train_lab!=1)[0])
        # max_bound, min_bound = find_region(train_pts)
        # max_x = max_bound[1]-min_bound[1]
        # max_y = max_bound[2]-min_bound[2]
        if np.max(train_lab) > 1 and num_apple_points>500:
            gt = cmap[train_lab, :]
            show3d.showpoints(train_pts, gt, ballradius=1)
        else:
            print("ss")
        # seed_neighbour.points = o3d.utility.Vector3dVector(points[idx, :])
        # seed_neighbour.colors = o3d.utility.Vector3dVector(colors[idx, :])
        # o3d.visualization.draw_geometries([seed_neighbour],
        #                                   lookat=[2.0839173373147659, -0.16145597655437618, -0.076792589935532757],
        #                                   up=[-0.15122911613742016, 0.060546159945558287, 0.98664275041584415],
        #                                   front=[-0.98335812024751634, 0.092445977745939628, -0.15639868459123479],
        #                                   zoom=3.0)

    # visualization
    if FLAGS.if_vis:
        points[:, 0] = -points[:, 0]
        points[:, 2] = points[:, 2]
        points[:, 1] = -points[:, 1]
        cmap = plt.cm.get_cmap("hsv", 4)
        cmap = np.array([cmap(i) for i in range(30)])[:, :3]
        gt = cmap[labels, :]
        show3d.showpoints(points, gt, ballradius=1)
