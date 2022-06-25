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
        if colors[i, 0] >= 0.6 and colors[i, 1] <= 0.4 and colors[i, 2] <= 0.4:
            seed.append(i)
    return np.asarray(seed)


def seed_from_clusrer(points, seed):
    eps = 0.05
    min_points = 40
    cluster = dbscan(np.transpose(points[seed]), eps, min_points)
    cluster = np.array(cluster)
    index = np.argwhere(cluster == None)
    cluster[index] = 0
    cluster.astype("int32")
    cluster_index = []
    for i in range(np.max(cluster)):
        index = np.where(cluster == i)[0]
        if i != 0:
            cluster_index.append(index[-1])
    return np.array(cluster_index)
    # print(np.max(cluster))
    # return cluster


if __name__ == "__main__":
    # file name
    filename = "0001.pcd"
    filepath = os.path.join(DATA_DIR, filename)
    # read data from pcd
    points, colors = read_points(filepath)
    seed = simple_segmentation(colors)
    print("%s seed points are choosen" % len(seed))
    cluster = seed_from_clusrer(points, seed)
    print("%s clusters are choosen" % len(cluster))
    # read label
    labels = read_label(filepath)
    # #
    pcd = o3d.geometry.PointCloud()
    seed_neighbour = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    seed_points = points[seed]
    o3d.visualization.draw_geometries([pcd],
                                      lookat=[2.0839173373147659, -0.16145597655437618, -0.076792589935532757],
                                      up=[-0.15122911613742016, 0.060546159945558287, 0.98664275041584415],
                                      front=[-0.98335812024751634, 0.092445977745939628, -0.15639868459123479],
                                      zoom=0.59999999999999964)
    for i in range(len(cluster)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(seed_points[cluster[i]], 8000)
        seed_neighbour.points = o3d.utility.Vector3dVector(points[idx, :])
        seed_neighbour.colors = o3d.utility.Vector3dVector(colors[idx, :])
        o3d.visualization.draw_geometries([seed_neighbour],
                                          lookat=[2.0839173373147659, -0.16145597655437618, -0.076792589935532757],
                                          up=[-0.15122911613742016, 0.060546159945558287, 0.98664275041584415],
                                          front=[-0.98335812024751634, 0.092445977745939628, -0.15639868459123479],
                                          zoom=3.0)

    # visualization
    if FLAGS.if_vis:
        points[:, 0] = -points[:, 0]
        points[:, 2] = points[:, 2]
        points[:, 1] = -points[:, 1]
        cmap = plt.cm.get_cmap("hsv", 4)
        cmap = np.array([cmap(i) for i in range(30)])[:, :3]
        gt = cmap[labels, :]
        show3d.showpoints(points, gt, ballradius=1)
