#! /usr/bin/env python
import os
import sys
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "label")
label_list = glob.glob(os.path.join(DATA_DIR, "*.pcd"))

sys.path.append(BASE_DIR)
import show3d_balls as show3d


cmap = plt.cm.get_cmap("hsv", 4)
cmap = np.array([cmap(i) for i in range(30)])[:, :3]


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class MarsDataset():
    def __init__(self, batch_size=32, npoints=8000, normalize=True, color_channel=False, random_num=5, load_data = True):
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.color_channel = color_channel
        self.random_num = random_num
        ##
        self.batch_idx = 0
        ##
        self.pts_set = []
        self.lab_set = []
        ##
        if load_data:
            self.pts_data = np.load("pts_data.npy")
            self.lab_data = np.load("cls_data.npy")
        else:
            self.load_data_from_pcd()
        #
        self.num_batches = (len(self.pts_data)) // self.batch_size

    def load_data_from_pcd(self):
        pcd = o3d.geometry.PointCloud()
        for i in range(len(label_list)):
            points, colors, labels = self._get_item(label_list[i])
            print(label_list[i])
            seed = np.where(labels==20)[0]
            # seed = self.simple_segmentation(colors)
            ##
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            total_index = 0
            it_index = 0
            while total_index < self.random_num:
                it_index = it_index+1
                index = int(len(seed) * np.random.random([1, 1]))
                [k, idx, _] = pcd_tree.search_knn_vector_3d(points[index, :], self.npoints)
                pts = points[idx, :]
                lab = labels[idx]
                num_apple_points = len(np.where(lab != 1)[0])
                if num_apple_points > 250:
                    # gt = cmap[lab, :]
                    # show3d.showpoints(pts, gt, ballradius=1)
                    self.pts_set.append(pts)
                    self.lab_set.append(lab)
                    total_index = total_index+1
                    print("%f data are loaded"%(total_index/self.random_num))
                if it_index > 4*self.random_num:
                    break
        self.pts_data = np.array(self.pts_set)
        self.lab_data = np.array(self.lab_set)
        np.save('pts_data.npy', self.pts_data)
        np.save('cls_data.npy', self.lab_data)

    def _get_item(self, file_path):
        points, colors = self.read_points(file_path)
        labels = self.read_label(file_path)
        return points, colors, labels

    def read_points(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        NUM_POINTS = len(np.asarray(pcd.colors))
        print("%s points are loaded" % NUM_POINTS)
        # load data into numpy
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        return points, colors

    def read_label(self, file_path):
        labels = []
        with open(file_path, 'r') as f:
            i = 0
            for line in f:
                ls = line.strip().split()
                i = i + 1
                if i > 10:
                    if ls[-2] == "null":
                        labels.append(1)
                    else:
                        labels.append(int(ls[-2]) + 1)
        labels = np.array(labels).astype(np.int32)
        return labels

    def simple_segmentation(self, colors):
        seed = []
        for i in range(len(colors)):
            if colors[i, 0] >= 0.6 and colors[i, 1] <= 0.35 and colors[i, 2] <= 0.35:
                seed.append(i)
        return np.asarray(seed)

    def _augment_batch_data(self, batch_data):
        return

    def reset(self):
        self.batch_idx = 0

    def num_channel(self):
        if self.color_channel:
            return 6
        else:
            return 3

    def next_batch(self, augment=False):
        start_idx = self.batch_size*self.batch_idx
        end_idx = self.batch_size*(self.batch_idx+1)
        pts = self.pts_data[start_idx:end_idx]
        cls = self.lab_data[start_idx:end_idx]
        self.batch_idx = self.batch_idx + 1
        return pts, cls

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def return_dataset(self):
        return self.pts_data, self.lab_data


if __name__ == "__main__":
    dataset = MarsDataset(batch_size=6,random_num=5,load_data=True)
    pts_data, cls_data = dataset.return_dataset()
    print(len(pts_data))
    index = 0
    while dataset.has_next_batch():
        pts,cls = dataset.next_batch()
        index = index+1
    print(index)
