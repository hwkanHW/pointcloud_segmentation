import open3d as o3d
import numpy as np

class DataLoader(object):
    def __init__(self, data_dir, anno_file):
        self.data_dir = data_dir

        with open(anno_file, 'r') as f:
            self.annos = f.readlines()

    def __getitem__(self, index):
        anno = self.annos[index].strip().split('\t')
        file_name = anno[0]

        instance_num = (len(anno) - 1) // 8
        for i in range(instance_num):
            label, item, xmin, xmax, ymin, ymax, zmin, zmax = anno[i*8+1:(i+1)*8+1]

if __name__ == "__main__":
    pass