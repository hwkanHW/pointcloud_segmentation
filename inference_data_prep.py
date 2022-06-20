#! /usr/bin/env python
import numpy as np
import open3d as o3d
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,"data")

if_visualization = False

def pcd_subdivision(pcd, grid_num_x, grid_num_y):
    max_bound = pcd.get_max_bound()[1:]
    min_bound = pcd.get_min_bound()[1:]
    grid_x = np.linspace(min_bound[0]-0.1,max_bound[0]+0.1,grid_num_x+1)
    grid_y = np.linspace(min_bound[1]-0.1,max_bound[1]+0.1,grid_num_y+1)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    ##
    points_list = []
    colors_list = []
    for i in range(grid_num_x*grid_num_y):
        points_list.append([])
        colors_list.append([])
    ##
    for i in range(len(points)):
        index_x = np.where(points[i,1]>=grid_x)[0][-1]
        index_y = np.where(points[i,2]>=grid_y)[0][-1]
        # print(index_y[0][-1]*grid_num_x+(index_x[0][-1]-1))
        # print(index_y*grid_num_x+(index_x-1))
        points_list[index_y*grid_num_x+(index_x-1)].append(points[i].tolist())
        colors_list[index_y*grid_num_x+(index_x-1)].append(colors[i].tolist())
        # print(points_list[index_y*grid_num_x+(index_x-1)])
    ##
    for i in range(len(points_list)):
        print(len(points_list[i]))
        pcd.points = o3d.utility.Vector3dVector(np.array(points_list[i]))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors_list[i]))
        o3d.visualization.draw_geometries([pcd],
                                          lookat=[ -0.00071885912563767005, 0.036710214427356704, 0.098602843157838918 ],
                                          up = [ 0.036150451334005476, 0.064092308544618698, 0.99728898562742052 ],
                                          front = [ -0.99912036372943447, 0.023539986228374905, 0.034704003076456005 ],
                                          zoom = 0.10)
    return points_list, colors_list
    
    


if __name__ == "__main__":
    
    # read data from pcd
    pcd = o3d.io.read_point_cloud(os.path.join(DATA_DIR,"0102.pcd"))
    # load data into numpy
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # subdivide into small region
    points_list, colors_list = pcd_subdivision(pcd, 4, 3)
    # visualizaion
    if if_visualization:
        o3d.visualization.draw_geometries([pcd],
                                          lookat=[ -0.00071885912563767005, 0.036710214427356704, 0.098602843157838918 ],
                                          up = [ 0.036150451334005476, 0.064092308544618698, 0.99728898562742052 ],
                                          front = [ -0.99912036372943447, 0.023539986228374905, 0.034704003076456005 ],
                                          zoom = 0.10)