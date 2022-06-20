import os
import open3d as o3d
import numpy as np
import mayavi.mlab
import argparse
import glob

def generate_bbox(input_list, output_path):
    with open(output_path, 'w') as f:
        for input_path in input_list:
            pcd = o3d.io.read_point_cloud(input_path, format='pcd')
            point_cloud = np.asarray(pcd.points)

            labels, objects = extract(input_path, ['label', 'object'])
            labels = [int(item) for item in labels]
            labels = np.array(labels)
            objects = [int(item) for item in objects]
            objects = np.array(objects)

            instance = set(objects)
            annos = []
            for item in instance:
                if item < 0:
                    continue
                points_instance = point_cloud[objects==item]
                label = labels[objects==item][0]

                x = points_instance[:, 0]  # x position of point
                xmin = np.amin(x, axis=0)
                xmax = np.amax(x, axis=0)
                y = points_instance[:, 1]  # y position of point
                ymin = np.amin(y, axis=0)
                ymax = np.amax(y, axis=0)
                z = points_instance[:, 2]  # z position of point
                zmin = np.amin(z, axis=0)
                zmax = np.amax(z, axis=0)
                # print(xmin, xmax, ymin, ymax, zmin, zmax)

                annos.append("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                             .format(label, item, xmin, xmax, ymin, ymax, zmin, zmax))

            f.write("{}\t{}\n".format(os.path.basename(input_path), "\t".join(annos)))

def extract(file_path, extract_fileds):
    with open(file_path, 'r') as f:
        data = f.readlines()

    data_fileds = data[1].strip().split(' ')[1:]
    idxs = [data_fileds.index(item) for item in extract_fileds]

    output = [[] for i in range(len(extract_fileds))]
    for line in data[10:]:
        line = line.strip().split(' ')
        for i, idx in enumerate(idxs):
            output[i].append(line[idx])
    return output

def pcd_vis(input_list):
    file_path = input_list[0]
    pcd = o3d.io.read_point_cloud(file_path, format='pcd')
    # o3d.visualization.draw_geometries([pcd])

    labels, objects = extract(file_path, ['label', 'object'])
    objects = [int(item) for item in objects]
    objects = np.array(objects)

    point_cloud = np.asarray(pcd.points)
    point_color = np.asarray(pcd.colors)

    x = point_cloud[:, 0]  # x position of point
    y = point_cloud[:, 1]  # y position of point
    z = point_cloud[:, 2]  # z position of point

    col = (objects + 1) / 100.0
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z, col, mode="point", figure=fig)

    instance = set(objects)
    for item in instance:
        if item < 0:
            continue
        points_instance = point_cloud[objects == item]
        print("points_instance", item, objects)
        x = points_instance[:, 0]  # x position of point
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)
        y = points_instance[:, 1]  # y position of point
        ymin = np.amin(y, axis=0)
        ymax = np.amax(y, axis=0)
        z = points_instance[:, 2]  # z position of point
        zmin = np.amin(z, axis=0)
        zmax = np.amax(z, axis=0)

        draw_bbox(fig, xmin, xmax, ymin, ymax, zmin, zmax)



    mayavi.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.001)
    axes = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    mayavi.mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    mayavi.mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    mayavi.mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )

    mayavi.mlab.show()

def draw_bbox(fig, xmin, xmax, ymin, ymax, zmin, zmax):
    mayavi.mlab.plot3d(
        [xmin, xmax], [ymin, ymin], [zmin, zmin],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmax, xmax], [ymin, ymax], [zmin, zmin],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmax, xmin], [ymax, ymax], [zmin, zmin],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmin, xmin], [ymax, ymin], [zmin, zmin],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )

    mayavi.mlab.plot3d(
        [xmin, xmax], [ymin, ymin], [zmax, zmax],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmax, xmax], [ymin, ymax], [zmax, zmax],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmax, xmin], [ymax, ymax], [zmax, zmax],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmin, xmin], [ymax, ymin], [zmax, zmax],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )

    mayavi.mlab.plot3d(
        [xmin, xmin], [ymin, ymin], [zmin, zmax],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmax, xmax], [ymax, ymax], [zmin, zmax],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmin, xmin], [ymax, ymax], [zmin, zmax],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )
    mayavi.mlab.plot3d(
        [xmax, xmax], [ymin, ymin], [zmin, zmax],
        color=(1, 0, 0), tube_radius=None, figure=fig
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Datasets configuration
    parser.add_argument('--data_dir', type=str, default="./")
    parser.add_argument('--anno_file', type=str, default='./anno.txt')
    parser.add_argument('--vis', action="store_true")

    args = parser.parse_args()
    if args.vis:
        print(args.data_dir)
        print(glob.glob(os.path.join(args.data_dir, "*.pcd")))
        pcd_vis(glob.glob(os.path.join(args.data_dir, "*.pcd")))
    else:
        generate_bbox(glob.glob(os.path.join(args.data_dir, "*.pcd")), args.anno_file)
