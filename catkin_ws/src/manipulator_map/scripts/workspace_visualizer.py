#!/usr/bin/env python3

import rospy
import rospkg
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import PyKDL as kdl
import numpy as np

# for the map stuff
import pandas as pd
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.neighbors import KDTree

# modes for visualization
UNIQUE_FIRST_POINTS=0
UNIQUE_SECOND_POINTS=1
COMBINED_POINTS=2

class WorkspaceVisualizer:
    def __init__(self, first_map_name, second_map_name):
        rospy.init_node('workspace_visualizer_node')

        rospack = rospkg.RosPack()
        self.first_map_name = first_map_name
        self.first_ee_map_path = f"{rospack.get_path('manipulator_map')}/robots/{first_map_name}/{first_map_name}_ee_map.csv"
        self.first_orientation_map_path = f"{rospack.get_path('manipulator_map')}/robots/{first_map_name}/{first_map_name}_orientation_map.csv"
        self.first_ee_df = pd.read_csv(self.first_ee_map_path)
        self.first_orientation_df = pd.read_csv(self.first_orientation_map_path)
        self.first_orientation_tree = KDTree(self.first_orientation_df[['qx', 'qy', 'qz']].to_numpy())

        self.second_map_name = second_map_name
        self.second_ee_map_path = f"{rospack.get_path('manipulator_map')}/robots/{second_map_name}/{second_map_name}_ee_map.csv"
        self.second_orientation_map_path = f"{rospack.get_path('manipulator_map')}/robots/{second_map_name}/{second_map_name}_orientation_map.csv"
        self.second_ee_df = pd.read_csv(self.second_ee_map_path)
        self.second_orientation_df = pd.read_csv(self.second_orientation_map_path)
        self.second_orientation_tree = KDTree(self.second_orientation_df[['qx', 'qy', 'qz']].to_numpy())

        self.findScalingMap()

    # how much to scale leader workspace by to get follower (assuming leader is first robot)
    def findScalingMap(self):
        first_orientations = self.first_orientation_df.to_numpy()
        second_orientations = self.second_orientation_df.to_numpy()

        # need 3 indices to store scaling information
        scaling_map = first_orientations.copy()
        scaling_map = np.hstack((scaling_map, np.zeros((len(scaling_map), 1))))

        for first_i in range(len(first_orientations)):
            _, second_i = self.second_orientation_tree.query(first_orientations[first_i, :3].reshape(1, -1), k=1)
            second_i = second_i[0][0]

            first_min = first_orientations[first_i, 3]
            first_max = first_orientations[first_i, 4]
            second_min = second_orientations[second_i, 3]
            second_max = second_orientations[second_i, 4]

            # scaling offset
            scaling_map[first_i, 3] = first_min
            scaling_map[first_i, 4] = second_min
            scaling_map[first_i, 5] = (second_max - second_min) / (first_max - first_min) # might be div by 0. Assume for now we have full  sphere workspace. If not, then adjust num bin in scale map

        self.scaling_map = scaling_map

    def scalePoints(self, points):
        center = self.first_ee_df[['x', 'y', 'z']].mean().values
        dir_vecs = points - center
        norms = np.linalg.norm(dir_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid div by zero
        directions = dir_vecs / norms

        # we can use the first robot tree for finding the scaling array indices
        _, idxs = self.first_orientation_tree.query(directions, k=1)

        scaling_params = self.scaling_map[idxs.flatten()]
        first_mins = scaling_params[:, 3:4] # need to do weird indexing to get shapes to work
        second_mins = scaling_params[:, 4:5]
        scales = scaling_params[:, 5:6]

        # do the actual scalin'
        scaled_points = (norms - first_mins) * scales + second_mins
        scaled_dir_vecs = directions * scaled_points

        return center + scaled_dir_vecs

    def forwardKinematics(self, chain, joints):
        # Create FK solver
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)

        # Compute forward kinematics
        frame = kdl.Frame()
        fk_solver.JntToCart(joints, frame)

        return frame
    
    def inverseKinematics(self, solver, target, guess):
        pos = target.p
        quat = target.M.GetQuaternion()

        solution = solver.get_ik([*guess], *pos, *quat)
        return solution
    
    def arrayToKdlJoints(self, joint_array):
        n_joints = len(joint_array)
        jnt_kdl = kdl.JntArray(n_joints)
        for i in range(n_joints):
            jnt_kdl[i] = joint_array[i]
        return jnt_kdl

    def slicePoints(self, points, axis='y', value=0.0, thickness=0.02):
        if axis == 'x':
            idx = 0
        elif axis == 'y':
            idx = 1
        elif axis == 'z':
            idx = 2
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        # Filter points within slice
        mask = np.abs(points[:, idx] - value) <= thickness
        return points[mask]

    def visualize(self, mode=COMBINED_POINTS, axis='y', slice_loc=0.0, thickness=0.02, shared_dist_thresh=0.05, concave_radius=0.08, scaling=True):
        first_points = self.first_ee_df[['x', 'y', 'z']].to_numpy()
        second_points = self.second_ee_df[['x', 'y', 'z']].to_numpy()

        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # only want crosssection
        first_points = self.slicePoints(first_points, axis, slice_loc, thickness)
        second_points = self.slicePoints(second_points, axis, slice_loc, thickness)

        if scaling:
            first_points = self.scalePoints(first_points)

        first_tree = KDTree(first_points)
        second_tree = KDTree(second_points)

        # robot 1
        distSecondtoFirst, _ = second_tree.query(first_points, k=1)
        first_shared_mask = (distSecondtoFirst[:, 0] <= shared_dist_thresh)
        first_shared_points = first_points[first_shared_mask]
        first_unique_points = first_points[~first_shared_mask]

        # robot 2
        distSecondtoFirst, _ = first_tree.query(second_points, k=1)
        second_shared_mask = (distSecondtoFirst[:, 0] <= shared_dist_thresh)
        second_shared_points = second_points[second_shared_mask]
        second_unique_points = second_points[~second_shared_mask]

        shared_points = np.vstack((first_shared_points, second_shared_points))

        if mode == COMBINED_POINTS:
            self.plotConcaveApprox(ax, shared_points, 'deepskyblue', concave_radius)
        elif mode == UNIQUE_FIRST_POINTS:
            self.plotConcaveApprox(ax, first_unique_points, 'deepskyblue', concave_radius)
        elif mode == UNIQUE_SECOND_POINTS:
            self.plotConcaveApprox(ax, second_unique_points, 'orangered', concave_radius)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title('Approximate Concave Robot Workspace (no trimesh, no alphashape)')
        ax.grid(False)
        plt.tight_layout()
        plt.show()

    # got directly from gpt dont know if its right but works good enough
    def plotConcaveApprox(self, ax, points, color, radius_threshold):
        tri = Delaunay(points)

        faces = []
        for simplex in tri.simplices:
            verts = points[simplex]

            # Calculate max edge length
            edges = [
                np.linalg.norm(verts[i] - verts[j])
                for i in range(4) for j in range(i+1, 4)
            ]
            max_edge = max(edges)

            # Keep small tetrahedra (local tight triangles)
            if max_edge < radius_threshold:
                faces.append([verts[0], verts[1], verts[2]])
                faces.append([verts[0], verts[1], verts[3]])
                faces.append([verts[0], verts[2], verts[3]])
                faces.append([verts[1], verts[2], verts[3]])

        if faces:
            mesh = Poly3DCollection(faces, alpha=0.1)
            mesh.set_facecolor(color)
            mesh.set_edgecolor('k')
            ax.add_collection3d(mesh)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        rospy.logerr("Usage: rosrun manipulator_visualizer workspace_visualizer.py first_map_name(eg: panda) second_map_name")
        sys.exit(1)

    first_map_name = sys.argv[1]
    second_map_name = sys.argv[2]

    try:
        visualizer = WorkspaceVisualizer(first_map_name, second_map_name)
        visualizer.visualize(mode=UNIQUE_SECOND_POINTS, axis='x', scaling=True)
    except rospy.ROSInterruptException:
        pass