#!/usr/bin/env python3

import rospy
import rospkg
import sys
import kdl_parser_py.urdf
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from trac_ik_python.trac_ik import IK
from sklearn.neighbors import KDTree

# for the map stuff
import pandas as pd
import numpy as np

class WorkspaceMap:
    def __init__(self, param, start_link, end_link):
        rospy.init_node('workspace_map_node')

        self.param = param
        self.start_link = start_link
        self.end_link = end_link

        self.setupChain()

        rospack = rospkg.RosPack()
        self.ee_map_path = f"{rospack.get_path('manipulator_map')}/robots/{self.robot_name}/{self.robot_name}_ee_map.csv"
        self.orientation_map_path = f"{rospack.get_path('manipulator_map')}/robots/{self.robot_name}/{self.robot_name}_orientation_map.csv"

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
    
    def getChain(self, param, base, ee):
        # Load and parse URDF
        if not rospy.has_param(param):
            rospy.logerr(f"Parameter '{param}' not found!")
            return
        urdf = rospy.get_param(param)
        (ok, tree) = kdl_parser_py.urdf.treeFromString(urdf)
        robot = URDF.from_parameter_server(param)

        if not ok:
            rospy.logerr("Failed to parse URDF into KDL tree.")
            return None

        # TODO: find an automatic way to extract these
        ik_solver = IK(base, ee, urdf_string=urdf, timeout=0.05, solve_type="Speed")

        chain = tree.getChain(base, ee)

        joint_infos = self.getJointInfos(robot)


        return chain, ik_solver, joint_infos, robot.name

    def arrayToKdlJoints(self, joint_array):
        n_joints = len(joint_array)
        jnt_kdl = kdl.JntArray(n_joints)
        for i in range(n_joints):
            jnt_kdl[i] = joint_array[i]
        return jnt_kdl
    
    def getJointInfos(self, robot):
        joint_infos = []

        for joint_name in robot.joint_map:
            joint_info = {}
            joint = robot.joint_map.get(joint_name)
            if joint.joint_type != "revolute" and joint.joint_type != "continuous":
                continue

            joint_info["name"] = joint_name
            
            # 0 and 0 if no lower and upper limit stated
            if joint.limit.lower == 0 and joint.limit.upper == 0:
                joint_info["lower_limit"] = -3.14
                joint_info["upper_limit"] = 3.14
            else:
                joint_info["lower_limit"] = joint.limit.lower
                joint_info["upper_limit"] = joint.limit.upper
            joint_infos.append(joint_info)

        return joint_infos
    
    def setupChain(self):
        self.chain, self.ik_solver, self.joint_infos, self.robot_name = self.getChain(self.param, self.start_link, self.end_link)

    def createEeMap(self, chain, joint_infos, n=500000):
        num_joints = len(joint_infos)

        configs = np.zeros((n, num_joints))

        # setup all joint configs
        for i, joint in enumerate(joint_infos):
            lower = joint['lower_limit']
            upper = joint['upper_limit']
            configs[:, i] = np.random.uniform(lower, upper, n)

        # remember end effector information
        data = []
        for cfg in configs:
            joints = self.arrayToKdlJoints(cfg)
            frame = self.forwardKinematics(chain, joints)
            pos = np.array([*frame.p])
            quat = np.array(frame.M.GetQuaternion())
            row = np.concatenate([pos, quat, cfg])
            data.append(row)

        columns = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] + [info['name'] for info in joint_infos]
        df = pd.DataFrame(data, columns=columns)

        df.to_csv(self.ee_map_path, index=False)
        rospy.loginfo(f"Saved map of size {len(df)} to {self.robot_name}_ee_map.csv")

        return df

    def createOrientationMap(self, df, n=10000):
        center = df[['x', 'y', 'z']].mean().values

        positions = df[['x', 'y', 'z']].to_numpy()

        # 3D angle from a point to the center
        directions = positions - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        directions /= norms

        # map of n rotations
        samples = np.random.normal(size=(n, 3))
        samples /= np.linalg.norm(samples, axis=1)[:, np.newaxis]

        tree = KDTree(samples)

        min_dist = np.full(n, np.inf)
        max_dist = np.full(n, -np.inf)

        # find max and min of each sample rotation using point cloud
        for pos, direc in zip(positions, directions):
            _, idx = tree.query(direc.reshape(1, -1), k=1)
            idx = idx[0][0]

            distance = np.linalg.norm(pos - center)

            if distance < min_dist[idx]:
                min_dist[idx] = distance
            if distance > max_dist[idx]:
                max_dist[idx] = distance

        orientation_map = pd.DataFrame({
            'qx': samples[:,0],
            'qy': samples[:,1],
            'qz': samples[:,2],
            'min_dist': min_dist,
            'max_dist': max_dist
        })

        orientation_map = orientation_map[~orientation_map.isin([np.inf, -np.inf]).any(axis=1)]
        
        orientation_map.to_csv(self.orientation_map_path, index=False)
        rospy.loginfo(f"Saved orientation map to {self.robot_name}_orientation_map.csv")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        rospy.logerr("Usage: rosrun manipulator_map workspace_map.py param_name start_link_name end_link_name")
        sys.exit(1)

    param = sys.argv[1]
    start_link = sys.argv[2]
    end_link = sys.argv[3]

    try:
        mapper = WorkspaceMap(param, start_link, end_link)
        df = mapper.createEeMap(mapper.chain, mapper.joint_infos, 500000)
        mapper.createOrientationMap(df, 1000)
    except rospy.ROSInterruptException:
        pass