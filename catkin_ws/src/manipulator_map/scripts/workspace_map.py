#!/usr/bin/env python3

import rospy
import rospkg
import sys
import kdl_parser_py.urdf
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from trac_ik_python.trac_ik import IK

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
        self.map_path = f"{rospack.get_path('manipulator_map')}/robots/{self.robot_name}/{self.robot_name}.csv"

        self.granularity = 1

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

    # arbitrary DOF find all possible joint states
    def findMapJointConfigurations(self, joint_infos, curr_idx, configuration):
        lower = joint_infos[curr_idx]['lower_limit']
        upper = joint_infos[curr_idx]['upper_limit']
        step = self.granularity
        configs = []
        angle = lower

        # go through each joint position for a joint
        while angle <= upper:
            cfg = configuration.copy()
            cfg[curr_idx] = angle

            if curr_idx == len(joint_infos) - 1:
                # we are done at the last joint
                configs.append(cfg)
            else:
                # fill in the configuration recursively
                child_configs = self.findMapJointConfigurations(
                    joint_infos, curr_idx + 1, cfg
                )
                configs.extend(child_configs)

            angle += step

        return configs

    def createMap(self, chain, joint_infos):
        configuration_map = self.findMapJointConfigurations(joint_infos, 0, [0]*len(joint_infos))

        # find the ee position and orientations for each configuration
        data = []
        for configuration in configuration_map:
            joints = self.arrayToKdlJoints(configuration)
            frame = self.forwardKinematics(chain, joints)
            pos = np.array([*frame.p])
            quat = np.array(frame.M.GetQuaternion())
            conf = np.array(configuration)
            data.append(np.concatenate([pos, quat, conf]))
        
        columns = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'] + [joint_info["name"] for joint_info in joint_infos]
        df = pd.DataFrame(data, columns=columns)

        df.to_csv(self.map_path, index=False)
        rospy.loginfo(f"Saved map of size {len(df)} to {self.robot_name}.csv")

        return df

if __name__ == '__main__':
    if len(sys.argv) != 4:
        rospy.logerr("Usage: rosrun manipulator_map workspace_map.py param_name start_link_name end_link_name")
        sys.exit(1)

    param = sys.argv[1]
    start_link = sys.argv[2]
    end_link = sys.argv[3]

    try:
        mapper = WorkspaceMap(param, start_link, end_link)
        mapper.createMap(mapper.chain, mapper.joint_infos)
    except rospy.ROSInterruptException:
        pass