#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import kdl_parser_py.urdf
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from trac_ik_python.trac_ik import IK

# for the map stuff
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class LeaderFollowerSync:
    def __init__(self):
        rospy.init_node('leader_follower_sync_node')

        self.leader_pub = rospy.Publisher('leader/joint_states', JointState, queue_size=10)
        self.follower_pub = rospy.Publisher('follower/joint_states', JointState, queue_size=10)

        self.setupChains()

        self.follower_joints = kdl.JntArray(self.follower_chain.getNrOfJoints())
        self.leader_joints = kdl.JntArray(self.leader_chain.getNrOfJoints())
        self.leader_joints[3] = -1.57
        self.leader_joints[5] = -1.57

        self.granularity = 1

    def constructJointState(self, infos, position):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()

        joint_state.name = [info["name"] for info in infos]
        joint_state.position = [*position]
        return joint_state

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

        #TODO: find an automatic way to extract these
        ik_solver = IK(base, ee, urdf_string=urdf, timeout=0.05, solve_type="Speed")

        chain = tree.getChain(base, ee)

        joint_infos = self.getJointInfos(robot)

        return chain, ik_solver, joint_infos

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

    
    def setupChains(self):
        self.leader_chain, self.leader_ik_solver, self.leader_joint_infos = self.getChain("leader/robot_description", "base_link", "panda_hand")
        self.follower_chain, self.follower_ik_solver, self.follower_joint_infos = self.getChain("follower/robot_description", "base_link", "end_effector_link")


    def sync(self):
        # rate = rospy.Rate(1000)  # 1 KHz
        rate = rospy.Rate(1)  # 1 KHz

        # wait for subscriber to get some data
        rospy.sleep(0.5)


        # follower_df = self.createMap(self.follower_chain, self.follower_joint_infos, "kinova_map")
        # self.createMap(self.leader_chain, self.leader_joint_infos, "panda_map")

        follower_df = pd.read_csv('kinova_map.csv')
        # leader_df = pd.read_csv('panda_map.csv')

        # follower_frame = self.forwardKinematics(self.follower_chain, self.follower_joints)
        leader_frame = self.forwardKinematics(self.leader_chain, self.leader_joints)

        query = [*leader_frame.p, *leader_frame.M.GetQuaternion()]
        nearest = self.kNearestInMap(query, follower_df, 20)
        pose_features = follower_df[[info["name"] for info in self.follower_joint_infos]]

        nearest = np.array([row for row in pose_features.to_numpy()])
        print(nearest)

        # # target = kdl.Frame(leader_frame.M, leader_frame.p)
        # self.follower_joints = self.inverseKinematics(self.follower_ik_solver, leader_frame, self.follower_joints)
        # self.leader_joints = self.inverseKinematics(self.leader_ik_solver, leader_frame, self.leader_joints)

        # # print(new_follower_joints, new_leader_joints)

        i = 0
        while not rospy.is_shutdown():
            curr = nearest[i % 20]
            print(curr)
            self.follower_joints = self.inverseKinematics(self.follower_ik_solver, leader_frame, curr)
            # new_joint = [curr[info["name"]] for info in self.follower_joint_infos]
            # print(new_joint)
            new_follower_joint_state = self.constructJointState(self.follower_joint_infos, self.follower_joints)
            self.follower_pub.publish(new_follower_joint_state)
            new_leader_joint_state = self.constructJointState(self.leader_joint_infos, self.leader_joints)
            self.leader_pub.publish(new_leader_joint_state)
            # if self.follower_joints is not None:
            #     new_follower_joint_state = self.constructJointState(self.follower_joint_names, self.follower_joints)
            #     self.follower_pub.publish(new_follower_joint_state)
            # if self.leader_joints is not None:
            #     new_leader_joint_state = self.constructJointState(self.leader_joint_names, self.leader_joints)
            #     self.leader_pub.publish(new_leader_joint_state)
            i += 1
            rate.sleep()

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

    # geodesic distances to measure quaternions
    def quaternionDistance(self, q1, q2):
        dot_product = np.abs(np.dot(q1, q2))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return 2 * np.arccos(dot_product)

    def jointDistance(self, p1, p2, pos_weight=1.0, rot_weight=1.0):
        pos_dist = np.linalg.norm(p1[:3] - p2[:3])
        
        rot_dist = self.quaternionDistance(p1[3:], p2[3:])

        # combine pos and angle distance
        return pos_weight * pos_dist + rot_weight * rot_dist

    def createMap(self, chain, joint_infos, map_name):
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

        df.to_csv(f"{map_name}.csv", index=False)
        rospy.loginfo(f"Saved map of size {len(df)} to {map_name}.csv")

        return df

    def kNearestInMap(self, query, df, k):
        pose_features = df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]

        distances = np.array([self.jointDistance(query, row) for row in pose_features.to_numpy()])
        nearest_indices = np.argsort(distances)[:k]
        nearest_poses = df.iloc[nearest_indices]

        return nearest_poses


if __name__ == '__main__':
    try:
        syncer = LeaderFollowerSync()
        syncer.sync()
    except rospy.ROSInterruptException:
        pass