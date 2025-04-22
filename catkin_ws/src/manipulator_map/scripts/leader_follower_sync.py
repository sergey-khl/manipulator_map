#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import kdl_parser_py.urdf
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from trac_ik_python.trac_ik import IK

# for the map stuff
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree

class KNearestSearch:
    def __init__(self, df, pos_weight, rot_weight, prune, k):
        self.df = df
        self.positions = self.df[['x','y','z']].to_numpy()  # shape (N,3)
        self.quats     = self.df[['qx','qy','qz','qw']].to_numpy()  # shape (N,4)
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.prune = prune
        self.k = k

        self.pos_tree = KDTree(self.positions, leaf_size=40)

    # vectorized geodesic distances to measure quaternions
    # q2s will be the candidate quaternions we want to compare with respect to q1 (query)
    def _quaternionDistance(self, q1, q2s):
        dots = np.clip(np.abs(q2s.dot(q1)), -1.0, 1.0)
        return 2.0 * np.arccos(dots)

    def kNearestInMap(self, query):
        qpos = np.array(query[:3]).reshape(1, -1)
        qquat = np.array(query[3:])
        
        # use the positions to prune. note that m >> k
        M = self.prune * self.k
        dist_pos, idxs = self.pos_tree.query(qpos, k=M)
        idxs = idxs[0]
        dist_pos = dist_pos[0]
        
        # find the quaternion distance with respect to our query
        cand_quats = self.quats[idxs]
        dist_rot = self._quaternionDistance(qquat, cand_quats)
        
        # combining M positions and M rotations
        combined = self.pos_weight * dist_pos + self.rot_weight * dist_rot
        best = np.argpartition(combined, self.k)[:self.k] # returns k smallest but not necessarily sorted
        best_idxs = idxs[best]
        
        return self.df.iloc[best_idxs]


class LeaderFollowerSync:
    def __init__(self, k=20, pos_weight=1, rot_weight=1, prune=5):
        rospy.init_node('leader_follower_sync_node')

        self.leader_pub = rospy.Publisher('leader/joint_states', JointState, queue_size=10)
        self.follower_pub = rospy.Publisher('follower/joint_states', JointState, queue_size=10)

        self.setupChains()

        self.follower_joints = kdl.JntArray(self.follower_chain.getNrOfJoints())
        self.leader_joints = kdl.JntArray(self.leader_chain.getNrOfJoints())
        self.leader_joints[3] = -1.57
        self.leader_joints[5] = -1.57

        rospack = rospkg.RosPack()
        self.follower_map_path = f"{rospack.get_path('manipulator_map')}/robots/{self.follower_robot_name}/{self.follower_robot_name}.csv"
        self.leader_map_path = f"{rospack.get_path('manipulator_map')}/robots/{self.leader_robot_name}/{self.leader_robot_name}.csv"

        self.follower_df = pd.read_csv(self.follower_map_path)
        self.leader_df = pd.read_csv(self.leader_map_path)

        self.follower_ksearch = KNearestSearch(self.follower_df, pos_weight, rot_weight, prune, k)
        self.leader_ksearch = KNearestSearch(self.leader_df, pos_weight, rot_weight, prune, k)


        # the joint(s) we care about matching the follower to the leader.
        # higher the number for the joint index the more we care.
        # needs to be tweeked for each robot so TODO: make more generalizable
        self.criteria_weights = np.array([0, 0, 0, 1, 1, 0, 0])

    def constructJointState(self, infos, position):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()

        joint_state.name = [info["name"] for info in infos]
        joint_state.position = [*position]
        return joint_state

    def forwardKinematics(self, chain, joints, segment=None):
        # Create FK solver
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)

        # Compute forward kinematics
        frame = kdl.Frame()
        if segment is not None:
            fk_solver.JntToCart(joints, frame, segment)
        else:
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
            
            # 0 and 0 if no lower and upper limit stated. probably need something more robust
            if joint.limit.lower == 0 and joint.limit.upper == 0:
                joint_info["lower_limit"] = -3.14
                joint_info["upper_limit"] = 3.14
            else:
                joint_info["lower_limit"] = joint.limit.lower
                joint_info["upper_limit"] = joint.limit.upper
            joint_infos.append(joint_info)

        return joint_infos
    
    def setupChains(self):
        self.leader_chain, self.leader_ik_solver, self.leader_joint_infos, self.leader_robot_name = self.getChain("leader/robot_description", "base_link", "panda_hand")
        self.follower_chain, self.follower_ik_solver, self.follower_joint_infos, self.follower_robot_name = self.getChain("follower/robot_description", "base_link", "end_effector_link")

    # find the positions of the ends of all the links not including start of base and end effector
    def _getLinksPos(self, chain, joint_infos, joints):
        positions = np.zeros((len(joint_infos), 3))
        for i in range(1, len(joint_infos) + 1):
            frame = self.forwardKinematics(chain, joints, i)
            positions[i-1] = np.array([*frame.p])

        return positions

    # assuming the follower leader relationship, TODO: probably should do this for other places as well
    def findBestNearest(self, nearest, criteria):
        values = np.array(nearest[[info["name"] for info in self.follower_joint_infos]])
        values = np.array([self._getLinksPos(self.follower_chain, self.follower_joint_infos, self.arrayToKdlJoints(val)) for val in values]) # shape of (20, 7 , 3)
        target = self._getLinksPos(self.leader_chain, self.leader_joint_infos, criteria)
        target = target[np.newaxis, :, :] # shape of (1, 7, 3)

        # squared difference
        diffs = (values - target) ** 2

        # modify errors by weights
        weighted = diffs * self.criteria_weights[np.newaxis, :, np.newaxis]

        # link (axis=1) and coord (axis=2) to get one score per sample
        scores = np.sum(weighted, axis=(1, 2))

        # choose lowest error
        best_idx = np.argmin(scores)
        # print(best_idx)
        # best_idx = 1
        best_nearest = np.array(nearest.iloc[best_idx][[info["name"] for info in self.follower_joint_infos]])
        return best_nearest

    def sync(self):
        # rate = rospy.Rate(1000)  # 1 KHz
        rate = rospy.Rate(1)  # 1 KHz

        # wait for subscriber to get some data
        rospy.sleep(0.5)
        old = rospy.Time.now()

        # follower_frame = self.forwardKinematics(self.follower_chain, self.follower_joints)
        leader_frame = self.forwardKinematics(self.leader_chain, self.leader_joints)

        query = [*leader_frame.p, *leader_frame.M.GetQuaternion()]
        nearest = self.follower_ksearch.kNearestInMap(query)

        curr = self.findBestNearest(nearest, self.leader_joints)
        print(curr)

        new = rospy.Time.now()
        print((new-old).to_sec())

        # # target = kdl.Frame(leader_frame.M, leader_frame.p)
        # self.follower_joints = self.inverseKinematics(self.follower_ik_solver, leader_frame, self.follower_joints)
        # self.leader_joints = self.inverseKinematics(self.leader_ik_solver, leader_frame, self.leader_joints)

        # # print(new_follower_joints, new_leader_joints)

        self.follower_joints = self.inverseKinematics(self.follower_ik_solver, leader_frame, curr)
        print(self.follower_joints)

        while not rospy.is_shutdown():
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
            # might not need to sleep because of how long the search already takes
            # rate.sleep()

if __name__ == '__main__':
    try:
        syncer = LeaderFollowerSync()
        syncer.sync()
    except rospy.ROSInterruptException:
        pass