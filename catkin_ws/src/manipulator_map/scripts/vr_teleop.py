#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
import kdl_parser_py.urdf
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from trac_ik_python.trac_ik import IK
import random

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


class VRTeleop:
    def __init__(self, k=20, pos_weight=1, rot_weight=1, prune=5):
        rospy.init_node('vr_teleop_node')

        self.follower_pub = rospy.Publisher('follower/joint_states', JointState, queue_size=10)

        self.vr_sub = rospy.Subscriber('/vr_position', Vector3, self.vrCallback)
        self.vr_pos = None
        self.new_pos = None
        self.old_pos = None
        self.scale = 0.1

        self.setupChains()

        self.follower_joints = kdl.JntArray(self.follower_chain.getNrOfJoints())
        self.follower_joints[3] = 2

        rospack = rospkg.RosPack()

        # self.follower_ee_map_path = f"{rospack.get_path('manipulator_map')}/robots/{self.follower_robot_name}/{self.follower_robot_name}_ee_map.csv"
        # self.follower_orientation_map_path = f"{rospack.get_path('manipulator_map')}/robots/{self.follower_robot_name}/{self.follower_robot_name}_orientation_map.csv"
        # self.follower_ee_df = pd.read_csv(self.follower_ee_map_path)
        # self.follower_orientation_df = pd.read_csv(self.follower_orientation_map_path)
        # self.follower_orientation_tree = KDTree(self.follower_orientation_df[['qx', 'qy', 'qz']].to_numpy())

        # self.follower_ksearch = KNearestSearch(self.follower_ee_df, pos_weight, rot_weight, prune, k)

        # self.findScalingMap()


        # # the joint(s) we care about matching the follower to the leader.
        # # higher the number for the joint index the more we care.
        # # needs to be tweeked for each robot so TODO: make more generalizable
        # self.criteria_weights = np.array([1, 1, 1, 1, 1, 1, 1])
    
    def vrCallback(self, msg):
        msg = np.array([msg.x, msg.y, msg.z])
        if np.all(msg) == 0:
            return
        msg *= self.scale
        if self.old_pos is None:
            self.old_pos = msg
            return
        if self.new_pos is None:
            self.new_pos = msg
            return
        self.old_pos = self.new_pos
        self.new_pos = msg

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
        if solution is None:
            return None
        return self.arrayToKdlJoints(solution)
    
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
        # self.leader_chain, self.leader_ik_solver, self.leader_joint_infos, self.leader_robot_name = self.getChain("leader/robot_description", "base_link", "panda_hand")
        self.follower_chain, self.follower_ik_solver, self.follower_joint_infos, self.follower_robot_name = self.getChain("follower/robot_description", "base_link", "end_effector_link")

    # find the positions of the ends of all the links not including start of base and end effector
    def _getLinksPos(self, chain, joint_infos, joints):
        positions = np.zeros((len(joint_infos), 3))
        for i in range(1, len(joint_infos) + 1):
            frame = self.forwardKinematics(chain, joints, i)
            positions[i-1] = np.array([*frame.p])

        return positions

    def angleDifference(self, a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    # assuming the follower leader relationship, TODO: probably should do this for other places as well
    # TODO: update this with scaling
    def findBestNearest(self, nearest, criteria):
        joint_names = [info["name"] for info in self.follower_joint_infos]
        values = np.array(nearest[joint_names])
        values = np.array([self._getLinksPos(self.follower_chain, self.follower_joint_infos, self.arrayToKdlJoints(val)) for val in values]) # shape of (k, DOF , 3)
        target = self._getLinksPos(self.leader_chain, self.leader_joint_infos, criteria)
        target = self.scalePoints(target)
        target = target[np.newaxis, :, :] # shape of (1, DOF, 3)

        # squared difference
        diffs = (values - target) ** 2

        # modify errors by weights
        weighted = diffs * self.criteria_weights[np.newaxis, :, np.newaxis]

        joint_diffs = self.angleDifference(np.array(nearest[joint_names]), np.array([*self.follower_joints])[np.newaxis, :]) ** 2


        # link (axis=1) and coord (axis=2) to get one score per sample
        scores = np.sum(weighted, axis=(1, 2)) + np.sum(joint_diffs, axis=1)
        # scores = np.sum(weighted, axis=(1, 2))
        # scores = np.sum(joint_diffs, axis=1)

        # choose lowest error
        best_idx = np.argmin(scores)
        best_nearest = np.array(nearest.iloc[best_idx][joint_names])
        return best_nearest


    def findNextSeed(self, leader_frame):
        query = [*leader_frame.p, *leader_frame.M.GetQuaternion()]
        nearest = self.follower_ksearch.kNearestInMap(query)

        return self.findBestNearest(nearest, self.leader_joints)

    def sync(self):
        rate = rospy.Rate(1000)  # 1 KHz

        while not rospy.is_shutdown():
            if self.new_pos is None:
                continue
            pos_update = kdl.Vector(*self.new_pos) - kdl.Vector(*self.old_pos)
            self.follower_frame = self.forwardKinematics(self.follower_chain, self.follower_joints)
            
            self.follower_frame.p = self.follower_frame.p + pos_update
            print(self.follower_frame.p, pos_update)
            # self.follower_joints = self.inverseKinematics(self.follower_ik_solver, self.follower_frame, self.follower_joints)
            self.follower_joints = self.inverseKinematics(self.follower_ik_solver, self.follower_frame, [0, 0, 0, 0, 0, 0, 0])

            # publish new joints
            if self.follower_joints is not None:
                # seed = self.follower_joints
                new_follower_joint_state = self.constructJointState(self.follower_joint_infos, self.follower_joints)
                self.follower_pub.publish(new_follower_joint_state)
            else:
                rospy.logerr("Could not find IK solution")

            rate.sleep()

if __name__ == '__main__':
    try:
        syncer = VRTeleop(k=1000)
        syncer.sync()
    except rospy.ROSInterruptException:
        pass