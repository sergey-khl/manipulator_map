#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import kdl_parser_py.urdf
import PyKDL as kdl
from trac_ik_python.trac_ik import IK

class LeaderFollowerSync:
    def __init__(self):
        rospy.init_node('leader_follower_sync_node')

        self.leader_pub = rospy.Publisher('leader/joint_states', JointState, queue_size=10)
        self.follower_pub = rospy.Publisher('follower/joint_states', JointState, queue_size=10)

        self.setupChains()

        self.follower_joints = kdl.JntArray(self.follower_chain.getNrOfJoints())
        self.leader_joints = kdl.JntArray(self.leader_chain.getNrOfJoints())
        self.leader_joints[3] = -1.57

    def constructJointState(self, name, position):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()

        joint_state.name = name
        joint_state.position = [*position]
        return joint_state

    def forwardKinematics(self, chain, joints):
        # Create FK solver
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)

        # Compute forward kinematics
        frame = kdl.Frame()
        fk_solver.JntToCart(joints, frame)

        print("End-effector position:", frame.p)
        print("End-effector rotation:", frame.M)

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

        if not ok:
            rospy.logerr("Failed to parse URDF into KDL tree.")
            return None

        #TODO: find an automatic way to extract these
        ik_solver = IK(base, ee, urdf_string=urdf, timeout=0.05, solve_type="Speed")

        chain = tree.getChain(base, ee)

        return chain, ik_solver
    
    def getNames(self, chain):
        joint_names = []
        for idx in range(chain.getNrOfSegments()):
            segment = chain.getSegment(idx)
            joint = segment.getJoint()

            if joint.getType() == 0:
                joint_names.append(joint.getName())

        return joint_names
    
    def setupChains(self):
        self.leader_chain, self.leader_ik_solver = self.getChain("leader/robot_description", "base_link", "panda_hand")
        self.follower_chain, self.follower_ik_solver = self.getChain("follower/robot_description", "base_link", "end_effector_link")

        self.leader_joint_names = self.getNames(self.leader_chain)
        self.follower_joint_names = self.getNames(self.follower_chain)

    def sync(self):
        rate = rospy.Rate(1000)  # 1 KHz

        # wait for subscriber to get some data
        rospy.sleep(0.5)
        follower_frame = self.forwardKinematics(self.follower_chain, self.follower_joints)
        leader_frame = self.forwardKinematics(self.leader_chain, self.leader_joints)

        # target = kdl.Frame(leader_frame.M, leader_frame.p)
        self.follower_joints = self.inverseKinematics(self.follower_ik_solver, leader_frame, self.follower_joints)
        self.leader_joints = self.inverseKinematics(self.leader_ik_solver, leader_frame, self.leader_joints)

        # print(new_follower_joints, new_leader_joints)

        while not rospy.is_shutdown():
            if self.follower_joints is not None:
                new_follower_joint_state = self.constructJointState(self.follower_joint_names, self.follower_joints)
                self.follower_pub.publish(new_follower_joint_state)
            if self.leader_joints is not None:
                new_leader_joint_state = self.constructJointState(self.leader_joint_names, self.leader_joints)
                self.leader_pub.publish(new_leader_joint_state)
            rate.sleep()



if __name__ == '__main__':
    try:
        syncer = LeaderFollowerSync()
        syncer.sync()
    except rospy.ROSInterruptException:
        pass