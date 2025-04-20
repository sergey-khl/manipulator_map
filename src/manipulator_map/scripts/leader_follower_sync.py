#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import kdl_parser_py.urdf
import PyKDL as kdl

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
        joint_state.position = [j for j in position]
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
    
    def inverseKinematics(self, chain, target, guess):
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)
        ik_solver = kdl.ChainIkSolverVel_pinv(chain)
        ik_solver_pos = kdl.ChainIkSolverPos_NR(chain, fk_solver, ik_solver)

        # Output joint angles
        result = kdl.JntArray(chain.getNrOfJoints())

        ret = ik_solver_pos.CartToJnt(guess, target, result)
        print(ret)
        if ret >= 0:
            # print("Inverse Kinematics result:")
            # for i in range(result.rows()):
            #     print(f"Joint {i}: {result[i]}")
            
            return result
        else:
            print("IK solver failed")
            return None
    
    def getTree(self, param):
        # Load and parse URDF
        if not rospy.has_param(param):
            rospy.logerr(f"Parameter '{param}' not found!")
            return
        urdf = rospy.get_param(param)
        (ok, tree) = kdl_parser_py.urdf.treeFromString(urdf)

        if not ok:
            rospy.logerr("Failed to parse URDF into KDL tree.")
            return None

        return tree
    
    def getNames(self, chain):
        joint_names = []
        for idx in range(chain.getNrOfSegments()):
            segment = chain.getSegment(idx)
            joint = segment.getJoint()

            if joint.getType() == 0:
                joint_names.append(joint.getName())

        return joint_names
    
    def setupChains(self):
        leader_tree = self.getTree("leader/robot_description")
        follower_tree = self.getTree("follower/robot_description")

        #TODO: find an automatic way to extract these
        self.leader_chain = leader_tree.getChain("base_link", "panda_hand")
        self.follower_chain = follower_tree.getChain("base_link", "end_effector_link")

        self.leader_joint_names = self.getNames(self.leader_chain)
        self.follower_joint_names = self.getNames(self.follower_chain)

    def sync(self):
        rate = rospy.Rate(1000)  # 1 KHz

        # wait for subscriber to get some data
        rospy.sleep(2.5)
        follower_frame = self.forwardKinematics(self.follower_chain, self.follower_joints)
        leader_frame = self.forwardKinematics(self.leader_chain, self.leader_joints)

        # target = kdl.Frame(leader_frame.M, leader_frame.p)
        self.follower_joints = self.inverseKinematics(self.follower_chain, leader_frame, self.follower_joints)
        self.leader_joints = self.inverseKinematics(self.leader_chain, leader_frame, self.leader_joints)

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