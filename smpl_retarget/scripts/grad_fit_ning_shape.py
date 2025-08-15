import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from phc.utils.torch_ning_humanoid_batch import Humanoid_Batch, NING_ROTATION_AXIS

from smplx import SMPL
np.random.seed(0)
torch.manual_seed(0)

# h1_joint_names = ['pelvis',
#  'left_hip_yaw_link',
#  'left_hip_pitch_link',
#  'left_hip_roll_link',
#  'left_knee_link',
#  'left_ankle_pitch_link',
#  'left_ankle_roll_link',
#  'right_hip_yaw_link',
#  'right_hip_pitch_link',
#  'right_hip_roll_link',
#  'right_knee_link',
#  'right_ankle_pitch_link',
#  'right_ankle_roll_link',
#  'torso_link',
#  'left_shoulder_pitch_link',
#  'left_shoulder_roll_link',
#  'left_shoulder_yaw_link',
#  'left_elbow_pitch_link',
#  'left_elbow_roll_link',
#  'left_wrist_pitch_link',
#  'left_wrist_yaw_link',
#  'right_shoulder_pitch_link',
#  'right_shoulder_roll_link',
#  'right_shoulder_yaw_link',
#  'right_elbow_pitch_link',
#  'right_elbow_roll_link',
#  'right_wrist_pitch_link',
#  'right_wrist_yaw_link']

# ning_joint_names = ['base_link', 
#                     'L_arm_shoulder_pitch_Link',  
#                     'L_arm_shoulder_roll_Link',  
#                     'L_arm_shoulder_yaw_Link',  
#                     'L_arm_elbow_Link',  
#                     'L_leg_hip_yaw_Link',  
#                     'L_leg_hip_roll_Link', 
#                     'L_leg_hip_pitch_Link',  
#                     'L_leg_knee_Link',  
#                     'L_leg_ankle_Link',  
#                     'R_arm_shoulder_pitch_Link',  
#                     'R_arm_shoulder_roll_Link',  
#                     'R_arm_shoulder_yaw_Link',  
#                     'R_arm_elbow_Link',
#                     'R_leg_hip_yaw_Link',  
#                     'R_leg_hip_roll_Link',      
#                     'R_leg_hip_pitch_Link',      
#                     'R_leg_knee_Link',  
#                     'R_leg_ankle_Link',  
#                     'L_arm_hand_Link',
#                     'R_arm_hand_Link', 
#                     'Head_Link']
ning_joint_names = ['waist_link', 'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll', 'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll', 'waist_roll', 'waist_pitch', 'waist_yaw', 'right_arm_pitch1', 'right_arm_roll', 'right_arm_yaw', 'right_arm_pitch2', 'left_arm_pitch1', 'left_arm_roll', 'left_arm_yaw', 'left_arm_pitch2', ]

h1_fk = Humanoid_Batch(mjcf_file = f"resources/robots/Mia/mjcf/Mia.xml",extend_head = True, extend_hand=True) # load forward kinematics model
#### Define corresonpdances between h1 and smpl joints
ning_joint_names_augment = ning_joint_names
# ning_joint_pick = ["base_link","L_arm_shoulder_roll_Link", "L_arm_elbow_Link", "L_arm_hand_Link", 'L_leg_hip_pitch_Link', "L_leg_knee_Link", "L_leg_ankle_Link","R_arm_shoulder_roll_Link", "R_arm_elbow_Link", "R_arm_hand_Link", 'R_leg_hip_pitch_Link', 'R_leg_knee_Link', 'R_leg_ankle_Link', "Head_Link"]
ning_joint_pick = ["waist_link","left_arm_roll", "left_arm_pitch2","left_hand_link",'left_hip_yaw', 'left_knee', 'left_ankle_roll', "right_arm_roll", "right_arm_pitch2","right_hand_link", 'right_hip_pitch', "right_knee", "right_ankle_roll","head_link"]
ning_joint_names_augment = ning_joint_names + ["left_hand_link", "right_hand_link", "head_link"]
smpl_joint_pick = ["Pelvis","L_Shoulder", "L_Elbow", "L_Hand", "L_Hip",  "L_Knee", "L_Ankle", "R_Shoulder", "R_Elbow", "R_Hand", "R_Hip", "R_Knee", "R_Ankle", "Head"]
ning_joint_pick_idx = [ ning_joint_names_augment.index(j) for j in ning_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]


#### Preparing fitting varialbes
device = torch.device("cpu")
pose_aa_h1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 21, axis = 2), 1, axis = 1)
pose_aa_h1 = torch.from_numpy(pose_aa_h1).float()

dof_pos = torch.zeros((1, 23))
pose_aa_h1 = torch.cat([torch.zeros((1, 1, 3)), NING_ROTATION_AXIS * dof_pos[..., None], torch.zeros((1, 2, 3))], axis = 1)


root_trans = torch.zeros((1, 1, 3))    

###### prepare SMPL default pause for H1
pose_aa_stand = np.zeros((1, 72))   
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand[:, :3] = rotvec
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0],  degrees = False).as_rotvec()
# pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0],  degrees = False).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

###### Shape fitting
trans = torch.zeros([1, 3])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)
offset = joints[:, 0] - trans
root_trans_offset = trans + offset

fk_return = h1_fk.fk_batch(pose_aa_h1[None, ], root_trans_offset[None, 0:1])

# import ipdb; ipdb.set_trace()
shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(torch.tensor([1.0]).to(device), requires_grad=True)
optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.5)

for iteration in range(1000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, 0]
    import ipdb; ipdb.set_trace()
    joints = (joints - joints[:, 0]) * scale + root_pos
    diff = fk_return.global_translation_extend[:, :, ning_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
    loss_g = diff.norm(dim = -1).mean() 
    loss = loss_g
    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000)
        # print(root_pos)
        # print(fk_return.global_translation_extend[:, :, [0,15,21]])
        # print(joints[:, [0,7,15]])

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()

# import ipdb; ipdb.set_trace()
joblib.dump(shape_new.detach(), "../mink_retarget/shape_optimized_N2.pkl") # V2 has hip jointsrea  