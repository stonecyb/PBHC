import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
# from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch

import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_ning_humanoid_batch import Humanoid_Batch, NING_ROTATION_AXIS
from torch.autograd import Variable
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Path to the folder containing FBX files.")
parser.add_argument("-o", "--output", type=str, help="Path to the folder to save NPY files.")
args = parser.parse_args()


device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

ning_rotation_axis = torch.tensor([[ 
    [0, 0, 1], # l_shoulder_pitch
    [1, 0, 0], # l_roll_pitch
    [0, 0, 1], # l_yaw_pitch
    
    [0, 1, 0], # l_elbow

    [0, 0, 1], # l_hip_yaw
    [1, 0, 0], # l_hip_roll
    [0, 1, 0], # l_hip_pitch
    
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle
    
    [0, 0, 1], # r_shoulder_pitch
    [1, 0, 0], # r_roll_pitch
    [0, 0, 1], # r_yaw_pitch
    
    [0, 1, 0], # r_elbow
    
    [0, 0, 1], # r_hip_yaw
    [1, 0, 0], # r_hip_roll
    [0, 1, 0], # r_hip_pitch
    
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle

]]).to(device)
ning_joint_names =["base_link", "L_arm_shoulder_pitch_Link", "L_arm_shoulder_roll_Link" ,"L_arm_shoulder_yaw_Link",
                   "L_arm_elbow_Link","L_hand_Link","L_leg_hip_yaw_Link",  "L_leg_hip_roll_Link" ,
                   "L_leg_hip_pitch_Link", "L_leg_knee_Link", "L_leg_ankle_Link","R_arm_shoulder_pitch_Link","R_arm_shoulder_roll_Link",
                   "R_arm_shoulder_yaw_Link", "R_arm_elbow_Link", "R_hand_Link", "R_leg_hip_yaw_Link", "R_leg_hip_roll_Link",
                   "R_leg_hip_pitch_Link", "R_leg_knee_Link", "R_leg_ankle_Link","Head_Link"]

h1_joint_pick = [ "L_arm_shoulder_roll_Link", "L_arm_elbow_Link", 'L_hand_Link', "L_leg_knee_Link", "L_leg_ankle_Link",'R_arm_shoulder_roll_Link', "R_arm_elbow_Link", "R_hand_Link", "R_leg_knee_Link", "R_leg_ankle_Link"]

smpl_joint_pick = [ "L_Shoulder", "L_Elbow", "L_Hand", "L_Knee", "L_Ankle","R_Shoulder", "R_Elbow", "R_Hand", "R_Knee", "R_Ankle"]
h1_joint_pick_idx = [ ning_joint_names.index(j) for j in h1_joint_pick]
fbx_joint_names = ["LeftArm","LeftForeArm","LeftHand","LeftLeg","LeftFoot","RightArm",  "RightForeArm",  "RightHand",  "RightLeg", "RightFoot"]
smpl_joint_names = [
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine1", "LeftFoot", "RightFoot",
    "Neck", "LeftShoulder", "RightShoulder", "Head",
    "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"
]
gt_joint_pick_idx = [smpl_joint_names.index(j) for j in fbx_joint_names]
# print(gt_joint_pick_idx)
# smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

# smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
# smpl_parser_n.to(device)
amass_data = {}
# amass_data = joblib.load('data/sample_data/amass_copycat_take5_test_small.pkl') # From PHC
# for filename in os.listdir('data/final_ning/'):
#     file_path = os.path.join('data/final_ning/', filename)
#     file_key = os.path.splitext(filename)[0]
#     data = joblib.load(file_path)
#     amass_data[file_key] = data["motive_001"]
# amass_data = joblib.load('data/final_ning/step_forward_ning.pkl')

amass_data = joblib.load('data/ning/res.pkl')
# shape_new = joblib.load("data/ning/shape_optimized_v1.pkl").to(device)

h1_fk = Humanoid_Batch(mjcf_file = f"resources/robots/N2/mjcf/N2.xml", extend_hand = False, device = device)
data_dump = {}
pbar = tqdm(amass_data.keys())
def compute_angle_between_vectors(v1, v2):
    """
    计算两个向量之间的夹角，单位为弧度。
    Args:
        v1: 第一个向量 (n_frames, 3)
        v2: 第二个向量 (n_frames, 3)
    Returns:
        夹角，单位为弧度 (n_frames,)
    """
    # 计算点积
    dot_product = torch.sum(v1 * v2, dim=-1)  # (n_frames,)
    
    # 计算模长
    norm_v1 = torch.norm(v1, dim=-1)  # (n_frames,)
    norm_v2 = torch.norm(v2, dim=-1)  # (n_frames,)
    
    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # 由于数值误差，确保 cos_theta 在 [-1, 1] 范围内
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    
    # 计算角度（弧度）
    angle = torch.acos(cos_theta)  # (n_frames,)
    return angle

for data_key in pbar:

    trans = torch.from_numpy(amass_data[data_key]['pose_pos'][:,:3]).float().to(device)
    N = trans.shape[0]
    global_pos = torch.from_numpy(amass_data[data_key]['pose_pos']).float().reshape([N,-1,3]).to(device)
    # import ipdb; ipdb.set_trace()
    
    # print(global_pos[:,8,:])
    # print(trans)
#     pose_aa_walk = torch.from_numpy(np.concatenate((amass_data[data_key]['pose_aa'][:, :66], np.zeros((N, 6))), axis = -1)).float().to(device)
#     verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
#     offset = joints[:, 0] - trans
    root_trans_offset = trans

    # print(amass_data[data_key]['pose_rot'][:, :3])
#     # gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3])).as_rotvec()).float().to(device)
    # * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    rotation_matrix = sRot.from_quat(amass_data[data_key]['pose_rot'][:, :4]).as_matrix()

    # Step 2: 调整旋转矩阵以适应 YZX 坐标系
    # 重新排列行
    rotation_matrix = rotation_matrix[:,[1, 2, 0], :]
    # 重新排列列
    rotation_matrix = rotation_matrix[:, :, [1, 2, 0]]

    # Step 3: 将调整后的旋转矩阵转换回四元数
    new_rotation = sRot.from_matrix(rotation_matrix)
    new_quaternion = new_rotation.as_quat()  # 返回为 (x, y, z, w)
    gt_root_rot = torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:, :4])* sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device) 
    # * torch.tensor([0,0,1]).to(device)
    # import ipdb; ipdb.set_trace()
    dof_pos = torch.zeros((1, N, 18, 1)).to(device)
    # print(gt_root_rot)
    # dof_pos_new = Variable(dof_pos.to(device), requires_grad=True)
    # R_A_to_B = np.array([
    # [0,      -1 , 0,  ],
    # [-0.143, 0, 1],
    # [-0.157, 0, 0]
    # ])
    # q_A = amass_data[data_key]['pose_rot'].reshape([N, -1, 4])[:, 13, :] 
    # R_A = sRot.from_quat(q_A).as_matrix()  # 转换为旋转矩阵

    # # 新的旋转矩阵 (在B坐标系)
    # R_B = R_A_to_B @ R_A @ R_A_to_B.T

    # # 转回四元数（可选）
    # q_B = sRot.from_matrix(R_B).as_euler('zyx')
    # 2025.3.19 ning（N2）跑步动作调整
    dof_pos[0,:,:3,0] = -torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,13*4:14*4])).as_euler('zxy')).float().to(device)
    dof_pos[0,:,9:12,0] = torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,14*4:15*4])).as_euler('zxy')).float().to(device)
    dof_pos[0,:,[11],0] = - dof_pos[0,:,[11],0]

    # 2025.3.19舞蹈动作调整
    # dof_pos[0,:,:3,0] = torch.from_numpy(sRot.from_quat(amass_data[data_key]['pose_rot'][:,13*4:14*4]).as_euler('xzy')).float().to(device)
    # dof_pos[0,:,[2],0] = -dof_pos[0,:,[2],0]
    # dof_pos[0,:,9:12,0] = torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,14*4:15*4])).as_euler('xzy')).float().to(device)
    # dof_pos[0,:,[11],0] = - dof_pos[0,:,[11],0]

    # import ipdb; ipdb.set_trace()
    # print(torch.rad2deg(dof_pos[0,:,:3,0]))
    # dof_pos[0,:,[1],0] = -dof_pos[0,:,[1],0]
    # 计算每一帧的旋转角度（弧度）
    angles_left_elbow = compute_angle_between_vectors(global_pos[:, 15].reshape([-1,3]) - global_pos[:, 13].reshape([-1,3]), global_pos[:, 17].reshape([-1,3])   - global_pos[:, 15].reshape([-1,3]))
    angles_right_elbow = compute_angle_between_vectors(global_pos[:, 16].reshape([-1,3]) - global_pos[:, 14].reshape([-1,3]), global_pos[:, 18].reshape([-1,3])   - global_pos[:, 16].reshape([-1,3]))
    
    # dof_pos[0,:,3,0] = -angles_left_elbow
    # dof_pos[0,:,12,0] = - angles_right_elbow
    
    #import ipdb; ipdb.set_trace()
    dof_pos[0,:,3,0] = - torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,15*4:16*4])).as_euler('zyx')[:,0]).float().to(device)
    dof_pos[0,:,12,0] = torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,16*4:17*4])).as_euler('zyx')[:,0]).float().to(device)
    
    dof_pos[0,:,4:7,0] = torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,4:8])).as_euler('zyx')).float().to(device)
    dof_pos[0,:,7,0] = - torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,16:20])).as_euler('xyz')[:,0]).float().to(device)
    dof_pos[0,:,8,0] = - torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,7*4:8*4])).as_euler('xyz')[:,0]).float().to(device)

    # import ipdb; ipdb.set_trace()
    dof_pos[0,:,13:16,0] = torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,2*4:3*4])).as_euler('zyx')).float().to(device)
    # import ipdb; ipdb.set_trace()
    dof_pos[0,:,15,0] = -dof_pos[0,:,15,0]
    dof_pos[0,:,6,0] = -dof_pos[0,:,6,0]

    # dof_pos[0,:,13,0] = 0
    # dof_pos[0,:,4,0] = 0

    # print(torch.rad2deg(dof_pos[0,552,4,0]))
    # print(torch.from_numpy((sRot.from_rotvec(amass_data[data_key]['pose_rot'][:,4*3:5*3])).as_euler('zyx')).float().to(device))
    angles_left_knee = compute_angle_between_vectors(global_pos[:, 4].reshape([-1,3]) - global_pos[:, 1].reshape([-1,3]), global_pos[:, 7].reshape([-1,3])   - global_pos[:, 4].reshape([-1,3]))
    # dof_pos[0,:,7,0] = angles_left_knee
    angles_right_knee = compute_angle_between_vectors(global_pos[:, 5].reshape([-1,3]) - global_pos[:, 2].reshape([-1,3]), global_pos[:, 8].reshape([-1,3])   - global_pos[:, 5].reshape([-1,3]))
    # dof_pos[0,:,7,0] = angles_left_knee
    # dof_pos[0,:,16,0] = angles_right_knee
    dof_pos[0,:,16,0] = - torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,5*4:6*4])).as_euler('xyz')[:,0]).float().to(device)
    dof_pos[0,:,17,0] = - torch.from_numpy((sRot.from_quat(amass_data[data_key]['pose_rot'][:,8*4:9*4])).as_euler('xyz')[:,0]).float().to(device)
    # dof_pos[0,:,8,0] = torch.from_numpy((sRot.from_rotvec(amass_data[data_key]['pose_rot'][:,7*3:8*3])).as_euler('zyx')).float().to(device)[:,2]
    # dof_pos[0,:,17,0] = torch.from_numpy((sRot.from_rotvec(amass_data[data_key]['pose_rot'][:,8*3:9*3])).as_euler('zyx')).float().to(device)[:,2]
    # dof_pos[0,:,13:16,0] = -dof_pos[0,:,13:16,0] 
    # print(dof_pos[0,:,4:7,0])
    # dof_pos[0,:,5,0] = -dof_pos[0,:,5,0]
    # # dof_pos[0,:,7,0] = -torch.from_numpy(sRot.from_rotvec(amass_data[data_key]['pose_rot'].reshape([N,-1,3])[:,4,:]).as_euler('yzx')[:,1]).float().to(device)
    # root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
    # root_pos_offset = Variable(torch.zeros(1, 3).to(device), requires_grad=True)

    # optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)
    # optimizer_root = torch.optim.Adam([root_rot_new, root_pos_offset],lr=0.1)

    # kernel_size = 5  # Size of the Gaussian kernel
    # sigma = 0.75  # Standard deviation of the Gaussian kernel
    # B, T, J, D = dof_pos_new.shape    

#     for iteration in range(500):
# #         verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
#         pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], ning_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
#         fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
#         # print(global_pos.shape)
#         diff = fk_return.global_translation_extend[:, :, h1_joint_pick_idx] - global_pos[:,:,gt_joint_pick_idx]
#         loss_g = diff.norm(dim = -1).mean() 
#         loss = loss_g
        
#         if iteration % 50 == 0:
#             pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

#         optimizer_pose.zero_grad()
#         # optimizer_root.zero_grad()
#         optimizer_pose.zero_grad()
#         # optimizer_root.zero_grad()
#         optimizer_pose.zero_grad()
#         # optimizer_root.zero_grad()
#         loss.backward()
#         optimizer_pose.step()
#         optimizer_root.step()

#         # dof_pos_new.data.clamp_(h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None])
#         # print(dof_pos_new) 
#         dof_pos_new.data = gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None, ], kernel_size, sigma).transpose(2, 1)[..., None]

    # dof_pos_new.data.clamp_(h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None])
    # print(h1_fk.joints_range)
    # # 调整根节点旋转到站立方向
    # upright_quat = sRot.from_euler('xyz', [0, 0, 0]).as_quat()
    # current_root_quat = sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat()
    # adjustment_quat = sRot.from_quat(current_root_quat) * sRot.from_quat(upright_quat).inv()
    # adjusted_root_rot = (sRot.from_rotvec(gt_root_rot.cpu().numpy()) * adjustment_quat).as_rotvec()
    # gt_root_rot = torch.from_numpy(adjusted_root_rot).float().to(device)

    # 重新生成站立姿态
    pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], ning_rotation_axis * dof_pos, torch.zeros((1, N, 2, 3)).to(device)], axis=2)
    root_trans_offset_dump = (root_trans_offset).clone()
    # import ipdb; ipdb.set_trace()
    fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
    root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08
    # print(pose_aa_h1_new)
    # print(gt_root_rot)
    data_dump[data_key]={
            "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().cpu().detach().numpy(), 
            "dof": dof_pos.squeeze().detach().cpu().numpy(), 
            "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
            "fps": 180,
            }
    # joblib.dump(data_dump,f"data/ning/{data_key}.pkl")
# import ipdb; ipdb.set_trace()
joblib.dump(data_dump,"data/ning/res_n2.pkl")