import torch
import numpy as np
import smplx
import trimesh
import pyrender
import matplotlib.pyplot as plt
import joblib
# Step 1: 加载SMPL模型
# 假设 smpl_neutral.pkl 位于路径 "./models/smpl_neutral.pkl"
model_path = "/home/cheng/Downloads/PBHC/smpl_retarget/smpl_model/smpl/SMPL_NEUTRAL.pkl"
smpl_model = smplx.create(model_path, model_type='smpl', gender='neutral', use_pca=False)

# Step 2: 设定Shape参数 (10维)
# shape_parameters 是您已有的10个shape参数
betas = joblib.load('/home/cheng/Downloads/PBHC/smpl_retarget/mink_retarget/shape_optimized_neutral_N2.pkl')
# betas = joblib.load('/home/cheng/Downloads/PBHC/smpl_retarget/retargeted_motion_data/phc/shape_optimized_v1.pkl')

# shape_parameters = np.array([0, 0, 0, 0, -0, 0, -0, 0, -0, 0])
# betas = torch.tensor(shape_parameters, dtype=torch.float32).unsqueeze(0)
print(betas)
# 通过模型生成3D顶点
output = smpl_model(betas=betas[0], return_verts=True)
vertices = output.vertices.detach().cpu().numpy().squeeze()

# Step 4: 可视化SMPL模型
# 使用 Trimesh 和 Pyrender 来渲染
mesh = trimesh.Trimesh(vertices, smpl_model.faces, process=False)
mesh.visual.vertex_colors = [200, 200, 200, 255]

# 使用Pyrender进行可视化
scene = pyrender.Scene()
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_pyrender)

viewer = pyrender.Viewer(scene, use_raymond_lighting=True)