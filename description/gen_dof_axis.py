import xml.etree.ElementTree as ET
import numpy as np

mjcf_path = "/home/cheng/Downloads/PBHC/description/robots/taihu/mjcf/mjmodel.xml"
output_path = "/home/cheng/Downloads/PBHC/description/robots/taihu/dof_axis.npy"

tree = ET.parse(mjcf_path)
root = tree.getroot()

# MuJoCo joints are usually under <worldbody>/<body>/<joint>
joint_axes = []
joint_names = []

for joint in root.findall(".//joint"):
    axis_str = joint.get("axis")
    name = joint.get("name")
    if axis_str is not None:
        axis = [float(x) for x in axis_str.strip().split()]
        joint_axes.append(axis)
        joint_names.append(name)

dof_axis = np.array(joint_axes)
np.save(output_path, dof_axis)
print(f"Saved dof_axis.npy with shape {dof_axis.shape} to {output_path}")