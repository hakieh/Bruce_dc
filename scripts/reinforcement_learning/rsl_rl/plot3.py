import matplotlib
from matplotlib import pyplot as plt
import numpy as np

path = "/home/kh/hkh/data/excle/4.27/afop_bruce_actions_history_2025-04-27_12-04-23.npy"
actions = np.load(path)*0.5
print(actions.shape
      )



x = actions[:,2]
y = actions[:,6]
z = actions[:,8]
fig = plt.figure(figsize=(10,10),dpi=600)
ax = fig.add_subplot(projection='3d')

ax.set(xlabel="hip roll (rad)", ylabel="thigh (rad)", zlabel="calf (rad)")
# ax.set_zlabel('Z')
# c颜色，marker：样式*雪花
ax.plot(xs=x, ys=y, zs=z, c="b",)
# plt.xlabel("hip (rad)")
# plt.ylabel("thigh (rad)")
# plt.zlabel("calf (rad)")
plt.savefig('/home/kh/hkh/data/excle/4.27/双足优化后.png')
plt.show()



# path = "/home/kh/hkh/data/excle/4.27/bfop_actions_history_2025-04-27_11-43-46.npy"
# actions = np.load(path)*0.25
# print(actions.shape
#       )



# x = actions[:,0]
# y = actions[:,4]
# z = actions[:,8]
# fig = plt.figure(figsize=(10,10),dpi=600)
# ax = fig.add_subplot(projection='3d')

# ax.set(xlabel="hip (rad)", ylabel="thigh (rad)", zlabel="calf (rad)")
# # ax.set_zlabel('Z')
# # c颜色，marker：样式*雪花
# ax.plot(xs=x, ys=y, zs=z, c="b",)
# # plt.xlabel("hip (rad)")
# # plt.ylabel("thigh (rad)")
# # plt.zlabel("calf (rad)")
# plt.savefig('/home/kh/hkh/data/excle/4.27/四足优化前.png')
# plt.show()
