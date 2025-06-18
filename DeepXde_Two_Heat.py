import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zhplot
import os  # 导入os模块用于文件和目录操作

os.environ["DDE_BACKEND"] = "pytorch"  # 关键修改：在导入deepxde前设置后端
# # 显式设置PyTorch后端（根据您的选择）
# dde.config.set_default_backend("pytorch")

# 创建data子目录
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"已创建目录: {data_dir}")

# 定义计算域
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])

# 温度归一化参数
temp_min = 0.0
temp_max = 75.0

# 温度归一化和反归一化
def normalize_temperature(temp, temp_min=temp_min, temp_max=temp_max):
    return (temp - temp_min) / (temp_max - temp_min)

def denormalize_temperature(temp_norm, temp_min=temp_min, temp_max=temp_max):
    return temp_norm * (temp_max - temp_min) + temp_min

# 定义PDE
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return dy_xx + dy_yy

# 边界条件
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], x_min)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], x_max)

def boundary_bottom(x, on_boundary):
    return on_boundary and np.isclose(x[1], y_min)

def boundary_top(x, on_boundary):
    return on_boundary and np.isclose(x[1], y_max)

# 边界条件的归一化处理
def func_left(x):
    return normalize_temperature(75.0)

def func_right(x):
    return normalize_temperature(0.0)

def func_bottom(x):
    return normalize_temperature(50.0)

def func_top(x):
    return normalize_temperature(0.0)

bc_left = dde.DirichletBC(geom, func_left, boundary_left)
bc_right = dde.DirichletBC(geom, func_right, boundary_right)
bc_bottom = dde.DirichletBC(geom, func_bottom, boundary_bottom)
bc_top = dde.DirichletBC(geom, func_top, boundary_top)

# 定义数据和模型
data = dde.data.PDE(
    geom,
    pde,
    [bc_left, bc_right, bc_bottom, bc_top],
    num_domain=2000,
    num_boundary=100,
    num_test=10000,
)

net = dde.nn.FNN([2] + [20] * 4 + [1], "tanh", "Glorot normal")

# 模型
model = dde.Model(data, net)

# 编译并训练模型
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=40000)

model.compile("L-BFGS")
losshistory, train_state = model.train()

# 保存训练好的模型
model_path = os.path.join(data_dir, "heat_equation_model")
model.save(model_path)
print(f"模型已保存到: {model_path}")

# 提取损失函数数据
losses = np.array(losshistory.loss_train)
epochs = np.array(losshistory.steps)

# 检查损失数组的维度
num_losses = losses.shape[1]
print(f"检测到 {num_losses} 个损失函数分量")

# 绘制损失函数分解图
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.semilogy(epochs, losses[:, 0], 'k-', label='总损失')
plt.xlabel('迭代次数')
plt.ylabel('损失值 (对数尺度)')
plt.title('训练过程中的损失函数变化')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogy(epochs, losses[:, 1], 'b-', label='PDE 损失')

if num_losses > 2:
    plt.semilogy(epochs, losses[:, 2], 'r-', label='左边界损失 (x=-1)')
if num_losses > 3:
    plt.semilogy(epochs, losses[:, 3], 'g-', label='右边界损失 (x=1)')
if num_losses > 4:
    plt.semilogy(epochs, losses[:, 4], 'm-', label='下边界损失 (y=-1)')
if num_losses > 5:
    plt.semilogy(epochs, losses[:, 5], 'c-', label='上边界损失 (y=1)')

plt.xlabel('迭代次数')
plt.ylabel('损失值 (对数尺度)')
plt.title('各部分损失函数变化')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
loss_plot_path = os.path.join(data_dir, "loss_function_evolution.png")
plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
plt.show()

# 保存模型
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# 评估模型
x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
points = np.array([X.flatten(), Y.flatten()]).T

# 预测温度分布
predictions_norm = model.predict(points).reshape(X.shape)

# 反归一化温度
predictions = denormalize_temperature(predictions_norm)

# 保存等高图温度分布数据到Excel
temperature_df = pd.DataFrame(predictions, index=y, columns=x)
temperature_df.index.name = 'y/x'
temperature_excel_path = os.path.join(data_dir, "temperature_contour_data.xlsx")
temperature_df.to_excel(temperature_excel_path)
print(f"等高图温度分布数据已保存到 {temperature_excel_path}")

# 绘制温度分布图
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, predictions, levels=100, cmap="jet")
plt.colorbar(label="温度")
plt.contour(X, Y, predictions, levels=20, colors="k", linewidths=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("二维稳态热传导方程温度分布")
temp_dist_plot_path = os.path.join(data_dir, "temperature_distribution.png")
plt.savefig(temp_dist_plot_path, dpi=300, bbox_inches="tight")
plt.show()

# 绘制不同y值的温度分布
plt.figure(figsize=(10, 6))
y_values = [0, 0.25, 0.5, 0.75]
colors = ['blue', 'green', 'red', 'purple']
markers = ['o', 's', '^', 'D']
data_to_excel = {}

for i, y_val in enumerate(y_values):
    y_idx = np.abs(y - y_val).argmin()
    actual_y = y[y_idx]
    temp_distribution = predictions[y_idx, :]
    data_to_excel[f'y={actual_y}'] = temp_distribution
    
    plt.plot(x, temp_distribution, color=colors[i], marker=markers[i], markevery=10, 
             linewidth=2, label=f'y={actual_y}')

plt.title('不同y位置的温度分布对比')
plt.xlabel('x 坐标')
plt.ylabel('温度')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(x_min, x_max)
plt.ylim(0, 80)
plt.legend(loc='best')
plt.tight_layout()
temp_dist_y_plot_path = os.path.join(data_dir, "temperature_distribution_at_different_y.png")
plt.savefig(temp_dist_y_plot_path, dpi=300, bbox_inches="tight")
plt.show()

# 保存特定y值的温度分布数据
df = pd.DataFrame(data_to_excel, index=x)
df.index.name = 'x'
specific_temp_excel_path = os.path.join(data_dir, "temperature_data_at_different_y.xlsx")
df.to_excel(specific_temp_excel_path)
print(f"特定y位置的温度分布数据已保存到 {specific_temp_excel_path}")