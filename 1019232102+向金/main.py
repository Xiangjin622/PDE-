import numpy as np
import matplotlib.pyplot as plt
from Mesh import Mesh
from BoundaryCondition import BoundaryCondition
from FiniteDifferenceSolver import FiniteDifferenceSolver
from FiniteElementSolver import FiniteElementSolver
from SpectralSolver import SpectralSolver

def main():
    # 网格设置
    num_nodes = 50  # 网格节点数
    node_positions = np.linspace(0, 1, num_nodes)  # 一维网格
    mesh = Mesh(num_nodes, node_positions)
    
    # 边界条件
    boundary_conditions = {
        'left': 0.0,   # 左边界温度
        'right': 1.0   # 右边界温度
    }

    # 初始条件
    mesh.initial_condition = lambda: np.zeros(num_nodes)  # 初始温度场为0

    # 创建边界条件对象
    bc_left = BoundaryCondition(type='Dirichlet', value=boundary_conditions['left'])
    bc_right = BoundaryCondition(type='Dirichlet', value=boundary_conditions['right'])

    # 设置时间步长
    dx = (1.0 - 0.0) / (num_nodes - 1)  # 网格间距
    dt = 0.01  # 时间步长
    time_max = 1.0  # 最大时间

    # 使用有限差分法求解
    fdm_solver = FiniteDifferenceSolver(mesh, boundary_conditions, dx, dt, method='explicit')
    result_fdm = fdm_solver.solve()

    # 使用有限元法求解（此处仅为示例，实际FEM求解需要实现具体的单元组装等细节）
    fem_solver = FiniteElementSolver(mesh, boundary_conditions, element_type='linear', material_properties=None)
    result_fem = fem_solver.solve()

    # 使用谱方法求解
    spectral_solver = SpectralSolver(mesh, boundary_conditions, basis_function='fourier')
    result_spectral = spectral_solver.solve()

    # 绘制结果
    plt.figure(figsize=(10, 6))

    plt.plot(mesh.node_positions, result_fdm, label='Finite Difference Method', linestyle='-', marker='o')
    plt.plot(mesh.node_positions, result_fem, label='Finite Element Method', linestyle='--', marker='x')
    plt.plot(mesh.node_positions, result_spectral, label='Spectral Method', linestyle='-.', marker='s')

    plt.title("PDE Solving Methods Comparison (Heat Equation)")
    plt.xlabel("Position (x)")
    plt.ylabel("Temperature (u)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
