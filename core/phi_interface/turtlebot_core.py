# -*- coding: utf-8 -*-
"""
EICPS Core Logic Library
对应论文中的 Φ (Phi) 算子实现
"""

import numpy as np
import osqp
from scipy import sparse

class EICPS_Interface:
    """
    EICPS 具身接口核心类
    实现 HNN -> CBF-QP -> PDT 的级联处理
    """
    
    def __init__(self):
        # --- 物理参数 (Process P) ---
        self.v_max = 0.26       # m/s, TurtleBot3 Waffle Pi
        self.w_max = 1.82       # rad/s
        self.d_min = 0.20       # m, 安全距离阈值
        self.robot_radius = 0.15 # m
        
        # --- 控制参数 ---
        self.alpha = 1.0        # CBF 衰减系数 (h >= 0 时生效)
        self.beta = 0.5         # PDT 恢复速率 (h < 0 时生效)
        
        # --- QP求解器初始化 (OSQP) ---
        self.prob = osqp.OSQP()
        self.P_base = sparse.csc_matrix(np.eye(2)) # H = I
        self.is_setup = False
        
        self.last_u_safe = np.zeros(2)

    def phi_hnn_projection(self, u_ai):
        """
        Φ1: HNN 动力学一致性投影 (简化版: 限幅)
        """
        v_dyn = np.clip(u_ai[0], -self.v_max, self.v_max)
        w_dyn = np.clip(u_ai[1], -self.w_max, self.w_max)
        return np.array([v_dyn, w_dyn])

    def solve_safety_filter(self, x, u_dyn, scan_min_dist):
        """
        Φ2 + Φ3: CBF-QP 安全过滤 + PDT 时间恢复
        """
        h_val = scan_min_dist - self.d_min
        
        # --- 约束构建 ---
        if h_val >= 0:
            # [Case 1: 安全区 - Φ2 CBF] v <= alpha * h
            A_cbf = np.array([[1.0, 0.0]])
            b_cbf = np.array([self.alpha * h_val])
        else:
            # [Case 2: 危险区 - Φ3 PDT] v <= beta * h (强制后退)
            A_cbf = np.array([[1.0, 0.0]])
            b_cbf = np.array([self.beta * h_val]) 
            
        # --- 输入约束 ---
        A_limits = np.array([
            [ 1.0,  0.0], [-1.0,  0.0],
            [ 0.0,  1.0], [ 0.0, -1.0]
        ])
        b_limits = np.array([self.v_max, self.v_max, self.w_max, self.w_max])
        
        A = sparse.csc_matrix(np.vstack([A_cbf, A_limits]))
        l = np.array([-np.inf] * 5)
        u = np.hstack([b_cbf, b_limits])
        
        q = -u_dyn
        
        if not self.is_setup:
            self.prob.setup(self.P_base, q, A, l, u, verbose=False)
            self.is_setup = True
        else:
            self.prob.update(q=q, u=u)
            
        res = self.prob.solve()
        
        if res.info.status != 'solved':
            return np.array([0.0, 0.0]), "infeasible"
            
        self.last_u_safe = res.x
        return res.x, "optimal"
