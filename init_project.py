import os

# 定义要创建的核心文件内容
TURTLEBOT_CORE_PY = r'''# -*- coding: utf-8 -*-
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
'''

EICPS_NODE_PY = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import sys
import os

# 动态添加 core 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设目录结构为 ros_ws/src/eicps_phi/scripts/
core_path = os.path.join(current_dir, '../../../../core')
sys.path.append(core_path)

# 尝试导入，如果在非ROS环境下可能会失败，这里做个保护
try:
    from phi_interface.turtlebot_core import EICPS_Interface
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float32
except ImportError:
    pass

class EICPS_Node:
    def __init__(self):
        rospy.init_node('eicps_phi_node', anonymous=True)
        self.eicps = EICPS_Interface()
        
        # ROS 参数
        self.eicps.d_min = rospy.get_param('~d_min', 0.20)
        self.eicps.v_max = rospy.get_param('~v_max', 0.26)
        
        self.min_scan_dist = 99.9
        
        # 通信接口
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/cmd_vel_ai', Twist, self.ai_cmd_cb)
        self.pub_safe = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_h = rospy.Publisher('/eicps/debug/h_val', Float32, queue_size=1)

        rospy.loginfo("EICPS Phi Node Running...")

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        valid = ranges[np.isfinite(ranges)]
        if len(valid) > 0:
            self.min_scan_dist = np.min(valid)
        else:
            self.min_scan_dist = 99.9

    def ai_cmd_cb(self, msg):
        u_ai = np.array([msg.linear.x, msg.angular.z])
        
        # Φ1: HNN
        u_dyn = self.eicps.phi_hnn_projection(u_ai)
        
        # Φ2 + Φ3: CBF + PDT
        u_safe, status = self.eicps.solve_safety_filter(None, u_dyn, self.min_scan_dist)
        
        # Debug
        self.pub_h.publish(self.min_scan_dist - self.eicps.d_min)
        
        # Publish
        t = Twist()
        t.linear.x = u_safe[0]
        t.angular.z = u_safe[1]
        self.pub_safe.publish(t)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        EICPS_Node().run()
    except Exception:
        pass
'''

LAUNCH_XML = r'''<launch>
  <arg name="d_min" default="0.20" doc="安全距离阈值 (m)"/>
  <arg name="v_max" default="0.26" doc="最大线速度 (m/s)"/>
  
  <node pkg="eicps_phi" type="eicps_phi_node.py" name="eicps_phi_node" output="screen">
    <param name="d_min" value="$(arg d_min)"/>
    <param name="v_max" value="$(arg v_max)"/>
  </node>
</launch>
'''

REQUIREMENTS_TXT = r'''numpy>=1.20.0
scipy>=1.7.0
osqp>=0.6.2
rospkg>=1.3.0
matplotlib>=3.5.0
'''

README_MD = r'''# 📦 EICPS-Stack: 具身智能信息物理系统框架

**Embodied Intelligent Cyber-Physical System Framework**

[![ROS](https://img.shields.io/badge/ROS-Noetic%2FHumble-blue)](http://wiki.ros.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**EICPS-Stack** 是一个统一的具身智能机器人软件栈，实现了 **$\mathbb{P}-\mathcal{E}-\Phi$** 理论框架。它充当 AI 控制器（RL/LLM）与物理机器人（TurtleBot3/USV）之间的“数字脊髓”，提供动力学一致性、形式化安全保证（CBF）与有限时间恢复（PDT）能力。

## 🏗️ 核心架构 (The P-E-Phi Framework)

本框架由三个核心数学模块构成：

1.  **$\mathbb{P}$ (Modeling Process)**: 物理动力学建模与约束定义。
2.  **$\mathcal{E}$ (Embodied Space)**: 具身空间，包含几何安全集 $h(x) \ge 0$ 与 PDT 时间场 $T(x)$。
3.  **$\Phi$ (Embodied Interface)**: 具身接口，实现 AI $\to$ Physics 的安全投影。
    * $\Phi_1$: **HNN** (动力学一致性投影)
    * $\Phi_2$: **CBF-QP** (安全过滤)
    * $\Phi_3$: **PDT** (有限时间恢复)

## 📂 目录结构

```text
EICPS-Stack/
├── core/                   # [核心] EICPS 数学算法库 (平台无关)
│   └── phi_interface/      # Φ 接口实现 (HNN/QP/PDT)
├── ros_ws/                 # [部署] ROS 工作空间
│   └── src/
│       └── eicps_phi/      # ROS 节点: 桥接 Core 与 Robot
├── sim/                    # [仿真] Gazebo/MWORKS 场景
├── training/               # [训练] RL/HNN 离线训练代码
└── docs/                   # [文档] 论文与教程
```

## 🚀 快速开始 (Quick Start)

### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 运行 EICPS 安全接口

**步骤 A: 启动仿真环境**
```bash
export TURTLEBOT3_MODEL=waffle_pi
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```

**步骤 B: 启动 EICPS Φ 节点**
```bash
# 该节点会拦截 /cmd_vel_ai，处理后发布到 /cmd_vel
roslaunch eicps_phi eicps_turtlebot.launch
```

---
**Maintainer:** Zhou Lab @ BIT
'''

GITIGNORE = r'''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# ROS
/ros_ws/build/
/ros_ws/devel/
/ros_ws/install/
/ros_ws/log/
/ros_ws/.catkin_workspace
*.bag

# IDEs
.vscode/
.idea/

# System
.DS_Store
Thumbs.db
'''

def create_file(path, content):
    """辅助函数：创建文件并写入内容"""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 文件已创建: {path}")
    except Exception as e:
        print(f"❌ 创建文件失败 {path}: {str(e)}")

def main():
    print("🚀 开始初始化 EICPS-Stack 项目 (Windows 兼容版)...")
    
    # 1. 创建目录结构
    dirs = [
        "core/phi_interface",
        "core/p_model",
        "core/e_space",
        "ros_ws/src/eicps_phi/scripts",
        "ros_ws/src/eicps_phi/launch",
        "sim/turtlebot3_gazebo",
        "sim/usv_mworks",
        "docs/papers",
        "training",
        "deployment"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"📂 目录已创建: {d}")

    # 2. 创建 __init__.py 使其成为 Python 包
    open("core/__init__.py", 'a').close()
    open("core/phi_interface/__init__.py", 'a').close()

    # 3. 写入文件
    create_file("core/phi_interface/turtlebot_core.py", TURTLEBOT_CORE_PY)
    create_file("ros_ws/src/eicps_phi/scripts/eicps_phi_node.py", EICPS_NODE_PY)
    create_file("ros_ws/src/eicps_phi/launch/eicps_turtlebot.launch", LAUNCH_XML)
    create_file("requirements.txt", REQUIREMENTS_TXT)
    create_file("README.md", README_MD)
    create_file(".gitignore", GITIGNORE