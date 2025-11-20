# Innovation Project: MARL for Vehicular Edge Computing

基于多智能体强化学习的车辆边缘计算卸载决策系统。

## 项目结构
- `src/`: 核心源代码
- `scripts/`: 运行脚本
- `configs/`: 配置文件
- `outputs/`: 输出文件


## 环境要求
- Python 3.8+
- PyTorch 1.9+
- SUMO 1.8.0
- VEINS 5.2



## 归一化需要重点讨论，关于最大值为多少等，具体涉及在 veins_interface.py 以及 environment.yaml 中

## 模拟运行在 dqn_learning.py ,至少能确定可以在训练后得到成果，但是仅限于场景未改变，并未考虑小车移动

## mock_veins_env可以模拟输入但是纯随机

Socket连接信息
地址: localhost

端口: 8888

协议: TCP Socket

数据格式: JSON
