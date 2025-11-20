"""
修复的调试入口
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.agents.agent_factory import AgentFactory
from src.environment.veins_interface import VeinsInterface
from src.agents.state_manager import StateManager
from src.utils.config_loader import ConfigLoader

def debug_simple_workflow():
    """工作流程测试"""
    print("=" * 50)
    print("修复的工作流程测试")
    print("=" * 50)

    try:
        # 初始化组件
        config_loader = ConfigLoader()
        veins_interface = VeinsInterface()
        state_manager = StateManager(veins_interface)

        # 创建智能体
        agent_config = config_loader.load_agent_config("agent_1")
        agent = AgentFactory.create_agent("test_vehicle", agent_config)

        # 模拟VEINS状态数据
        veins_state_data = {
            "car_id": 385,
            "msg_type": "state",
            "servers": [
                {
                    "dist": 265.4778030882422,
                    "load": 0.0,
                    "name": "server[0]",
                    "rate": 12091930.572322471
                },
                {
                    "dist": 136.91559848884904,
                    "load": 0.0,
                    "name": "server[1]",
                    "rate": 38398684.78841574
                }
            ],
            "task_info": {
                "demand": 500.0,
                "input_size": 1000.0,
                "is_busy": 0.0
            }
        }

        print("=== 1. 状态构建 ===")
        # 构建状态
        state_vector, valid_actions, task_info = state_manager.build_state_from_veins(veins_state_data)
        print(f" 状态构建成功")
        print(f" 状态维度: {state_vector.shape}")
        print(f" 有效动作: {valid_actions}")
        print(f" 任务信息: {task_info}")

        print("\n=== 2. 智能体决策 ===")
        # 智能体决策
        action_idx, action_name = agent.select_action(state_vector, valid_actions, exploration=True)
        print(f" 动作选择成功")
        print(f" 选择动作: {action_name} (索引: {action_idx})")
        print(f" 探索率: {agent.epsilon:.3f}")

        # 转换为VEINS输出格式 - 修复参数顺序
        veins_action = veins_interface.convert_action_to_veins(385, action_name)
        print(f"   VEINS输出: {veins_action}")

        # 解释动作含义
        server_name = veins_interface.get_server_name_from_action(action_name, veins_state_data["servers"])
        print(f"   动作含义: {action_name} → {server_name}")

        print("\n=== 3. 奖励处理 ===")
        # 模拟VEINS奖励数据
        veins_reward_data = {
            "latency": 0.024,
            "msg_type": "reward",
            "value": -0.024  # 这个值会被忽略
        }

        # 计算奖励 - 现在使用新的归一化方法
        reward = veins_interface.parse_reward_message(veins_reward_data)

        # 显示奖励分析
        analysis = veins_interface.get_latency_analysis(veins_reward_data["latency"])

        print(f"✅ 奖励计算成功")
        print(f"   延迟: {veins_reward_data['latency']}s")
        print(f"   归一化延迟: {analysis['normalized_latency']:.3f}")
        print(f"   计算奖励: {reward:.3f}")
        print(f"   VEINS原始奖励: {veins_reward_data['value']} (被忽略)")
        print("\n=== 4. 智能体学习 ===")
        # 使用新的方法更新智能体
        agent.update_with_reward(reward)
        print(f" 智能体更新成功")

        # 获取训练状态
        training_status = agent.get_training_status()
        print(f" 训练步数: {training_status['train_step']}")
        print(f" 探索率: {training_status['epsilon']:.3f}")

        print(f"\n 工作流程测试成功!")
        print(f" 最终输出: {veins_action}")
        print(f" 学习状态: 探索率 {training_status['epsilon']:.3f}, 步数 {training_status['train_step']}")

        return True

    except Exception as e:
        print(f" 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_simple_workflow()