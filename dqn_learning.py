"""
DQN学习能力验证脚本
测试智能体是否能够通过经验学习改进策略
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from src.agents.agent_factory import AgentFactory
from src.environment.veins_interface import VeinsInterface
from src.agents.state_manager import StateManager
from src.utils.config_loader import ConfigLoader


class DQNLearningTester:
    """DQN学习能力测试器"""

    def __init__(self):
        self.config_loader = ConfigLoader()
        self.veins_interface = VeinsInterface()
        self.state_manager = StateManager(self.veins_interface)
        self.agent = None
        self.training_history = []

    def setup_agent(self):
        """设置测试智能体"""
        agent_config = self.config_loader.load_agent_config("agent_1")
        self.agent = AgentFactory.create_agent("learning_test_agent", agent_config)
        print(f" 创建测试智能体: {self.agent.agent_id}")

    def generate_test_scenarios(self, num_scenarios=5):
        """生成测试场景"""
        scenarios = []

        # 场景1: 近距离低负载服务器 (应该选择服务器)
        scenarios.append({
            "name": "近距离低负载",
            "servers": [
                {"dist": 50.0, "load": 0.1, "name": "server[0]", "rate": 50000000},
                {"dist": 300.0, "load": 0.8, "name": "server[1]", "rate": 10000000}
            ],
            "task": {"demand": 300.0, "input_size": 2000.0, "is_busy": 0.0},
            "expected_best_action": "1",  # 应该选择server[0]
            "good_latency": 0.01,
            "bad_latency": 0.5
        })

        # 场景2: 所有服务器高负载 (应该选择本地)
        scenarios.append({
            "name": "高负载服务器",
            "servers": [
                {"dist": 100.0, "load": 0.9, "name": "server[0]", "rate": 30000000},
                {"dist": 150.0, "load": 0.95, "name": "server[1]", "rate": 20000000}
            ],
            "task": {"demand": 200.0, "input_size": 1000.0, "is_busy": 0.0},
            "expected_best_action": "0",  # 应该选择本地
            "good_latency": 0.02,
            "bad_latency": 1.2
        })

        # 场景3: 远距离但高速服务器
        scenarios.append({
            "name": "高速远距离服务器",
            "servers": [
                {"dist": 400.0, "load": 0.2, "name": "server[0]", "rate": 80000000},
                {"dist": 100.0, "load": 0.6, "name": "server[1]", "rate": 20000000}
            ],
            "task": {"demand": 500.0, "input_size": 5000.0, "is_busy": 0.0},
            "expected_best_action": "1",  # 应该选择高速服务器
            "good_latency": 0.05,
            "bad_latency": 0.3
        })

        return scenarios[:num_scenarios]

    def test_initial_performance(self, scenarios):
        """测试初始性能（随机策略）"""
        print("\n" + "=" * 60)
        print("阶段1: 测试初始性能 (随机策略)")
        print("=" * 60)

        initial_scores = []

        for i, scenario in enumerate(scenarios):
            # 构建状态
            state_data = {
                "car_id": 999,
                "msg_type": "state",
                "servers": scenario["servers"],
                "task_info": scenario["task"]
            }

            state_vector, valid_actions, _ = self.state_manager.build_state_from_veins(state_data)

            # 测试10次决策，统计选择最优动作的比例
            correct_choices = 0
            total_tests = 10

            for _ in range(total_tests):
                action_idx, action_name = self.agent.select_action(state_vector, valid_actions, exploration=True)
                if action_name == scenario["expected_best_action"]:
                    correct_choices += 1

            accuracy = correct_choices / total_tests
            initial_scores.append(accuracy)

            print(f"场景 {i + 1} ({scenario['name']}):")
            print(f"  预期最优动作: {scenario['expected_best_action']}")
            print(f"  随机选择准确率: {accuracy:.1%}")
            print(f"  当前探索率: {self.agent.epsilon:.3f}")

        avg_initial_accuracy = np.mean(initial_scores)
        print(f"\n 初始平均准确率: {avg_initial_accuracy:.1%} (随机策略)")

        return avg_initial_accuracy

    def train_agent(self, scenarios, training_episodes=100):
        """训练智能体"""
        print(f"\n" + "=" * 60)
        print(f"阶段2: 训练智能体 ({training_episodes}回合)")
        print("=" * 60)

        episode_rewards = []
        exploration_rates = []
        accuracy_history = []

        for episode in range(training_episodes):
            episode_reward = 0
            scenario = scenarios[episode % len(scenarios)]  # 循环使用场景

            # 构建状态
            state_data = {
                "car_id": 999,
                "msg_type": "state",
                "servers": scenario["servers"],
                "task_info": scenario["task"]
            }

            state_vector, valid_actions, _ = self.state_manager.build_state_from_veins(state_data)

            # 智能体决策
            action_idx, action_name = self.agent.select_action(state_vector, valid_actions, exploration=True)

            # 根据决策质量给予奖励
            if action_name == scenario["expected_best_action"]:
                # 好决策 - 低延迟奖励
                reward = self.veins_interface.parse_reward_message({"latency": scenario["good_latency"]})
            else:
                # 坏决策 - 高延迟惩罚
                reward = self.veins_interface.parse_reward_message({"latency": scenario["bad_latency"]})

            # 更新智能体
            self.agent.update_with_reward(reward)
            episode_reward += reward

            # 每10回合评估一次性能
            if episode % 10 == 0:
                accuracy = self.evaluate_current_performance(scenarios)
                accuracy_history.append(accuracy)

                print(f"回合 {episode:3d}: 探索率={self.agent.epsilon:.3f}, "
                      f"本轮奖励={reward:.3f}, 准确率={accuracy:.1%}")

            episode_rewards.append(episode_reward)
            exploration_rates.append(self.agent.epsilon)

        return episode_rewards, exploration_rates, accuracy_history

    def evaluate_current_performance(self, scenarios, test_runs=5):
        """评估当前性能（关闭探索）"""
        self.agent.training = False  # 关闭探索
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # 完全利用

        correct_choices = 0
        total_tests = 0

        for scenario in scenarios:
            state_data = {
                "car_id": 999,
                "msg_type": "state",
                "servers": scenario["servers"],
                "task_info": scenario["task"]
            }

            state_vector, valid_actions, _ = self.state_manager.build_state_from_veins(state_data)

            for _ in range(test_runs):
                action_idx, action_name = self.agent.select_action(state_vector, valid_actions, exploration=False)
                if action_name == scenario["expected_best_action"]:
                    correct_choices += 1
                total_tests += 1

        # 恢复训练状态
        self.agent.training = True
        self.agent.epsilon = original_epsilon

        return correct_choices / total_tests

    def test_final_performance(self, scenarios):
        """测试最终性能（学习后）"""
        print(f"\n" + "=" * 60)
        print("阶段3: 测试最终性能 (学习后)")
        print("=" * 60)

        final_scores = []
        q_value_analysis = []

        for i, scenario in enumerate(scenarios):
            state_data = {
                "car_id": 999,
                "msg_type": "state",
                "servers": scenario["servers"],
                "task_info": scenario["task"]
            }

            state_vector, valid_actions, _ = self.state_manager.build_state_from_veins(state_data)

            # 测试10次决策（关闭探索）
            self.agent.training = False
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0

            correct_choices = 0
            total_tests = 10

            for _ in range(total_tests):
                action_idx, action_name = self.agent.select_action(state_vector, valid_actions, exploration=False)
                if action_name == scenario["expected_best_action"]:
                    correct_choices += 1

            accuracy = correct_choices / total_tests
            final_scores.append(accuracy)

            # 分析Q值
            q_values = self.agent.get_q_values(state_vector, valid_actions)
            best_action_q = q_values.get(scenario["expected_best_action"], 0)

            q_value_analysis.append({
                "scenario": scenario["name"],
                "best_action": scenario["expected_best_action"],
                "best_action_q": best_action_q,
                "all_q_values": q_values
            })

            print(f"场景 {i + 1} ({scenario['name']}):")
            print(f"  预期最优动作: {scenario['expected_best_action']}")
            print(f"  学习后准确率: {accuracy:.1%}")
            print(f"  最优动作Q值: {best_action_q:.3f}")
            print(f"  所有Q值: {q_values}")

            # 恢复训练状态
            self.agent.training = True
            self.agent.epsilon = original_epsilon

        avg_final_accuracy = np.mean(final_scores)
        print(f"\n 最终平均准确率: {avg_final_accuracy:.1%} (学习后)")

        return avg_final_accuracy, q_value_analysis

    def plot_learning_curves(self, episode_rewards, exploration_rates, accuracy_history):
        """绘制学习曲线"""
        plt.figure(figsize=(15, 5))

        # 奖励曲线
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards, alpha=0.7)
        plt.title('每回合奖励')
        plt.xlabel('回合')
        plt.ylabel('奖励')
        plt.grid(True, alpha=0.3)

        # 探索率曲线
        plt.subplot(1, 3, 2)
        plt.plot(exploration_rates, color='orange')
        plt.title('探索率衰减')
        plt.xlabel('回合')
        plt.ylabel('探索率 (ε)')
        plt.grid(True, alpha=0.3)

        # 准确率曲线
        plt.subplot(1, 3, 3)
        episodes = range(0, len(episode_rewards), 10)
        plt.plot(episodes[:len(accuracy_history)], accuracy_history, color='green')
        plt.title('学习准确率')
        plt.xlabel('回合')
        plt.ylabel('准确率')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('dqn_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_complete_test(self, training_episodes=200):
        """运行完整的学习测试"""
        print(" 开始DQN学习能力验证")
        print("=" * 60)

        # 设置智能体
        self.setup_agent()

        # 生成测试场景
        scenarios = self.generate_test_scenarios(3)

        # 阶段1: 初始性能
        initial_accuracy = self.test_initial_performance(scenarios)

        # 阶段2: 训练
        episode_rewards, exploration_rates, accuracy_history = self.train_agent(
            scenarios, training_episodes
        )

        # 阶段3: 最终性能
        final_accuracy, q_value_analysis = self.test_final_performance(scenarios)

        # 绘制学习曲线
        self.plot_learning_curves(episode_rewards, exploration_rates, accuracy_history)

        # 最终报告
        print(f"\n" + "=" * 60)
        print(" DQN学习验证报告")
        print("=" * 60)
        print(f"初始准确率: {initial_accuracy:.1%} (随机策略)")
        print(f"最终准确率: {final_accuracy:.1%} (学习后策略)")
        print(f"性能提升: {((final_accuracy - initial_accuracy) / initial_accuracy * 100):+.1f}%")
        print(f"训练回合数: {training_episodes}")
        print(f"最终探索率: {self.agent.epsilon:.3f}")
        print(f"经验回放大小: {len(self.agent.memory)}")

        # 学习成功判断
        if final_accuracy > initial_accuracy + 0.2:  # 提升20%以上
            print("\n DQN学习验证: 成功")
            print("   智能体成功通过经验学习改进了决策策略")
        elif final_accuracy > initial_accuracy:
            print("\n⚠ DQN学习验证: 部分成功")
            print("   智能体有学习迹象，但提升不明显")
        else:
            print("\n DQN学习验证: 失败")
            print("   智能体未能有效学习")

        return initial_accuracy, final_accuracy


def main():
    """主函数"""
    tester = DQNLearningTester()

    try:
        initial_acc, final_acc = tester.run_complete_test(training_episodes=300)

        # 学习效果总结
        improvement = final_acc - initial_acc
        if improvement > 0.3:
            print("\n 优秀的学习效果! 智能体显著改进了决策能力")
        elif improvement > 0.15:
            print("\n 良好的学习效果! 智能体明显学习了更好的策略")
        elif improvement > 0.05:
            print("\n 基本的学习效果! 智能体有学习迹象")
        else:
            print("\n 学习效果有限，可能需要调整超参数或更多训练")

    except Exception as e:
        print(f" 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()