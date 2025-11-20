from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """
    智能体基类
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config

        # 配置参数
        self.state_config = config.get('state_space', {})
        self.action_config = config.get('action_space', {})
        self.reward_config = config.get('reward', {})

        # 状态信息
        self.current_state = None
        self.current_action = None
        self.current_valid_actions = []
        self.last_state = None
        self.last_action = None
        self.last_valid_actions = []

        # 训练状态
        self.training = True
        self.episode_count = 0
        self.step_count = 0

        # 性能统计
        self.performance_stats = {
            'total_reward': 0.0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'avg_response_time': 0.0,
            'action_distribution': {}
        }

        self._initialize_agent()

    def _initialize_agent(self):
        """初始化智能体"""
        for i in range(4):  # 0=local, 1-3=servers
            self.performance_stats['action_distribution'][str(i)] = 0

    @abstractmethod
    def build_network(self, state_dim: int, action_dim: int) -> nn.Module:
        pass

    @abstractmethod
    def select_action(self, state: np.ndarray, valid_actions: List[str],
                      exploration: bool = True) -> Tuple[int, str]:
        pass

    @abstractmethod
    def learn_from_experience(self, state: np.ndarray, action: int, reward: float,
                              next_state: np.ndarray, done: bool, valid_actions: List[str]):
        pass

    def store_transition(self, state: np.ndarray, action: int, valid_actions: List[str]):
        """存储当前状态和动作用于后续学习"""
        self.last_state = state
        self.last_action = action
        self.last_valid_actions = valid_actions

        self.current_state = state
        self.current_action = action
        self.current_valid_actions = valid_actions

    def update_with_reward(self, reward: float, next_state: np.ndarray = None, done: bool = False):
        """使用VEINS反馈的奖励更新智能体"""
        if self.last_state is None or self.last_action is None:
            print(f"警告: 智能体 {self.agent_id} 没有先前的状态用于学习")
            return

        if next_state is None:
            next_state = self.last_state  # 如果没有新状态，使用旧状态

        # 调用具体的学习方法
        self.learn_from_experience(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=next_state,
            done=done,
            valid_actions=self.last_valid_actions
        )

        # 更新统计
        self.step_count += 1
        self.performance_stats['total_reward'] += reward

        if reward > 0:
            self.performance_stats['successful_tasks'] += 1
        else:
            self.performance_stats['failed_tasks'] += 1

        # 更新动作分布
        action_name = str(self.last_action)
        if action_name in self.performance_stats['action_distribution']:
            self.performance_stats['action_distribution'][action_name] += 1

    def preprocess_state(self, state: np.ndarray, valid_actions: List[str]) -> np.ndarray:
        return state.astype(np.float32)

    def validate_action(self, action_idx: int, action_name: str,
                        valid_actions: List[str]) -> bool:
        return action_name in valid_actions

    def get_performance_report(self) -> Dict[str, Any]:
        total_tasks = self.performance_stats['successful_tasks'] + self.performance_stats['failed_tasks']
        success_rate = (self.performance_stats['successful_tasks'] / total_tasks * 100) if total_tasks > 0 else 0

        return {
            'agent_id': self.agent_id,
            'total_reward': self.performance_stats['total_reward'],
            'success_rate': success_rate,
            'action_distribution': self.performance_stats['action_distribution'],
            'episode_count': self.episode_count,
            'step_count': self.step_count
        }

    def get_agent_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.config.get('type', 'unknown'),
            'state_dim': self.state_config.get('dimensions', 'dynamic'),
            'action_dim': len(self.current_valid_actions) if self.current_valid_actions else 'dynamic',
            'training_mode': self.training,
            'episode_count': self.episode_count,
            'step_count': self.step_count
        }

    def reset_episode_stats(self):
        self.episode_count += 1

    def save_model(self, filepath: str):
        """保存模型"""
        print(f"Base save_model called for agent {self.agent_id}")

    def load_model(self, filepath: str):
        """加载模型"""
        print(f"Base load_model called for agent {self.agent_id}")