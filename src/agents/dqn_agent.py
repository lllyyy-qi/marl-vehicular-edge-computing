import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, List, Tuple, Optional, Any
from .base_agent import BaseAgent


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = None):
        super(DQNNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers = []
        input_size = state_dim

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent(BaseAgent):
    """DQN智能体 """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)

        # DQN参数
        self.dqn_config = config.get('dqn', {})
        self.network_config = config.get('network', {})

        # 网络参数
        self.state_dim = None
        self.action_dim = None
        self.q_network = None
        self.target_network = None
        self.optimizer = None

        # 训练参数
        self.gamma = self.dqn_config.get('gamma', 0.99)
        self.epsilon = self.dqn_config.get('epsilon_start', 1.0)
        self.epsilon_min = self.dqn_config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.dqn_config.get('epsilon_decay', 0.995)
        self.learning_rate = self.dqn_config.get('learning_rate', 0.0005)
        self.target_update_frequency = self.dqn_config.get('target_update_frequency', 100)

        # 经验回放
        self.memory_capacity = self.dqn_config.get('memory_capacity', 10000)
        self.memory = []
        self.batch_size = self.dqn_config.get('batch_size', 32)

        # 训练状态
        self.train_step = 0
        self.losses = []

        self._initialize_networks()

    def build_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """实现抽象方法 - 构建神经网络"""
        hidden_layers = self.network_config.get('hidden_layers', [64, 32])
        return DQNNetwork(state_dim, action_dim, hidden_layers)

    def _initialize_networks(self):
        pass

    def _create_networks(self, state_dim: int, action_dim: int):
        hidden_layers = self.network_config.get('hidden_layers', [64, 32])

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 使用build_network方法创建网络
        self.q_network = self.build_network(state_dim, action_dim)
        self.target_network = self.build_network(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        print(f"Agent {self.agent_id}: 创建网络 state_dim={state_dim}, action_dim={action_dim}")

    def select_action(self, state: np.ndarray, valid_actions: List[str],
                      exploration: bool = True) -> Tuple[int, str]:
        if self.q_network is None:
            self._create_networks(state.shape[0], len(valid_actions))

        processed_state = self.preprocess_state(state, valid_actions)
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)

        # 探索
        if exploration and self.training and random.random() < self.epsilon:
            action_idx = random.randint(0, len(valid_actions) - 1)
            action_name = valid_actions[action_idx]

            # 存储当前决策用于后续学习
            self.store_transition(state, action_idx, valid_actions)
            return action_idx, action_name

        # 利用
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze().numpy()

        # 选择Q值最大的有效动作
        valid_q_values = [-np.inf] * len(valid_actions)
        for i, action_name in enumerate(valid_actions):
            if self.validate_action(i, action_name, valid_actions):
                valid_q_values[i] = q_values[i]

        action_idx = np.argmax(valid_q_values)
        action_name = valid_actions[action_idx]

        # 存储当前决策用于后续学习
        self.store_transition(state, action_idx, valid_actions)
        return action_idx, action_name

    def learn_from_experience(self, state: np.ndarray, action: int, reward: float,
                              next_state: np.ndarray, done: bool, valid_actions: List[str]):
        """从单次经验中学习"""
        if not self.training or self.q_network is None:
            return

        # 存储经验
        self._store_experience(state, action, reward, next_state, done, valid_actions)

        # 从经验回放中学习
        if len(self.memory) >= self.batch_size:
            self._train_from_memory()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _store_experience(self, state: np.ndarray, action: int, reward: float,
                          next_state: np.ndarray, done: bool, valid_actions: List[str]):
        """存储经验到回放缓冲区"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'valid_actions': valid_actions
        }

        if len(self.memory) < self.memory_capacity:
            self.memory.append(experience)
        else:
            self.memory.pop(0)
            self.memory.append(experience)

    def _train_from_memory(self):
        """从经验回放中训练"""
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([exp['state'] for exp in batch])
        states = torch.FloatTensor(states)

        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])

        next_states = np.array([exp['next_state'] for exp in batch])
        next_states = torch.FloatTensor(next_states)

        dones = torch.BoolTensor([exp['done'] for exp in batch])

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones).float())

        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.losses.append(loss.item())

    def get_q_values(self, state: np.ndarray, valid_actions: List[str]) -> Dict[str, float]:
        if self.q_network is None:
            return {}

        processed_state = self.preprocess_state(state, valid_actions)
        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze().numpy()

        q_value_dict = {}
        for i, action_name in enumerate(valid_actions):
            if i < len(q_values):
                q_value_dict[action_name] = float(q_values[i])

        return q_value_dict

    def get_training_status(self) -> Dict[str, Any]:
        status = super().get_agent_info()
        status.update({
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-10:]) if self.losses else 0.0,
            'network_initialized': self.q_network is not None
        })
        return status

    def save_model(self, filepath: str):
        """保存模型"""
        if self.q_network is not None:
            checkpoint = {
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'train_step': self.train_step
            }
            torch.save(checkpoint, filepath)
            print(f"Agent {self.agent_id}: 模型保存到 {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath)

            if self.q_network is None:
                self._create_networks(
                    checkpoint.get('state_dim', 11),
                    checkpoint.get('action_dim', 4)
                )

            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.epsilon = checkpoint['epsilon']
            self.train_step = checkpoint['train_step']

            print(f"Agent {self.agent_id}: 从 {filepath} 加载模型")

        except Exception as e:
            print(f"Agent {self.agent_id}: 加载模型失败: {e}")

