import random
import numpy as np
from typing import Dict, List, Tuple, Any
import torch


class Experience:
    """经验数据 """

    def __init__(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool, valid_actions: List[str]):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.valid_actions = valid_actions


class ExperienceMemory:
    """经验回放 """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience: Experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], List[List[str]]]:
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        experiences = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.LongTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.BoolTensor([exp.done for exp in experiences])
        next_actions = [exp.valid_actions for exp in experiences]

        return {
            'states': states,
            'actions': actions.unsqueeze(1),
            'rewards': rewards.unsqueeze(1),
            'next_states': next_states,
            'dones': dones.unsqueeze(1)
        }, next_actions

    def __len__(self):
        return len(self.memory)