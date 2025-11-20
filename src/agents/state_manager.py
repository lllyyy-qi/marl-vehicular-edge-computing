import numpy as np
from typing import Dict, List, Tuple, Optional


class StateManager:
    """
    简化状态管理器 - 直接使用VEINS接口处理数据
    """

    def __init__(self, veins_interface):
        self.veins_interface = veins_interface

    def build_state_from_veins(self, veins_data: Dict) -> Tuple[np.ndarray, List[str], Dict]:
        """
        直接从VEINS数据构建状态

        Args:
            veins_data: VEINS原始数据

        Returns:
            (状态向量, 有效动作, 任务信息)
        """
        return self.veins_interface.parse_state_message(veins_data)