import json
from typing import Dict, List, Any, Tuple
import numpy as np


class VeinsInterface:
    """
    VEINS数据接口
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.max_acceptable_latency = 1.0  # 最大可接受延迟1秒
        self.min_acceptable_latency = 0.001  # 最小延迟1毫秒

    def parse_state_message(self, veins_data: Dict) -> Tuple[np.ndarray, List[str], Dict]:
        """解析VEINS状态消息"""
        # 提取基本信息
        car_id = veins_data["car_id"]
        servers = veins_data["servers"]
        task_info = veins_data["task_info"]

        # 构建状态向量 [任务大小, 任务计算密度, 服务器1距离, 服务器1负载, 服务器1速率, ...]
        state_features = []

        # 额外的本地车端特征（可选）
        # 优先从 task_info 中读取，如果 OMNeT 把字段放在顶层 car 字段也会兼容
        car_meta = veins_data.get("car", {})

        local_load = None
        local_proc_rate = None
        # check task_info first then car_meta
        if isinstance(task_info, dict):
            local_load = task_info.get("local_load", None)
            local_proc_rate = task_info.get("local_processing_rate", None)

        if local_load is None:
            local_load = car_meta.get("local_load", None)
        if local_proc_rate is None:
            local_proc_rate = car_meta.get("local_processing_rate", None)

        # 如果提供了本地特征，先加入 state 特征（保持 0..1 归一化）
        if local_load is not None:
            try:
                ll = float(local_load)
            except Exception:
                ll = 0.0
            # 保证 0..1
            state_features.append(min(max(ll, 0.0), 1.0))
        
        if local_proc_rate is not None:
            try:
                lpr = float(local_proc_rate)
            except Exception:
                lpr = 0.0
            # 归一化到 Mbps 级别（cycles/s 可能更大，使用 1e6 作为缩放因子，防止过大数值）
            state_features.append(min(lpr / 1_000_000.0, 1.0))

        # 任务特征 (2维)
        input_size_mb = task_info["input_size"] / 8_000_000  # bits to MB
        compute_density = task_info["demand"]  # cycles/bit

        state_features.append(input_size_mb / 10.0)  # 归一化
        state_features.append(compute_density / 1000.0)  # 归一化

        # 服务器特征 (每个服务器3维: 距离, 负载, 速率)
        sorted_servers = sorted(servers, key=lambda x: x["name"])

        for server in sorted_servers:
            # 距离归一化 (0-500m -> 0-1)
            dist_norm = min(server["dist"] / 500.0, 1.0)
            state_features.append(dist_norm)

            # 负载直接使用 (0-1)
            state_features.append(server["load"])

            # 传输速率归一化 (bits/s to Mbps, 0-150Mbps -> 0-1)
            rate_mbps = server["rate"] / 1_000_000
            rate_norm = min(rate_mbps / 150.0, 1.0)
            state_features.append(rate_norm)

        # 如果服务器数量不足3个，用0填充
        max_servers = 3
        current_servers = len(sorted_servers)
        if current_servers < max_servers:
            for _ in range(max_servers - current_servers):
                state_features.extend([0.0, 0.0, 0.0])

        state_vector = np.array(state_features, dtype=np.float32)

        # 构建有效动作列表 - 使用数字索引
        valid_actions = ["0"]  # local = 0

        # 添加可用的服务器动作 - 按名称排序
        for i, server in enumerate(sorted_servers):
            action_name = str(i + 1)  # server[0] = 1, server[1] = 2, ...
            valid_actions.append(action_name)

        # 保存状态用于奖励计算
        self.last_state = {
            "car_id": car_id,
            "task_info": task_info,
            "servers": sorted_servers
        }

        # 将解析到的本地特征回填到 task_info（便于外部查看/日志），并记录 best_uplink_rate
        try:
            # best uplink rate (bps) - 取可用服务器中最大的链路速率
            best_rate = 0.0
            for s in sorted_servers:
                try:
                    r = float(s.get("rate", 0.0))
                    if r > best_rate:
                        best_rate = r
                except Exception:
                    continue

            # 写回 task_info 字段（不覆盖已有字段）
            if isinstance(task_info, dict):
                if "local_load" not in task_info and local_load is not None:
                    task_info["local_load"] = float(local_load)
                if "local_processing_rate" not in task_info and local_proc_rate is not None:
                    task_info["local_processing_rate"] = float(local_proc_rate)
                task_info["best_uplink_rate"] = best_rate
        except Exception:
            pass

        return state_vector, valid_actions, task_info

    def parse_reward_message(self, veins_reward_data: Dict) -> float:
        """
        解析VEINS奖励消息，基于延迟计算归一化奖励

        Args:
            veins_reward_data: VEINS奖励数据，包含latency

        Returns:
            归一化奖励值 [-1, 1]
        """
        latency = veins_reward_data.get("latency", 0.0)

        # 忽略VEINS提供的value，直接基于latency计算奖励
        return self._calculate_reward_from_latency(latency)

    def _calculate_reward_from_latency(self, latency: float) -> float:
        """
        基于延迟计算归一化奖励

        设计原则:
        - 延迟越小，奖励越大
        - 延迟在可接受范围内时，奖励为正
        - 延迟超过阈值时，奖励为负
        - 奖励范围归一化到 [-1, 1]
        """
        # 确保latency是合理的正值
        latency = max(float(latency), self.min_acceptable_latency)

        # 计算归一化延迟 (0到1之间，1表示最差情况)
        normalized_latency = min(latency / self.max_acceptable_latency, 1.0)

        # 使用指数衰减函数计算奖励
        # 当latency=0.001s时，reward≈0.9
        # 当latency=0.1s时，reward≈0.37
        # 当latency=0.5s时，reward≈0.006
        # 当latency=1.0s时，reward≈0.0
        # 当latency>1.0s时，reward为负
        if latency <= self.max_acceptable_latency:
            # 在可接受范围内，使用指数衰减的正奖励
            reward = np.exp(-5 * normalized_latency)  # 衰减因子控制曲线陡峭程度
        else:
            # 超过可接受范围，使用线性负奖励
            excess_ratio = (latency - self.max_acceptable_latency) / self.max_acceptable_latency
            reward = -min(excess_ratio, 1.0)  # 最大负奖励为-1

        # 确保奖励在[-1, 1]范围内
        reward = max(min(reward, 1.0), -1.0)

        return float(reward)

    def get_latency_analysis(self, latency: float) -> Dict[str, Any]:
        """分析延迟对应的奖励计算"""
        reward = self._calculate_reward_from_latency(latency)
        normalized_latency = min(latency / self.max_acceptable_latency, 1.0)

        return {
            "latency": latency,
            "normalized_latency": normalized_latency,
            "reward": reward,
            "max_acceptable_latency": self.max_acceptable_latency
        }

    def get_action_mapping(self, valid_actions: List[str]) -> Dict[int, str]:
        """获取动作映射"""
        action_mapping = {}
        for i, action in enumerate(valid_actions):
            action_mapping[i] = action
        return action_mapping

    def convert_action_to_veins(self, car_id: int, action_name: str) -> Dict:
        """将动作名称转换为VEINS输出格式"""
        return {
            "car_id": int(car_id),
            "action": str(action_name)
        }

    def get_server_name_from_action(self, action_name: str, servers: List[Dict]) -> str:
        """根据动作获取服务器名称"""
        if action_name == "0":
            return "local"
        else:
            server_index = int(action_name) - 1
            sorted_servers = sorted(servers, key=lambda x: x["name"])
            if 0 <= server_index < len(sorted_servers):
                return sorted_servers[server_index]["name"]
            else:
                return "unknown"