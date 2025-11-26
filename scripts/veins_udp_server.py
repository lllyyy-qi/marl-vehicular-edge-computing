import sys
import os

# Ensure project root is on sys.path so `from src...` imports work when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import json
import threading
import time
from src.agents.agent_factory import AgentFactory
from src.environment.veins_interface import VeinsInterface
from src.agents.state_manager import StateManager
from src.utils.config_loader import ConfigLoader


class VeinsUDPServer:
    """UDP 版本的 VEINS-代理服务器

    功能：
    - 监听来自 OMNeT/VEINS 的 UDP JSON 消息（state / reward）
    - 基于 state 调用 agent 做决策，并将动作以数字字符串发送到 OMNET_HOST:OMNET_BASE_PORT+car_id
    - 基于 reward 调用 agent.update_with_reward(reward)
    """

    def __init__(self,
                 agent_host: str = '127.0.0.1',
                 agent_port: int = 5000,
                 omnet_host: str = '127.0.0.1',
                 omnet_base_port: int = 4000,
                 buffer_size: int = 8192):
        self.agent_host = agent_host
        self.agent_port = agent_port
        self.omnet_host = omnet_host
        self.omnet_base_port = omnet_base_port
        self.buffer_size = buffer_size

        self.config_loader = ConfigLoader()
        self.veins_interface = VeinsInterface()
        self.state_manager = StateManager(self.veins_interface)

        # agents: key = car_id, value = agent instance
        self.agents = {}

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.agent_host, self.agent_port))

        print(f"Veins UDP Agent initialized. Listening on {self.agent_host}:{self.agent_port}")
        print(f"Will send actions to {self.omnet_host}:{self.omnet_base_port}+car_id")

    def setup_agent(self, car_id: int):
        agent_id = f"vehicle_{car_id}"
        if agent_id not in self.agents:
            # 默认使用 agent_1 配置；如果需要按车不同配置，可在 Configs 中调整并实现映射
            agent_config = self.config_loader.load_agent_config("agent_1")
            self.agents[agent_id] = AgentFactory.create_agent(agent_id, agent_config)
            print(f"Created agent for car {car_id} -> {agent_id}")
        return self.agents[agent_id]

    def handle_message(self, data: bytes, addr):
        try:
            text = data.decode('utf-8').strip()
        except Exception as e:
            print(f"Failed to decode incoming bytes from {addr}: {e}")
            return

        # 尝试解析为 JSON
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            print(f"JSON decode error from {addr}. Raw: {text}")
            return

        msg_type = message.get('msg_type', '').lower()
        car_id = message.get('car_id')

        if car_id is None:
            print(f"Received message without car_id from {addr}: {message}")
            return

        # 状态消息：做决策并通过 UDP 发送数字字符串动作到 OMNeT (omnet_base_port + car_id)
        if msg_type == 'state':
            print(f"Received state message from car {car_id}: {message}")
            try:
                agent = self.setup_agent(int(car_id))

                state_vector, valid_actions, task_info = self.state_manager.build_state_from_veins(message)

                action_idx, action_name = agent.select_action(state_vector, valid_actions, exploration=True)

                # OMNeT/C++ 端期望接收数字字符串（例如 "0" 或 "1"），所以直接发送 action_name
                send_port = self.omnet_base_port + int(car_id)
                self.sock.sendto(str(action_name).encode('utf-8'), (self.omnet_host, send_port))

                print(f"[car {car_id}] state processed -> action {action_name} sent to {self.omnet_host}:{send_port}")

            except Exception as e:
                print(f"Error processing state for car {car_id}: {e}")

        # 奖励消息：解析 reward 并传递给 agent 做学习更新
        elif msg_type == 'reward':
            try:
                agent = self.setup_agent(int(car_id))
                # parse_reward_message 以 latency field 为准
                reward = self.veins_interface.parse_reward_message(message)

                # 将 reward 交给 agent 更新（agent 会使用之前 store 的 last_state/last_action）
                agent.update_with_reward(reward)

                # 可选：向 OMNeT 发送 ACK
                ack = {"status": "reward_received", "car_id": int(car_id)}
                send_port = addr[1]
                try:
                    self.sock.sendto(json.dumps(ack).encode('utf-8'), (addr[0], send_port))
                except Exception:
                    # 不关键，忽略发送 ACK 错误
                    pass

                print(f"[car {car_id}] reward processed -> reward={reward:.4f}")

            except Exception as e:
                print(f"Error processing reward for car {car_id}: {e}")

        else:
            print(f"Unknown msg_type from {addr}: {msg_type} | raw={message}")

    def start(self):
        print("Veins UDP Server started. Waiting for messages...")
        try:
            while True:
                data, addr = self.sock.recvfrom(self.buffer_size)
                # 为短处理创建线程，避免阻塞接收循环
                t = threading.Thread(target=self.handle_message, args=(data, addr))
                t.daemon = True
                t.start()

        except KeyboardInterrupt:
            print("Shutting down Veins UDP Server")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.sock.close()


def main():
    server = VeinsUDPServer(agent_host='127.0.0.1', agent_port=5000,
                            omnet_host='127.0.0.1', omnet_base_port=4000,
                            buffer_size=8192)
    server.start()


if __name__ == '__main__':
    main()
