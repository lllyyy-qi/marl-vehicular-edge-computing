import socket
import json
import threading
from src.agents.agent_factory import AgentFactory
from src.environment.veins_interface import VeinsInterface
from src.agents.state_manager import StateManager
from src.utils.config_loader import ConfigLoader

class VeinsSocketServer:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.config_loader = ConfigLoader()
        self.veins_interface = VeinsInterface()
        self.state_manager = StateManager(self.veins_interface)
        self.agents = {}
        self.pending_decisions = {}

        print(f"MARL Socket服务器初始化完成")
        print(f"监听地址: {host}:{port}")

    def setup_agent(self, car_id):
        agent_id = f"vehicle_{car_id}"
        if agent_id not in self.agents:
            agent_config = self.config_loader.load_agent_config("agent_1")
            self.agents[agent_id] = AgentFactory.create_agent(agent_id, agent_config)
            print(f"为车辆 {car_id} 创建智能体")
        return self.agents[agent_id]

    def process_state_message(self, data):
        car_id = data["car_id"]
        agent = self.setup_agent(car_id)

        state_vector, valid_actions, task_info = self.state_manager.build_state_from_veins(data)
        action_idx, action_name = agent.select_action(state_vector, valid_actions, exploration=True)

        response = self.veins_interface.convert_action_to_veins(car_id, action_name)

        print(f"车辆 {car_id}: 决策动作 {action_name}")
        return response

    def process_reward_message(self, data):
        latency = data.get("latency", 0.0)
        reward = self.veins_interface.parse_reward_message(data)

        print(f"收到奖励: 延迟 {latency}s -> 奖励 {reward:.3f}")
        return {"status": "reward_received", "reward": reward}

    def handle_client(self, conn, addr):
        print(f"客户端连接: {addr}")

        try:
            while True:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break

                try:
                    message = json.loads(data)
                    msg_type = message.get("msg_type")

                    if msg_type == "state":
                        response = self.process_state_message(message)
                    elif msg_type == "reward":
                        response = self.process_reward_message(message)
                    else:
                        response = {"error": f"未知消息类型: {msg_type}"}

                    conn.send(json.dumps(response).encode('utf-8'))

                except json.JSONDecodeError:
                    error_msg = {"error": "JSON解析错误"}
                    conn.send(json.dumps(error_msg).encode('utf-8'))
                except Exception as e:
                    error_msg = {"error": f"处理错误: {str(e)}"}
                    conn.send(json.dumps(error_msg).encode('utf-8'))

        except ConnectionResetError:
            print(f"客户端断开连接: {addr}")
        finally:
            conn.close()

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"服务器开始监听 {self.host}:{self.port}")

            while True:
                conn, addr = s.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr)
                )
                client_thread.start()

if __name__ == "__main__":
    server = VeinsSocketServer()
    server.start_server()