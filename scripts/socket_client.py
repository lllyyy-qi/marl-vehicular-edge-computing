import socket
import json
import time

def test_socket_communication():
    # 测试状态消息
    state_message = {
        "car_id": 385,
        "msg_type": "state",
        "servers": [
            {
                "dist": 265.48,
                "load": 0.0,
                "name": "server[0]",
                "rate": 12091930.57
            },
            {
                "dist": 136.92,
                "load": 0.0,
                "name": "server[1]",
                "rate": 38398684.79
            }
        ],
        "task_info": {
            "demand": 500.0,
            "input_size": 1000.0,
            "is_busy": 0.0
        }
    }

    # 测试奖励消息
    reward_message = {
        "latency": 0.024,
        "msg_type": "reward"
    }

    # 连接服务器
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 8888))

            # 发送状态消息获取决策
            print("发送状态消息...")
            s.send(json.dumps(state_message).encode('utf-8'))
            response = s.recv(1024).decode('utf-8')
            print(f"收到决策响应: {response}")

            time.sleep(1)

            # 发送奖励消息
            print("发送奖励消息...")
            s.send(json.dumps(reward_message).encode('utf-8'))
            response = s.recv(1024).decode('utf-8')
            print(f"收到奖励响应: {response}")

    except ConnectionRefusedError:
        print("错误: 无法连接到服务器，请确保服务器正在运行")
    except Exception as e:
        print(f"通信错误: {e}")

if __name__ == "__main__":
    test_socket_communication()