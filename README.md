# Innovation Project: MARL for Vehicular Edge Computing

基于多智能体强化学习的车辆边缘计算卸载决策系统。

## 项目结构
- `src/`: 核心源代码
- `scripts/`: 运行脚本
- `configs/`: 配置文件
- `outputs/`: 输出文件


## 环境要求
- Python 3.8+
- PyTorch 1.9+
- SUMO 1.8.0
- VEINS 5.2



## 归一化需要重点讨论，关于最大值为多少等，具体涉及在 veins_interface.py 以及 environment.yaml 中

## 模拟运行在 dqn_learning.py ,至少能确定可以在训练后得到成果，但是仅限于场景未改变，并未考虑小车移动

新增文件，作为和udp通信的接口veins_udp_server.py，运行：`python veins_udp_server.py`

当前能和omnet进行通信
{
"msg_type": "state",
"car_id": 0,
"task_info": {
"input_size": 2000.0,
"demand": 1000,
"is_busy": 0.0
},
"servers": [
{
"name": "server[0]",
"dist": 250.3,
"rate": 2692125.1966334726,
"load": 0.12
},
{
"name": "server[1]",
"dist": 400.7,
"rate": 1234567.89,
"load": 0.05
}
],
"car": {
"load": 0.001, // currentQueueLoad_/maxQueueCapacity_, 0..1
"rate": 2692125.1966334726 // best uplink rate (bps) among servers
}
}消息格式如上

然而问题是：我们目前使用UDP协议进行通信，任务reward到达的顺序不能得到保证，很可能得到错误的匹配。所以在omnet端我们设置了在carload 不等于 0 的时候需要等待任务结束，这样虽然保证了任务每次只处理一个不会匹配错误，但是造成的问题是：RL收到的小时永远carload = 0,不能有效学习知识。

所以下一步骤的解决方法是为每个任务reward添加id和nextstate，对应task和action，以供智能体学习。形成完整的强化学习回路。
