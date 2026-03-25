import json
import cv2
import math
import heapq
import numpy as np

def load_graph_data(json_path):
    """读取拓扑地图 JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['nodes'], data['edges']

def build_adjacency_list(nodes, edges):
    """将边列表转换为邻接表，方便 A* 算法快速查找邻居"""
    adj = {i: [] for i in range(len(nodes))}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)  # 无向图，双向添加
    return adj

def heuristic_cost_estimate(node1, node2, nodes):
    """A* 的启发式函数：计算两点之间的欧几里得直线距离"""
    x1, y1 = nodes[node1]
    x2, y2 = nodes[node2]
    return math.hypot(x1 - x2, y1 - y2)

def a_star_search(start_id, goal_id, nodes, adj):
    """A* 核心算法实现"""
    if start_id not in adj or goal_id not in adj:
        print("错误：起点或终点序号不存在于地图中！")
        return None

    # 优先队列 (f_score, node_id)
    open_set = []
    heapq.heappush(open_set, (0, start_id))
    
    # 记录路径来源
    came_from = {}
    
    # g_score: 起点到当前点的实际代价
    g_score = {i: float('inf') for i in range(len(nodes))}
    g_score[start_id] = 0

    while open_set:
        # 取出 f_score 最小的节点
        current_f, current = heapq.heappop(open_set)

        # 找到终点，回溯路径
        if current == goal_id:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1] # 反转列表，得到从起点到终点的顺序

        # 遍历所有相连的邻居节点
        for neighbor in adj[current]:
            # 相邻两点的距离代价
            cost = heuristic_cost_estimate(current, neighbor, nodes)
            tentative_g_score = g_score[current] + cost

            if tentative_g_score < g_score[neighbor]:
                # 找到更优路径，更新记录
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_cost_estimate(neighbor, goal_id, nodes)
                heapq.heappush(open_set, (f_score, neighbor))
                
    # 队列空了还没找到，说明不连通
    return None

def draw_topology_and_path(img, nodes, edges, path=None):
    """在原图上绘制带有序号的节点、连线以及最终路径"""
    canvas = img.copy()

    # 1. 画出所有底层的连线 (浅蓝色)
    for u, v in edges:
        pt1 = tuple(nodes[u])
        pt2 = tuple(nodes[v])
        cv2.line(canvas, pt1, pt2, (255, 200, 150), 1)

    # 2. 画出所有节点，并标上序号 ID (深蓝色文字)
    for i, pt in enumerate(nodes):
        pt_tuple = tuple(pt)
        cv2.circle(canvas, pt_tuple, 3, (0, 0, 0), -1)
        # 在节点右上方稍微偏移的位置写上序号，使用深蓝色 (139, 0, 0)
        cv2.putText(canvas, str(i), (pt[0] + 5, pt[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139, 0, 0), 1, cv2.LINE_AA)

    # 3. 如果找到了路径，用醒目的红色粗线高亮显示
    if path:
        print(f"找到路径，经过的节点依次为: {path}")
        for i in range(len(path) - 1):
            p1 = tuple(nodes[path[i]])
            p2 = tuple(nodes[path[i+1]])
            cv2.line(canvas, p1, p2, (0, 0, 255), 3) # 红色路径线
            
        for node_id in path:
            cv2.circle(canvas, tuple(nodes[node_id]), 5, (0, 0, 255), -1) # 红色路径点

        # 特别标记起点(绿色)和终点(紫色)
        start_pt = tuple(nodes[path[0]])
        goal_pt = tuple(nodes[path[-1]])
        cv2.circle(canvas, start_pt, 8, (0, 255, 0), -1)
        cv2.putText(canvas, " ", (start_pt[0]-15, start_pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(canvas, goal_pt, 8, (255, 0, 255), -1)
        cv2.putText(canvas, " ", (goal_pt[0]-15, goal_pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    else:
        print("未找到有效路径！(可能两点之间不连通)")

    return canvas

if __name__ == "__main__":
    # --- 配置文件路径 ---
    image_path = "./3/3.png"
    json_path = "./3/map_graph.json"

    # 1. 加载数据
    img0 = cv2.imread(image_path)
    if img0 is None:
        print(f"图片加载失败，请检查路径: {image_path}")
        exit()

    try:
        nodes, edges = load_graph_data(json_path)
    except FileNotFoundError:
        print(f"找不到 JSON 文件 {json_path}，请先运行生成脚本！")
        exit()

    adj_list = build_adjacency_list(nodes, edges)

    # ==========================================
    # 在这里手动设置起点和终点序号
    # ==========================================
    START_NODE = 24   
    END_NODE = 134  # len(nodes) - 1 # 默认设为最后一个节点

    # 2. 运行 A* 算法
    print(f"正在规划从节点 {START_NODE} 到节点 {END_NODE} 的最短路径...")
    shortest_path = a_star_search(START_NODE, END_NODE, nodes, adj_list)

    # 3. 绘制并显示
    result_img = draw_topology_and_path(img0, nodes, edges, shortest_path)

    # 缩放窗口显示
    window_name = "A-Star Path Planning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = result_img.shape[:2]
    scale = min(1280 / w, 720 / h, 1.0)
    cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    
    cv2.imshow(window_name, result_img)
    print("按任意键退出程序...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
