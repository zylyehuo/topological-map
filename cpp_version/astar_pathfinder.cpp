#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <queue>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Node {
    int x, y;
};

double heuristic(Node a, Node b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

int main() {
    // --- 1. 配置路径与参数 ---
    std::string img_path = "./5/5.png";
    std::string json_path = "./5/map_graph.json";
    
    int START_NODE = 922;
    int END_NODE = 703;

    // --- 2. 加载图片和 JSON 数据 ---
    cv::Mat canvas = cv::imread(img_path, cv::IMREAD_COLOR);
    if (canvas.empty()) {
        std::cerr << "错误：无法读取图片 " << img_path << std::endl;
        return -1;
    }

    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        std::cerr << "错误：找不到 JSON 文件 " << json_path << std::endl;
        return -1;
    }

    json j_graph;
    ifs >> j_graph;

    std::vector<Node> nodes;
    for (auto& n : j_graph["nodes"]) {
        nodes.push_back({n[0], n[1]});
    }

    std::map<int, std::vector<int>> adj;
    for (auto& e : j_graph["edges"]) {
        int u = e[0], v = e[1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // --- 3. 运行 A* 算法 ---
    if (START_NODE >= nodes.size() || END_NODE >= nodes.size()) {
        std::cerr << "起点或者终点索引越界！" << std::endl;
        return -1;
    }

    // 优先队列：<f_score, node_id>，使用小顶堆
    typedef std::pair<double, int> pq_elem;
    std::priority_queue<pq_elem, std::vector<pq_elem>, std::greater<pq_elem>> open_set;
    
    std::map<int, int> came_from;
    std::vector<double> g_score(nodes.size(), INFINITY);
    
    open_set.push({0.0, START_NODE});
    g_score[START_NODE] = 0.0;

    std::vector<int> path;
    bool found = false;

    std::cout << "正在规划从 " << START_NODE << " 到 " << END_NODE << " 的路径..." << std::endl;

    while (!open_set.empty()) {
        int current = open_set.top().second;
        open_set.pop();

        if (current == END_NODE) {
            found = true;
            while (came_from.find(current) != came_from.end()) {
                path.push_back(current);
                current = came_from[current];
            }
            path.push_back(START_NODE);
            std::reverse(path.begin(), path.end());
            break;
        }

        for (int neighbor : adj[current]) {
            double cost = heuristic(nodes[current], nodes[neighbor]);
            double tentative_g_score = g_score[current] + cost;

            if (tentative_g_score < g_score[neighbor]) {
                came_from[neighbor] = current;
                g_score[neighbor] = tentative_g_score;
                double f_score = tentative_g_score + heuristic(nodes[neighbor], nodes[END_NODE]);
                open_set.push({f_score, neighbor});
            }
        }
    }

    // --- 4. 可视化绘制 ---
    // 绘制浅蓝色底图连线
    for (auto& e : j_graph["edges"]) {
        int u = e[0], v = e[1];
        cv::line(canvas, cv::Point(nodes[u].x, nodes[u].y), cv::Point(nodes[v].x, nodes[v].y), cv::Scalar(250, 200, 150), 1);
    }

    // 绘制所有节点和深蓝色 ID
    for (int i = 0; i < nodes.size(); i++) {
        cv::Point pt(nodes[i].x, nodes[i].y);
        cv::circle(canvas, pt, 3, cv::Scalar(0, 0, 0), -1);
        cv::putText(canvas, std::to_string(i), cv::Point(pt.x + 5, pt.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(139, 0, 0), 1, cv::LINE_AA);
    }

    // 绘制高亮红色路径
    if (found) {
        for (size_t i = 0; i < path.size() - 1; i++) {
            cv::Point p1(nodes[path[i]].x, nodes[path[i]].y);
            cv::Point p2(nodes[path[i+1]].x, nodes[path[i+1]].y);
            cv::line(canvas, p1, p2, cv::Scalar(0, 0, 255), 3);
            cv::circle(canvas, p1, 5, cv::Scalar(0, 0, 255), -1);
        }
        cv::circle(canvas, cv::Point(nodes[path.back()].x, nodes[path.back()].y), 5, cv::Scalar(0, 0, 255), -1);

        // 标记起终点
        cv::circle(canvas, cv::Point(nodes[START_NODE].x, nodes[START_NODE].y), 8, cv::Scalar(0, 255, 0), -1);
        cv::circle(canvas, cv::Point(nodes[END_NODE].x, nodes[END_NODE].y), 8, cv::Scalar(255, 0, 255), -1);
        std::cout << "规划成功！" << std::endl;
    } else {
        std::cout << "未找到有效路径！" << std::endl;
    }

    // 显示窗口
    cv::namedWindow("A-Star Path", cv::WINDOW_NORMAL);
    int max_w = 1280, max_h = 720;
    double scale = std::min((double)max_w / canvas.cols, (double)max_h / canvas.rows);
    cv::resizeWindow("A-Star Path", canvas.cols * scale, canvas.rows * scale);
    cv::imshow("A-Star Path", canvas);
    
    cv::waitKey(0);
    return 0;
}
