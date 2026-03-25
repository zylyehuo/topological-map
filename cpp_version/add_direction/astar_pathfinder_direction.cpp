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

struct Node { int x, y; };
enum Direction { UP, DOWN, LEFT, RIGHT, NONE };

struct State {
    int id;
    Direction dir;
    bool operator<(const State& other) const {
        if (id != other.id) return id < other.id;
        return dir < other.dir;
    }
    bool operator==(const State& other) const {
        return id == other.id && dir == other.dir;
    }
};

Direction get_heading(Node u, Node v) {
    int dx = v.x - u.x;
    int dy = v.y - u.y;
    if (std::abs(dx) > std::abs(dy)) return dx > 0 ? RIGHT : LEFT;
    else return dy > 0 ? DOWN : UP;
}

Direction str_to_dir(const std::string& s) {
    if (s == "up" || s == "w" || s == "W") return UP;
    if (s == "down" || s == "s" || s == "S") return DOWN;
    if (s == "left" || s == "a" || s == "A") return LEFT;
    if (s == "right" || s == "d" || s == "D") return RIGHT;
    return NONE;
}

std::string dir_to_str(Direction d) {
    if (d == UP) return "上 (w)";
    if (d == DOWN) return "下 (s)";
    if (d == LEFT) return "左 (a)";
    if (d == RIGHT) return "右 (d)";
    return "无 (-)";
}

bool is_opposite(Direction d1, Direction d2) {
    if (d1 == NONE || d2 == NONE) return false;
    if (d1 == UP && d2 == DOWN) return true;
    if (d1 == DOWN && d2 == UP) return true;
    if (d1 == LEFT && d2 == RIGHT) return true;
    if (d1 == RIGHT && d2 == LEFT) return true;
    return false;
}

double heuristic(Node a, Node b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

// 提取出的单段 A* 搜索函数
bool find_path_segment(int start_id, Direction start_dir, int end_id, Direction target_dir, 
                       const std::vector<Node>& nodes, const std::map<int, std::vector<int>>& adj, 
                       std::vector<State>& out_path) {
    
    typedef std::pair<double, State> pq_elem;
    std::priority_queue<pq_elem, std::vector<pq_elem>, std::greater<pq_elem>> open_set;
    std::map<State, State> came_from;
    std::map<State, double> g_score;
    
    State start_state = {start_id, start_dir};
    open_set.push({0.0, start_state});
    g_score[start_state] = 0.0;

    while (!open_set.empty()) {
        State current = open_set.top().second;
        open_set.pop();

        if (current.id == end_id && (current.dir == target_dir || start_id == end_id)) {
            std::vector<State> path;
            State curr = current;
            while (!(curr == start_state)) {
                path.push_back(curr);
                curr = came_from[curr];
            }
            path.push_back(start_state);
            std::reverse(path.begin(), path.end());
            out_path = path;
            return true;
        }

        for (int neighbor_id : adj.at(current.id)) {
            Direction move_dir = get_heading(nodes[current.id], nodes[neighbor_id]);
            if (is_opposite(current.dir, move_dir)) continue; 

            State neighbor_state = {neighbor_id, move_dir};
            double cost = heuristic(nodes[current.id], nodes[neighbor_id]);
            double tentative_g_score = g_score[current] + cost;

            if (g_score.find(neighbor_state) == g_score.end() || tentative_g_score < g_score[neighbor_state]) {
                came_from[neighbor_state] = current;
                g_score[neighbor_state] = tentative_g_score;
                double f_score = tentative_g_score + heuristic(nodes[neighbor_id], nodes[end_id]);
                open_set.push({f_score, neighbor_state});
            }
        }
    }
    return false;
}

// 鼠标悬停交互回调
struct MouseData {
    cv::Mat base_canvas;
    std::vector<std::pair<cv::Point, int>> node_info;
};

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_MOUSEMOVE) {
        MouseData* data = static_cast<MouseData*>(userdata);
        cv::Mat display = data->base_canvas.clone();
        
        for (const auto& node : data->node_info) {
            int dx = node.first.x - x;
            int dy = node.first.y - y;
            if (dx * dx + dy * dy <= 100) { 
                cv::circle(display, node.first, 6, cv::Scalar(0, 255, 0), -1);
                cv::putText(display, std::to_string(node.second), 
                            cv::Point(node.first.x + 8, node.first.y - 8),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(139, 0, 0), 2, cv::LINE_AA);
                break; 
            }
        }
        cv::imshow("Multi-Waypoint A-Star", display);
    }
}

int main(int argc, char* argv[]) {
    // --- 0. 解析无限参数序列 ---
    if (argc < 5 || argc % 2 == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "启动失败：参数数量错误！\n";
        std::cout << "用法: ./astar_pathfinder <节点1> <朝向1> <节点2> <朝向2> ... <节点N> <朝向N>\n";
        std::cout << "示例: ./astar_pathfinder 1185 w 333 a 85 w 3 d\n";
        std::cout << "========================================================\n\n";
        return -1;
    }

    std::vector<std::pair<int, Direction>> route;
    for (int i = 1; i < argc; i += 2) {
        int node_id = std::stoi(argv[i]);
        Direction dir = str_to_dir(argv[i+1]);
        if (dir == NONE) {
            std::cerr << "错误：无效的朝向参数 '" << argv[i+1] << "'，请使用 w, s, a, d。\n";
            return -1;
        }
        route.push_back({node_id, dir});
    }

    // --- 1. 加载数据 ---
    std::string img_path = "./5/5.png";
    std::string json_path = "./5/map_graph.json"; 

    cv::Mat canvas = cv::imread(img_path, cv::IMREAD_COLOR);
    std::ifstream ifs(json_path);
    if (!ifs.is_open() || canvas.empty()) {
        std::cerr << "错误：找不到文件，请确认 ./5/5.png 和 map_graph.json 存在！\n";
        return -1;
    }

    const int PADDING = 40;
    cv::Mat padded_canvas;
    cv::copyMakeBorder(canvas, padded_canvas, PADDING, PADDING, PADDING, PADDING, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    canvas = padded_canvas;

    json j_graph;
    ifs >> j_graph;
    ifs.close();

    std::vector<Node> nodes;
    for (auto& n : j_graph["nodes"]) nodes.push_back({n[0], n[1]});

    std::map<int, std::vector<int>> adj;
    for (auto& e : j_graph["edges"]) {
        int u = e[0], v = e[1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    std::cout << "==================================================\n";
    std::cout << "正在规划多途经点姿态约束路径...\n";
    for (size_t i = 0; i < route.size(); ++i) {
        std::cout << "   " << (i==0 ? "起点" : (i==route.size()-1 ? "终点" : "途经")) 
                  << " " << route[i].first << "，要求车头驶入朝向: " << dir_to_str(route[i].second) << "\n";
    }
    std::cout << "==================================================\n";

    // --- 2. 逐段执行 A* 算法 ---
    std::vector<State> full_path;
    bool overall_success = true;

    for (size_t i = 0; i < route.size() - 1; ++i) {
        std::vector<State> segment_path;
        bool success = find_path_segment(route[i].first, route[i].second, 
                                         route[i+1].first, route[i+1].second, 
                                         nodes, adj, segment_path);
        
        if (!success) {
            std::cout << "规划中断！无法找到从 " << route[i].first << " 到 " << route[i+1].first << " 且满足姿态要求的路线。\n";
            overall_success = false;
            break;
        }

        // 拼接路径，去掉重叠的连接点
        if (i == 0) {
            full_path = segment_path;
        } else {
            full_path.insert(full_path.end(), segment_path.begin() + 1, segment_path.end());
        }
    }

    // --- 绘图 ---
    MouseData mouse_data;
    for (auto& e : j_graph["edges"]) {
        cv::line(canvas, cv::Point(nodes[e[0]].x + PADDING, nodes[e[0]].y + PADDING), 
                         cv::Point(nodes[e[1]].x + PADDING, nodes[e[1]].y + PADDING), cv::Scalar(250, 200, 150), 1);
    }
    
    for (int i = 0; i < nodes.size(); i++) {
        cv::Point display_pt(nodes[i].x + PADDING, nodes[i].y + PADDING);
        cv::circle(canvas, display_pt, 3, cv::Scalar(0, 0, 0), -1);
        mouse_data.node_info.push_back({display_pt, i});
    }

    if (overall_success) {
        // 画出完整的红色主线
        for (size_t i = 0; i < full_path.size() - 1; i++) {
            cv::Point p1(nodes[full_path[i].id].x + PADDING, nodes[full_path[i].id].y + PADDING);
            cv::Point p2(nodes[full_path[i+1].id].x + PADDING, nodes[full_path[i+1].id].y + PADDING);
            cv::line(canvas, p1, p2, cv::Scalar(0, 0, 255), 3);
            cv::circle(canvas, p1, 5, cv::Scalar(0, 0, 255), -1);
        }

        // 绘制所有的关键节点标记与姿态箭头
        for (size_t i = 0; i < route.size(); ++i) {
            int node_id = route[i].first;
            Direction dir = route[i].second;
            cv::Point pt(nodes[node_id].x + PADDING, nodes[node_id].y + PADDING);
            
            cv::Scalar circle_color;
            cv::Scalar arrow_color;
            
            if (i == 0) {
                circle_color = cv::Scalar(0, 255, 0);       // 起点：绿色圈
                arrow_color = cv::Scalar(0, 200, 0);        // 起点箭头也是绿色
            } else if (i == route.size() - 1) {
                circle_color = cv::Scalar(255, 0, 255);     // 终点：紫色圈
                arrow_color = cv::Scalar(0, 255, 255);      // 终点箭头黄色
            } else {
                circle_color = cv::Scalar(0, 165, 255);     // 途经点：橙色圈
                arrow_color = cv::Scalar(0, 165, 255);      // 途经点箭头橙色
            }

            cv::circle(canvas, pt, 8, circle_color, -1);

            int dx = 0, dy = 0, l = 35;
            if(dir == UP) dy = l; else if(dir == DOWN) dy = -l;
            else if(dir == LEFT) dx = l; else if(dir == RIGHT) dx = -l;
            
            if (i == 0) {
                // 起点是“驶出方向”，箭头从圆心向外指
                cv::arrowedLine(canvas, pt, cv::Point(pt.x - dx, pt.y - dy), arrow_color, 3, 8, 0, 0.3);
            } else {
                // 途经点和终点都是“驶入方向”，箭头指向圆心
                cv::arrowedLine(canvas, cv::Point(pt.x + dx, pt.y + dy), pt, arrow_color, 4, 8, 0, 0.3);
            }
        }

        std::cout << "多点路线规划成功！\n";
    }

    mouse_data.base_canvas = canvas.clone();
    cv::namedWindow("Multi-Waypoint A-Star", cv::WINDOW_NORMAL);
    int max_w = 1280, max_h = 720;
    double scale = std::min((double)max_w / canvas.cols, (double)max_h / canvas.rows);
    cv::resizeWindow("Multi-Waypoint A-Star", canvas.cols * scale, canvas.rows * scale);
    cv::setMouseCallback("Multi-Waypoint A-Star", onMouse, &mouse_data);
    cv::imshow("Multi-Waypoint A-Star", canvas);
    
    cv::waitKey(0);
    return 0;
}
