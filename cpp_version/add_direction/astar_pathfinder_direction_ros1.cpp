#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

// ROS 头文件
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>

using json = nlohmann::json;

struct Node { int x, y; };
enum Direction { UP, DOWN, LEFT, RIGHT, NONE };
enum Gear { FORWARD, REVERSE, NEUTRAL }; 

struct State {
    int id;
    Direction front_dir; 
    Gear gear;           

    bool operator<(const State& other) const {
        if (id != other.id) return id < other.id;
        if (front_dir != other.front_dir) return front_dir < other.front_dir;
        return gear < other.gear;
    }
    bool operator==(const State& other) const {
        return id == other.id && front_dir == other.front_dir && gear == other.gear;
    }
};

Direction get_heading(Node u, Node v) {
    int dx = v.x - u.x;
    int dy = v.y - u.y;
    if (std::abs(dx) > std::abs(dy)) return dx > 0 ? RIGHT : LEFT;
    else return dy > 0 ? DOWN : UP;
}

Direction get_opposite(Direction d) {
    if (d == UP) return DOWN;
    if (d == DOWN) return UP;
    if (d == LEFT) return RIGHT;
    if (d == RIGHT) return LEFT;
    return NONE;
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

// 辅助函数：将方向转换为 ROS 的四元数 (基于图像坐标系，+X 向右，+Y 向下)
geometry_msgs::Quaternion dir_to_quat(Direction d) {
    tf2::Quaternion q;
    double yaw = 0.0;
    if (d == RIGHT) yaw = 0.0;
    else if (d == DOWN) yaw = M_PI / 2.0;    // 图像中向下是 Y 轴正方向
    else if (d == LEFT) yaw = M_PI;
    else if (d == UP) yaw = -M_PI / 2.0;     // 图像中向上是 Y 轴负方向
    
    q.setRPY(0.0, 0.0, yaw);
    geometry_msgs::Quaternion msg;
    msg.x = q.x();
    msg.y = q.y();
    msg.z = q.z();
    msg.w = q.w();
    return msg;
}

double heuristic(Node a, Node b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

bool find_path_segment(int start_id, Direction start_dir, Gear start_gear, int end_id, Direction target_dir, 
                       const std::vector<Node>& nodes, const std::map<int, std::vector<int>>& adj, 
                       std::vector<State>& out_path) {
    
    typedef std::pair<double, State> pq_elem;
    std::priority_queue<pq_elem, std::vector<pq_elem>, std::greater<pq_elem>> open_set;
    std::map<State, State> came_from;
    std::map<State, double> g_score;
    
    State start_state = {start_id, start_dir, start_gear};
    open_set.push({0.0, start_state});
    g_score[start_state] = 0.0;

    while (!open_set.empty()) {
        State current = open_set.top().second;
        open_set.pop();

        if (current.id == end_id && (current.front_dir == target_dir || start_id == end_id)) {
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
            double dist_cost = heuristic(nodes[current.id], nodes[neighbor_id]);

            if (current.gear == NEUTRAL) {
                // 起步允许挂前进或倒车
                if (move_dir != get_opposite(current.front_dir)) {
                    State next_state = {neighbor_id, move_dir, FORWARD};
                    double tentative_g = g_score[current] + dist_cost;
                    if (g_score.find(next_state) == g_score.end() || tentative_g < g_score[next_state]) {
                        came_from[next_state] = current; g_score[next_state] = tentative_g;
                        open_set.push({tentative_g + heuristic(nodes[neighbor_id], nodes[end_id]), next_state});
                    }
                }
                if (move_dir != current.front_dir) {
                    State next_state = {neighbor_id, get_opposite(move_dir), REVERSE};
                    double tentative_g = g_score[current] + dist_cost;
                    if (g_score.find(next_state) == g_score.end() || tentative_g < g_score[next_state]) {
                        came_from[next_state] = current; g_score[next_state] = tentative_g;
                        open_set.push({tentative_g + heuristic(nodes[neighbor_id], nodes[end_id]), next_state});
                    }
                }
            } 
            else if (current.gear == FORWARD) {
                // 动作 A：保持前进挡
                if (move_dir != get_opposite(current.front_dir)) {
                    State next_state = {neighbor_id, move_dir, FORWARD};
                    double tentative_g = g_score[current] + dist_cost;
                    if (g_score.find(next_state) == g_score.end() || tentative_g < g_score[next_state]) {
                        came_from[next_state] = current; g_score[next_state] = tentative_g;
                        open_set.push({tentative_g + heuristic(nodes[neighbor_id], nodes[end_id]), next_state});
                    }
                }
                // 动作 B：切换为倒车档
                if (move_dir == get_opposite(current.front_dir)) {
                    State next_state = {neighbor_id, current.front_dir, REVERSE}; 
                    double tentative_g = g_score[current] + dist_cost + 500.0; 
                    if (g_score.find(next_state) == g_score.end() || tentative_g < g_score[next_state]) {
                        came_from[next_state] = current; g_score[next_state] = tentative_g;
                        open_set.push({tentative_g + heuristic(nodes[neighbor_id], nodes[end_id]), next_state});
                    }
                }
            } 
            else if (current.gear == REVERSE) {
                // 动作 A：保持倒车档
                if (move_dir != current.front_dir) {
                    State next_state = {neighbor_id, get_opposite(move_dir), REVERSE};
                    double tentative_g = g_score[current] + dist_cost;
                    if (g_score.find(next_state) == g_score.end() || tentative_g < g_score[next_state]) {
                        came_from[next_state] = current; g_score[next_state] = tentative_g;
                        open_set.push({tentative_g + heuristic(nodes[neighbor_id], nodes[end_id]), next_state});
                    }
                }
                // 动作 B：切换为前进挡
                if (move_dir == current.front_dir) {
                    State next_state = {neighbor_id, current.front_dir, FORWARD}; 
                    double tentative_g = g_score[current] + dist_cost + 500.0; 
                    if (g_score.find(next_state) == g_score.end() || tentative_g < g_score[next_state]) {
                        came_from[next_state] = current; g_score[next_state] = tentative_g;
                        open_set.push({tentative_g + heuristic(nodes[neighbor_id], nodes[end_id]), next_state});
                    }
                }
            }
        }
    }
    return false;
}

struct MouseData {
    cv::Mat base_canvas;
    std::vector<std::pair<cv::Point, int>> node_info;
    std::set<int> gear_shift_nodes; 
};

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_MOUSEMOVE) {
        MouseData* data = static_cast<MouseData*>(userdata);
        cv::Mat display = data->base_canvas.clone();
        
        int min_dist = 2500; 
        int best_idx = -1;

        for (size_t i = 0; i < data->node_info.size(); i++) {
            int dx = data->node_info[i].first.x - x;
            int dy = data->node_info[i].first.y - y;
            int dist = dx * dx + dy * dy;
            if (dist < min_dist) { min_dist = dist; best_idx = i; }
        }

        if (best_idx != -1) {
            cv::Point pt = data->node_info[best_idx].first;
            int node_id = data->node_info[best_idx].second;

            cv::circle(display, pt, 6, cv::Scalar(0, 255, 0), -1);

            std::string text = std::to_string(node_id);
            if (data->gear_shift_nodes.count(node_id)) {
                text += " [Shift Gear]"; 
            }

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
            cv::Point text_org(pt.x + 10, pt.y - 5);

            cv::rectangle(display, text_org + cv::Point(-2, baseline + 2), 
                          text_org + cv::Point(text_size.width + 2, -text_size.height - 2), 
                          cv::Scalar(255, 255, 255), cv::FILLED);

            cv::Scalar text_color = data->gear_shift_nodes.count(node_id) ? cv::Scalar(255, 0, 0) : cv::Scalar(139, 0, 0);
            cv::putText(display, text, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv::LINE_AA);
        }
        cv::imshow("Multi-Waypoint A-Star (Kinematics)", display);
    }
}

int main(int argc, char* argv[]) {
    // 1. 初始化 ROS 节点（需在解析自定义参数前调用，以剥离 ROS 自带参数）
    ros::init(argc, argv, "astar_pathfinder_node");
    ros::NodeHandle nh;
    
    // 创建一个发布者，发布 nav_msgs::Path 消息。由于可能只发布一次，将 latch 设为 true
    ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("/path", 1, true);

    if (argc < 5 || argc % 2 == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "启动失败：参数数量错误！\n";
        std::cout << "用法: rosrun <包名> <节点名> <节点1> <朝向1> <节点2> <朝向2> ... <节点N> <朝向N>\n";
        std::cout << "========================================================\n\n";
        return -1;
    }

    std::vector<std::pair<int, Direction>> route;
    for (int i = 1; i < argc; i += 2) {
        int node_id = std::stoi(argv[i]);
        Direction dir = str_to_dir(argv[i+1]);
        if (dir == NONE) return -1;
        route.push_back({node_id, dir});
    }

    std::string img_path = "./9/9.png";
    std::string json_path = "./9/processed_graph.json"; 

    cv::Mat canvas = cv::imread(img_path, cv::IMREAD_COLOR);
    std::ifstream ifs(json_path);
    if (!ifs.is_open() || canvas.empty()) {
        std::cerr << "错误：找不到文件，请确认 " << img_path << " 和 " << json_path << " 存在！\n";
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

    std::vector<State> full_path;
    bool overall_success = true;

    for (size_t i = 0; i < route.size() - 1; ++i) {
        std::vector<State> segment_path;
        Direction current_start_dir = (i == 0) ? route[0].second : full_path.back().front_dir;
        Gear current_start_gear = (i == 0) ? NEUTRAL : full_path.back().gear;
        
        bool success = find_path_segment(route[i].first, current_start_dir, current_start_gear,
                                         route[i+1].first, route[i+1].second, 
                                         nodes, adj, segment_path);
        
        if (!success) {
            std::cout << "规划中断！无法找到可满足姿态要求的连通路线。\n";
            overall_success = false;
            break;
        }

        if (i == 0) full_path = segment_path;
        else full_path.insert(full_path.end(), segment_path.begin() + 1, segment_path.end());
    }

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
        
        // 2. 组装并发布 nav_msgs::Path
        nav_msgs::Path ros_path;
        ros_path.header.stamp = ros::Time::now();
        ros_path.header.frame_id = "map";

        for (size_t i = 0; i < full_path.size(); ++i) {
            geometry_msgs::PoseStamped pose;
            pose.header = ros_path.header;
            // 填入计算出的节点坐标 (这里直接使用解析出的 x 和 y，如果需要换算实际物理单位请按比例相乘)
            pose.pose.position.x = nodes[full_path[i].id].x;
            pose.pose.position.y = nodes[full_path[i].id].y;
            pose.pose.position.z = 0.0;
            // 填入姿态朝向四元数
            pose.pose.orientation = dir_to_quat(full_path[i].front_dir);
            
            ros_path.poses.push_back(pose);
            
            // 下方是原本的 UI 显示逻辑
            if (i < full_path.size() - 1) {
                if (full_path[i].gear != full_path[i+1].gear && full_path[i].gear != NEUTRAL) {
                    mouse_data.gear_shift_nodes.insert(full_path[i].id);
                }
            }
        }

        // 3. 发布数据到 ROS 系统中
        path_pub.publish(ros_path);
        ROS_INFO("publish to /path");

        for (size_t i = 0; i < full_path.size() - 1; i++) {
            cv::Point p1(nodes[full_path[i].id].x + PADDING, nodes[full_path[i].id].y + PADDING);
            cv::Point p2(nodes[full_path[i+1].id].x + PADDING, nodes[full_path[i+1].id].y + PADDING);
            
            cv::Scalar color = (full_path[i+1].gear == REVERSE) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
            cv::line(canvas, p1, p2, color, 3);
            cv::circle(canvas, p1, 5, color, -1);
        }

        if (!mouse_data.gear_shift_nodes.empty()) {
            for (int shift_id : mouse_data.gear_shift_nodes) {
                cv::Point pt(nodes[shift_id].x + PADDING, nodes[shift_id].y + PADDING);
                cv::circle(canvas, pt, 12, cv::Scalar(255, 255, 0), 2);
                cv::putText(canvas, "R", cv::Point(pt.x - 5, pt.y + 5), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            }
        }

        for (size_t i = 0; i < route.size(); ++i) {
            int node_id = route[i].first;
            Direction dir = route[i].second;
            cv::Point pt(nodes[node_id].x + PADDING, nodes[node_id].y + PADDING);
            
            cv::Scalar circle_color = (i == 0) ? cv::Scalar(0, 255, 0) : ((i == route.size() - 1) ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 165, 255));
            cv::circle(canvas, pt, 8, circle_color, -1);

            int dx = 0, dy = 0, l = 35;
            if(dir == UP) dy = l; else if(dir == DOWN) dy = -l;
            else if(dir == LEFT) dx = l; else if(dir == RIGHT) dx = -l;
            
            if (i == 0) {
                cv::arrowedLine(canvas, pt, cv::Point(pt.x - dx, pt.y - dy), circle_color, 3, 8, 0, 0.3);
            } else {
                cv::arrowedLine(canvas, cv::Point(pt.x + dx, pt.y + dy), pt, circle_color, 4, 8, 0, 0.3);
            }
        }
    }

    mouse_data.base_canvas = canvas.clone();
    cv::namedWindow("Multi-Waypoint A-Star (Kinematics)", cv::WINDOW_NORMAL);
    int max_w = 1280, max_h = 720;
    double scale = std::min((double)max_w / canvas.cols, (double)max_h / canvas.rows);
    cv::resizeWindow("Multi-Waypoint A-Star (Kinematics)", canvas.cols * scale, canvas.rows * scale);
    cv::setMouseCallback("Multi-Waypoint A-Star (Kinematics)", onMouse, &mouse_data);
    cv::imshow("Multi-Waypoint A-Star (Kinematics)", canvas);
    
    // 如果想要在 OpenCV 界面打开时依然能够响应 ROS 事件（例如接收订阅消息等），
    // 建议把原有的 cv::waitKey(0) 改成带超时时间的循环
    while (ros::ok()) {
        int key = cv::waitKey(30); // 30ms 刷新率
        if (key == 27 || key == 'q' || key == 'Q') { // 按 ESC 或 Q 退出
            break;
        }
        ros::spinOnce();
    }
    
    return 0;
}