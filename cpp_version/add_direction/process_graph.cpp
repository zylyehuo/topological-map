#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "json.hpp" // 引入 nlohmann json

using json = nlohmann::json;

struct Node {
    int x, y;
    bool active;
    Node(int _x, int _y, bool _active = true) : x(_x), y(_y), active(_active) {}
};

struct Edge {
    int u, v;
    Edge(int _u, int _v) : u(_u), v(_v) {}
};

// 检查线段是否不仅可通行，且距离障碍物有一定安全距离 (避免贴墙)
bool isLineClearThick(cv::Point p1, cv::Point p2, const cv::Mat& dt, float min_dist = 2.0f) {
    cv::LineIterator it(dt, p1, p2, 8);
    for (int i = 0; i < it.count; i++, ++it) {
        if (dt.at<float>(it.pos()) < min_dist) {
            return false;
        }
    }
    return true;
}

// 寻找 L 型路径的有效拐点。如果两种拐法都可行，选择处于更宽阔区域的那个
cv::Point getValidLShapeCorner(cv::Point A, cv::Point B, const cv::Mat& dt, float min_dist = 2.0f) {
    cv::Point C1(A.x, B.y);
    cv::Point C2(B.x, A.y);

    bool ok1 = isLineClearThick(A, C1, dt, min_dist) && isLineClearThick(C1, B, dt, min_dist);
    bool ok2 = isLineClearThick(A, C2, dt, min_dist) && isLineClearThick(C2, B, dt, min_dist);

    if (ok1 && ok2) {
        return dt.at<float>(C1) > dt.at<float>(C2) ? C1 : C2;
    } else if (ok1) {
        return C1;
    } else if (ok2) {
        return C2;
    }
    return cv::Point(-1, -1);
}

// 核心优化算法
void optimize_graph(std::vector<Node>& nodes, std::vector<Edge>& edges, const cv::Mat& img) {
    // 1. 生成距离变换图
    cv::Mat dt;
    cv::distanceTransform(img > 127, dt, cv::DIST_L2, 3);

    // 2. 节点全局居中：将节点安全地推向通道距离最大处
    int radius = 5;
    for (int iter = 0; iter < 3; ++iter) {
        for (auto& n : nodes) {
            if (!n.active) continue;
            int bx = n.x, by = n.y;
            float bval = dt.at<float>(n.y, n.x);
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = n.x + dx, ny = n.y + dy;
                    if (nx >= 0 && nx < img.cols && ny >= 0 && ny < img.rows) {
                        float val = dt.at<float>(ny, nx);
                        if (val > bval) {
                            // 确保居中过程不会穿墙
                            if(isLineClearThick(cv::Point(n.x, n.y), cv::Point(nx, ny), dt, 1.0f)) {
                                bval = val; bx = nx; by = ny;
                            }
                        }
                    }
                }
            }
            n.x = bx; n.y = by;
        }
    }

    // 3. 构建邻接表用于拓扑修改
    std::vector<std::unordered_set<int>> adj(nodes.size());
    for (const auto& e : edges) {
        if (e.u != e.v) {
            adj[e.u].insert(e.v);
            adj[e.v].insert(e.u);
        }
    }

    // 4. 核心化简：链式路径拉直 (消除锯齿，强制正交)
    bool global_changed = true;
    float min_clearance = 3.0f; // 要求新线条距离障碍物的最小安全距离

    while (global_changed) {
        global_changed = false;
        std::vector<bool> visited(nodes.size(), false);

        for (int i = 0; i < nodes.size(); ++i) {
            if (!nodes[i].active || visited[i]) continue;

            // 从端点或交叉路口开始追踪 "链" (Chain)
            if (adj[i].size() == 1 || adj[i].size() >= 3 || (adj[i].size() == 2 && i == 0)) {
                for (int neighbor : adj[i]) {
                    if (visited[neighbor]) continue;

                    std::vector<int> chain;
                    chain.push_back(i);
                    int curr = neighbor;
                    int prev = i;

                    // 追踪连续的度为 2 的节点
                    while (true) {
                        chain.push_back(curr);
                        visited[curr] = true;
                        if (adj[curr].size() != 2) break; 

                        int next_node = -1;
                        for (int nxt : adj[curr]) {
                            if (nxt != prev) { next_node = nxt; break; }
                        }
                        if (next_node == -1) break;
                        prev = curr;
                        curr = next_node;
                    }

                    // 对整条链尝试直接拉直或使用最少的 L 型连接
                    if (chain.size() > 2) {
                        int c_idx = 0;
                        std::vector<cv::Point> new_points;
                        new_points.push_back(cv::Point(nodes[chain[0]].x, nodes[chain[0]].y));

                        while (c_idx < chain.size() - 1) {
                            int best_next = -1;
                            int path_type = 0; // 1: 一条直线, 2: 一个L型直角
                            cv::Point best_corner(-1, -1);

                            // 从最远处向回尝试跳跃，贪心寻找最长的笔直或单拐点路径
                            for (int k = chain.size() - 1; k > c_idx; --k) {
                                cv::Point A(nodes[chain[c_idx]].x, nodes[chain[c_idx]].y);
                                cv::Point B(nodes[chain[k]].x, nodes[chain[k]].y);

                                // 尝试直线
                                if (A.x == B.x || A.y == B.y) {
                                    if (isLineClearThick(A, B, dt, min_clearance)) {
                                        best_next = k; path_type = 1; break;
                                    }
                                }
                                // 尝试 L 型
                                cv::Point corner = getValidLShapeCorner(A, B, dt, min_clearance);
                                if (corner.x != -1) {
                                    best_next = k; path_type = 2; best_corner = corner; break;
                                }
                            }

                            if (best_next != -1) {
                                if (path_type == 2) new_points.push_back(best_corner);
                                if (best_next < chain.size() - 1) new_points.push_back(cv::Point(nodes[chain[best_next]].x, nodes[chain[best_next]].y));
                                c_idx = best_next;
                            } else {
                                c_idx++;
                                if (c_idx < chain.size() - 1) new_points.push_back(cv::Point(nodes[chain[c_idx]].x, nodes[chain[c_idx]].y));
                            }
                        }
                        new_points.push_back(cv::Point(nodes[chain.back()].x, nodes[chain.back()].y));

                        // 如果路径被大幅简化（节点数减少），更新拓扑图
                        if (new_points.size() < chain.size()) {
                            // 移除旧边和旧节点
                            for (size_t k = 0; k < chain.size() - 1; ++k) {
                                adj[chain[k]].erase(chain[k+1]);
                                adj[chain[k+1]].erase(chain[k]);
                            }
                            for (size_t k = 1; k < chain.size() - 1; ++k) nodes[chain[k]].active = false;

                            // 插入新结构
                            int last_id = chain[0];
                            for (size_t p = 1; p < new_points.size() - 1; ++p) {
                                nodes.push_back(Node(new_points[p].x, new_points[p].y, true));
                                int new_id = nodes.size() - 1;
                                adj.push_back(std::unordered_set<int>());
                                adj[last_id].insert(new_id); adj[new_id].insert(last_id);
                                last_id = new_id;
                            }
                            adj[last_id].insert(chain.back()); adj[chain.back()].insert(last_id);

                            global_changed = true;
                            break; // 拓扑改变，跳出当前循环重新检测
                        }
                    }
                }
                if (global_changed) break;
            }
        }
    }

    // 5. 将处理后的邻接表转换回边集，并执行最终的绝对正交保护（针对极少数直接相连但不平行的路口）
    std::vector<Edge> temp_edges;
    for (int u = 0; u < adj.size(); ++u) {
        for (int v : adj[u]) {
            if (u < v && nodes[u].active && nodes[v].active) {
                temp_edges.push_back(Edge(u, v));
            }
        }
    }

    std::vector<Edge> final_ortho_edges;
    for (const auto& e : temp_edges) {
        Node& u = nodes[e.u]; Node& v = nodes[e.v];
        if (u.x == v.x || u.y == v.y) {
            final_ortho_edges.push_back(e);
        } else {
            // 兜底策略：强制插入90度直角
            cv::Point A(u.x, u.y), B(v.x, v.y);
            cv::Point corner = getValidLShapeCorner(A, B, dt, 1.0f); // 兜底放宽限制
            if (corner.x == -1) corner = cv::Point(A.x, B.y); // 极端情况忽略碰撞墙体强制成角

            nodes.push_back(Node(corner.x, corner.y, true));
            int c_id = nodes.size() - 1;
            final_ortho_edges.push_back(Edge(e.u, c_id));
            final_ortho_edges.push_back(Edge(c_id, e.v));
        }
    }
    edges = final_ortho_edges;

    // 6. 清理冗余的共线中间节点
    bool cleanup = true;
    while (cleanup) {
        cleanup = false;
        std::vector<std::vector<int>> cur_adj(nodes.size());
        for (auto& e : edges) {
            cur_adj[e.u].push_back(e.v);
            cur_adj[e.v].push_back(e.u);
        }

        for (int i = 0; i < nodes.size(); ++i) {
            if (!nodes[i].active || cur_adj[i].size() != 2) continue;
            int u = cur_adj[i][0];
            int v = cur_adj[i][1];
            Node& N = nodes[i]; Node& U = nodes[u]; Node& V = nodes[v];

            if ((U.x == N.x && N.x == V.x) || (U.y == N.y && N.y == V.y)) {
                if (isLineClearThick(cv::Point(U.x, U.y), cv::Point(V.x, V.y), dt, 1.0f)) {
                    N.active = false;
                    cleanup = true;
                    std::vector<Edge> new_edges_set;
                    for (auto& e : edges) {
                        if (e.u == i || e.v == i) continue;
                        new_edges_set.push_back(e);
                    }
                    new_edges_set.push_back(Edge(u, v));
                    edges = new_edges_set;
                    break;
                }
            }
        }
    }
}

int main() {
    // 1. 读取图像与 JSON
    cv::Mat img = cv::imread("./9/9.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "错误：无法读取图片 ./9/9.png" << std::endl;
        return -1;
    }
    cv::threshold(img, img, 127, 255, cv::THRESH_BINARY);

    std::ifstream ifs("./9/map_graph.json");
    if (!ifs.is_open()) {
        std::cerr << "错误：无法读取 map_graph.json" << std::endl;
        return -1;
    }
    json j; ifs >> j;

    std::vector<Node> raw_nodes;
    for (auto& n : j["nodes"]) raw_nodes.push_back(Node(n[0].get<int>(), n[1].get<int>(), true));
    
    int num_raw_nodes = raw_nodes.size();
    std::vector<std::vector<int>> adj(num_raw_nodes);
    for (auto& e : j["edges"]) {
        adj[e[0]].push_back(e[1]);
        adj[e[1]].push_back(e[0]);
    }

    // 2. 提取主骨骼 (仅保留连通性最大的一组)
    std::vector<bool> visited(num_raw_nodes, false);
    std::vector<int> largest_component;

    for (int i = 0; i < num_raw_nodes; ++i) {
        if (!visited[i]) {
            std::vector<int> comp;
            std::queue<int> q;
            q.push(i); visited[i] = true;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                comp.push_back(u);
                for (int v : adj[u]) {
                    if (!visited[v]) {
                        visited[v] = true;
                        q.push(v);
                    }
                }
            }
            if (comp.size() > largest_component.size()) largest_component = comp;
        }
    }

    std::vector<Node> main_nodes;
    std::unordered_map<int, int> old_to_new;
    for (size_t i = 0; i < largest_component.size(); ++i) {
        old_to_new[largest_component[i]] = i;
        main_nodes.push_back(raw_nodes[largest_component[i]]);
    }

    std::vector<Edge> main_edges;
    for (auto& e : j["edges"]) {
        if (old_to_new.count(e[0]) && old_to_new.count(e[1])) {
            main_edges.push_back(Edge(old_to_new[e[0]], old_to_new[e[1]]));
        }
    }

    // 3. 执行全套优化算法：居中、正交化与折线消除
    std::vector<Node> final_nodes = main_nodes;
    std::vector<Edge> final_edges = main_edges;
    optimize_graph(final_nodes, final_edges, img);

    // 4. 结果可视化展示（仅预览用，不保存图片）
    cv::Mat out_img;
    cv::cvtColor(img, out_img, cv::COLOR_GRAY2BGR);

    for (const auto& e : final_edges) {
        if(final_nodes[e.u].active && final_nodes[e.v].active) {
            cv::line(out_img, cv::Point(final_nodes[e.u].x, final_nodes[e.u].y), 
                             cv::Point(final_nodes[e.v].x, final_nodes[e.v].y), cv::Scalar(255, 0, 0), 2);
        }
    }

    for (const auto& n : final_nodes) {
        if(n.active) {
            cv::circle(out_img, cv::Point(n.x, n.y), 4, cv::Scalar(0, 255, 0), -1);
        }
    }

    cv::namedWindow("Processed Graph", cv::WINDOW_NORMAL);
    int max_w = 1280, max_h = 720;
    double scale = std::min((double)max_w / out_img.cols, (double)max_h / out_img.rows);
    cv::resizeWindow("Processed Graph", out_img.cols * scale, out_img.rows * scale);
    cv::imshow("Processed Graph", out_img);
    
    std::cout << "优化处理完毕。请查看可视化结果，按任意键保存 JSON 并退出..." << std::endl;
    cv::waitKey(0);

    // 5. 将处理后干净的拓扑图保存为新 JSON
    json out_j;
    std::unordered_map<int, int> final_index_map;
    int current_index = 0;
    
    // 只输出有效的活跃节点
    for (size_t i = 0; i < final_nodes.size(); ++i) {
        if (final_nodes[i].active) {
            out_j["nodes"].push_back({final_nodes[i].x, final_nodes[i].y});
            final_index_map[i] = current_index++;
        }
    }
    
    // 映射连线关系
    for (const auto& e : final_edges) {
        if (final_nodes[e.u].active && final_nodes[e.v].active) {
            out_j["edges"].push_back({final_index_map[e.u], final_index_map[e.v]});
        }
    }

    std::ofstream ofs("./9/processed_map_graph.json");
    ofs << out_j.dump(4);
    ofs.close();
    std::cout << "干净的骨架拓扑结构已保存至 ./9/processed_map_graph.json" << std::endl;

    return 0;
}