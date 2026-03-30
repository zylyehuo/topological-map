#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <map>
#include <set>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;

// --- 可调参数 ---
const double MERGE_DIST_THRESH = 20.0; // 节点合并距离阈值 (像素)
const double BLACK_DIST_THRESH = 25.0; // 末端节点距离黑色区域的最小允许距离 (像素)
const int PADDING = 40;                // 可视化时的边缘留白

// 用于鼠标交互的数据结构
struct MouseData {
    cv::Mat base_canvas; 
    std::vector<std::pair<cv::Point, int>> node_info; 
};

// 鼠标回调函数：悬停显示节点 ID
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_MOUSEMOVE) {
        MouseData* data = static_cast<MouseData*>(userdata);
        cv::Mat display = data->base_canvas.clone(); 
        
        int min_dist = 2500; // 距离阈值的平方
        int best_idx = -1;

        // 遍历所有节点，找到距离鼠标最近的那一个
        for (size_t i = 0; i < data->node_info.size(); i++) {
            int dx = data->node_info[i].first.x - x;
            int dy = data->node_info[i].first.y - y;
            int dist = dx * dx + dy * dy;
            
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }

        if (best_idx != -1) {
            cv::Point pt = data->node_info[best_idx].first;
            int node_id = data->node_info[best_idx].second;

            // 高亮当前节点
            cv::circle(display, pt, 6, cv::Scalar(0, 255, 0), -1);

            std::string text = std::to_string(node_id);
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
            
            cv::Point text_org(pt.x + 10, pt.y - 5);

            // 绘制文字背景框
            cv::rectangle(display, 
                          text_org + cv::Point(-2, baseline + 2), 
                          text_org + cv::Point(text_size.width + 2, -text_size.height - 2), 
                          cv::Scalar(255, 255, 255), cv::FILLED);

            // 绘制文字
            cv::putText(display, text, text_org,
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(139, 0, 0), 2, cv::LINE_AA);
        }

        cv::imshow("Processed Topology Map Preview", display);
    }
}

// 辅助函数：计算当前图中每个节点的度数
map<int, int> calculate_degrees(const vector<pair<int, int>>& edges) {
    map<int, int> deg;
    for (const auto& e : edges) {
        deg[e.first]++;
        deg[e.second]++;
    }
    return deg;
}

// 辅助函数：清理无效边（去除自环、去除重复边）
void clean_edges(vector<pair<int, int>>& edges) {
    set<pair<int, int>> unique_edges;
    for (const auto& e : edges) {
        if (e.first != e.second) {
            unique_edges.insert({min(e.first, e.second), max(e.first, e.second)});
        }
    }
    edges.assign(unique_edges.begin(), unique_edges.end());
}

int main() {
    // --- 1. 配置路径 ---
    string img_path = "./8/8.pgm"; 
    string json_in_path = "./8/map_graph.json";
    string json_out_path = "./8/processed_graph.json";

    // --- 2. 读取 JSON 文件 ---
    ifstream ifs(json_in_path);
    if (!ifs.is_open()) {
        cerr << "错误：无法打开 " << json_in_path << endl;
        return -1;
    }
    json j;
    ifs >> j;
    
    vector<Point2f> nodes;
    for (const auto& n : j["nodes"]) {
        nodes.push_back(Point2f(n[0].get<float>(), n[1].get<float>()));
    }
    
    vector<pair<int, int>> edges;
    for (const auto& e : j["edges"]) {
        edges.push_back({e[0].get<int>(), e[1].get<int>()});
    }

    // --- 3. 加载图片并计算距离变换 ---
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "错误：无法打开图片 " << img_path << endl;
        return -1;
    }
    
    Mat binary, dist_map;
    threshold(img, binary, 127, 255, THRESH_BINARY);
    distanceTransform(binary, dist_map, DIST_L2, 3);

    cout << "开始处理：初始节点数 " << nodes.size() << ", 边数 " << edges.size() << endl;

    // --- 4. 算法核心：合并相近节点 (优先保留端点) ---
    bool merging = true;
    while (merging) {
        merging = false;
        double min_d = 1e9;
        int best_u = -1, best_v = -1;
        
        for (const auto& e : edges) {
            double d = norm(nodes[e.first] - nodes[e.second]);
            if (d < MERGE_DIST_THRESH && d < min_d) {
                min_d = d;
                best_u = e.first;
                best_v = e.second;
            }
        }
        
        if (best_u != -1) {
            map<int, int> degrees = calculate_degrees(edges);
            int deg_u = degrees[best_u];
            int deg_v = degrees[best_v];
            
            // 优先级：端点(3) > 分叉点(2) > 普通节点(1)
            int score_u = (deg_u == 1) ? 3 : (deg_u >= 3 ? 2 : 1);
            int score_v = (deg_v == 1) ? 3 : (deg_v >= 3 ? 2 : 1);
            
            int keep_idx = (score_u >= score_v) ? best_u : best_v;
            int remove_idx = (score_u >= score_v) ? best_v : best_u;
            
            for (auto& e : edges) {
                if (e.first == remove_idx) e.first = keep_idx;
                if (e.second == remove_idx) e.second = keep_idx;
            }
            clean_edges(edges);
            merging = true;
        }
    }

    // --- 5. 检查并删除距离黑色区域过近的末端节点 ---
    bool pruning = true;
    while (pruning) {
        pruning = false;
        map<int, int> degrees = calculate_degrees(edges);
        set<int> to_remove;
        
        for (const auto& kv : degrees) {
            if (kv.second == 1) { // 仅检查末端节点
                int node_idx = kv.first;
                Point2f p = nodes[node_idx];
                
                int x = max(0, min(img.cols - 1, (int)round(p.x)));
                int y = max(0, min(img.rows - 1, (int)round(p.y)));
                
                float dist_to_black = dist_map.at<float>(y, x);
                if (dist_to_black < BLACK_DIST_THRESH) {
                    to_remove.insert(node_idx);
                }
            }
        }
        
        if (!to_remove.empty()) {
            vector<pair<int, int>> new_edges;
            for (const auto& e : edges) {
                if (to_remove.count(e.first) == 0 && to_remove.count(e.second) == 0) {
                    new_edges.push_back(e);
                }
            }
            edges = new_edges;
            pruning = true;
        }
    }

    // --- 6. 重新映射节点索引并导出 JSON ---
    set<int> active_node_indices;
    for (const auto& e : edges) {
        active_node_indices.insert(e.first);
        active_node_indices.insert(e.second);
    }
    
    vector<Point2f> final_nodes;
    map<int, int> old_to_new;
    int new_index = 0;
    
    for (int old_idx : active_node_indices) {
        final_nodes.push_back(nodes[old_idx]);
        old_to_new[old_idx] = new_index++;
    }
    
    vector<pair<int, int>> final_edges;
    for (const auto& e : edges) {
        final_edges.push_back({old_to_new[e.first], old_to_new[e.second]});
    }

    json out_j;
    out_j["nodes"] = json::array();
    out_j["edges"] = json::array();
    for (const auto& p : final_nodes) {
        out_j["nodes"].push_back({(int)round(p.x), (int)round(p.y)});
    }
    for (const auto& e : final_edges) {
        out_j["edges"].push_back({e.first, e.second});
    }
    
    ofstream ofs(json_out_path);
    ofs << setw(4) << out_j << endl;
    ofs.close();
    
    cout << "处理完成！最终节点数 " << final_nodes.size() << ", 边数 " << final_edges.size() << endl;
    cout << "JSON 拓扑地图成功保存至: " << json_out_path << endl;

    // --- 7. 准备交互式可视化 ---
    Mat original_canvas;
    Mat color_src = imread(img_path, IMREAD_COLOR);
    if (color_src.empty()) cvtColor(img, original_canvas, COLOR_GRAY2BGR);
    else original_canvas = color_src.clone();

    // 增加 Padding 留白
    Mat canvas;
    copyMakeBorder(original_canvas, canvas, PADDING, PADDING, PADDING, PADDING, BORDER_CONSTANT, Scalar(255, 255, 255));

    // 画边（蓝色）
    for (const auto& e : final_edges) {
        Point p1(final_nodes[e.first].x + PADDING, final_nodes[e.first].y + PADDING);
        Point p2(final_nodes[e.second].x + PADDING, final_nodes[e.second].y + PADDING);
        line(canvas, p1, p2, Scalar(255, 0, 0), 2, LINE_AA);
    }

    // 画节点（红色）并录入鼠标交互数据
    MouseData mouse_data;
    for (size_t i = 0; i < final_nodes.size(); i++) {
        Point pt(final_nodes[i].x + PADDING, final_nodes[i].y + PADDING);
        circle(canvas, pt, 4, Scalar(0, 0, 255), -1, LINE_AA);
        mouse_data.node_info.push_back({pt, (int)i});
    }

    mouse_data.base_canvas = canvas.clone();

    // --- 8. 启动交互窗口 ---
    namedWindow("Processed Topology Map Preview", WINDOW_NORMAL);
    int max_w = 1280, max_h = 720;
    double scale = min((double)max_w / canvas.cols, (double)max_h / canvas.rows);
    resizeWindow("Processed Topology Map Preview", canvas.cols * scale, canvas.rows * scale);
    
    setMouseCallback("Processed Topology Map Preview", onMouse, &mouse_data);   

    imshow("Processed Topology Map Preview", canvas);
    waitKey(0);

    return 0;
}
