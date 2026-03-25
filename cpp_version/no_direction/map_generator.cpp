#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "trace_skeleton.cpp"

using json = nlohmann::json;

struct PointComp {
    bool operator() (const std::pair<int, int>& a, const std::pair<int, int>& b) const {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    }
};

int main() {
    // --- 1. 配置路径 ---
    std::string img_path = "./5/5.png"; // 请确保路径对应你正在测的图
    std::string json_path = "./5/map_graph.json";

    cv::Mat src = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "错误：无法读取图片 " << img_path << std::endl;
        return -1;
    }

    cv::Mat bin;
    cv::threshold(src, bin, 200, 1, cv::THRESH_BINARY);

    // ==========================================
    // 边缘保护 (Border Protection)
    // 强制给图片加一圈 1 像素的黑边，解决进出口贴边导致的细化死角问题
    // ==========================================
    cv::Mat padded_bin;
    cv::copyMakeBorder(bin, padded_bin, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));

    skeleton_tracer_t* T = new skeleton_tracer_t();
    T->W = padded_bin.cols;
    T->H = padded_bin.rows;
    T->im = (unsigned char*)malloc(sizeof(unsigned char) * T->W * T->H);

    // ==========================================
    // 内存对齐 (Memory Alignment)
    // 绝对禁止使用 memcpy(bin.data)！必须使用 ptr 逐行安全获取内存，过滤掉 OpenCV 的 Padding
    // ==========================================
    for (int r = 0; r < T->H; r++) {
        memcpy(T->im + r * T->W, padded_bin.ptr<uchar>(r), T->W);
    }

    std::cout << "正在提取骨架 (Skeletonization)..." << std::endl;
    T->thinning_zs(); 
    
    std::cout << "正在追踪拓扑图..." << std::endl;
    skeleton_tracer_t::polyline_t* polys = T->trace_skeleton(0, 0, T->W, T->H, 0);

    // --- 构建图结构 (Nodes & Edges) ---
    std::vector<std::pair<int, int>> nodes_list;
    std::set<std::pair<int, int>> edges_set;
    std::map<std::pair<int, int>, int, PointComp> node_to_idx;

    auto get_node_idx = [&](int x, int y) -> int {
        // 因为之前加了 1 像素黑边，这里将坐标减 1，完美还原回原图坐标空间
        int real_x = std::max(0, x - 1);
        int real_y = std::max(0, y - 1);
        std::pair<int, int> pt = {real_x, real_y};
        if (node_to_idx.find(pt) == node_to_idx.end()) {
            node_to_idx[pt] = nodes_list.size();
            nodes_list.push_back(pt);
        }
        return node_to_idx[pt];
    };

    skeleton_tracer_t::polyline_t* it = polys;
    while(it) {
        skeleton_tracer_t::point_t* jt = it->head;
        while(jt && jt->next) {
            int idx1 = get_node_idx(jt->x, jt->y);
            int idx2 = get_node_idx(jt->next->x, jt->next->y);
            if (idx1 != idx2) {
                edges_set.insert({std::min(idx1, idx2), std::max(idx1, idx2)});
            }
            jt = jt->next;
        }
        it = it->next;
    }

    // --- 导出为 JSON ---
    json j_graph;
    j_graph["nodes"] = json::array();
    for (const auto& n : nodes_list) {
        j_graph["nodes"].push_back({n.first, n.second});
    }
    j_graph["edges"] = json::array();
    for (const auto& e : edges_set) {
        j_graph["edges"].push_back({e.first, e.second});
    }

    std::ofstream o(json_path);
    o << std::setw(4) << j_graph << std::endl;
    std::cout << "JSON 拓扑地图成功保存至: " << json_path << " (生成节点数: " << nodes_list.size() << ")" << std::endl;

    // --- 可视化绘制 ---
    cv::Mat canvas;
    cv::Mat color_src = cv::imread(img_path, cv::IMREAD_COLOR);
    if(color_src.empty()) cv::cvtColor(src, canvas, cv::COLOR_GRAY2BGR);
    else canvas = color_src.clone();

    it = polys;
    cv::RNG rng(12345);
    while(it) {
        cv::Scalar color(rng.uniform(50, 200), rng.uniform(50, 200), rng.uniform(50, 200));
        skeleton_tracer_t::point_t* jt = it->head;
        while(jt && jt->next) {
            // 画线时同样要减去 padding 偏移
            int x1 = std::max(0, jt->x - 1);
            int y1 = std::max(0, jt->y - 1);
            int x2 = std::max(0, jt->next->x - 1);
            int y2 = std::max(0, jt->next->y - 1);
            cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
            jt = jt->next;
        }
        it = it->next;
    }

    for (const auto& pair : node_to_idx) {
        cv::Point pt(pair.first.first, pair.first.second);
        int node_id = pair.second;
        cv::circle(canvas, pt, 4, cv::Scalar(0, 0, 255), -1);
        cv::putText(canvas, std::to_string(node_id), cv::Point(pt.x + 5, pt.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(139, 0, 0), 1, cv::LINE_AA);
    }

    cv::namedWindow("C++ Topology Map Generator", cv::WINDOW_NORMAL);
    int max_w = 1280, max_h = 720;
    double scale = std::min((double)max_w / canvas.cols, (double)max_h / canvas.rows);
    cv::resizeWindow("C++ Topology Map Generator", canvas.cols * scale, canvas.rows * scale);
    cv::imshow("C++ Topology Map Generator", canvas);
    
    cv::waitKey(0);

    // --- 清理 ---
    T->destroy_polylines(polys);
    free(T->im);
    delete T;

    return 0;
}
