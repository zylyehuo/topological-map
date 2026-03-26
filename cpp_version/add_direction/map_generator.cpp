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

struct MouseData {
    cv::Mat base_canvas; 
    std::vector<std::pair<cv::Point, int>> node_info; 
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
            
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }

        if (best_idx != -1) {
            cv::Point pt = data->node_info[best_idx].first;
            int node_id = data->node_info[best_idx].second;

            cv::circle(display, pt, 6, cv::Scalar(0, 255, 0), -1);

            std::string text = std::to_string(node_id);
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
            
            cv::Point text_org(pt.x + 10, pt.y - 5);

            cv::rectangle(display, 
                          text_org + cv::Point(-2, baseline + 2), 
                          text_org + cv::Point(text_size.width + 2, -text_size.height - 2), 
                          cv::Scalar(255, 255, 255), cv::FILLED);

            cv::putText(display, text, text_org,
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(139, 0, 0), 2, cv::LINE_AA);
        }

        cv::imshow("C++ Topology Map Generator", display);
    }
}

int main() {
    // --- 1. 配置路径 ---
    std::string img_path = "./10/10.png"; 
    std::string json_path = "./10/map_graph.json";

    const int PADDING = 40; 

    cv::Mat src = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "错误：无法读取图片 " << img_path << std::endl;
        return -1;
    }

    cv::Mat bin;
    cv::threshold(src, bin, 200, 1, cv::THRESH_BINARY);

    cv::Mat padded_bin;
    cv::copyMakeBorder(bin, padded_bin, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));

    skeleton_tracer_t* T = new skeleton_tracer_t();
    T->W = padded_bin.cols;
    T->H = padded_bin.rows;
    T->im = (unsigned char*)malloc(sizeof(unsigned char) * T->W * T->H);

    for (int r = 0; r < T->H; r++) {
        memcpy(T->im + r * T->W, padded_bin.ptr<uchar>(r), T->W);
    }

    std::cout << "正在提取骨架 (Skeletonization)..." << std::endl;
    T->thinning_zs(); 
    
    std::cout << "正在追踪拓扑图..." << std::endl;
    skeleton_tracer_t::polyline_t* polys = T->trace_skeleton(0, 0, T->W, T->H, 0);

    std::vector<std::pair<int, int>> nodes_list;
    std::set<std::pair<int, int>> edges_set;
    std::map<std::pair<int, int>, int, PointComp> node_to_idx;

    auto get_node_idx = [&](int x, int y) -> int {
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

    json j_graph;
    j_graph["nodes"] = json::array();
    for (const auto& n : nodes_list) j_graph["nodes"].push_back({n.first, n.second});
    j_graph["edges"] = json::array();
    for (const auto& e : edges_set) j_graph["edges"].push_back({e.first, e.second});

    std::ofstream o(json_path);
    o << std::setw(4) << j_graph << std::endl;
    std::cout << "JSON 拓扑地图成功保存至: " << json_path << " (生成节点数: " << nodes_list.size() << ")" << std::endl;

    cv::Mat original_canvas;
    cv::Mat color_src = cv::imread(img_path, cv::IMREAD_COLOR);
    if(color_src.empty()) cv::cvtColor(src, original_canvas, cv::COLOR_GRAY2BGR);
    else original_canvas = color_src.clone();

    cv::Mat canvas;
    cv::copyMakeBorder(original_canvas, canvas, PADDING, PADDING, PADDING, PADDING, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    it = polys;
    cv::RNG rng(12345);
    while(it) {
        cv::Scalar color(rng.uniform(50, 200), rng.uniform(50, 200), rng.uniform(50, 200));
        skeleton_tracer_t::point_t* jt = it->head;
        while(jt && jt->next) {
            int x1 = std::max(0, jt->x - 1) + PADDING;
            int y1 = std::max(0, jt->y - 1) + PADDING;
            int x2 = std::max(0, jt->next->x - 1) + PADDING;
            int y2 = std::max(0, jt->next->y - 1) + PADDING;
            cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
            jt = jt->next;
        }
        it = it->next;
    }

    MouseData mouse_data;
    
    for (const auto& pair : node_to_idx) {
        cv::Point pt(pair.first.first, pair.first.second);
        int node_id = pair.second;
        cv::Point display_pt(pt.x + PADDING, pt.y + PADDING); 
        cv::circle(canvas, display_pt, 4, cv::Scalar(0, 0, 255), -1); 
        mouse_data.node_info.push_back({display_pt, node_id});
    }

    mouse_data.base_canvas = canvas.clone();

    cv::namedWindow("C++ Topology Map Generator", cv::WINDOW_NORMAL);
    int max_w = 1280, max_h = 720;
    double scale = std::min((double)max_w / canvas.cols, (double)max_h / canvas.rows);
    cv::resizeWindow("C++ Topology Map Generator", canvas.cols * scale, canvas.rows * scale);
    
    cv::setMouseCallback("C++ Topology Map Generator", onMouse, &mouse_data);   

    cv::imshow("C++ Topology Map Generator", canvas);
    cv::waitKey(0);

    T->destroy_polylines(polys);
    free(T->im);
    delete T;

    return 0;
}
