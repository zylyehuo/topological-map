# merged_road_segmentation.py
# 目标：提取边缘 -> 初步连接与填充 -> 高级骨架端点连接 -> 最终闭合填充 -> 最终结果：道路黑色，背景白色

import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

def _remove_small_components(binary_img: np.ndarray, min_area: int = 50) -> np.ndarray:
    """移除面积小于 min_area 的连通域噪声点。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    filtered = np.zeros_like(binary_img)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered

def connect_endpoints_on_skeleton(binary_img, max_gap_px=150):
    """
    寻找单像素宽线条的端点并连接。
    输入：黑底白线（1像素宽）
    输出：缺口已补齐的黑底白线
    """
    # 骨架化以获得精确端点
    skeleton = skeletonize(binary_img > 0).astype(np.uint8) * 255
    
    # 端点检测 (命中或击中变换 kernel)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    # 注意：如果原本 skeleton 是 0和255，这里的滤波可能会和 11 不匹配
    # 但为了绝对保持你原有的算法逻辑，此处不做修改。
    neighbor_counts = cv2.filter2D(skeleton, -1, kernel)
    endpoints = np.argwhere(neighbor_counts == 11) # 11表示中心点(10) + 1个邻居(1)

    # 简单的贪心连接算法
    connected_skeleton = skeleton.copy()
    used = set()
    for i in range(len(endpoints)):
        if i in used: continue
        p1 = endpoints[i]
        
        best_dist = max_gap_px
        best_idx = -1
        
        for j in range(i + 1, len(endpoints)):
            if j in used: continue
            p2 = endpoints[j]
            # 计算距离
            dist = np.sqrt(np.sum((p1 - p2)**2))
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        if best_idx != -1:
            # 连接缺口 (注意 OpenCV 坐标系是 (x,y), 而 numpy 是 (row,col))
            cv2.line(connected_skeleton, (p1[1], p1[0]), (endpoints[best_idx][1], endpoints[best_idx][0]), 255, 2)
            used.add(i)
            used.add(best_idx)
            
    return connected_skeleton

def process_road_image(
    input_path: str,
    output_path: str,
    # Phase 1 参数 (来自 22.py)
    low_thresh: int = 40,
    high_thresh: int = 120,
    phase1_min_area: int = 500,
    phase1_morph_kernel: int = 11,
    # Phase 2 参数 (来自 33.py)
    phase2_min_area: int = 30,
    max_gap: int = 50000
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"无法读取图片: {input_path}")

    # ==========================================
    # 第一阶段：基础边缘提取与初步填充 (对应原 22.py)
    # ==========================================
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Canny 边缘提取（得到：白线黑底）
    edges = cv2.Canny(blur, low_thresh, high_thresh)

    # 过滤细小噪声
    edges_clean = _remove_small_components(edges, min_area=phase1_min_area)

    # 形态学闭运算：连接相近的线条
    kernel_p1 = cv2.getStructuringElement(cv2.MORPH_RECT, (phase1_morph_kernel, phase1_morph_kernel))
    closed_p1 = cv2.morphologyEx(edges_clean, cv2.MORPH_CLOSE, kernel_p1)

    # 寻找轮廓并填充在黑色背景上
    contours_p1, _ = cv2.findContours(closed_p1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    road_mask_white_p1 = np.zeros_like(closed_p1)
    cv2.drawContours(road_mask_white_p1, contours_p1, -1, 255, thickness=cv2.FILLED)

    # 【优化点】：此时 road_mask_white_p1 正好是 33.py 中需要的“黑底白线”图。
    # 我们直接跳过原代码中反相保存再读取反相的过程，直接进入第二阶段。

    # ==========================================
    # 第二阶段：高级端点连接与最终闭合 (对应原 33.py)
    # ==========================================
    # 清理掉可能存在的微小颗粒噪点
    mask_wob_clean = _remove_small_components(road_mask_white_p1, min_area=phase2_min_area)
    
    # 形态学闭合预处理：连接极近的缝隙，加粗线条
    pre_close_kernel = np.ones((7, 7), np.uint8)
    pre_closed = cv2.morphologyEx(mask_wob_clean, cv2.MORPH_CLOSE, pre_close_kernel)
    
    # 端点连接逻辑：解决路口的断口
    connected_lines = connect_endpoints_on_skeleton(pre_closed, max_gap_px=max_gap)
    
    # 形态学增强：确保连接线完全封闭，没有气孔
    kernel_p2 = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(connected_lines, kernel_p2, iterations=1)
    
    # 重新寻找外部轮廓并填充
    contours_p2, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_road_mask_white = np.zeros_like(gray)
    cv2.drawContours(final_road_mask_white, contours_p2, -1, 255, thickness=cv2.FILLED)
    
    # 【核心最后一步】：反相变回“道路黑，背景白”
    final_inverted_mask = cv2.bitwise_not(final_road_mask_white)
    
    # 保存最终结果
    ok = cv2.imwrite(output_path, final_inverted_mask)
    if ok:
        print(f"合并处理完成，道路已成功闭合！\n结果保存至: {output_path}")
    else:
        print("保存失败，请检查路径。")

if __name__ == "__main__":
    # --- 请根据实际路径修改 ---
    input_img = "/home/yehuo/topological_map/img_process/ROAD/origin/0.png"
    output_img = "/home/yehuo/topological_map/img_process/ROAD/process/0_clear_process.png"
    
    # 运行整合后的流程
    process_road_image(input_img, output_img, max_gap=50000)
