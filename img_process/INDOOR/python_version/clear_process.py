# universal_line_connector.py
# 目标：通用型边缘提取与断线连接，兼容“真实道路照片”与“白底黑线平面图”

import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

def _remove_small_components(binary_img: np.ndarray, min_area: int = 50) -> np.ndarray:
    """移除面积过小的连通域噪声点"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    filtered = np.zeros_like(binary_img)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered

def get_connection_lines(binary_img, max_gap_px=150):
    """
    寻找单像素宽线条的端点并返回连接这些端点的线段Mask。
    """
    # 骨架化以获得精确端点
    skeleton = skeletonize(binary_img > 0).astype(np.uint8) * 255
    
    # 端点检测 (命中或击中变换 kernel)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_counts = cv2.filter2D(skeleton, -1, kernel)
    endpoints = np.argwhere(neighbor_counts == 11) 

    connections_mask = np.zeros_like(binary_img)
    used = set()
    
    # 贪心连接算法
    for i in range(len(endpoints)):
        if i in used: continue
        p1 = endpoints[i]
        
        best_dist = max_gap_px
        best_idx = -1
        
        for j in range(i + 1, len(endpoints)):
            if j in used: continue
            p2 = endpoints[j]
            dist = np.sqrt(np.sum((p1 - p2)**2))
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        if best_idx != -1:
            # 连接缺口 (OpenCV 坐标系是 (x,y), numpy 是 (row,col))
            cv2.line(connections_mask, (p1[1], p1[0]), (endpoints[best_idx][1], endpoints[best_idx][0]), 255, 2)
            used.add(i)
            used.add(best_idx)
            
    return connections_mask

def process_universal_image(
    input_path: str,
    output_path: str,
    max_gap: int = 150,         # 最大允许连接的断口距离
    fill_external: bool = False # 核心开关：是否将整个轮廓内部填满（道路适用 True，平面图适用 False）
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"无法读取图片: {input_path}")

    # 1. 读取并预处理
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ==========================================
    # 2. 【核心通用提取】：融合 Canny 与 二值化
    # 这样既能抓取灰度图的边缘，又能把黑线图直接转为"实心白块"避免空心双线
    # ==========================================
    edges = cv2.Canny(blur, 40, 120)
    # 对于像 0.png 这种图，较暗的线会直接被提取为白色的实心区域
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV) 
    
    # 融合两者
    combined_mask = cv2.bitwise_or(edges, thresh)

    # 3. 过滤细小噪声
    clean_mask = _remove_small_components(combined_mask, min_area=100)

    # ==========================================
    # 4. 【去毛刺预处理】：平滑边缘，防止骨架化崩溃
    # ==========================================
    # 闭运算：填补线段内部的小坑洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 开运算：极度重要！去除边缘凸起的毛刺，确保后续骨架化是一条干净的单线
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # ==========================================
    # 5. 端点连接与原始厚度融合
    # ==========================================
    # 只寻找连接线
    connection_lines = get_connection_lines(smoothed, max_gap_px=max_gap)
    
    # 膨胀一下连接线，使其符合墙壁/道路的厚度
    connection_lines = cv2.dilate(connection_lines, np.ones((5,5), np.uint8))
    
    # 将原始的实心图像与新连上的线合并（保留了原图的线宽！）
    full_network = cv2.bitwise_or(smoothed, connection_lines)
    full_network = cv2.morphologyEx(full_network, cv2.MORPH_CLOSE, kernel_close) # 再次平滑接口处

    # ==========================================
    # 6. 最终填充与反相
    # ==========================================
    if fill_external:
        # 道路模式：提取最外围轮廓并全部填满内部
        contours, _ = cv2.findContours(full_network, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(gray)
        cv2.drawContours(final_mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        # 平面图模式：保留墙体/线条本身
        final_mask = full_network

    # 反相：目标物体变为黑色(0)，背景变为白色(255)
    final_inverted = cv2.bitwise_not(final_mask)
    
    ok = cv2.imwrite(output_path, final_inverted)
    if ok:
        print(f"通用处理完成！结果保存至: {output_path}")
    else:
        print("保存失败，请检查路径。")

if __name__ == "__main__":
    # --- 请修改为你自己的路径 ---
    input_img = "/home/yehuo/img_process/origin/0.png"
    output_img = "/home/yehuo/img_process/process/0_clear_process.png"
    
    # 关键参数：
    # max_gap: 平面图建议 100-200，道路如果有大缺口可以调大
    # fill_external: 对于 0.png 这种平面图务必设为 False，否则整个房子内部全变黑。如果是之前的纯块状道路可以设为 True
    process_universal_image(input_img, output_img, max_gap=150, fill_external=False)
