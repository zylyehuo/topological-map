import cv2
import numpy as np

def keep_largest_white_and_clean_black(input_path, output_path, min_black_area=1000):
    """
    只保留最大的白色连通域，并去除内部微小的黑色噪点，保留大的黑色区块
    """
    # 1. 读取并二值化图片
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("无法读取图片，请检查路径。")
        return
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 2. 核心逻辑一：只保留面积【最大】的白色连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    clean_white = np.zeros_like(binary)
    if num_labels > 1:
        # stats 包含 [x, y, width, height, area]
        # 提取除背景(索引0)以外的所有白色连通域的面积
        areas = stats[1:, cv2.CC_STAT_AREA] 
        # 找到面积最大的那一个连通域的索引 (注意索引要 +1)
        max_label = np.argmax(areas) + 1 
        
        # 只把这一个最大的白色主体画上去，其他全部抛弃
        clean_white[labels == max_label] = 255
    else:
        # 如果整张图全黑，直接返回
        clean_white = binary 

    # 3. 核心逻辑二：清理白色主体内部的细小黑色噪点，保留您说的大块黑色区域
    # 反色：白变黑，黑变白
    binary_inv = cv2.bitwise_not(clean_white)
    num_labels_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    
    clean_black_inv = np.zeros_like(binary)
    for i in range(1, num_labels_inv):
        # 这里的阈值设大一点(如1000)，过滤掉黑点，但保留图中大面积的黑洞
        if stats_inv[i, cv2.CC_STAT_AREA] >= min_black_area:
            clean_black_inv[labels_inv == i] = 255
            
    # 再反色回来，得到最终结果
    result = cv2.bitwise_not(clean_black_inv)

    # 4. 保存结果
    cv2.imwrite(output_path, result)
    print(f"处理完成，结果已保存至: {output_path}")

# --- 路径设置区域 ---
input_image_path = '/home/yehuo/topological_map/img_process/INDOOR/process/0_draw_process.png'
output_image_path = '/home/yehuo/topological_map/img_process/INDOOR/process/0_open_process.png'

# 调用函数 (min_black_area 设为 1000 左右，足以过滤黑色碎点，同时保留中间大块的黑色空洞)
keep_largest_white_and_clean_black(input_image_path, output_image_path, min_black_area=1000)
