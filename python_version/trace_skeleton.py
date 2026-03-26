# trace_skeleton_final.py
# Trace skeletonization result into polylines and save as JSON Graph
#
# Original Algorithm: Lingdong Huang 2020
# Modified with JSON export and OpenCV visualization fixes

import numpy as np
import cv2
import random
import json
import os

# ==========================================
# 1. 核心骨架提取算法 (保持原逻辑)
# ==========================================

def thinningZS(im):
    prev = np.zeros(im.shape, np.uint8)
    while True:
        im = thinningZSIteration(im, 0)
        im = thinningZSIteration(im, 1)
        diff = np.sum(np.abs(prev - im))
        if not diff:
            break
        prev = im.copy()
    return im

def thinningZSIteration(im, iter):
    marker = np.zeros(im.shape, np.uint8)
    for i in range(1, im.shape[0]-1):
        for j in range(1, im.shape[1]-1):
            if im[i, j] == 0: continue
            p2 = im[(i-1), j];   p3 = im[(i-1), j+1]; p4 = im[(i), j+1]; p5 = im[(i+1), j+1]
            p6 = im[(i+1), j];   p7 = im[(i+1), j-1]; p8 = im[(i), j-1]; p9 = im[(i-1), j-1]
            A  = (p2 == 0 and p3) + (p3 == 0 and p4) + \
                 (p4 == 0 and p5) + (p5 == 0 and p6) + \
                 (p6 == 0 and p7) + (p7 == 0 and p8) + \
                 (p8 == 0 and p9) + (p9 == 0 and p2)
            B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            m1 = (p2 * p4 * p6) if (iter == 0) else (p2 * p4 * p8)
            m2 = (p4 * p6 * p8) if (iter == 0) else (p2 * p6 * p8)
            if (A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0):
                marker[i, j] = 1
    return np.bitwise_and(im, np.bitwise_not(marker))

def thinningSkimage(im):
    from skimage.morphology import skeletonize
    return skeletonize(im).astype(np.uint8)

def thinning(im):
    try:
        return thinningSkimage(im)
    except:
        return thinningZS(im)

# ==========================================
# 2. 追踪算法核心 (修复并保留原逻辑)
# ==========================================

def notEmpty(im, x, y, w, h):
    return np.sum(im[y:y+h, x:x+w]) > 0

def mergeImpl(c0, c1, i, sx, isv, mode):
    B0 = (mode >> 1 & 1) > 0
    B1 = (mode >> 0 & 1) > 0
    mj = -1
    md = 4 
    
    p1 = c1[i][0 if B1 else -1]
    if abs(p1[isv] - sx) > 0: return False
    
    for j in range(len(c0)):
        p0 = c0[j][0 if B0 else -1]
        if abs(p0[isv] - sx) > 1: continue
        d = abs(p0[not isv] - p1[not isv])
        if d < md:
            mj = j
            md = d

    if mj != -1:
        if B0 and B1:
            c0[mj] = list(reversed(c1[i])) + c0[mj]
        elif not B0 and B1:
            c0[mj] += c1[i]
        elif B0 and not B1:
            c0[mj] = c1[i] + c0[mj]
        else:
            c0[mj] += list(reversed(c1[i]))
        c1.pop(i)
        return True
    return False

HORIZONTAL = 1
VERTICAL = 2

def mergeFrags(c0, c1, sx, dr):
    for i in range(len(c1)-1, -1, -1):
        if dr == HORIZONTAL:
            if mergeImpl(c0, c1, i, sx, False, 1): continue
            if mergeImpl(c0, c1, i, sx, False, 3): continue
            if mergeImpl(c0, c1, i, sx, False, 0): continue
            if mergeImpl(c0, c1, i, sx, False, 2): continue
        else:
            if mergeImpl(c0, c1, i, sx, True, 1): continue
            if mergeImpl(c0, c1, i, sx, True, 3): continue
            if mergeImpl(c0, c1, i, sx, True, 0): continue
            if mergeImpl(c0, c1, i, sx, True, 2): continue       
    c0 += c1

def chunkToFrags(im, x, y, w, h):
    frags = []
    on = False
    li, lj = -1, -1
    
    for k in range(h+h+w+w-4):
        i, j = 0, 0
        if k < w:
            i, j = y, x+k
        elif k < w+h-1:
            i, j = y+k-w+1, x+w-1
        elif k < w+h+w-2:
            i, j = y+h-1, x+w-(k-w-h+3) 
        else:
            i, j = y+h-(k-w-h-w+4), x+0
        
        if im[i, j]:
            if not on:
                on = True
                frags.append([[j, i], [x+w//2, y+h//2]])
        else:
            if on:
                frags[-1][0][0] = (frags[-1][0][0] + lj) // 2
                frags[-1][0][1] = (frags[-1][0][1] + li) // 2
                on = False
        li, lj = i, j
    
    if len(frags) == 2:
        f = [frags[0][0], frags[1][0]]
        frags = [f]
    elif len(frags) > 2:
        ms, mi, mj = 0, -1, -1
        for i in range(y+1, y+h-1):
            for j in range(x+1, x+w-1):
                s = np.sum(im[i-1:i+2, j-1:j+2])
                if s > ms:
                    mi, mj, ms = i, j, s
                elif s == ms and abs(j-(x+w//2))+abs(i-(y+h//2)) < abs(mj-(x+w//2))+abs(mi-(y+h//2)):
                    mi, mj, ms = i, j, s
        if mi != -1:
            for i in range(len(frags)):
                frags[i][1] = [mj, mi]
    return frags

def traceSkeleton(im, x, y, w, h, csize, maxIter, rects):
    frags = []
    if maxIter == 0: return frags
    if w <= csize and h <= csize:
        return chunkToFrags(im, x, y, w, h)
    
    ms = im.shape[0] + im.shape[1]
    mi, mj = -1, -1
    
    if h > csize:
        for i in range(y+3, y+h-3):
            if im[i,x] or im[i-1,x] or im[i,x+w-1] or im[i-1,x+w-1]: continue
            s = np.sum(im[i-1:i+1, x:x+w])
            if s < ms:
                ms, mi = s, i
            elif s == ms and abs(i-(y+h//2)) < abs(mi-(y+h//2)):
                ms, mi = s, i
                
    if w > csize:
        for j in range(x+3, x+w-2):
            if im[y,j] or im[y+h-1,j] or im[y,j-1] or im[y+h-1,j-1]: continue
            s = np.sum(im[y:y+h, j-1:j+1])
            if s < ms:
                ms, mi, mj = s, -1, j
            elif s == ms and abs(j-(x+w//2)) < abs(mj-(x+w//2)):
                ms, mi, mj = s, -1, j

    nf = []
    if h > csize and mi != -1:
        L = [x, y, w, mi-y]
        R = [x, mi, w, y+h-mi]
        if notEmpty(im, *L):
            if rects is not None: rects.append(L)
            nf += traceSkeleton(im, *L, csize, maxIter-1, rects)
        if notEmpty(im, *R):
            if rects is not None: rects.append(R)
            mergeFrags(nf, traceSkeleton(im, *R, csize, maxIter-1, rects), mi, VERTICAL)
            
    elif w > csize and mj != -1:
        L = [x, y, mj-x, h]
        R = [mj, y, x+w-mj, h]
        if notEmpty(im, *L):
            if rects is not None: rects.append(L)
            nf += traceSkeleton(im, *L, csize, maxIter-1, rects)
        if notEmpty(im, *R):
            if rects is not None: rects.append(R)
            mergeFrags(nf, traceSkeleton(im, *R, csize, maxIter-1, rects), mj, HORIZONTAL)
            
    frags += nf
    if mi == -1 and mj == -1:
        frags += chunkToFrags(im, x, y, w, h)
    return frags

# ==========================================
# 3. 拓扑图保存模块
# ==========================================

def save_topology_to_json(polys, filename):
    """
    将多段线转换为 Graph (Nodes 和 Edges) 结构并保存为 JSON，
    同时返回 node_to_idx 字典，供后续可视化绘制使用。
    """
    nodes_list = []
    edges_set = set()
    node_to_idx = {}

    def get_node_idx(pt):
        pt_tuple = (int(pt[0]), int(pt[1]))
        if pt_tuple not in node_to_idx:
            node_to_idx[pt_tuple] = len(nodes_list)
            nodes_list.append(pt_tuple)
        return node_to_idx[pt_tuple]

    for poly in polys:
        for i in range(len(poly) - 1):
            idx1 = get_node_idx(poly[i])
            idx2 = get_node_idx(poly[i+1])
            if idx1 != idx2:
                edge = (min(idx1, idx2), max(idx1, idx2))
                edges_set.add(edge)

    graph_data = {
        "nodes": nodes_list,
        "edges": [list(e) for e in edges_set] 
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4)

    print(f"拓扑地图成功保存至: {filename}")
    print(f"   - 提取到节点数 (Nodes): {len(nodes_list)}")
    print(f"   - 提取到路径段 (Edges): {len(edges_set)}\n")
    
    # 核心修改：返回映射字典
    return node_to_idx

# ==========================================
# 4. 主程序运行逻辑
# ==========================================

if __name__ == "__main__":
    img_path = "./3/3.png"
    im_src = cv2.imread(img_path)
    
    if im_src is None:
        print(f"错误：无法加载图片，请检查路径: {img_path}")
    else:
        # 1. 预处理 (适应白底路面)
        gray = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
        _, im_bin = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY)
        
        # 2. 细化处理
        print("正在进行细化 (Skeletonization)...")
        im_skel = thinning(im_bin)
        
        # 3. 追踪为多段线
        print("正在追踪并生成拓扑图...")
        polys = traceSkeleton(im_skel, 0, 0, im_skel.shape[1], im_skel.shape[0], 35, 999, [])
        
        # 4. 自动匹配路径并保存到文件
        img_dir = os.path.dirname(img_path)
        json_save_path = os.path.join(img_dir, "map_graph.json")
        # 拿到返回的节点映射字典
        node_to_idx = save_topology_to_json(polys, json_save_path)

        # 5. 可视化绘制
        canvas = im_src.copy()
        # 5.1 画线
        for poly in polys:
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            for i in range(len(poly) - 1):
                p1 = tuple(map(int, poly[i]))
                p2 = tuple(map(int, poly[i+1]))
                cv2.line(canvas, p1, p2, color, 2)
            
        # 5.2 画点和文字序号
        for pt_tuple, node_id in node_to_idx.items():
            # 画红色圆点
            cv2.circle(canvas, pt_tuple, 4, (0, 0, 255), -1)
            # 画深蓝色序号 (139, 0, 0)
            cv2.putText(canvas, str(node_id), (pt_tuple[0] + 5, pt_tuple[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139, 0, 0), 1, cv2.LINE_AA)

        # 6. 自适应窗口显示
        cv2.namedWindow("Topology Trace Result", cv2.WINDOW_NORMAL)
        h, w = canvas.shape[:2]
        scale = min(1200 / w, 800 / h, 1.0)
        cv2.resizeWindow("Topology Trace Result", int(w * scale), int(h * scale))
        
        cv2.imshow("Topology Trace Result", canvas)
        print("可视化已启动，可以任意拉伸窗口。按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
