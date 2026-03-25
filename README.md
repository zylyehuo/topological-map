# 工具描述
> 基于项目[skeleton-tracing](https://github.com/LingDong-/skeleton-tracing)基础上实现

> 主要用于将已有的“二值化”图片自动构建一张拓扑地图，以供A*等算法使用

# 具体效果
![hand](./assets/hand.jpg)
![maze](./assets/maze.jpg)
![road](./assets/road.jpg)
# 使用步骤
## 第一步：下载项目
```
git clone git@github.com:zylyehuo/topological-map.git

```
## 第二步：准备所需的图片

## 第三步：根据文件路径修改源码中的相关部分

## 第四步：设置 astar_pathfinder.py 文件中的起点与终点序号

## 第五步：运行指令
### Python 版本
```
python3 ./trace_skeleton.py

```

```
python3 ./astar_pathfinder.py

```
### C++ 版本
```
g++ map_generator.cpp -o map_generator `pkg-config --cflags --libs opencv4` -O3

./map_generator

```

```
g++ astar_pathfinder.cpp -o astar_pathfinder `pkg-config --cflags --libs opencv4` -O3

./astar_pathfinder

```