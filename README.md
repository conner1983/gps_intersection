# GPS多线交点计算与可视化工具  
*（项目名称：gps-intersection-calculator）*  


## 📌 项目概览  
- **核心功能**：通过多个GPS观测点的方位角数据，计算目标交点坐标并可视化  
- **技术栈**：Python + NumPy（数值计算） + Folium（地图可视化）  
- **应用场景**：测绘工程、目标定位、地理信息分析  


## 🚀 功能特点  
- **坐标转换**：WGS84经纬度与笛卡尔坐标系互转  
- **交点求解**：基于最小二乘法的多直线交点优化计算  
- **可视化**：生成交互式地图，展示观测点、方位线和交点  


## ⚙️ 快速开始  
### 环境要求  
```bash  
Python 3.7+  
依赖库：numpy, folium  