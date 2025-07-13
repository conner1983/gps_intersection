import numpy as np

import folium
import math



# GPS坐标转换为笛卡尔坐标（简化）
def gps_to_cartesian(latitude, longitude,height=0):
    # WGS-84 geodetic constants
    a = 6378137.0        # Earth's radius in meters
    b = 6356752.314245
    e = math.sqrt(1 - (b**2 / a**2))  # Eccentricity

    # Convert latitude and longitude to radians
    latitude = math.radians(latitude)
    longitude = math.radians(longitude)

    # Auxiliary values
    N = a / math.sqrt(1 - e**2 * math.sin(latitude)**2)

    # Cartesian coordinates
    x = (N + height) * math.cos(latitude) * math.cos(longitude)
    y = (N + height) * math.cos(latitude) * math.sin(longitude)
    z = ((1 - e**2) * N + height) * math.sin(latitude)

    return x, y, z
    return latitude, longitude

# 笛卡尔坐标转换回GPS坐标（简化）
def cartesian_to_gps(x, y,z=0):
    # WGS-84 geodetic constants
    a = 6378137.0        # WGS-84 Earth semimajor axis (m)
    b = 6356752.31424518   # Derived Earth semiminor axis (m)
    e = math.sqrt(1 - (b**2 / a**2))  # Eccentricity

    # Calculations
    p = math.sqrt(x**2 + y**2)
    theta = math.atan2(z * a, p * b)
    longitude = math.atan2(y, x)
    latitude = math.atan2(z + e**2 * b * math.sin(theta)**3, p - e**2 * a * math.cos(theta)**3)
    N = a / math.sqrt(1 - e**2 * math.sin(latitude)**2)
    height = p / math.cos(latitude) - N

    # Convert from radians to degrees
    latitude = math.degrees(latitude)
    longitude = math.degrees(longitude)

    return longitude,latitude,  height

# 将角度转换为向量
def bearing_to_vector(bearing):
    rad = np.radians(bearing)
    return np.array([np.sin(rad), np.cos(rad)])

# 从点和方位角创建直线方程
def line_from_point_bearing(point, bearing):
    vector = bearing_to_vector(bearing)
    print("vector",vector)
    a = -vector[1]
    b = vector[0]
    c = -(a * point[0] + b * point[1])
    return a, b, c

# 找到多条直线的交点
def find_intersection(lines):
    A = np.array([line[:2] for line in lines])
    b = -np.array([line[2] for line in lines])
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x, residuals

# 计算从一个点出发，给定长度和方位角的终点坐标
def calculate_line_endpoint(start_lat, start_lon, bearing, distance):
    # 地球半径（单位：米）
    R = 6371e3 
    bearing_rad = np.radians(bearing)

    start_lat_rad = np.radians(start_lat)
    start_lon_rad = np.radians(start_lon)

    end_lat_rad = np.arcsin(np.sin(start_lat_rad) * np.cos(distance/R) + np.cos(start_lat_rad) * np.sin(distance/R) * np.cos(bearing_rad))
    end_lon_rad = start_lon_rad + np.arctan2(np.sin(bearing_rad) * np.sin(distance/R) * np.cos(start_lat_rad), np.cos(distance/R) - np.sin(start_lat_rad) * np.sin(end_lat_rad))

    end_lat = np.degrees(end_lat_rad)
    end_lon = np.degrees(end_lon_rad)

    return end_lat, end_lon

# 主程序
if __name__ == "__main__":
    # 输入的GPS坐标和角度
    points = [
        (39.952778,116.822021, 135), 
        (39.95286,116.823362,225), 
        (39.950673,116.823529, 335),
        (39.950592,116.822183,1)
    ]

    # 转换和计算交点
    lines = []
    for lat, lon, bearing in points:
        x, y,z = gps_to_cartesian(lat, lon,bearing)
        print("x, y,z",x, y,z)
        lines.append(line_from_point_bearing((x, y,z), bearing))

    intersection_point_cartesian, _ = find_intersection(lines)
    print("intersection_point_cartesian",intersection_point_cartesian)
    intersection_point_gps = cartesian_to_gps(*intersection_point_cartesian)

    # 输出交点的GPS坐标
    print(f"交点的GPS坐标为: {intersection_point_gps}")
    # 创建地图对象，中心设置在计算的重心坐标上
    m = folium.Map(location=[intersection_point_gps[0], intersection_point_gps[1]], zoom_start=13)

    # 在地图上添加一个标记
    folium.Marker(
        location=[intersection_point_gps[0], intersection_point_gps[1]],
        popup='重心位置'+str(intersection_point_gps[0])+","+ str(intersection_point_gps[1]),
        icon=folium.Icon(icon='cloud')
    ).add_to(m)

    for lat, lon, bearing in points:
        # 在地图上添加一个标记
        folium.Marker(
            location=[lat, lon],
            popup='重心位置'+str(lat)+","+ str(lon),
            icon=folium.Icon(icon='cloud')
        ).add_to(m)
    

    # 在地图上添加线段
    for lat, lon, bearing in points:
        # 计算线段的终点，这里假设每个线段的长度为1000米
        end_lat, end_lon = calculate_line_endpoint(lat, lon, bearing, 1000)

        folium.PolyLine(
            locations=[(lat, lon), (end_lat, end_lon)],
            color='blue',
            weight=2.5,
            opacity=1
        ).add_to(m)

    # 保存地图
    m.save("map2.html")



