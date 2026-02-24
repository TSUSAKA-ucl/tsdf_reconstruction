#!/usr/bin/env python3
import os
import argparse
import sys
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np

# --- 設定 ---
# .mcapファイルそのものではなく、bagデータが含まれるディレクトリを指定してください
# BAG_PATH = 'path/to/your_bag_directory' 
# TOPIC_NAME = '/camera/camera/depth/color/points'
OUTPUT_DIR = './out_ply'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run(BAG_PATH, TOPIC_NAME):
    reader = rosbag2_py.SequentialReader()
    
    # storage_idを指定しないことで、metadata.yamlから自動判別させます
    storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', 
        output_serialization_format='cdr'
    )
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        return

    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        if topic == TOPIC_NAME:
            # PointCloud2として復元
            msg = deserialize_message(data, PointCloud2)
            
            print(f"height & width: {msg.height} x {msg.width}")
            for field in msg.fields:
                print(f"Field: {field.name}, offset: {field.offset}, datatype: {field.datatype}, count: {field.count}")
            # PointCloud2 からデータを抽出
            gen = pc2.read_points(msg, skip_nans=True, field_names=['x', 'y', 'z', 'rgb'])
            
            # 構造化されたデータを numpy の float32 行列に一括変換
            data_np = np.fromiter(gen, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
            
            if data_np.size == 0:
                continue

            # 位置 (XYZ) を抽出して (N, 3) の行列にする
            xyz = np.stack([data_np['x'], data_np['y'], data_np['z']], axis=1).astype(np.float64)
            
            # Open3D PCD作成
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # 色 (RGB) の処理
            if 'rgb' in data_np.dtype.names:
                rgb_floats = data_np['rgb']
                # float32のビットパターンをuint32として解釈
                rgb_ints = rgb_floats.view(np.uint32)
                r = ((rgb_ints >> 16) & 0xFF) / 255.0
                g = ((rgb_ints >> 8) & 0xFF) / 255.0
                b = (rgb_ints & 0xFF) / 255.0
                colors = np.stack([r, g, b], axis=1).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # ファイル名生成
            filename = f"frame_{timestamp}.ply"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # 保存 (ascii=Falseで高速バイナリ保存)
            o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)
            print(f"Exported: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Convert ROS 2 bag to PLY')
    parser.add_argument('bag_path', help='Path to the ROS 2 bag directory')
    parser.add_argument('--topic', default='/camera/camera/depth/color/points', help='Topic name')
    
    # ROS 2 の引数（--ros-args）を無視してパースする場合
    args, unknown = parser.parse_known_args()
    
    print(f"Processing bag: {args.bag_path}")
    run(args.bag_path, args.topic);

if __name__ == '__main__':
    main()
