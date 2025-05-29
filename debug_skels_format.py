#!/usr/bin/env python3
"""
调试脚本：分析.skels文件的确切格式
"""

import re

def analyze_skels_format():
    """分析.skels文件格式"""
    
    # 读取第一行数据
    with open("datasets/final_data/final_data/dev.skels", 'r') as f:
        first_line = f.readline().strip()
    
    parts = first_line.split()
    print(f"总数据点数: {len(parts)}")
    
    # 查找所有可能的时间戳
    timestamps = []
    for i, part in enumerate(parts):
        try:
            value = float(part)
            # 检查是否是时间戳（0-1之间且有小数点）
            if 0 <= value <= 1 and '.' in part:
                timestamps.append((i, value, part))
        except ValueError:
            continue
    
    print(f"找到 {len(timestamps)} 个可能的时间戳:")
    for i, (pos, val, orig) in enumerate(timestamps[:20]):  # 只显示前20个
        print(f"  位置 {pos}: {orig} ({val})")
    
    if len(timestamps) >= 2:
        # 计算帧间距离
        distances = []
        for i in range(1, min(10, len(timestamps))):
            dist = timestamps[i][0] - timestamps[i-1][0] - 1  # 减1是因为要排除时间戳本身
            distances.append(dist)
        
        print(f"\n前几帧的数据点数: {distances}")
        print(f"平均每帧数据点数: {sum(distances)/len(distances):.1f}")
    
    # 分析数值范围
    values = []
    for part in parts[:1000]:  # 只分析前1000个数值
        try:
            values.append(float(part))
        except ValueError:
            continue
    
    print(f"\n数值统计 (前1000个):")
    print(f"  最小值: {min(values):.5f}")
    print(f"  最大值: {max(values):.5f}")
    print(f"  平均值: {sum(values)/len(values):.5f}")
    
    # 查找时间戳模式
    print(f"\n时间戳模式分析:")
    if len(timestamps) >= 2:
        time_diffs = []
        for i in range(1, min(10, len(timestamps))):
            diff = timestamps[i][1] - timestamps[i-1][1]
            time_diffs.append(diff)
        
        print(f"时间戳差值: {[f'{d:.5f}' for d in time_diffs]}")
        if time_diffs:
            print(f"平均时间间隔: {sum(time_diffs)/len(time_diffs):.5f}")
    
    # 尝试按固定长度分割
    print(f"\n尝试按固定长度分割:")
    for frame_size in [150, 135, 120]:
        num_frames = len(parts) // frame_size
        remainder = len(parts) % frame_size
        print(f"  如果每帧{frame_size}维: {num_frames}帧, 剩余{remainder}个数据点")


if __name__ == "__main__":
    analyze_skels_format() 