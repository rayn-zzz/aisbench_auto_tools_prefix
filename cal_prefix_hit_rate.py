import subprocess
import logging
import re
from datetime import datetime
import os

def get_prefix_queries_total(ip_address, port):
    """
    获取查询token总数,返回{engine:tokens}
    """
    try:
        # 构建并执行命令
        url = f"http://{ip_address}:{port}/metrics"
        command = f"sleep 3s && curl -s {url} | grep 'prefix_cache_queries_total' | grep 'model_name'"
        # os.system(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0 or not result.stdout.strip():
            # print(f"未找到指标数据: {result.stderr}")
            return {}, {}
        lines = result.stdout.strip().split('\n')
        normal_stats = {}
        external_stats = {}
        
        for line in lines:
            # 提取engine值
            engine_match = re.search(r'engine="(\d+)"', line)
            if not engine_match:
                continue
                
            engine = engine_match.group(1)
            
            # 提取最后一个数字
            parts = line.split()
            if len(parts) < 2:
                continue
                
            value_str = parts[-1]
            try:
                value = float(value_str)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                continue
            
            # 根据指标名称分类
            if 'external_prefix_cache_queries_total' in line:
                external_stats[int(engine)] = value
            elif 'vllm:prefix_cache_queries_total' in line:
                normal_stats[int(engine)] = value
        print(normal_stats, external_stats)
        return normal_stats, external_stats
        
    except Exception as e:
        print(f"错误: {e}")
        return {}, {}
    
def get_prefix_hits_total(ip_address, port):
    """
    获取命中token总数,返回{engine:tokens}
    """
    try:
        # 构建并执行命令
        url = f"http://{ip_address}:{port}/metrics"
        command = f"sleep 3s && curl -s {url} | grep 'prefix_cache_hits_total' | grep 'model_name'"
        # os.system(command)
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0 or not result.stdout.strip():
            return {}, {}
        lines = result.stdout.strip().split('\n')
        normal_stats = {}
        external_stats = {}
        
        for line in lines:
            # 提取engine值
            engine_match = re.search(r'engine="(\d+)"', line)
            if not engine_match:
                continue
                
            engine = engine_match.group(1)
            
            # 提取最后一个数字
            parts = line.split()
            if len(parts) < 2:
                continue
                
            value_str = parts[-1]
            try:
                value = float(value_str)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                continue
            
            # 根据指标名称分类
            if 'external_prefix_cache_hits_total' in line:
                external_stats[int(engine)] = value
            elif 'vllm:prefix_cache_hits_total' in line:
                normal_stats[int(engine)] = value
        print(normal_stats, external_stats)
        return normal_stats, external_stats
        
    except Exception as e:
        print(f"错误: {e}")
        return {}, {}
