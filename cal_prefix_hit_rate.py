import logging
import re
from datetime import datetime
import requests

def _fetch_metrics(ip_address, port):
    """获取 /metrics 接口文本"""
    url = f"http://{ip_address}:{port}/metrics"
    resp = requests.get(url, proxies={"http": None, "https": None}, timeout=30)
    resp.raise_for_status()
    return resp.text


def get_prefix_queries_total(ip_address, port):
    """
    获取查询token总数,返回{engine:tokens}
    """
    try:
        text = _fetch_metrics(ip_address, port)
        lines = [l for l in text.split('\n') if 'model_name' in l and 'prefix_cache_queries_total' in l]
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
        text = _fetch_metrics(ip_address, port)
        lines = [l for l in text.split('\n') if 'model_name' in l and 'prefix_cache_hits_total' in l]
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
