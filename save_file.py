import os
import re
import logging
import traceback
import pandas as pd
from datetime import datetime
logging.getLogger().setLevel(logging.INFO)

def get_data(aisbench_log, req_rate, npu_num):
    log_dir=""
    current_time, Total_InputTokens, Total_GeneratedTokens, Total_requests, max_Concurrency, Concurrency, req_rate, AVG_first_token_time, slo_p90_first_token_time, AVG_token_time, slo_p90_token_time, total_time, GenerateSpeed, single_generatespeed,e2e_throughput, single_e2e_throughput, qps, qpm, input_token_throughput,prefill_throughput = [99999, 99999, 99999, 99999, 99999, 99999, 0, 99999, 99999, 99999, 99999, 99999, 99999, 99999,9999,9999,9999,9999,9999,9999]
    try:
        with open(aisbench_log, 'r') as f_streaming:
            txt = f_streaming.readlines()
            for i in range(len(txt)):
                if "Current exp folder" in txt[i]:
                    matches = re.search(r"Current exp folder:\s*(.+)$", txt[i])
                    log_dir = matches.group(1).strip()
                if "TTFT" in txt[i]:
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    slo_p90_first_token_time = list(map(float, matches))[5]
                    AVG_first_token_time = list(map(float, matches))[0]
                if "TPOT" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    slo_p90_token_time = list(map(float, matches))[5]
                    AVG_token_time = list(map(float, matches))[0]
                if "Benchmark Duration" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # tmp = int(matches[0]) / 1000
                    # print(matches)
                    total_time = list(map(float, matches))[0] / 1000
                    # total_time = tmp
                if "Concurrency" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    if matches:
                        Concurrency = list(map(float, matches))[0]
                    
                if "Max Concurrency" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r"[\w']+", txt[i])
                    # print(matches[-1])
                    max_Concurrency = matches[-1]
                    
                if "Output Token Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    GenerateSpeed = list(map(float, matches))[0]
                    single_generatespeed = GenerateSpeed / npu_num
                
                if "Input Token Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    input_token_throughput = list(map(float, matches))[0]
                    
                if "Total Token Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    e2e_throughput = list(map(float, matches))[0]
                    single_e2e_throughput = e2e_throughput / npu_num

                if "InputTokens" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.?\d*)', txt[i])
                    # print(matches)
                    Total_InputTokens = list(map(float, matches))[0]

                if "OutputTokens" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.?\d*)', txt[i])
                    # print(matches)
                    Total_GeneratedTokens = list(map(float, matches))[0]

                if "Total Requests" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.?\d*)', txt[i])
                    # print(matches)
                    Total_requests = list(map(float, matches))[0]
                    
                if "Request Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    qps = list(map(float, matches))[0]
                    qpm = qps * 60
                    
                if "Prefill Token Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    prefill_throughput = list(map(float, matches))[0]
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if AVG_first_token_time and slo_p90_first_token_time and total_time and slo_p90_token_time and GenerateSpeed and Total_InputTokens and Total_GeneratedTokens and Total_requests and AVG_first_token_time and AVG_token_time:
            ans = [current_time, Total_InputTokens, Total_GeneratedTokens, Total_requests, max_Concurrency, Concurrency, req_rate, AVG_first_token_time, slo_p90_first_token_time, AVG_token_time, slo_p90_token_time, total_time, GenerateSpeed, single_generatespeed,e2e_throughput, single_e2e_throughput, qps, qpm, input_token_throughput, prefill_throughput]
        else:
            ans = [current_time, 99999, 99999, 99999, 99999, 99999, 0, 99999, 99999, 99999, 99999, 99999, 99999, 99999,9999,9999,9999,9999,9999,9999]
    except Exception as e:
        logging.warning(traceback.format_exc())
        ans = [99999, 99999, 99999, 99999, 99999, 99999, 0, 99999, 99999, 99999, 99999, 99999, 99999, 99999,9999,9999,9999,9999,9999,9999]
    # print(ans)
    return ans, log_dir

def save_log(aisbench_log, log_dir):
    os.system(f"cp {aisbench_log} {log_dir}")
    source_file = aisbench_log
    target_file = "aisbench_all.log"
    try:
        # 读取源文件内容
        with open(source_file, 'r', encoding='utf-8') as src:
            content = src.read()
        
        # 追加到目标文件
        with open(target_file, 'a', encoding='utf-8') as tgt:
            # 添加分隔线和时间戳
            tgt.write(f"\n\n{'='*50}\n")
            tgt.write(f"{'='*50}\n\n")
            tgt.write(content)
            tgt.write(f"\n\n{'='*50}\n")
            tgt.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            tgt.write(f"{'='*50}\n")
        
        logging.info(f"成功将 {source_file} 的内容追加到 {target_file}")
        
    except FileNotFoundError:
        logging.error(f"错误：找不到文件，请检查文件路径是否正确")
    except Exception as e:
        logging.error(f"发生错误：{e}")

def save_csv(ans, filename):
    # headers = ["平均输入", "平均输出", "总请求数", "最大并发数", "系统并发数", "请求频率", "TTFT平均", "TTFT P90", "TPOT平均", "TPOT SLO_P90", "E2E时间", "输出吞吐", "单卡输出吞吐", "E2E吞吐", "单卡E2E吞吐", "qps", "qpm", "prefill吞吐"]
    headers = ["current_time","input_len", "output_len", "total_req", "max_cc", "cc", "rr", "TTFT avg", "TTFT P90", "TPOT avg", "TPOT SLO_P90", "E2E_time", "output_throughput", "single_output_throughput","E2E_throughput","single_E2E_throughput","qps","qpm","input_token_throughput","prefill_token_throughput"]
    # 检查文件是否存在
    file_exists = os.path.exists(filename)
    # print(ans)
    try:
        if file_exists:
            # 文件存在，读取现有数据
            df_existing = pd.read_csv(filename)
            logging.info("文件已存在，读取现有数据")
            # 创建新数据行（确保列名匹配）
            new_row = pd.DataFrame([ans], columns=headers)
            # 追加新行
            df_updated = pd.concat([df_existing, new_row], ignore_index=True)
            # 保存更新后的数据
            df_updated.to_csv(filename, index=False)
            logging.info("成功追加新行")
            
        else:
            # 文件不存在，创建新的 DataFrame
            # print("文件不存在，创建新的 DataFrame")
            if isinstance(ans, dict):
                df_new = pd.DataFrame([ans])
            else:
                df_new = pd.DataFrame([ans], columns=headers)
            
            # 保存为新文件
            df_new.to_csv(filename, index=False)
            logging.info("创建新文件并写入数据")
        
    except Exception as e:
        logging.error(f"操作失败: {e}")
