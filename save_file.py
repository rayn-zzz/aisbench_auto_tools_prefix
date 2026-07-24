import os
import re
import logging
import traceback
import pandas as pd
from datetime import datetime
logging.getLogger().setLevel(logging.INFO)

def get_data(aisbench_log, req_rate, npu_num):
    log_dir=""
    DEFAULT_METRICS = {
    "current_time": None,
    "input_len": 99999, #InputTokens
    "output_len": 99999, #OutputTokens
    "total_req": 99999, #Total Requests
    "max_cc": 99999,
    "cc": 99999,
    "rr": 0,
    "TTFT avg": 99999,
    "TTFT P90": 99999,
    "TPOT avg": 99999,
    "TPOT SLO_P90": 99999,
    "E2E_time": 99999, #Benchmark Duration 
    "output_throughput": 99999, #Output Token Throughput
    "single_output_throughput": 99999,
    "E2E_throughput": 9999, #Total Token Throughput
    "single_E2E_throughput": 9999,
    "qps": 9999,
    "qpm": 9999,
    "input_token_throughput": 9999, #Input Token Throughput
    "prefill_token_throughput": 9999, #Prefill Token Throughput
    "E2EL avg":9999,
    "E2EL P90":9999
    }

    try:
        with open(aisbench_log, 'r') as f_streaming:
            txt = f_streaming.readlines()
            for i in range(len(txt)):
                if "Current exp folder" in txt[i]:
                    matches = re.search(r"Current exp folder:\s*(.+)$", txt[i])
                    log_dir = matches.group(1).strip()
                if "E2EL" in txt[i]:
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    DEFAULT_METRICS["E2EL P90"]=list(map(float, matches))[5]
                    DEFAULT_METRICS["E2EL avg"]=list(map(float, matches))[0]
                if "TTFT" in txt[i]:
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    DEFAULT_METRICS["TTFT P90"] = list(map(float, matches))[5]
                    DEFAULT_METRICS["TTFT avg"] = list(map(float, matches))[0]
                if "TPOT" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    DEFAULT_METRICS["TPOT SLO_P90"] = list(map(float, matches))[5]
                    DEFAULT_METRICS["TPOT avg"] = list(map(float, matches))[0]
                if "Benchmark Duration" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # tmp = int(matches[0]) / 1000
                    # print(matches)
                    DEFAULT_METRICS["E2E_time"] = list(map(float, matches))[0] / 1000
                    # total_time = tmp
                if "Concurrency" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    if matches:
                        DEFAULT_METRICS["cc"] = list(map(float, matches))[0]
                    
                if "Max Concurrency" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r"[\w']+", txt[i])
                    # print(matches[-1])
                    DEFAULT_METRICS["max_cc"] = matches[-1]
                    
                if "Output Token Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    GenerateSpeed = list(map(float, matches))[0]
                    DEFAULT_METRICS["output_throughput"] = GenerateSpeed
                    DEFAULT_METRICS["single_output_throughput"] = GenerateSpeed / npu_num
                
                if "Input Token Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    DEFAULT_METRICS["input_token_throughput"] = list(map(float, matches))[0]
                    
                if "Total Token Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    e2e_throughput = list(map(float, matches))[0]
                    DEFAULT_METRICS["E2E_throughput"] = e2e_throughput
                    DEFAULT_METRICS["single_E2E_throughput"] =e2e_throughput / npu_num


                if "InputTokens" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.?\d*)', txt[i])
                    # print(matches)
                    DEFAULT_METRICS["input_len"] = list(map(float, matches))[0]

                if "OutputTokens" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.?\d*)', txt[i])
                    # print(matches)
                    DEFAULT_METRICS["output_len"] = list(map(float, matches))[0]

                if "Total Requests" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.?\d*)', txt[i])
                    # print(matches)
                    DEFAULT_METRICS["total_req"] = list(map(float, matches))[0]
                    
                if "Request Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    qps = list(map(float, matches))[0]
                    DEFAULT_METRICS["qps"]=qps
                    DEFAULT_METRICS["qpm"]=qps * 60
                    
                if "Prefill Token Throughput" in txt[i]:
                    # print(txt[i])
                    matches = re.findall(r'(\d+\.\d+)', txt[i])
                    # print(matches)
                    DEFAULT_METRICS["prefill_token_throughput"] = list(map(float, matches))[0] 
        
        DEFAULT_METRICS["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logging.warning(traceback.format_exc())
    print(DEFAULT_METRICS)
    return DEFAULT_METRICS, log_dir

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
    #headers = ["current_time","input_len", "output_len", "total_req", "max_cc", "cc", "rr", "TTFT avg", "TTFT P90", "TPOT avg", "TPOT SLO_P90", "E2E_time", "output_throughput", "single_output_throughput","E2E_throughput","single_E2E_throughput","qps","qpm","input_token_throughput","prefill_token_throughput","avg_e2e_latency","slo_p90_e2e_latency"]
    # 检查文件是否存在
    file_exists = os.path.exists(filename)
    # print(ans)
    df_new = pd.DataFrame([ans])
    try:
        if file_exists:
            # 文件存在，读取现有数据
            df_existing = pd.read_csv(filename)
            logging.info("文件已存在，读取现有数据")
            #new_row = pd.DataFrame(ans)
            # 追加新行
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            # 强制按旧文件的列顺序输出，防止新增字段导致列错乱
            df_updated = df_updated.reindex(columns=df_existing.columns.union(df_new.columns, sort=False))
            # 保存更新后的数据
            df_updated.to_csv(filename, index=False)
            logging.info("成功追加新行")
            
        else:
            # 保存为新文件
            df_new.to_csv(filename, index=False)
            logging.info("创建新文件并写入数据")
        
    except Exception as e:
        logging.error(f"操作失败: {e}")
