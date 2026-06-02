import os, errno
import argparse
import re
import logging
from config import *
from generate_dataset import *
from save_file import get_data, save_csv, save_log
from cal_prefix_hit_rate import *
logging.getLogger().setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=3500, help="input token length")
    parser.add_argument("--output_len", type=str, default="1500", help="output token length")
    parser.add_argument("--data_num", type=int, default=8192, help="dataset number")
    parser.add_argument("--concurrency", type=str, default="2048", help="max concurrency")
    parser.add_argument("--request_rate", type=str, default="0", help="request rate")
    parser.add_argument("--test_type", type=str, default="stream", help="text or stream")
    parser.add_argument("--dataset", type=str, default="none", help="dataset path")
    parser.add_argument("--repeat", type=int, default=1, help="number of test repeat times")
    parser.add_argument("--enable_think", action='store_true', default=False, help="enable thinking for ds v3.1")
    parser.add_argument("--test_accuracy", action='store_true', default=False, help="test accuracy")
    parser.add_argument("--npu_num", type=int, default=1, help="npu numbers")
    parser.add_argument("--dataset_type", type=str, default="normal", help="normal or prefix_cache")
    parser.add_argument("--prefix_num", type=int, default=1, help="prefix numbers")
    parser.add_argument("--repeat_rate", type=str, default="0", help="dataset repeat rate")
    parser.add_argument("--prefix_test", action='store_true', default=False, help="test prefix dataset firstly")
    parser.add_argument("--seed", type=int, default=1, help="dataset random seed")
    parser.add_argument("--dp", type=int, default=1, help="dp size")
    parser.add_argument("--length_mean", type=int, default=None, help="gaussian mean for variable length")
    parser.add_argument("--length_std", type=float, default=None, help="gaussian std for variable length")
    parser.add_argument("--length_min", type=int, default=None, help="min length for uniform range or gaussian clip")
    parser.add_argument("--length_max", type=int, default=None, help="max length for uniform range or gaussian clip")
    return parser.parse_args()

def symlink_force(target, link_name):
    logging.info(f"make symlink: {link_name} ==> {target}")
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def create_gsm8k_dataset(dataset_type, input_len, data_num, model_path, dataset_path, prefix_num, repeat_rate, seed,
                         length_mean=None, length_std=None, length_min=None, length_max=None):
    if not os.path.exists(dataset_path):
        logging.error(f"dataset work path {dataset_path} not exist. please create it first.")
        exit(0)

    base_name = os.path.basename(os.path.normpath(model_path))
    if dataset_type == "prefix_cache":
        prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,1,dp,repeat_rate,seed,prefix_num,
                                                                             length_mean, length_std, length_min, length_max)
        logging.info("[完成] 数据集已生成：")
        logging.info(f"  - 公共前缀：{prefix_jsonl_path}  (行数={dp*prefix_num})")
        logging.info(f"  - 数据集：  {dataset_jsonl_path} (行数={data_num})")
        logging.info("[信息] 配置：")
        logging.info(f"  tokens(单条长度)={input_len}, prefix_ratio(前缀重复率)={repeat_rate}")
        if length_mean is not None and length_std is not None:
            logging.info(f"  length_dist=gaussian(mean={length_mean}, std={length_std}, min={length_min}, max={length_max})")
        elif length_min is not None and length_max is not None:
            logging.info(f"  length_dist=uniform_int([{length_min}, {length_max}])")
        else:
            logging.info("  length_dist=fixed")
    else:
        dataset_name = "GSM8K-in" + str(input_len) + "-num" + str(data_num) + "-" + base_name + ".jsonl"
        logging.info(f"dataset_name: {dataset_name}")
        dataset_jsonl_path = os.path.join(dataset_path, dataset_name)
        prefix_jsonl_path = ""
        # 判断数据集是否存在
        if not os.path.exists(dataset_jsonl_path):
            logging.warning(f"Dataset {dataset_name} is not exist. Start create dataset")
            # create_data(input_len, data_num, model_path, dataset_path)
            prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,0,dp,0,seed,prefix_num,
                                                                                 length_mean, length_std, length_min, length_max)
            logging.info(f"Dataset {dataset_name} created.")
        else:
            logging.info(f"Dataset {dataset_name} exist.")
    return prefix_jsonl_path, dataset_jsonl_path

def generate_aisbench_command(DEFAULT_PERFORMANCE_TEST):
    if test_accuracy:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --work-dir {OUTPUT_DIR} --dump-eval-details"
    else:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --summarizer {DEFAULT_PERFORMANCE_TEST} --work-dir {OUTPUT_DIR} --debug --num-warmups 0 2>&1 | tee aisbench.log"
    return ais_bench_cmd

def generate_test_dataset(src_file, dst_dir):
    dst_file = os.path.join(dst_dir, "test.jsonl")
    logging.info(f"src_file: {src_file}")
    logging.info(f"dst_file: {dst_file}")
    # 使用软连接
    symlink_force(src_file, dst_file)
    return

def save_result(request_rate, npu_num):
    aisbench_log_dir = "aisbench.log"
    filename = "aisbench_result.csv"
    ans, log_dir=get_data(aisbench_log_dir,request_rate,npu_num)
    save_log(aisbench_log_dir, log_dir)
    save_csv(ans, filename)

def modify_aisbench_api(concurrency, output_len):
    file_default = open("default_api.py", 'r+')
    file_temp = open("temp_api.py", 'w+')
    logging.info("Api config file:")
    for ss in file_default.readlines():
        tt = re.sub("model_path_for_replace", MODEL_PATH, ss)
        tt = re.sub("model_name_for_replace", MODEL_NAME, tt)
        tt = re.sub("rr_for_replace", request_rate, tt)
        tt = re.sub("test_type_for_replace", api_test_type, tt)
        tt = re.sub("test_abbr_for_replace", api_test_abbr, tt)
        tt = re.sub("ip_for_replace", HOST_IP, tt)
        tt = re.sub("port_for_replace", HOST_PORT, tt)
        tt = re.sub("outputlen_for_replace", output_len, tt)
        tt = re.sub("concurrency_for_replace", concurrency, tt)
        if test_accuracy:
            generation_kwargs = "temperature=0.6,\n\t\t\ttop_p = 0.95"
        else:
            generation_kwargs = "temperature=0,\n\t\t\tignore_eos=True"
        if enable_think:
            generation_kwargs = generation_kwargs + ",\n\t\t\tchat_template_kwargs={\"enable_thinking\": True}"
        tt = re.sub("generation_kwargs_for_replace", generation_kwargs.expandtabs(4), tt)
        print(tt, end='')
        file_temp.write(tt)
    file_default.close()
    file_temp.close()
    symlink_force(
        os.path.join(os.getcwd(), "temp_api.py"),
        os.path.join(WORK_PATH, "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py")
    )

def get_pod_metrics_info(pod_info):
    query_tokens, query_tokens_external,hit_tokens,hit_tokens_external = {},{},{},{}
    for pod in pod_info:
        ip,port = pod.split(":")
        query_tokens[pod],query_tokens_external[pod] = get_prefix_queries_total(ip,port)
        hit_tokens[pod], hit_tokens_external[pod] = get_prefix_hits_total(ip,port)
    return query_tokens, query_tokens_external, hit_tokens, hit_tokens_external

def cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external,
                        query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new):
    if not query_tokens or not query_tokens_external or not hit_tokens or not hit_tokens_external:
        return
    
    # 定义列宽
    col1_width = 15   # engine id
    col2_width = 20   # hbm hit rate
    col3_width = 20   # hbm(hit/query)
    col4_width = 20   # external hit rate
    col5_width = 20   # external(hit/query)
    
    # 按POD分组遍历
    for pod, engines in sorted(query_tokens.items()):
        # 准备数据行
        data_rows = []
        for engine_id, token in sorted(engines.items()):
            query_hbm = query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
            hits_hbm = hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
            query_ex = query_tokens_external_new[pod][engine_id] - query_tokens_external[pod][engine_id]
            hits_ex = hit_tokens_external_new[pod][engine_id] - hit_tokens_external[pod][engine_id]
            
            if query_hbm == 0:
                hit_rate_str = "0%"
                hit_detail = "0/0"
            else:
                hit_rate_str = format(hits_hbm / query_hbm, '.2%')
                hit_detail = f"{hits_hbm}/{query_hbm}"
            
            if query_ex == 0:
                hit_rate_ex_str = "0%"
                hit_ex_detail = "0/0"
            else:
                hit_rate_ex_str = format(hits_ex / query_ex, '.2%')
                hit_ex_detail = f"{hits_ex}/{query_ex}"
            
            data_rows.append({
                'engine_id': str(engine_id),
                'hbm_rate': hit_rate_str,
                'hbm_detail': hit_detail,
                'external_rate': hit_rate_ex_str,
                'external_detail': hit_ex_detail
            })
        
        # 定义表头
        headers = ['engine_id', 'hbm_hit_rate', 'hbm(hit/query)', 'externel_hit_rate', 'externel(hit/query)']
        
        # 计算总宽度
        total_width = col1_width + col2_width + col3_width + col4_width + col5_width + 8
        
        # 打印POD信息
        print("\n" + "=" * total_width)
        print(f"POD: {pod}")
        print("=" * total_width)
        
        # 打印表头
        print(f"{headers[0]:<{col1_width}} {headers[1]:<{col2_width}} {headers[2]:<{col3_width}} {headers[3]:<{col4_width}} {headers[4]:<{col5_width}}")
        print("-" * total_width)
        
        # 打印数据行
        for row in data_rows:
            print(f"{row['engine_id']:<{col1_width}} {row['hbm_rate']:<{col2_width}} {row['hbm_detail']:<{col3_width}} {row['external_rate']:<{col4_width}} {row['external_detail']:<{col5_width}}")
        
        print("=" * total_width)

if __name__ == '__main__':
    args = parse_arguments()
    input_len = args.input_len
    output_len = args.output_len
    data_num = args.data_num
    concurrency = args.concurrency
    request_rate = args.request_rate
    test_type = args.test_type
    dataset_path_input = args.dataset
    test_times = args.repeat
    enable_think = args.enable_think
    test_accuracy = args.test_accuracy
    npu_num = args.npu_num
    prefix_num = args.prefix_num
    repeat_rate = parse_prefix_ratio(args.repeat_rate)
    prefix_test = args.prefix_test
    dataset_type = args.dataset_type
    seed = args.seed
    dp = args.dp
    length_mean = args.length_mean
    length_std = args.length_std
    length_min = args.length_min
    length_max = args.length_max

    # 变长参数校验
    if (length_mean is None) ^ (length_std is None):
        raise ValueError("length_mean 和 length_std 必须同时提供或同时不提供")
    if (length_min is None) ^ (length_max is None):
        raise ValueError("length_min 和 length_max 必须同时提供或同时不提供")
    if length_mean is not None and length_mean < 1:
        raise ValueError("length_mean 必须 >= 1")
    if length_std is not None and length_std < 0:
        raise ValueError("length_std 必须 >= 0")
    if length_min is not None and length_min < 1:
        raise ValueError("length_min 必须 >= 1")
    if length_max is not None and length_max < 1:
        raise ValueError("length_max 必须 >= 1")

    logging.info(f"input token length: {input_len}")
    logging.info(f"output token length: {output_len}")
    logging.info(f"number of dataset: {data_num}")
    logging.info(f"concurrency: {concurrency}")
    logging.info(f"request rate: {request_rate}")
    logging.info(f"test type: {test_type}")
    logging.info(f"test_times: {test_times}")
    logging.info(f"v3.1 enable_think: {enable_think}")
    logging.info(f"accuracy test: {test_accuracy}")
    logging.info(f"npu numbers: {npu_num}")
    logging.info(f"prefix numbers: {prefix_num}")
    logging.info(f"dataset repeat rate: {repeat_rate}")
    logging.info(f"test prefix dataset: {prefix_test}")
    logging.info(f"dataset type: {dataset_type}")
    logging.info(f"seed: {seed}")
    logging.info(f"dp size: {dp}")
    logging.info(f"length_mean: {length_mean}")
    logging.info(f"length_std: {length_std}")
    logging.info(f"length_min: {length_min}")
    logging.info(f"length_max: {length_max}")

    # 区分流式和非流式
    if test_type == "text":
        api_test_type = "VLLMCustomAPIChat"
        api_test_abbr = "vllm-api-general-chat"
    elif test_type == "stream":
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = "vllm-api-stream-chat"
    else:
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = "vllm-api-stream-chat"

    if dataset_path_input == "none":
        src_file_prefix,src_file_data = create_gsm8k_dataset(dataset_type, input_len, data_num, MODEL_PATH, DATASET_PATH, prefix_num, repeat_rate, seed,
                                                              length_mean, length_std, length_min, length_max)
    else:
        # 指定数据集路径逻辑
        if not os.path.exists(dataset_path_input):
            logging.error(f"Dataset {dataset_path_input} is not exist.")
            exit(0)
        src_file_data = dataset_path_input
        src_file_prefix = ""

    dst_dir = os.path.join(WORK_PATH, "ais_bench/datasets/gsm8k")

    # 判断 aisbench 的 gsm8k 文件夹是否存在在
    if not os.path.exists(dst_dir):
        logging.info("dataset work path not exist. creating.")
        os.makedirs(dst_dir)
        logging.info("dataset work path created.")
    # 判断 aisbench 的 gsm8k 文件夹是否存在在 train.jsonl 文件
    train_dataset = os.path.join(dst_dir, "train.jsonl")
    if not os.path.exists(train_dataset):
        logging.info("train dataset not exist. creating.")
        file = open(train_dataset, 'w')
        file.close()
        logging.info("train dataset created.")    
    
    # 生成 aisbench 命令
    ais_bench_cmd = generate_aisbench_command(DEFAULT_PERFORMANCE_TEST)
    logging.info(f"test start, use command: {ais_bench_cmd}")
    
    # 执行命令    
    if dataset_type == "prefix_cache":
        # 前缀数据集测试
        if prefix_test:
            if not POD_INFO:
                pod_info = [HOST_IP+":"+HOST_PORT]
            else:
                pod_info = POD_INFO
            logging.info(f"pod_info: {pod_info}")
            
            logging.info(f"[开始] 前缀数据集测试")
            modify_aisbench_api(str(dp),"1")
            dst_file = generate_test_dataset(src_file_prefix, dst_dir)

            # 命中率计算
            query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

            os. system(ais_bench_cmd)
            logging.info(f"[完成] 前缀数据集测试完成，结果保存在aisbench_result.csv")

            query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)
            cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external,query_tokens_new,
                                query_tokens_external_new, hit_tokens_new, hit_tokens_external_new)
            
            # 保存前缀测试结果
            save_result(request_rate, npu_num)
            logging.info(f"[开始] 全量数据集测试")
            # 命中率计算
            query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)
            
            modify_aisbench_api(concurrency,str(output_len))
            dst_file = generate_test_dataset(src_file_data, dst_dir)
            # 执行测试命令
            os. system(ais_bench_cmd)
            logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv")
            
            query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)
            cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external, query_tokens_new,
                                query_tokens_external_new, hit_tokens_new, hit_tokens_external_new)
            
        else:
            logging.info(f"[开始] 全量数据集测试")
            modify_aisbench_api(concurrency,str(output_len))
            dst_file = generate_test_dataset(src_file_data, dst_dir)
            os. system(ais_bench_cmd)
            logging.info(f"[完成] 全量数据集测试完成")

    else:
        logging.info(f"[开始] 全量数据集测试")
        modify_aisbench_api(concurrency,str(output_len))
        dst_file = generate_test_dataset(src_file_data, dst_dir)
        if test_times > 1:
            for test_time in range(test_times):
                logging.info(f"Execution rounds: {test_time + 1}")
                os.system(ais_bench_cmd)
        else:
            os.system(ais_bench_cmd)
        logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv")

    
    # 保存结果
    save_result(request_rate, npu_num)
