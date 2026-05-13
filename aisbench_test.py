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

def create_gsm8k_dataset(dataset_type, input_len, data_num, model_path, dataset_path, prefix_num, repeat_rate, seed):
    if not os.path.exists(dataset_path):
        logging.error(f"dataset work path {dataset_path} not exist. please create it first.")
        exit(0)
    
    base_name = os.path.basename(os.path.normpath(model_path))       
    if dataset_type == "prefix_cache":
        prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,1,dp,repeat_rate,seed,prefix_num)
        logging.info("[完成] 数据集已生成：")
        logging.info(f"  - 公共前缀：{prefix_jsonl_path}  (行数={dp*prefix_num})")
        logging.info(f"  - 数据集：  {dataset_jsonl_path} (行数={data_num})")
        logging.info("[信息] 配置：")
        logging.info(f"  tokens(单条长度)={input_len}, prefix_ratio(前缀重复率)={repeat_rate}")
    else:
        dataset_name = "GSM8K-in" + str(input_len) + "-num" + str(data_num) + "-" + base_name + ".jsonl"
        logging.info(f"dataset_name: {dataset_name}")
        dataset_jsonl_path = os.path.join(dataset_path, dataset_name)
        prefix_jsonl_path = ""
        # 判断数据集是否存在
        if not os.path.exists(dataset_jsonl_path):
            logging.warning(f"Dataset {dataset_name} is not exist. Start create dataset")
            # create_data(input_len, data_num, model_path, dataset_path)
            prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,0,dp,0,seed,prefix_num)
            logging.info(f"Dataset {dataset_name} created.")
        else:
            logging.info(f"Dataset {dataset_name} exist.")
    return prefix_jsonl_path, dataset_jsonl_path

def generate_aisbench_command(DEFAULT_PERFORMANCE_TEST):
    if test_accuracy:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --dump-eval-details --work-dir {OUTPUT_DIR}"
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
        src_file_prefix,src_file_data = create_gsm8k_dataset(dataset_type, input_len, data_num, MODEL_PATH, DATASET_PATH, prefix_num, repeat_rate, seed)
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
            logging.info(f"[开始] 前缀数据集测试")
            modify_aisbench_api(str(dp),"1")
            dst_file = generate_test_dataset(src_file_prefix, dst_dir)

            # 命中率计算
            query_tokens,query_tokens_external = get_prefix_queries_total(HOST_IP,HOST_PORT)
            hit_tokens,hit_tokens_external = get_prefix_hits_total(HOST_IP,HOST_PORT)

            os. system(ais_bench_cmd)
            logging.info(f"[完成] 前缀数据集测试完成")

            query_tokens_new,query_tokens_external_new = get_prefix_queries_total(HOST_IP,HOST_PORT)
            hit_tokens_new,hit_tokens_external_new = get_prefix_hits_total(HOST_IP,HOST_PORT)
            if query_tokens_new and hit_tokens_new:
                for key in query_tokens_new:
                    logging.info(f"----------------------prefix cache metrics: engine {key}----------------------")
                    # HBM 命中率
                    logging.info(f"[prefix cache metrics: engine {key}] prefix cache查询的token数：{query_tokens_new[key] - query_tokens[key]}")
                    logging.info(f"[prefix cache metrics: engine {key}] prefix cache命中的token数：{hit_tokens_new[key] - hit_tokens[key]}")
                    if query_tokens[key] != query_tokens_new[key]:
                        hit_rate = format((hit_tokens_new[key] - hit_tokens[key]) / (query_tokens_new[key] - query_tokens[key]), '.2%')
                    else:
                        hit_rate = 0
                    logging.info(f"[prefix cache metrics: engine {key}] prefix cache命中率（命中token/查询token）：{hit_rate}")
                    # DRAM 命中率
                    logging.info(f"[prefix cache metrics: engine {key}] external查询的token数：{query_tokens_external_new[key] - query_tokens_external[key]}")
                    logging.info(f"[prefix cache metrics: engine {key}] external命中的token数：{hit_tokens_external_new[key] - hit_tokens_external[key]}")
                    if query_tokens_external[key] != query_tokens_external_new[key]:
                        hit_rate_external = format((hit_tokens_external_new[key] - hit_tokens_external[key]) / (query_tokens_external_new[key] - query_tokens_external[key]), '.2%')
                    else:
                        hit_rate_external = 0
                    logging.info(f"[prefix cache metrics: engine {key}] external命中率（命中token/查询token）：{hit_rate_external}")
            
            # 保存前缀测试结果
            save_result(request_rate, npu_num)
            logging.info(f"[开始] 全量数据集测试")
            # 命中率计算
            query_tokens,query_tokens_external = get_prefix_queries_total(HOST_IP,HOST_PORT)
            hit_tokens,hit_tokens_external = get_prefix_hits_total(HOST_IP,HOST_PORT)
            
            modify_aisbench_api(concurrency,str(output_len))
            dst_file = generate_test_dataset(src_file_data, dst_dir)
            # 执行测试命令
            os. system(ais_bench_cmd)
            logging.info(f"[完成] 全量数据集测试完成")
            
            query_tokens_new,query_tokens_external_new = get_prefix_queries_total(HOST_IP,HOST_PORT)
            hit_tokens_new,hit_tokens_external_new = get_prefix_hits_total(HOST_IP,HOST_PORT)
            if query_tokens_new and hit_tokens_new:
                # print("---------------prefix metrics---------------")
                for key in query_tokens_new:
                    logging.info(f"----------------------prefix cache metrics: engine {key}----------------------")
                    # HBM 命中率
                    logging.info(f"[prefix cache metrics: engine {key}] prefix cache查询的token数：{query_tokens_new[key] - query_tokens[key]}")
                    logging.info(f"[prefix cache metrics: engine {key}] prefix cache命中的token数：{hit_tokens_new[key] - hit_tokens[key]}")
                    if query_tokens[key] != query_tokens_new[key]:
                        hit_rate = format((hit_tokens_new[key] - hit_tokens[key]) / (query_tokens_new[key] - query_tokens[key]), '.2%')
                    else:
                        hit_rate = 0
                    logging.info(f"[prefix cache metrics: engine {key}] prefix cache命中率（命中token/查询token）：{hit_rate}")
                    # DRAM 命中率
                    logging.info(f"[prefix cache metrics: engine {key}] external查询的token数：{query_tokens_external_new[key] - query_tokens_external[key]}")
                    logging.info(f"[prefix cache metrics: engine {key}] external命中的token数：{hit_tokens_external_new[key] - hit_tokens_external[key]}")
                    if query_tokens_external[key] != query_tokens_external_new[key]:
                        hit_rate_external = format((hit_tokens_external_new[key] - hit_tokens_external[key]) / (query_tokens_external_new[key] - query_tokens_external[key]), '.2%')
                    else:
                        hit_rate_external = 0
                    logging.info(f"[prefix cache metrics: engine {key}] external命中率（命中token/查询token）：{hit_rate_external}")
            
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
        logging.info(f"[完成] 全量数据集测试完成")

    
    # 保存结果
    save_result(request_rate, npu_num)
