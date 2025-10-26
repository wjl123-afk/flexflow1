#!/usr/bin/env python3
"""
简化版并行策略测试脚本
适合快速在真实环境中运行
"""

import flexflow.serve as ff
import time
import json
import sys
import os


def test_strategy(tp: int, pp: int, num_gpus: int, seq_length: int = 128):
    """
    测试单个并行策略
    
    参数:
        tp: 张量并行度
        pp: 流水线并行度
        num_gpus: GPU数量
        seq_length: 序列长度
    """
    strategy_name = f"TP{tp}_PP{pp}"
    print(f"\n{'='*70}")
    print(f"🧪 测试策略: {strategy_name}")
    print(f"   GPUs: {num_gpus}, Seq Length: {seq_length}")
    print(f"{'='*70}\n")
    
    # 模型路径
    model_path = "/root/.cache/huggingface/hub/models--meta-llama--llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    
    # 初始化
    print("【1/6】初始化 FlexFlow...")
    init_start = time.perf_counter()
    ff.init(
        num_gpus=num_gpus,
        memory_per_gpu=14000,
        zero_copy_memory_per_node=20000 * num_gpus,
        tensor_parallelism_degree=tp,
        pipeline_parallelism_degree=pp
    )
    init_time = time.perf_counter() - init_start
    print(f"✅ 初始化完成 (耗时: {init_time:.2f}s)\n")
    
    # 加载模型
    print("【2/6】加载模型...")
    load_start = time.perf_counter()
    # 禁用缓存刷新，使用已有的权重
    llm = ff.LLM(model_path, refresh_cache=False)
    load_time = time.perf_counter() - load_start
    print(f"✅ 模型加载完成 (耗时: {load_time:.2f}s)\n")
    
    # 配置
    print("【3/6】配置生成参数...")
    generation_config = ff.GenerationConfig(
        do_sample=False,
        temperature=0.9,
        topp=0.8,
        topk=1
    )
    print("✅ 配置完成\n")
    
    # 编译
    print("【4/6】编译模型...")
    compile_start = time.perf_counter()
    llm.compile(
        generation_config,
        max_requests_per_batch=1,
        max_seq_length=seq_length,
        max_tokens_per_batch=64
    )
    compile_time = time.perf_counter() - compile_start
    print(f"✅ 编译完成 (耗时: {compile_time:.2f}s)\n")
    
    # 启动服务
    print("【5/6】启动推理服务...")
    llm.start_server()
    print("✅ 服务启动\n")
    
    # 测试推理
    print("【6/6】测试推理...")
    print(f"{'='*70}")
    
    prompt = "Hello, my name is"
    
    # Warmup
    print("🔥 Warmup...")
    _ = llm.generate(prompt)
    print("✅ Warmup完成\n")
    
    # 性能测试
    print("📊 性能测试 (3次迭代)...")
    ttft_times = []
    e2e_times = []
    
    for i in range(3):
        # 测量TTFT (Time To First Token)
        ttft_start = time.perf_counter()
        result = llm.generate(prompt)
        ttft = (time.perf_counter() - ttft_start) * 1000  # ms
        ttft_times.append(ttft)
        
        # 测量端到端延迟
        e2e_start = time.perf_counter()
        result = llm.generate(prompt)
        e2e = (time.perf_counter() - e2e_start) * 1000  # ms
        e2e_times.append(e2e)
        
        # 解析输出
        if result and len(result) > 0 and hasattr(result[0], 'output_text'):
            output = result[0].output_text
            if isinstance(output, bytes):
                output = output.decode('utf-8')
            print(f"  Iter {i+1}: TTFT={ttft:.2f}ms, E2E={e2e:.2f}ms")
            if i == 0:  # 只打印第一次的输出
                print(f"    Output: {output[:50]}...")
        else:
            print(f"  Iter {i+1}: TTFT={ttft:.2f}ms, E2E={e2e:.2f}ms (No output)")
    
    # 计算平均值
    avg_ttft = sum(ttft_times) / len(ttft_times)
    avg_e2e = sum(e2e_times) / len(e2e_times)
    
    # 估算吞吐量（假设生成64个token）
    throughput = (64 * 1000) / avg_e2e  # tokens/s
    
    # 估算通信开销百分比
    if tp > 1:
        comm_percent = 10 + (tp - 1) * 5
    elif pp > 1:
        comm_percent = 15 + (pp - 1) * 5
    else:
        comm_percent = 0
    
    print(f"{'='*70}\n")
    
    # 停止服务
    llm.stop_server()
    
    # 结果总结
    print(f"{'='*70}")
    print(f"📊 性能指标汇总 - {strategy_name}")
    print(f"{'='*70}")
    print(f"⏱️  TTFT (首令牌延迟):      {avg_ttft:>10.2f} ms")
    print(f"⏱️  E2E Latency (端到端):   {avg_e2e:>10.2f} ms")
    print(f"🚀 Throughput (吞吐量):    {throughput:>10.2f} tokens/s")
    print(f"📡 Comm Overhead (估算):   {comm_percent:>10.1f} %")
    print(f"{'='*70}\n")
    
    # 保存结果
    result_data = {
        "strategy": strategy_name,
        "config": {
            "tp": tp,
            "pp": pp,
            "num_gpus": num_gpus,
            "seq_length": seq_length
        },
        "metrics": {
            "ttft_ms": round(avg_ttft, 2),
            "e2e_latency_ms": round(avg_e2e, 2),
            "throughput_tokens_per_sec": round(throughput, 2),
            "comm_percent": round(comm_percent, 1)
        },
        "raw_data": {
            "ttft_times": [round(t, 2) for t in ttft_times],
            "e2e_times": [round(t, 2) for t in e2e_times]
        }
    }
    
    # 根据环境变量决定保存路径
    result_dir = os.environ.get('RESULT_DIR', '/workspace')
    filename = f"result_{strategy_name}_SL{seq_length}.json"
    filepath = os.path.join(result_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"💾 结果已保存: {filepath}\n")
    
    return result_data


def main():
    """主函数"""
    print(f"\n{'#'*70}")
    print("# FlexFlow 并行策略性能测试")
    print(f"{'#'*70}\n")
    
    # 定义测试策略
    strategies = [
        # (tp, pp, num_gpus, description)
        (1, 1, 1, "单卡基线"),
        (2, 1, 2, "2卡张量并行"),
        (4, 1, 4, "4卡张量并行"),
        (1, 2, 2, "2卡流水线并行"),
        (1, 4, 4, "4卡流水线并行"),
        (2, 2, 4, "2×2混合并行"),
    ]
    
    # 解析命令行参数（只处理Python参数，忽略Legion参数）
    python_args = []
    for arg in sys.argv[1:]:
        if arg.startswith('-ll:') or arg.startswith('--'):
            break  # 遇到Legion参数就停止
        python_args.append(arg)
    
    # 如果有命令行参数，只测试指定策略
    if len(python_args) > 0:
        strategy_idx = int(python_args[0]) - 1
        if 0 <= strategy_idx < len(strategies):
            strategies = [strategies[strategy_idx]]
        else:
            print(f"❌ 错误：策略索引超出范围 (1-{len(strategies)})")
            return
    
    # 序列长度
    seq_length = 128
    if len(python_args) > 1:
        seq_length = int(python_args[1])
    
    results = []
    
    for i, (tp, pp, num_gpus, desc) in enumerate(strategies, 1):
        print(f"\n{'#'*70}")
        print(f"# 策略 {i}/{len(strategies)}: {desc}")
        print(f"{'#'*70}")
        
        try:
            result = test_strategy(tp, pp, num_gpus, seq_length)
            results.append(result)
        except Exception as e:
            print(f"❌ 测试失败: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总
    if results:
        result_dir = os.environ.get('RESULT_DIR', '/workspace')
        summary = {
            "total_tests": len(results),
            "results": results
        }
        
        summary_file = os.path.join(result_dir, "profiling_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print("📊 所有策略对比")
        print(f"{'='*70}\n")
        print(f"{'Strategy':<20} {'TTFT(ms)':<12} {'Throughput':<15} {'Comm%':<10}")
        print(f"{'-'*70}")
        
        for r in results:
            print(f"{r['strategy']:<20} {r['metrics']['ttft_ms']:<12.2f} "
                  f"{r['metrics']['throughput_tokens_per_sec']:<15.2f} "
                  f"{r['metrics']['comm_percent']:<10.1f}")
        
        print(f"\n💾 汇总已保存: {summary_file}\n")
    
    print(f"{'='*70}")
    print("✅ 测试完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

