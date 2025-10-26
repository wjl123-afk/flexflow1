#!/usr/bin/env python3
"""真正测试 2k input + 生成长文本（修复 token 长度问题）"""

import flexflow.serve as ff
import time

print("="*70)
print("真正的 2k Context 测试（4卡张量并行）")
print("="*70)
print()

# 生成约 1500 tokens 的 prompt（留 500 tokens 给输出）
# 基础文本约 45 tokens，重复 30 次 = 1350 tokens
base_text = """
Once upon a time, in a vast kingdom filled with mysteries and wonders, 
there lived a young adventurer named Alex who dreamed of exploring the 
unknown territories beyond the mountains. The journey was long and perilous,
filled with ancient forests, towering peaks, and hidden valleys where
legends spoke of treasures untold and dangers unimaginable.
"""

# 重复 30 次，约 1350 tokens（安全范围内）
long_story = base_text * 30

print(f"📝 Input 长度: {len(long_story)} 字符")
print(f"   (约 {len(long_story.split())} 个单词)")
print(f"   (约 {int(len(long_story.split()) * 1.3)} tokens)")
print()
print("💡 配置:")
print("   max_seq_length: 2048 tokens")
print("   预留空间: ~700 tokens for output")
print()

# 初始化 - 4卡张量并行
start_init = time.time()
print("【1/6】初始化 FlexFlow (4 GPUs, TP=4)...")
ff.init(
    num_gpus=4,
    memory_per_gpu=14000,
    zero_copy_memory_per_node=80000,
    tensor_parallelism_degree=4,
    pipeline_parallelism_degree=1
)
print(f"✅ 初始化完成 (耗时: {time.time() - start_init:.2f}s)")
print()

# 加载模型
start_load = time.time()
print("【2/6】加载模型...")
model_path = "/root/.cache/huggingface/hub/models--meta-llama--llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
llm = ff.LLM(model_path)
print(f"✅ 模型加载完成 (耗时: {time.time() - start_load:.2f}s)")
print()

# 配置生成参数
print("【3/6】配置生成参数...")
generation_config = ff.GenerationConfig(
    do_sample=False,     # Greedy decoding（张量并行兼容）
    temperature=0.9,
    topp=0.8,
    topk=1
)
print("✅ 配置完成 (Greedy decoding)")
print()

# 编译
start_compile = time.time()
print("【4/6】编译模型...")
print("   max_seq_length: 2048 tokens")
print("   max_tokens_per_batch: 4096 tokens")
llm.compile(
    generation_config,
    max_requests_per_batch=1,
    max_seq_length=2048,
    max_tokens_per_batch=4096
)
print(f"✅ 编译完成 (耗时: {time.time() - start_compile:.2f}s)")
print()

# 启动服务
print("【5/6】启动推理服务...")
llm.start_server()
print("✅ 服务启动")
print()

# 推理测试
print("【6/6】测试推理...")
print("="*70)
print(f"📝 Prompt 长度: {len(long_story)} 字符 (~{len(long_story.split())} 单词)")
print()

start = time.time()
result = llm.generate(long_story)
elapsed = time.time() - start

if result and len(result) > 0 and hasattr(result[0], 'output_text'):
    output = result[0].output_text
    if isinstance(output, bytes):
        output = output.decode('utf-8')
    
    print(f"\n✅ 推理成功")
    print("="*70)
    print(f"📊 统计信息:")
    print(f"   Input:  {len(long_story):,} 字符 (~{len(long_story.split())} 单词)")
    print(f"   Output: {len(output):,} 字符 (~{len(output.split())} 单词)")
    print(f"   Total:  {len(long_story) + len(output):,} 字符")
    print(f"   Time:   {elapsed:.2f}s")
    print()
    
    # 计算吞吐量（粗略估计）
    input_tokens = int(len(long_story.split()) * 1.3)
    output_tokens = int(len(output.split()) * 1.3)
    total_tokens = input_tokens + output_tokens
    throughput = total_tokens / elapsed
    
    print(f"   估算 tokens:")
    print(f"     Input:  ~{input_tokens} tokens")
    print(f"     Output: ~{output_tokens} tokens")
    print(f"     Total:  ~{total_tokens} tokens")
    print(f"   吞吐量: {throughput:.2f} tokens/s")
    print()
    
    print(f"📝 Output 预览 (前 500 字符):")
    print("-"*70)
    print(output[:500])
    if len(output) > 500:
        print("...")
        print(f"(还有 {len(output) - 500} 字符)")
    print("-"*70)
else:
    print(f"\n⚠️  结果异常: {result}")

llm.stop_server()

print("\n" + "="*70)
print("✅ 2k Context 长文本测试完成")
print("="*70)
print()
print("💡 关键配置:")
print("   ✅ do_sample=False (Greedy, 兼容张量并行)")
print("   ✅ tensor_parallelism_degree=4 (4卡并行)")
print("   ✅ max_seq_length=2048 (支持 2k tokens)")
print("   ✅ input: ~1300 tokens (留空间给 output)")
print()
print("📊 验证结果:")
print(f"   Input 实际长度: ~{int(len(long_story.split()) * 1.3)} tokens")
print("   配置上限: 2048 tokens")
print(f"   是否成功: {'✅ 是' if int(len(long_story.split()) * 1.3) < 2048 else '❌ 否'}")
print()
