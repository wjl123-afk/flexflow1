#!/usr/bin/env python3
"""真正测试 2k input + 生成长文本"""

import flexflow.serve as ff
import time

print("="*70)
print("真正的 2k Context 测试")
print("="*70)
print()

# 生成一个接近 2k tokens 的长 prompt（约 1500-1800 个单词）
# 1 token ≈ 0.75 个英文单词
long_story = """
Once upon a time, in a vast kingdom filled with mysteries and wonders, 
there lived a young adventurer named Alex who dreamed of exploring the 
unknown territories beyond the mountains. """ * 100  # 重复 100 次，生成长文本

print(f"📝 Input 长度: {len(long_story)} 字符")
print(f"   (约 {len(long_story.split())} 个单词)")
print(f"   (约 {len(long_story.split()) * 4 // 3} tokens)")
print()

# 初始化
ff.init(
    num_gpus=4,
    memory_per_gpu=14000,
    zero_copy_memory_per_node=80000,
    tensor_parallelism_degree=4,
    pipeline_parallelism_degree=1
)

model_path = "/root/.cache/huggingface/hub/models--meta-llama--llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
llm = ff.LLM(model_path)

# 注意：temperature 和 sampling 设置会影响生成长度
generation_config = ff.GenerationConfig(
    do_sample=True,      # 开启采样，可能生成更长
    temperature=0.9,
    topp=0.8,
    topk=40
)

llm.compile(
    generation_config,
    max_requests_per_batch=1,
    max_seq_length=2048,      # 支持 2k input
    max_tokens_per_batch=4096  # 支持生成更多
)

llm.start_server()

print("🚀 开始推理...")
start = time.time()
result = llm.generate(long_story)
elapsed = time.time() - start

if result and len(result) > 0 and hasattr(result[0], 'output_text'):
    output = result[0].output_text
    if isinstance(output, bytes):
        output = output.decode('utf-8')
    
    print(f"\n✅ 推理成功")
    print(f"   Input: {len(long_story)} 字符")
    print(f"   Output: {len(output)} 字符")
    print(f"   Total: {len(long_story) + len(output)} 字符")
    print(f"   Time: {elapsed:.2f}s")
    print()
    print(f"   Output 预览 (前 200 字符):")
    print(f"   {output[:200]}")
    print(f"   ...")
    print(f"   (还有 {len(output) - 200} 字符)")

llm.stop_server()

print("\n" + "="*70)
print("✅ 2k Context 测试完成")
print("="*70)
