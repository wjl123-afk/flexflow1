#!/usr/bin/env python3
"""测试 ModelScope 下载的 OPT-125M + FlexFlow (使用绝对路径)"""

import flexflow.serve as ff
import time

print("="*70)
print("🤖 FlexFlow + OPT-125M (使用绝对路径)")
print("="*70)
print()

# 初始化
start_time = time.time()
print("【1/6】初始化 FlexFlow...")
ff.init(
    num_gpus=1,
    memory_per_gpu=8000,
    zero_copy_memory_per_node=20000,
    tensor_parallelism_degree=1,
    pipeline_parallelism_degree=1
)
print(f"✅ 初始化完成 (耗时: {time.time() - start_time:.2f}s)")
print()

# 加载模型（使用绝对路径，而不是 HF repo ID）
start_time = time.time()
print("【2/6】加载模型...")
model_path = "/root/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"
print(f"路径: {model_path}")
llm = ff.LLM(model_path)
print(f"✅ 模型加载完成 (耗时: {time.time() - start_time:.2f}s)")
print()

# 配置生成参数
print("【3/6】配置生成参数...")
generation_config = ff.GenerationConfig(
    do_sample=False, 
    temperature=0.9, 
    topp=0.8, 
    topk=1
)
print("✅ 配置完成")
print()

# 编译
start_time = time.time()
print("【4/6】编译模型...")
print("⚠️  如果是首次运行，会自动转换权重（需要 2-3 分钟）")
llm.compile(
    generation_config,
    max_requests_per_batch=1,
    max_seq_length=128,
    max_tokens_per_batch=64
)
print(f"✅ 编译完成 (耗时: {time.time() - start_time:.2f}s)")
print()

# 启动服务
print("【5/6】启动推理服务...")
llm.start_server()
print("✅ 服务启动")
print()

# 测试推理
print("【6/6】测试推理...")
print("="*70)
prompt = "Hello, I am a language model"
print(f"📝 Prompt: {prompt}")
print()

start_time = time.time()
result = llm.generate(prompt)
inference_time = time.time() - start_time

print(f"🤖 Result: {result}")
if result and len(result) > 0:
    if hasattr(result[0], 'output_text'):
        output = result[0].output_text
        if isinstance(output, bytes):
            output = output.decode('utf-8')
        print(f"✅ Output: {output}")
        print(f"⏱️  推理时间: {inference_time:.2f}s")
    else:
        print(f"⚠️  Result attributes: {dir(result[0])}")

print("="*70)
print()

# 停止服务
llm.stop_server()
print("✅ 测试完成")
print()
print("="*70)
print("🎉 FlexFlow + OPT-125M 运行成功！")
print("="*70)
print()
print("📝 验证权重转换:")
print(f"   ls -lh {model_path}/half-precision/")
print()
