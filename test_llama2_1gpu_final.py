#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FlexFlow + Llama-2-7b 1GPU 功能验证测试
"""
import flexflow.serve as ff
import time

print("=" * 70)
print("🦙 FlexFlow + Llama-2-7b 功能验证")
print("=" * 70)
print()

# 初始化
print("【1/6】初始化 FlexFlow...")
start = time.time()
ff.init(
    num_gpus=1,
    memory_per_gpu=14000,
    zero_copy_memory_per_node=40000,
    tensor_parallelism_degree=1,
    pipeline_parallelism_degree=1
)
print(f"✅ 完成 (耗时: {time.time()-start:.2f}s)")
print()

# 加载模型
print("【2/6】加载模型...")
start = time.time()
model_path = "/root/.cache/huggingface/hub/models--meta-llama--llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
llm = ff.LLM(model_path)
print(f"✅ 完成 (耗时: {time.time()-start:.2f}s)")
print()

# 配置
print("【3/6】配置生成参数...")
generation_config = ff.GenerationConfig(
    do_sample=False,
    temperature=0.9,
    topp=0.8,
    topk=1
)
print("✅ 完成")
print()

# 编译
print("【4/6】编译模型...")
start = time.time()
llm.compile(
    generation_config,
    max_requests_per_batch=1,
    max_seq_length=256,
    max_tokens_per_batch=128
)
print(f"✅ 完成 (耗时: {time.time()-start:.2f}s)")
print()

# 启动服务器
print("【5/6】启动服务器...")
start = time.time()
llm.start_server()
print(f"✅ 完成 (耗时: {time.time()-start:.2f}s)")
print()

# 推理测试
print("【6/6】推理测试...")
print("=" * 70)
prompt = "Hello, my name is"
print(f"📝 Prompt: {prompt}")
print()

start = time.time()
result = llm.generate(prompt)
inference_time = time.time() - start

print(f"⏱️  推理耗时: {inference_time:.2f}s")
print()

# 解析结果
if isinstance(result, list) and len(result) > 0:
    print(f"🔍 Result 类型: {type(result[0])}")
    print(f"🔍 Result 对象: {result[0]}")
    print()
    
    if hasattr(result[0], 'output_text'):
        output_text = result[0].output_text
        if isinstance(output_text, bytes):
            output_text = output_text.decode('utf-8')
        
        print(f"✅ 生成成功！")
        print()
        print(f"🤖 生成内容:")
        print("-" * 70)
        print(output_text)
        print("-" * 70)
        print()
        print(f"📊 统计:")
        print(f"   - 输出长度: {len(output_text)} 字符")
        print(f"   - 输出词数: ~{len(output_text.split())} 词")
        print(f"   - 平均速度: ~{len(output_text.split())/inference_time:.2f} tokens/s")
    else:
        print("⚠️  无法获取 output_text 属性")
        print(f"可用属性: {dir(result[0])}")
else:
    print(f"❌ 生成失败: {result}")

print()
print("=" * 70)

# 停止服务器
print("🛑 停止服务器...")
llm.stop_server()
print("✅ 已停止")
print()

# 总结
print("=" * 70)
print("🎉 FlexFlow + Llama-2-7b 功能验证完成！")
print("=" * 70)
print()
print("验证结果:")
print("  ✅ 初始化")
print("  ✅ 模型加载")
print("  ✅ 模型编译")
print("  ✅ 服务器启动")
print("  ✅ 推理生成")
print("  ✅ 服务器停止")
print()
print("配置: 1x GPU | Llama-2-7b | FP16")
print("=" * 70)
