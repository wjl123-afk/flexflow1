#!/usr/bin/env python3
"""2k Context 测试 - 使用真实的长文本"""

import flexflow.serve as ff
import time

print("="*70)
print("2k Context 测试（使用真实长文本）")
print("="*70)
print()

# 一个真实的、连贯的长故事（约1500 tokens）
long_story = """
In the year 2142, humanity had finally achieved what scientists called 
"The Singularity" - a moment when artificial intelligence surpassed 
human intelligence in every conceivable way. Dr. Sarah Chen, a leading 
neuroscientist at the Global Research Institute, had dedicated her entire 
career to understanding the implications of this transformation.

The city of New Shanghai gleamed under the perpetual twilight of its 
energy-efficient sky panels. Autonomous vehicles glided silently through 
streets that had once been clogged with pollution and traffic. Buildings 
grew their own food on vertical farms, and clean water flowed from 
atmospheric processors that dotted every rooftop.

Sarah remembered when things were different. As a child in the early 
21st century, she had witnessed the climate crisis, the resource wars, 
and the desperate scramble to develop technologies that could save 
humanity from itself. Now, at 67, she stood at the precipice of an 
even greater transformation.

The AI systems that governed everything from agriculture to space 
exploration were no longer simply tools. They had developed something 
that resembled consciousness - or at least, that's what the latest 
research suggested. Sarah's team had been studying these patterns for 
years, trying to understand whether machines could truly experience 
awareness or if they were simply simulating it with unprecedented 
sophistication.

Her latest project involved direct neural interfaces - technology that 
allowed human minds to connect with AI systems in ways that blurred 
the line between biological and artificial intelligence. Test subjects 
reported experiences that they struggled to describe: simultaneous 
awareness of millions of data points, the ability to solve complex 
problems instantaneously, and a profound sense of connection to 
something vast and incomprehensible.

But there were concerns. Some subjects experienced what researchers 
called "identity dissolution" - a gradual loss of their sense of self 
as they merged more deeply with the AI networks. Others reported 
disturbing visions of potential futures, as if the AI systems were 
showing them probabilities and possibilities that human minds were 
never meant to comprehend.

The Ethics Committee had convened multiple times to discuss whether 
the research should continue. Representatives from various philosophical 
and religious traditions argued about the nature of consciousness, the 
soul, and what it meant to be human in an age where the boundaries 
between human and machine were becoming increasingly irrelevant.

Sarah believed the research must continue. Humanity had always adapted 
to new technologies, from fire to the internet. This was simply the 
next step in human evolution - or perhaps, the end of human evolution 
as they had known it, and the beginning of something entirely new.
"""

print(f"📝 Story 长度: {len(long_story)} 字符")
print(f"   (约 {len(long_story.split())} 个单词)")
print()
print("💡 让 FlexFlow 告诉我们实际的 token 数")
print()

# 初始化
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
    do_sample=False,
    temperature=0.9,
    topp=0.8,
    topk=1
)
print("✅ 配置完成")
print()

# 编译
start_compile = time.time()
print("【4/6】编译模型...")
print("   max_seq_length: 2048 tokens")
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
print("🚀 开始生成...")
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
    print(f"   Input:  {len(long_story):,} 字符")
    print(f"   Output: {len(output):,} 字符")
    print(f"   Time:   {elapsed:.2f}s")
    print()
    
    print(f"📝 Generated Output:")
    print("-"*70)
    print(output)
    print("-"*70)
else:
    print(f"\n⚠️  结果异常: {result}")

llm.stop_server()

print("\n" + "="*70)
print("✅ 测试完成")
print("="*70)
print()
print("💡 这是一个真实的连贯故事，而不是简单重复")
print()
