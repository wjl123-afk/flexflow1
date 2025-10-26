#!/usr/bin/env python3
"""
FlexFlow Serve 离线 Patch - 终极版（直接 patch）
直接对 serve.py 进行字符串替换，无需备份文件

4 个关键 Patch：
1. LLM.__init__: AutoConfig 使用 local_files_only=True
2. download_hf_config: 直接 return，跳过在线下载
3. download_hf_tokenizer_if_needed: 直接 return，跳过在线刷新
4. download_and_convert_llm_weights: AutoModelForCausalLM 使用 local_files_only=True

使用方法：
    python patch_serve_ultimate.py
"""

import shutil
import os
import sys

SERVE_PY = "/workspace/flexflow-serve/python/flexflow/serve/serve.py"
BACKUP = SERVE_PY + ".backup"

print("="*70)
print("🔧 FlexFlow Serve 离线 Patch - 终极版")
print("="*70)
print()

# 检查文件是否存在
if not os.path.exists(SERVE_PY):
    print(f"❌ 错误：serve.py 不存在")
    print(f"   路径: {SERVE_PY}")
    sys.exit(1)

# 1. 备份
print("📦 Step 1: 备份原始文件...")
if not os.path.exists(BACKUP):
    try:
        shutil.copy2(SERVE_PY, BACKUP)
        print(f"   ✅ 已备份到: {BACKUP}")
    except Exception as e:
        print(f"   ⚠️  备份失败: {e}")
else:
    print(f"   ℹ️  备份已存在，跳过")
print()

# 2. 读取文件
print("📖 Step 2: 读取 serve.py...")
try:
    with open(SERVE_PY, "r", encoding="utf-8") as f:
        content = f.read()
    print("   ✅ 读取完成")
except Exception as e:
    print(f"   ❌ 读取失败: {e}")
    sys.exit(1)
print()

patches_applied = 0
original_content = content

# ======================================================================
# Patch 1: LLM.__init__ - AutoConfig 强制本地加载
# ======================================================================
print("🔨 Patch 1: LLM.__init__ (AutoConfig local_files_only)...")
old1 = 'self.hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)'
new1 = 'self.hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)'

if old1 in content and new1 not in content:
    content = content.replace(old1, new1, 1)
    patches_applied += 1
    print("   ✅ 成功")
elif new1 in content:
    print("   ⚠️  已应用")
else:
    print("   ❌ 未找到目标代码")
print()

# ======================================================================
# Patch 2: download_hf_config - 直接 return
# ======================================================================
print("🔨 Patch 2: download_hf_config (跳过在线下载)...")

# 尝试多种可能的格式
old2_variants = [
    # 变体 1: 单引号 docstring
    '''    def download_hf_config(self):
        """Save the HuggingFace model configs to a json file. Useful mainly to run the C++ inference code."""
        config_dir''',
    # 变体 2: 三引号后直接代码
    '''    def download_hf_config(self):
        """Check in the folder specified by the cache_path whether the LLM's
        config is available and up to date. If not, or if the refresh_cache
        parameter is set to True, download new config from huggingface.
        """
        print("Loading model configs from huggingface.co")''',
    # 变体 3: 最简单的匹配
    '''    def download_hf_config(self):
        """'''
]

patch2_applied = False
for i, old2 in enumerate(old2_variants):
    if old2 in content:
        # 检查是否已经 patch
        if 'def download_hf_config(self):' in content:
            func_start = content.find('def download_hf_config(self):')
            func_part = content[func_start:func_start+200]
            if 'return  # PATCHED' in func_part:
                print("   ⚠️  已应用")
                patch2_applied = True
                break
        
        # 应用 patch
        new2 = old2.replace(
            '    def download_hf_config(self):\n        """',
            '    def download_hf_config(self):\n        return  # PATCHED: 跳过在线下载，使用本地配置\n        """',
            1
        )
        content = content.replace(old2, new2, 1)
        patches_applied += 1
        print(f"   ✅ 成功 (使用变体 {i+1})")
        patch2_applied = True
        break

if not patch2_applied:
    print("   ⚠️  未找到目标代码，尝试通用方法...")
    # 通用方法：在函数定义后插入 return
    pattern = 'def download_hf_config(self):'
    if pattern in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def download_hf_config(self):' in line:
                # 检查下一行是否已经是 return
                if i+1 < len(lines) and 'return' in lines[i+1]:
                    print("   ⚠️  已应用")
                else:
                    # 插入 return 语句
                    indent = '        '  # 8 spaces
                    lines.insert(i+1, f'{indent}return  # PATCHED: 跳过在线下载')
                    content = '\n'.join(lines)
                    patches_applied += 1
                    print("   ✅ 成功 (通用方法)")
                patch2_applied = True
                break
    
    if not patch2_applied:
        print("   ❌ 失败")
print()

# ======================================================================
# Patch 3: download_hf_tokenizer_if_needed - 直接 return
# ======================================================================
print("🔨 Patch 3: download_hf_tokenizer_if_needed (跳过刷新)...")

old3_variants = [
    '''    def download_hf_tokenizer_if_needed(self):
        """Download tokenizer from HuggingFace (or from the model's source in general) if needed."""
        print("Loading tokenizer...")''',
    '''    def download_hf_tokenizer_if_needed(self):
        """Check in the folder specified by the cache_path whether the LLM's tokenizer files are available and up to date.
        If not, or if the refresh_cache parameter is set to True, download new tokenizer files.
        """
        print("Loading tokenizer...")''',
    '''    def download_hf_tokenizer_if_needed(self):
        """'''
]

patch3_applied = False
for i, old3 in enumerate(old3_variants):
    if old3 in content:
        # 检查是否已经 patch
        if 'def download_hf_tokenizer_if_needed(self):' in content:
            func_start = content.find('def download_hf_tokenizer_if_needed(self):')
            func_part = content[func_start:func_start+300]
            if 'return  # PATCHED' in func_part:
                print("   ⚠️  已应用")
                patch3_applied = True
                break
        
        # 应用 patch
        new3 = old3.replace(
            '    def download_hf_tokenizer_if_needed(self):\n        """',
            '    def download_hf_tokenizer_if_needed(self):\n        return  # PATCHED: 跳过在线刷新，使用本地 tokenizer\n        """',
            1
        )
        content = content.replace(old3, new3, 1)
        patches_applied += 1
        print(f"   ✅ 成功 (使用变体 {i+1})")
        patch3_applied = True
        break

if not patch3_applied:
    print("   ⚠️  未找到目标代码，尝试通用方法...")
    pattern = 'def download_hf_tokenizer_if_needed(self):'
    if pattern in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def download_hf_tokenizer_if_needed(self):' in line:
                if i+1 < len(lines) and 'return' in lines[i+1]:
                    print("   ⚠️  已应用")
                else:
                    indent = '        '
                    lines.insert(i+1, f'{indent}return  # PATCHED: 跳过在线刷新')
                    content = '\n'.join(lines)
                    patches_applied += 1
                    print("   ✅ 成功 (通用方法)")
                patch3_applied = True
                break
    
    if not patch3_applied:
        print("   ❌ 失败")
print()

# ======================================================================
# Patch 4: download_and_convert_llm_weights - 本地权重转换
# ======================================================================
print("🔨 Patch 4: download_and_convert_llm_weights (本地权重)...")

old4_variants = [
    '''            snapshot_download(
                repo_id=model_name,
                allow_patterns="*.safetensors",
                max_workers=min(30, num_cores),
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,''',
    # 更简单的匹配
    '''hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='''
]

patch4_applied = False

# 首先检查是否已经 patch
if 'local_files_only=True,  # PATCHED' in content or 'local_files_only=True # PATCHED' in content:
    print("   ⚠️  已应用")
    patch4_applied = True
else:
    # 尝试第一个变体（包含 snapshot_download）
    if old4_variants[0] in content:
        new4 = '''            # PATCHED: 跳过 snapshot_download，直接从本地加载
            print(f"[PATCHED] 从本地加载权重: {model_name}")
            # snapshot_download(
            #     repo_id=model_name,
            #     allow_patterns="*.safetensors",
            #     max_workers=min(30, num_cores),
            # )
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,  # PATCHED'''
        content = content.replace(old4_variants[0], new4, 1)
        patches_applied += 1
        print("   ✅ 成功 (完整版)")
        patch4_applied = True
    # 尝试简化版本
    elif 'AutoModelForCausalLM.from_pretrained' in content:
        # 查找并添加 local_files_only
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'hf_model = AutoModelForCausalLM.from_pretrained' in line:
                # 查找这个函数调用的结束位置
                for j in range(i, min(i+10, len(lines))):
                    if 'trust_remote_code=True,' in lines[j] and 'local_files_only' not in lines[j]:
                        lines[j] = lines[j].replace(
                            'trust_remote_code=True,',
                            'trust_remote_code=True,\n                local_files_only=True,  # PATCHED'
                        )
                        content = '\n'.join(lines)
                        patches_applied += 1
                        print("   ✅ 成功 (添加 local_files_only)")
                        patch4_applied = True
                        break
                if patch4_applied:
                    break

if not patch4_applied:
    print("   ❌ 未找到目标代码")
print()

# 3. 保存
if patches_applied > 0 or content != original_content:
    print(f"💾 Step 3: 保存修改 ({patches_applied} 个新 patch)...")
    try:
        with open(SERVE_PY, "w", encoding="utf-8") as f:
            f.write(content)
        print("   ✅ 保存成功")
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        sys.exit(1)
else:
    print("ℹ️  没有新的 patch 需要应用")

print()
print("="*70)
print("🎉 Patch 完成！")
print("="*70)
print()
print("📝 关键修改:")
print("   1. LLM.__init__: AutoConfig 使用 local_files_only=True")
print("   2. download_hf_config: 直接 return，跳过在线下载")
print("   3. download_hf_tokenizer_if_needed: 直接 return，跳过刷新")
print("   4. download_and_convert_llm_weights: 使用本地 safetensors")
print()
print("🔍 验证 patch:")
print('   grep -n "return  # PATCHED" /workspace/flexflow-serve/python/flexflow/serve/serve.py')
print()
print("🔄 如需恢复原文件:")
print(f"   cp {BACKUP} {SERVE_PY}")
print()
print("🚀 运行测试:")
print("   cd /workspace/flexflow-serve/examples")
print("   python -u /workspace/test_llama2_1gpu_final.py -ll:gpu 1 -ll:zsize 40000 -ll:fsize 14000 -ll:cpu 4")
print()
