#!/bin/bash

# FlexFlow环境检查和配置脚本
# 使用方法: source check_flexflow_env.sh

echo "========================================="
echo "🔍 开始检查FlexFlow环境"
echo "========================================="
echo ""

# 步骤1: 检查FlexFlow文件结构
echo "📁 步骤1: 检查FlexFlow目录结构"
echo "----------------------------------------"
echo "检查FlexFlow目录:"
ls -la /workspace/flexflow-serve/ 2>/dev/null || echo "❌ FlexFlow目录不存在"

echo ""
echo "检查build目录:"
ls -la /workspace/flexflow-serve/build/ 2>/dev/null || echo "❌ build目录不存在"

echo ""
echo "检查set_python_envs.sh:"
if [ -f /workspace/flexflow-serve/build/set_python_envs.sh ]; then
    echo "✅ set_python_envs.sh存在"
    ls -la /workspace/flexflow-serve/build/set_python_envs.sh
else
    echo "❌ set_python_envs.sh不存在"
fi

echo ""

# 步骤2: 配置FlexFlow环境变量
echo "🔧 步骤2: 配置FlexFlow环境变量"
echo "----------------------------------------"
cd /workspace/flexflow-serve/build

# 检查set_python_envs.sh是否存在
if [ -f set_python_envs.sh ]; then
    echo "✅ 加载set_python_envs.sh..."
    source set_python_envs.sh
else
    echo "⚠️ set_python_envs.sh不存在，手动设置环境变量..."
    
    # 手动设置环境变量
    export BUILD_FOLDER=/workspace/flexflow-serve/build
    export PYTHON_FOLDER=/workspace/flexflow-serve/python
    export PYLIB_PATH=$(python $PYTHON_FOLDER/flexflow/findpylib.py 2>/dev/null)
    
    if [ -n "$PYLIB_PATH" ]; then
        export PYLIB_DIR=$(dirname $PYLIB_PATH)
        export LD_LIBRARY_PATH="$BUILD_FOLDER:$BUILD_FOLDER/deps/legion/lib:$PYLIB_DIR:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$BUILD_FOLDER:$BUILD_FOLDER/deps/legion/lib:$LD_LIBRARY_PATH"
    fi
    
    export PYTHONPATH="$PYTHON_FOLDER:$BUILD_FOLDER/deps/legion/bindings/python:$PYTHONPATH"
fi

echo "✅ 环境变量已设置"
echo ""

# 步骤3: 验证FlexFlow
echo "🔍 步骤3: 验证FlexFlow可用性"
echo "----------------------------------------"

# 检查Python版本
echo "检查Python版本:"
python --version

# 检查Python路径
echo ""
echo "检查Python路径:"
python -c "import sys; print('\n'.join(sys.path))"

# 检查FlexFlow导入
echo ""
echo "检查FlexFlow导入:"
if python -c "import flexflow; print('✅ FlexFlow导入成功'); print('FlexFlow路径:', flexflow.__file__)" 2>&1; then
    echo ""
else
    echo "❌ FlexFlow导入失败"
    echo "请检查环境变量:"
    echo "PYTHONPATH: $PYTHONPATH"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
fi

echo ""
echo "========================================="
echo "✅ 环境检查完成"
echo "========================================="

