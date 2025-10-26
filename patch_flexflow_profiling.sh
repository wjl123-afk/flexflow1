#!/bin/bash
# patch_flexflow_profiling.sh
# 自动为FlexFlow源码添加算子级profiling代码

set -e

echo "========================================================================"
echo "🔧 FlexFlow Operator Profiling Patch Script"
echo "========================================================================"
echo ""

# 配置
FLEXFLOW_ROOT="/workspace/flexflow-serve"
BACKUP_DIR="/workspace/flexflow-backup-$(date +%Y%m%d_%H%M%S)"
PROFILING_DIR="/workspace/parallel_strategy_profiling/method3_cuda_event"

# 检查环境
if [ ! -d "$FLEXFLOW_ROOT" ]; then
    echo "❌ FlexFlow目录不存在: $FLEXFLOW_ROOT"
    exit 1
fi

echo "📍 FlexFlow目录: $FLEXFLOW_ROOT"
echo "💾 备份目录: $BACKUP_DIR"
echo ""

# Step 1: 备份
echo "【Step 1/7】备份FlexFlow源码..."
mkdir -p "$BACKUP_DIR"
cp -r "$FLEXFLOW_ROOT/src" "$BACKUP_DIR/"
cp -r "$FLEXFLOW_ROOT/python" "$BACKUP_DIR/"
echo "✅ 备份完成: $BACKUP_DIR"
echo ""

# Step 2: 复制profiling工具类
echo "【Step 2/7】添加Profiling工具类..."
mkdir -p "$FLEXFLOW_ROOT/src/runtime/profiling"
cp "$PROFILING_DIR/profiling_utils.h" "$FLEXFLOW_ROOT/src/runtime/profiling/"
cp "$PROFILING_DIR/profiling_utils.cpp" "$FLEXFLOW_ROOT/src/runtime/profiling/"
echo "✅ 已添加: src/runtime/profiling/profiling_utils.h"
echo "✅ 已添加: src/runtime/profiling/profiling_utils.cpp"
echo ""

# Step 3: 修改CMakeLists.txt
echo "【Step 3/7】修改CMakeLists.txt..."

CMAKE_FILE="$FLEXFLOW_ROOT/src/runtime/CMakeLists.txt"

# 检查是否已经添加过
if grep -q "profiling_utils.cpp" "$CMAKE_FILE"; then
    echo "⚠️  CMakeLists.txt已经包含profiling_utils.cpp，跳过"
else
    # 在flexflow_runtime目标中添加profiling_utils.cpp
    sed -i '/add_library(flexflow_runtime/,/)/s/)$/  profiling\/profiling_utils.cpp\n)/' "$CMAKE_FILE"
    echo "✅ 已修改: $CMAKE_FILE"
fi
echo ""

# Step 4: 修改Linear算子
echo "【Step 4/7】修改Linear算子..."

LINEAR_FILE="$FLEXFLOW_ROOT/src/ops/linear.cc"

# 添加include
if ! grep -q "profiling/profiling_utils.h" "$LINEAR_FILE"; then
    sed -i '/#include/a #include "runtime/profiling/profiling_utils.h"' "$LINEAR_FILE"
    echo "✅ 已添加include到: $LINEAR_FILE"
fi

# 在forward函数中添加profiling
# 查找函数签名（这需要根据实际代码调整）
if ! grep -q "OperatorProfiler" "$LINEAR_FILE"; then
    cat > /tmp/linear_patch.txt << 'EOF'
  // === PROFILING START ===
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  std::string prof_name = std::string("Linear_") + this->name;
  OperatorProfiler::getInstance().start_timing(prof_name, stream);
  // === PROFILING END ===
EOF
    
    # 在forward_kernel_wrapper之前插入
    # 注意：这个sed命令需要根据实际代码结构调整
    echo "⚠️  需要手动编辑 $LINEAR_FILE 添加profiling代码"
    echo "    请在forward_kernel_wrapper()调用前后添加:"
    echo "    OperatorProfiler::getInstance().start_timing(op_name, stream);"
    echo "    OperatorProfiler::getInstance().end_timing(op_name, stream);"
else
    echo "✅ Linear算子已包含profiling代码"
fi
echo ""

# Step 5: 修改LayerNorm算子
echo "【Step 5/7】修改LayerNorm算子..."

LAYERNORM_FILE="$FLEXFLOW_ROOT/src/ops/layer_norm.cc"

if [ -f "$LAYERNORM_FILE" ]; then
    if ! grep -q "profiling/profiling_utils.h" "$LAYERNORM_FILE"; then
        sed -i '/#include/a #include "runtime/profiling/profiling_utils.h"' "$LAYERNORM_FILE"
        echo "✅ 已添加include到: $LAYERNORM_FILE"
    fi
    
    echo "⚠️  需要手动编辑 $LAYERNORM_FILE 添加profiling代码"
else
    echo "⚠️  文件不存在: $LAYERNORM_FILE"
fi
echo ""

# Step 6: 修改serve.cc或model.cc，在结束时输出profiling结果
echo "【Step 6/7】修改serve.cc以输出profiling结果..."

SERVE_FILE="$FLEXFLOW_ROOT/src/runtime/serve.cc"

if [ -f "$SERVE_FILE" ]; then
    if ! grep -q "profiling/profiling_utils.h" "$SERVE_FILE"; then
        sed -i '/#include/a #include "runtime/profiling/profiling_utils.h"' "$SERVE_FILE"
        echo "✅ 已添加include到: $SERVE_FILE"
    fi
    
    echo "⚠️  需要手动在LLM::stop_server()中添加:"
    echo "    OperatorProfiler::getInstance().print_results();"
    echo "    OperatorProfiler::getInstance().save_results(\"operator_profiling.json\");"
else
    echo "⚠️  文件不存在: $SERVE_FILE"
fi
echo ""

# Step 7: 生成手动patch指南
echo "【Step 7/7】生成手动patch指南..."

cat > "$PROFILING_DIR/MANUAL_PATCH_GUIDE.md" << 'EOF'
# FlexFlow Profiling手动Patch指南

由于FlexFlow代码结构复杂，部分修改需要手动完成。

## 需要手动修改的文件

### 1. src/ops/linear.cc

在 `Linear::forward()` 函数中：

```cpp
void Linear::forward(/* ... */) {
  // ... 原有代码 ...
  
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  
  // === 添加profiling START ===
  std::string prof_name = std::string("Linear_") + this->name;
  OperatorProfiler::getInstance().start_timing(prof_name, stream);
  // === 添加profiling END ===
  
  // 原有的kernel launch
  Internal::forward_kernel_wrapper(/* ... */);
  
  // === 添加profiling START ===
  OperatorProfiler::getInstance().end_timing(prof_name, stream);
  // === 添加profiling END ===
}
```

### 2. src/ops/layer_norm.cc

在 `LayerNorm::forward()` 函数中：

```cpp
void LayerNorm::forward(/* ... */) {
  // ... 原有代码 ...
  
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  
  // === 添加profiling ===
  OperatorProfiler::getInstance().start_timing("LayerNorm", stream);
  // 原有kernel launch
  OperatorProfiler::getInstance().end_timing("LayerNorm", stream);
}
```

### 3. src/ops/softmax.cc

类似LayerNorm的修改。

### 4. src/ops/element_unary.cc (SiLU)

在SiLU的forward函数中添加profiling。

### 5. src/runtime/serve.cc 或 src/runtime/model.cc

在推理结束时输出结果：

```cpp
void LLM::stop_server() {
  // ... 原有代码 ...
  
  // === 添加profiling输出 ===
  OperatorProfiler::getInstance().print_results();
  OperatorProfiler::getInstance().save_results("operator_profiling.json");
}
```

## 编译

```bash
cd /workspace/flexflow-serve/build
make -j$(nproc)
make install
cd ../python
pip install -e . --force-reinstall
```

## 测试

```bash
python test_llama2_1gpu_final.py -ll:gpu 1 -ll:zsize 40000 -ll:fsize 14000 -ll:cpu 4
```

应该看到profiling输出并生成 `operator_profiling.json`。
EOF

echo "✅ 手动patch指南已生成: $PROFILING_DIR/MANUAL_PATCH_GUIDE.md"
echo ""

echo "========================================================================"
echo "✅ Patch脚本执行完成！"
echo "========================================================================"
echo ""
echo "📝 下一步："
echo "   1. 阅读手动patch指南: $PROFILING_DIR/MANUAL_PATCH_GUIDE.md"
echo "   2. 手动编辑需要修改的算子文件"
echo "   3. 重新编译FlexFlow"
echo "   4. 运行测试验证profiling功能"
echo ""
echo "💾 原始代码备份: $BACKUP_DIR"
echo ""

