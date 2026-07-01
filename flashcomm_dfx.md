# FlashComm 特性 DFX 问题定位指南

## 文档概述

本文档面向 vllm-ascend 中 FlashComm 特性的问题定位场景，目标读者包括：
- **一线交付/运维人员**：快速判断问题是否与 FlashComm 相关，并执行初步定界
- **研发/二线支持人员**：深入理解 FlashComm 原理，进行根因分析

FlashComm 是 vllm-ascend 中用于优化分布式推理通信效率的特性集合。由于 FlashComm 的核心逻辑在 NPU Graph（ACL Graph）编译后 replay 执行，**图内无法插入运行时日志**（日志打印会触发图重新编译或破坏异步流水线，严重影响性能），因此本文档采用"**原理理解 + 图外观测 + 工具定位**"的方式来实现 DFX 能力。

---

## 第一层：快速决策树

> 目标：一线人员 3 分钟内判断问题是否与 FlashComm 相关，并做出定界动作。

### 1.1 你的场景是否涉及 FlashComm？

FlashComm 在以下条件下才会被激活：

| FlashComm 版本 | 激活条件 | 影响范围 |
|---|---|---|
| **FlashComm1 (Sequence Parallelism)** | `enable_flashcomm1=True` 且 `tp_size > 1` 且 `num_tokens > 1000`（dense）或 MoE 模型任意 token 数 | 所有 column/row parallel linear 层 |
| **FlashComm2** | `enable_flashcomm2_parallel_size > 0` 且 `tp_size > 1` | 仅 `o_proj`（attention 输出投影）层 |
| **DSA-CP** | `enable_dsa_cp=True` 且 FlashComm1 已开启（依赖 SP） | MLA/DSA 模型的 QKV 与 o_proj 层 |

**快速自查命令**：检查启动日志中是否包含以下关键词：
```
"Enable FLASHCOMM2 with flashcomm2_oproj_tensor_parallel_size"
"Flash Comm v1 is only supported when tp_size > 1"
"Linear layer sharding enabled with config"
```

### 1.2 决策树

```
问题症状
├── 服务启动失败 / 初始化报错
│   └── → 检查 FlashComm 配置是否合法（见 2.1 节）
│
├── 推理结果异常 / 精度下降
│   └── → 关闭 FlashComm 对比结果（见 2.3 节）
│
├── 推理过程挂死（hang）
│   └── → 判断是通信挂死还是计算挂死（见 2.2 节）
│
└── 推理性能劣化（比不开 FlashComm 还慢）
    └── → profiler 分析通信耗时占比（见 2.4 节）
```

### 1.3 快速验证开关

将以下配置从 `--additional-config` 中移除或显式置为 False/0，然后重新启动服务对比行为：

```bash
# 关闭 FlashComm1
--additional-config '{"enable_flashcomm1": false}'

# 关闭 FlashComm2（将值设为 0）
--additional-config '{"enable_flashcomm2_parallel_size": 0}'
```

或通过环境变量：
```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
```

> **关键判断**：如果关闭 FlashComm 后问题消失，则强烈指向 FlashComm 相关逻辑。如果问题依然存在，则继续排查其他方向。

---

## 第二层：排查 SOP

> 目标：提供每种故障模式的详细排查步骤和信息收集清单。

### 2.1 初始化失败

#### 2.1.1 常见配置约束

FlashComm 的配置合法性检查分散在多个模块中，以下为关键约束：

| 约束 | 代码位置 | 说明 |
|---|---|---|
| FlashComm1 要求 `tp_size > 1` | `platform.py:L732-735` | 单卡场景 flashcomm 无意义 |
| FlashComm1 在 MoE 模型下要求 `enable_expert_parallel=True` | `platform.py:L737-739` | MoE 通信方式冲突 |
| FlashComm2 要求 `global_tp_size > flashcomm2_otp_size` | `utils.py:L1212-1216` | OTP 子组不能超过全局 TP |
| FlashComm2 要求 `global_tp_size % flashcomm2_otp_size == 0` | `utils.py:L1217-1221` | TP size 必须被 OTP size 整除 |
| FlashComm2 不能与 `oproj_tensor_parallel_size` 同时开启 | `utils.py:L1208-1211` | Finegrained TP 冲突 |
| FlashComm2 不支持 D-scenario（纯 decode 节点） | `utils.py:L1227-1231` | 仅 P 节点或 PD 混合 |
| `layer_sharding` 仅支持 `["o_proj"]` | `utils.py:L1196-1203` | 不支持其他层的 sharding |
| `ASCEND_LAUNCH_BLOCKING=1` 与 Graph 模式互斥 | `platform.py:L636-645` | 同步调试与图捕获不兼容 |
| DSA-CP 要求 FlashComm1 已开启 | `utils.py:L1370-1373` | DSA-CP 依赖 SP |

#### 2.1.2 排查步骤

1. **收集启动日志**：完整保留 `vllm serve` 的输出，搜索 `ERROR` 和 `AssertionError`
2. **确认配置来源**：检查是通过 `--additional-config` 还是环境变量传入（代码 `ascend_config.py:L291-302` 会打印日志指明来源）
3. **检查 TP/DP/PP 拓扑**：确认 `tensor_parallel_size`、`data_parallel_size`、`pipeline_parallel_size` 组合合法
4. **检查 KV transfer 角色**：确认当前节点是 P（kv_producer）、D（kv_consumer）还是 mixed（kv_both）

### 2.2 挂死（Hang）

FlashComm 挂死通常发生在集合通信操作上。由于通信操作已被编译进 NPU Graph，挂死时 Python 层无法提供栈信息，需要从通信库和系统层面定位。

#### 2.2.1 挂死点分析

FlashComm 中涉及的通信操作及挂死风险：

| FlashComm 版本 | 通信操作 | 所在 Op 类 | 挂死风险场景 |
|---|---|---|---|
| FlashComm1 | `all_gather` (column) | `SequenceColumnParallelOp` | TP 组内某 rank 提前退出 |
| FlashComm1 | `reduce_scatter` / `npu_mm_reduce_scatter_base` (row) | `SequenceRowParallelOp` | batch size 不一致导致通信不匹配 |
| FlashComm2 | `all_to_all_single` | `Flashcomm2OProjRowParallelOp` | ODP 域内的 batch 重组出错 |
| FlashComm2 | `reduce_scatter` | `Flashcomm2OProjRowParallelOp` | OTP 子组通信失败 |
| FlashComm2 | `all_gather` | `Flashcomm2OProjRowParallelOp` | TP 组 all_gather 卡死 |
| DSA-CP | `all_gather_async` | `AscendSFAImpl` | 异步通信未正确等待 |
| OShard | `broadcast`（异步） | `Flashcomm2OshardQKVParallelOp` | 异步 broadcast 未完成就使用数据 |

#### 2.2.2 挂死定位步骤

**步骤 1：确定挂死范围**

```bash
# 在所有 NPU 节点上执行，看哪些 rank 卡住
pstack <vllm_pid> | grep -E "hccl|all_reduce|all_gather|all_to_all|reduce_scatter"
```

或者使用 `gdb attach`：
```bash
gdb -p <pid> -batch -ex "thread apply all bt" 2>/dev/null | grep -E "hccl|Hccl|npu"
```

**步骤 2：开启 HCCL 通信库调试日志**

```bash
# 设置 HCCL 调试级别（0=关闭，1=INFO，2=DEBUG，3=TRACE）
export HCCL_DEBUG=2
export HCCL_DEBUG_FILE=/path/to/hccl_debug.log

# 重新启动服务，挂死后查看日志
```

> **注意**：HCCL 日志完全独立于 NPU Graph，不会影响图编译和 replay 性能。这是 flashcomm 挂死定位的**首选手段**。

**步骤 3：使用 msprof 获取通信 timeline**

```bash
# 开启 profiling 重新运行
msprof --output=/path/to/output --application="vllm serve <model> ..."
```

查看 timeline 中通信算子的发起时间和完成时间，判断哪个 rank 的通信未完成。

**步骤 4：检查 pad_size 是否正确**

FlashComm 在 `ascend_forward_context.py:L141-144` 中计算 pad_size：

```python
pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
```

如果不同 rank 的 `num_tokens` 不同（例如在 DP > 1 场景下，`num_tokens_across_dp` 导致 pad 不一致），会导致通信尺寸不匹配，引发挂死。检查方式：

```python
# 在 Flashcomm2OProjRowParallelOp.apply_impl 入口处（图外）
# 添加 device_print 验证 num_tokens 是否各 rank 一致
from vllm_ascend.utils import device_print
device_print(num_actual_tokens)
```

> 注意：`device_print` 仅在开发调试时使用，不要在生产环境长期开启（utils.py 中有明确警告）。

**步骤 5：验证通信域拓扑**

FlashComm2 创建了两层嵌套通信域（`parallel_state.py:L152-189`）：

```
全局 TP Group (size=8)
├── OTP Group (flashcomm2_otp_size=2) — 用于 reduce_scatter
│   ├── Group 0: ranks [0, 4]
│   ├── Group 1: ranks [1, 5]
│   ├── Group 2: ranks [2, 6]
│   └── Group 3: ranks [3, 7]
└── ODP Group — 用于 all-to-all
    ├── Group 0: ranks [0, 1, 2, 3]
    └── Group 1: ranks [4, 5, 6, 7]
```

出现挂死时，需要确认：
- 每个 OTP 子组内的 rank 是否都在参与通信
- ODP 域的 all-to-all 是否所有 rank 的发送/接收尺寸匹配

### 2.3 精度异常

#### 2.3.1 精度问题来源分析

FlashComm 通过改变通信和计算的执行顺序来优化性能，这种重排可能导致数值差异：

| 机制 | 潜在精度影响 |
|---|---|
| Pad 机制 | pad 位置的 token 参与了 all-reduce / reduce_scatter，可能引入额外的浮点累积误差 |
| MM + ReduceScatter Fusion (mmrs) | 融合算子内部精度路径与分开执行不同 |
| all-to-all 替代 all-reduce | 通信数据重排改变了浮点运算的加和顺序 |

#### 2.3.2 精度定位步骤

**步骤 1：二分法确认是否为 FlashComm 引入**

```bash
# 对同一输入，分别以开/关 flashcomm 运行，对比输出
# 关闭 FlashComm1
vllm serve <model> --additional-config '{"enable_flashcomm1": false, "enable_flashcomm2_parallel_size": 0}'
# 开启 FlashComm（原始配置）
vllm serve <model> --additional-config '{"enable_flashcomm1": true, ...}'
```

对比两次运行的输出 logits，如果差异超过预期范围（如 `max_diff > 1e-3`），则确认与 FlashComm 相关。

**步骤 2：逐层精度对比**

使用 `dump_config` 机制逐层 dump 中间结果：

```json
// additional_config 中的 dump 配置
{
    "dump_config": {
        "dump_path": "/path/to/dump",
        "dump_mode": "all",
        "dump_layers": ["o_proj", "gate_up_proj", "down_proj"]
    }
}
```

对比开/关 FlashComm 时每一层的输出，定位精度偏差首次出现的层。

**步骤 3：检查 pad token 影响**

FlashComm 的 pad 机制在 token 不足时会填充虚拟 token，这些 token 虽然最终被裁剪掉，但它们参与中间通信可能会影响其他 token 的数值。验证方式：

- 选择一个 `num_tokens` 能被 `tp_size` 整除的输入（此时 pad_size=0）
- 与 `num_tokens` 不能被整除的输入对比
- 如果两种情况下精度表现不同，则 pad 机制是精度偏差来源

### 2.4 性能劣化

#### 2.4.1 性能预期

FlashComm 的优化在以下条件下收益最大：
- **FlashComm1**：大 batch 场景（`num_tokens > 1000`）。小 batch 时通信切换本身的开销可能超过收益
- **FlashComm2**：P-scenario（prefill 节点）。在 D-scenario 下禁用
- **DSA-CP**：MLA/DSA 模型的 prefill 阶段

如果经验上应该优化的场景反而变慢，按以下步骤排查。

#### 2.4.2 性能排查步骤

**步骤 1：确认 FlashComm 是否真正被激活**

代码 `ascend_forward_context.py:L133-139` 中，FlashComm 的激活有多个条件。检查 forward context 中的标志位：

```python
# 在 Flashcomm2OProjRowParallelOp 的 update_attrs 中添加临时日志
logger.info("FlashComm2 enabled: otp_size=%d, odp_size=%d", self.otp_size, self.odp_size)
```

如果 `otp_size == 1` 且使用全局 TP group 的 all_gather，则 FlashComm2 并未真正生效。

**步骤 2：使用 msprof 分析通信耗时**

```bash
msprof --output=/tmp/flashcomm_profile --application="vllm serve <model> ..."
```

在 profiler timeline 中关注：
- `all_to_all_single` 耗时（FlashComm2 独有）
- `reduce_scatter` 耗时
- `all_gather` 耗时
- 通信和计算是否有有效 overlap

如果通信耗时占比超过 30%，说明通信是瓶颈，需要重新评估是否适合开启 FlashComm。

**步骤 3：检查 FlashComm1 的 `num_tokens > 1000` 阈值**

`ascend_forward_context.py:L134` 中有一个经验阈值：

```python
flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000
```

如果你的场景中 `num_tokens` 在 1000 边界附近波动，FlashComm1 会频繁切换，导致性能不稳定。建议：
- 如果稳定小于 1000，建议关闭 FlashComm1
- 如果实际并发很大但 `num_tokens` 被调度器分片导致偏小，需调整 `max_num_batched_tokens`

---

## 第三层：原理深度解析

> 目标：为研发/二线支持人员提供完整的 FlashComm 内部机制理解，支撑根因分析。

### 3.1 FlashComm 整体架构

FlashComm 是 vllm-ascend 中的**通信-计算融合优化特性集合**，包含三个子特性：

```
FlashComm 家族
├── FlashComm1 (Sequence Parallelism)
│   └── 将标准 TP 中的 all-reduce 替换为更优的通信模式
├── FlashComm2
│   └── 对 o_proj 层进一步拆分 TP，引入 all-to-all 通信
└── DSA-CP (DSA with Context Parallelism)
    └── 在 MLA/DSA 注意力中实现通信-计算 overlap
```

三者的共同特点：
1. **修改了模型 layer 的通信算子和计算逻辑**，替换了标准的 `ColumnParallelLinear` / `RowParallelLinear` 的 forward
2. **在图编译（Graph Capture）阶段录制**，运行时直接 replay，无法插入 Python 层日志
3. **通过 `_EXTRA_CTX`（Extra Forward Context）传递运行时标志位**，控制不同的执行分支

### 3.2 FlashComm 的算子替换机制

#### 3.2.1 CustomOp 注册与替换

FlashComm 通过 `ops/linear_op.py` 中的 `get_parallel_op` 函数在模型初始化阶段将标准 TP 线性层替换为自定义算子：

```python
# linear_op.py:L699-736
def get_parallel_op(disable_tp, prefix, layer, direct):
    # ... 省略共享专家判断 ...

    if direct == "row":
        custom_op = _get_row_parallel_op(prefix, layer)
    if direct == "column":
        custom_op = _get_column_parallel_op(prefix, layer)

    # 返回 (custom_op, tp_rank, tp_size)
```

替换规则（按优先级）：

| 条件 | Column Parallel Op | Row Parallel Op |
|---|---|---|
| `mlp_tp_enable()` + gate_up_proj | `MLPColumnParallelOp` | - |
| `mlp_tp_enable()` + down_proj | - | `MLPRowParallelOp` |
| `oproj_tp_enable()` + o_proj | - | `OProjRowParallelOp` |
| `matmul_allreduce_enable()` | - | `MatmulAllreduceRowParallelOp` |
| **`flashcomm2_enable()` + o_proj** | - | **`Flashcomm2OProjRowParallelOp`** |
| `flashcomm2_oshard_enable()` + qkv | **`Flashcomm2OshardQKVParallelOp`** | - |
| **`enable_sp()` (FlashComm1)** + 各层 | **`SequenceColumnParallelOp`** | **`SequenceRowParallelOp`** |
| `enable_dsa_cp()` + q_b/kv_b | `ShardedCPColumnParallelOp` | `ShardedCPRowParallelOp` |

#### 3.2.2 算子继承体系

```
CustomLinearOp
├── CustomColumnParallelOp
│   ├── MLPColumnParallelOp      (MLP TP 子组的 column parallel)
│   ├── SequenceColumnParallelOp  (FlashComm1 column parallel)
│   └── Flashcomm2OshardQKVParallelOp (FlashComm2 OShard QKV)
├── CustomRowParallelOp
│   ├── MLPRowParallelOp         (MLP TP 子组的 row parallel)
│   ├── OProjRowParallelOp       (OProj TP 子组的 row parallel)
│   ├── Flashcomm2OProjRowParallelOp (FlashComm2 o_proj)
│   ├── MatmulAllreduceRowParallelOp (Matmul+AllReduce 融合)
│   └── SequenceRowParallelOp    (FlashComm1 row parallel)
└── CustomReplicatedOp           (无 TP 的复制层)
```

### 3.3 FlashComm1：Sequence Parallelism 详解

#### 3.3.1 核心思想

标准 TP 的通信模式是：Column Parallel → `all_gather` → Row Parallel → `all_reduce`（或 `reduce_scatter`）。

FlashComm1 的核心改动：
- **Column parallel 层**：使用 `torch.ops.vllm.maybe_all_gather_and_maybe_unpad` 融合了 padding + all_gather + unpadding
- **Row parallel 层**：使用 `torch.ops.vllm.matmul_and_reduce` 或 `npu_mm_reduce_scatter_base` 融合了 matmul + reduce_scatter

#### 3.3.2 激活条件（`ascend_forward_context.py:L124-137`）

```python
if is_context_moe_model:
    flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
elif is_draft_model:
    flash_comm_v1_enabled = False  # draft 模型不启用
else:
    flash_comm_v1_enabled = (
        enable_sp(vllm_config) and
        num_tokens is not None and
        num_tokens > 1000      # 密集模型的经验阈值
    )
```

#### 3.3.3 pad_size 机制

```python
# ascend_forward_context.py:L141-144
if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
    pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
    forward_context.pad_size = pad_size
```

如果 batch 中有 `num_tokens` 个 token，而 `num_tokens` 不能被 `tp_size` 整除，则 pad 到最近的 `tp_size` 倍数。pad token 的值为 0，参与通信但最终被裁剪掉。

在 DP > 1 场景下，pad 计算更为复杂（`ascend_forward_context.py:L167-172`）：
```python
if dp_world_size > 1 and forward_context.dp_metadata is not None:
    max_tokens_across_dp = dp_meta.num_tokens_across_dp_cpu.max().item()
    padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
    pad_size = padded_length - num_tokens
```

这里引入了**跨 DP 维度的 max_tokens 对齐**——如果各 DP rank 的 token 数不同，需要以最大值为基准进行 padding。这可能导致小 batch rank 被大量无效 token 填充，是潜在的性能和精度问题来源。

#### 3.3.4 SequenceRowParallelOp 的 mmrs_fusion 分支

`SequenceRowParallelOp.matmul_and_reduce`（`linear_op.py:L501-582`）包含两条路径：

| mmrs_fusion | 量化方式 | 使用的算子 |
|---|---|---|
| True | Unquantized | `npu_mm_reduce_scatter_base`（HCCL 融合） |
| True | W8A8 | `npu_mm_reduce_scatter_base` + 量化参数 |
| False | 任意 | `quant_method.apply` + `tensor_model_parallel_reduce_scatter` |

`mmrs_fusion` 的开启条件为 `tp_world_size <= 8`（`ascend_forward_context.py:L115`）。

### 3.4 FlashComm2 详解

#### 3.4.1 核心思想

FlashComm2 在 FlashComm1 的基础上，对 `o_proj`（attention 输出投影层）进一步拆分 TP 维度：

```
标准 TP (size=8):  o_proj 权重沿列切分为 8 份
FlashComm2:        o_proj 进一步切分
  ├── OTP 域 (flashcomm2_otp_size=2): 用于 reduce_scatter
  └── ODP 域 (size=4): 用于 all-to-all 重排
```

通信拓扑（以 tp_size=8, flashcomm2_otp_size=2 为例）：

```python
# parallel_state.py:L152-189
# 全局 TP: ranks [0, 1, 2, 3, 4, 5, 6, 7]

# OTP Groups (用于 reduce_scatter):
#   Group 0: [0, 4]
#   Group 1: [1, 5]
#   Group 2: [2, 6]
#   Group 3: [3, 7]

# ODP Groups (用于 all-to-all):
#   Group 0: [0, 1, 2, 3]
#   Group 1: [4, 5, 6, 7]
```

这种嵌套通信域的"交错"组织方式，目的是让 all-to-all 的数据在 ODP 域内完整，同时 reduce_scatter 在 OTP 域内高效完成。

#### 3.4.2 前向流程

`Flashcomm2OProjRowParallelOp.apply_impl`（`linear_op.py:L299-376`）的执行流程：

```
input (batch_size, head_num * head_dim / tp_size)
    │
    ▼
1. get_input_parallel (TP split)
    │
    ▼
2. Padding (pad_size > 0 时补零)
    │
    ▼
3. 重排 batch 索引 (group_indices)
    │  使 batch_id 与 ODP rank_id 对应
    ▼
4. all_to_all_single (ODP 域内)
    │  重分布数据使其在 ODP 域内对齐
    ▼
5. quant_method.apply (matmul)
    │
    ▼
6. reduce_scatter (OTP 域内, 如果 otp_size > 1)
    │
    ▼
7. all_gather (全局 TP 域, 如果非 flashcomm1 模式)
    │
    ▼
8. 裁剪 padding
    │
    ▼
output
```

#### 3.4.3 关键参数

| 参数 | 来源 | 含义 |
|---|---|---|
| `flashcomm2_oproj_tensor_parallel_size` | `enable_flashcomm2_parallel_size` | OTP 子组的 TP size |
| `num_oproj_tensor_parallel_groups` | `global_tp_size // flashcomm2_otp_size` | OTP 子组的数量 |
| `reorgnized_batch_ids` | `get_flashcomm2_reorgnized_batch_ids()` | batch 重排的索引映射 |

`reorgnized_batch_ids` 的生成逻辑（`utils.py:L1236-1252`）：
```python
def get_flashcomm2_reorgnized_batch_ids(global_tp_size):
    flashcomm2_otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size
    num_oproj_tensor_parallel_groups = global_tp_size // flashcomm2_otp_size

    reorgnized_batch_ids = []
    for i in range(num_oproj_tensor_parallel_groups):
        ranks = []
        for j in range(flashcomm2_otp_size):
            rank_idx = i + j * num_oproj_tensor_parallel_groups
            ranks.append(rank_idx)
        reorgnized_batch_ids.append(ranks)
    return reorgnized_batch_ids
```

例如 `global_tp_size=8, flashcomm2_otp_size=2`：
```
reorgnized_batch_ids = [[0, 4], [1, 5], [2, 7], [3, 7]]
```

### 3.5 FlashComm2 OShard 详解

#### 3.5.1 核心思想

OShard（O-proj Sharding）是 FlashComm2 的一个附加优化，对 QKV 层的 column parallel 进行权重 sharding + 异步广播（`Flashcomm2OshardManager`, `flashcomm2_oshard_manager.py`）：

1. **注册阶段**：`register_layer` 将 attention 的 QKV 层注册到 shard 系列
2. **前向阶段**：`trigger_broadcast_for_layer` 在 QKV matmul 前触发异步广播，使通信与计算 overlap
3. **加载后**：`post_process_after_loading` 批量处理权重

#### 3.5.2 激活条件

```python
# flashcomm2_oshard_manager.py:L35-36
def flashcomm2_oshard_enable(self):
    return flashcomm2_enable() and o_shard_enable()
```

其中 `o_shard_enable()` 检查 `layer_sharding` 是否包含 `"o_proj"`：
```python
# utils.py:L1179-1183
def o_shard_enable():
    layer_sharding = get_ascend_config().layer_sharding
    if layer_sharding is None:
        return False
    return "o_proj" in layer_sharding
```

#### 3.5.3 异步广播的风险

`trigger_broadcast_for_layer` 触发了异步 broadcast，但没有显式的同步等待点。如果 broadcast 未完成就使用权重数据做 matmul，会导致数值错误或挂死。`LayerShardWeightSeries` 的同步点可能在：
1. `reach_layer_for_shard_weight_series` 内部
2. 权重预取流（prefetch_stream）与计算流之间的 stream 同步

排查此类问题时需要重点关注 stream 同步是否正确。

### 3.6 DSA-CP 详解

#### 3.6.1 核心思想

DSA-CP（DSA with Context Parallelism）用于 MLA/DSA 注意力模型的 prefill 场景，将 KV 计算沿 TP 维度切分，使用 `all_gather_async` 实现通信和计算 overlap。

#### 3.6.2 通信模式

在 `sfa_v1.py` 的 `forward` 方法中（L1221-1537），DSA-CP 路径的关键通信：

```python
# sfa_v1.py:L1318-1382
if self.enable_dsa_cp:
    async_op = self.enable_dsa_cp_with_layer_shard or full_gather_o_proj_enabled
    # 异步 all_gather kv_no_split（包含 k_pe, k_nope, k_li）
    fused_kv_no_split, kv_ag_handle = all_gather_async(
        torch.cat([k_pe, k_nope, k_li], dim=1),
        get_tp_group(),
        async_op=async_op,
    )
    # ... 中间计算与 all_gather 并行 ...
    if kv_ag_handle is not None:
        kv_ag_handle.wait()  # 在使用前等待通信完成

    # 如果 full_gather_o_proj_enabled，还会异步 all_gather o_proj 权重
    _, o_proj_full_handle = all_gather_async(
        self.o_proj_tp_weight_gather_input,
        get_tp_group(),
        output=self.o_proj_full_gather_pool,
    )
```

关键的同步点：
1. `kv_ag_handle.wait()` — 在使用 gathered kv 之前
2. `o_proj_full_handle.wait()` — 在使用 gathered o_proj 之前

如果这些 wait 缺失或提前返回，会导致数据竞争。

### 3.7 Graph 模式对 FlashComm 的影响

#### 3.7.1 为什么 Graph 内无法加日志

FlashComm 的通信操作在 Graph Capture 阶段通过 `torch.npu.graph_task_group_begin/end` 录制为图任务：

```python
# attention_v1.py:L802-827
torch.npu.graph_task_group_begin(stream)
torch_npu.npu_fused_infer_attention_score.out(...)
handle = torch.npu.graph_task_group_end(stream)
```

编译后的图在 replay 时不经过 Python 解释器，所有操作直接由 ACL runtime 调度执行。因此：
- `print()` 语句在 graph capture 时执行一次（warmup），之后不再执行
- `logger.info()` 同理
- 唯一的例外是 `device_print`（`utils.py:L161-206`），它通过 `torch.fx.node.has_side_effect` 标记防止被图优化消除

#### 3.7.2 `device_print` 的使用限制

`device_print` 注册为 side-effectful 操作（`utils.py:L130-134`），可以在 graph 内使用。但存在严重限制：

1. **保留所有 payload**：每次调用产生的回调数据不会被 GC，长时间运行会导致 NPU 内存泄漏
2. **性能开销**：每次调用都会从 device 拷贝数据到 host
3. **仅用于调试**：不适合生产环境

#### 3.7.3 Graph 兼容性约束

`ASCEND_LAUNCH_BLOCKING=1` 会强制所有 operator 同步执行，与 graph_task 不兼容（`platform.py:L636-645`）：
```python
if (
    compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    and os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1"
):
    raise ValueError(
        "ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. "
        "..."
    )
```

---

## 第四层：已知问题与案例

> 目标：持续积累实际问题的定位过程和解决方案，形成可复用的案例库。

### 4.1 FlashComm 配置冲突案例

**症状**：服务启动失败，报错 `flashcomm2_oproj_tensor_parallel_size cannot exceed global tensor parallel size`

**原因**：`enable_flashcomm2_parallel_size` 设置的值大于 `tensor_parallel_size`

**解决**：确保 `flashcomm2_oproj_tensor_parallel_size < tensor_parallel_size` 且 `tensor_parallel_size % flashcomm2_oproj_tensor_parallel_size == 0`

### 4.2 DP 场景下 pad 不一致导致挂死

**场景**：FlashComm2 在 `data_parallel_size > 1` 时各 DP rank 的 `num_tokens` 不同

**现象**：某些 rank 挂死在 `all_to_all_single` 通信中

**原因**：`ascend_forward_context.py:L167-172` 中以 `max_tokens_across_dp` 为基准计算 padding，但如果在某些边界条件下 pad 计算出现偏差，各 rank 的 send/recv 尺寸不匹配，all-to-all 会死锁

**排查手段**：HCCL debug 日志、msprof timeline

### 4.3 OShard 异步广播未完成

**场景**：开启了 `layer_sharding: ["o_proj"]`，使用 FlashComm2 OShard

**现象**：QKV 层的 matmul 结果偶发性异常或随机值

**原因**：`trigger_broadcast_for_layer` 触发的异步 broadcast 还没有完成，QKV matmul 就已经开始读取权重

**排查**：检查 `prefetch_step` 参数是否合理配置，确认 stream 同步逻辑正确

### 4.4 （待补充）

> 更多案例将随着实际问题排查的积累持续更新。

---

## 附录

### A. 关键代码文件索引

| 文件 | 内容 |
|---|---|
| `vllm_ascend/ascend_forward_context.py` | FlashComm 标志位和 pad_size 计算 |
| `vllm_ascend/ops/linear_op.py` | 所有 FlashComm CustomOp 实现 |
| `vllm_ascend/ops/flashcomm2_oshard_manager.py` | FlashComm2 OShard 管理器 |
| `vllm_ascend/distributed/parallel_state.py` | FlashComm2 OTP/ODP 通信域初始化 |
| `vllm_ascend/attention/sfa_v1.py` | DSA-CP 通信逻辑 |
| `vllm_ascend/attention/attention_v1.py` | 基础 attention 的 graph capture 逻辑 |
| `vllm_ascend/ascend_config.py` | FlashComm 配置解析和合法性校验 |
| `vllm_ascend/utils.py` | FlashComm 开关函数和辅助逻辑 |
| `vllm_ascend/envs.py` | 环境变量定义 |
| `vllm_ascend/platform.py` | 平台级配置检查和约束 |

### B. 关键环境变量速查

| 环境变量 | 用途 | 默认值 |
|---|---|---|
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 开启 FlashComm1 (SP) | 0 |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | FlashComm2 OTP size | 0（关闭） |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | Matmul-AllReduce 融合 | 0 |
| `ASCEND_LAUNCH_BLOCKING` | 同步执行模式（与 Graph 互斥） | 0 |
| `HCCL_DEBUG` | HCCL 调试日志级别 | - |
| `MSMONITOR_USE_DAEMON` | msMonitor 守护进程 | 0 |
| `PYTORCH_NPU_ALLOC_CONF` | NPU 内存分配配置 | `expandable_segments:True` |

### C. 关键 additional_config 配置项速查

| 配置项 | 用途 | 默认值 |
|---|---|---|
| `enable_flashcomm1` | 开启 FlashComm1 | False |
| `enable_flashcomm2_parallel_size` | FlashComm2 OTP size | 0 |
| `enable_matmul_allreduce` | Matmul-AllReduce 融合 | False |
| `layer_sharding` | 层权重 sharding 配置 | None |
| `finegrained_tp_config.oproj_tensor_parallel_size` | Finegrained OProj TP | 0 |
| `enable_dsa_cp` | 开启 DSA-CP | False |
| `dump_config` / `dump_config_path` | 精度 dump 配置 | None |
| `ascend_log_path` | Ascend 日志路径 | `~/ascend/log/vllm_ascend` |
