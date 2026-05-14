# FlashComm v1 整改需求分析文档

## 1. 背景与目标

本需求来源于 `flashcomm需求描述.txt` / `需求+进展+问题+规划.txt` 中关于 vLLM Ascend FlashComm v1 整改的讨论。

当前 `vllm-ascend` 中 FlashComm v1 与多个模型、特性和 runtime 分支耦合较深，包括：

- dense 模型中的 TP all-reduce 替换；
- MoE 模型中的 routed expert / shared expert 通信；
- `shared_expert_dp`；
- `multistream_moe` / 多流 MoE；
- 多模态模型第一层；
- 量化场景；
- FlashComm 自定义算子中的 pad / unpad / slice；
- runtime FlashComm1 与基于 FX pass 的 sequence parallel 双轨并存。

最终整改目标可以概括为：

1. 通过 FX graph pass 机制实现算子 pattern 替换，逐步替代当前大量通过自定义算子和 layer if/else 分支显式改 layer 代码的实现方式。
2. 与主社区 vLLM 的 `sequence_parallel` 方案对齐，明确 FlashComm / sequence parallel 的接口、命名和适用范围。
3. 解决或拆解 `shared_expert_dp`、`multistream_moe`、多模态、量化等特性叠加时与 FlashComm 的严重耦合问题。
4. 将 FlashComm 内部 pad / unpad / slice 逻辑上移到 attention metadata / runner 等图外阶段，避免与 ACL graph capture 冲突。

## 2. 当前主线理解

FlashComm v1 的基础语义是把 TP 域内的：

```text
all_reduce -> rmsnorm/layernorm
```

替换为：

```text
reduce_scatter -> rmsnorm/layernorm -> all_gather
```

这样可以让 norm 在 token shard 上执行，降低 norm 计算负载，并为后续通信和计算融合提供空间。

分析所有复杂特性时，建议始终围绕一个核心问题展开：

```text
当前 hidden_states 在 TP/DP/EP 域内处于什么分布状态？

- full token / replicated
- token shard
- hidden shard
- padded
- unpadded
- quantized
- shared expert / routed expert 是否已完成聚合
```

后续所有耦合问题本质上都来自“张量分布状态”与“通信点替换规则”没有被集中表达，而是散落在 layer、自定义算子、MoE prepare/finalize、forward context、patch 代码中。

## 3. 现有实现分层

当前代码中实际存在三套容易混淆的路径。

| 层级 | 含义 | 典型入口 |
|---|---|---|
| Runtime FlashComm1 | 通过 `enable_sp()`、`_EXTRA_CTX.flash_comm_v1_enabled`、`maybe_*` 自定义算子控制通信行为 | `VLLM_ASCEND_ENABLE_FLASHCOMM1`、`VLLM_ASCEND_ENABLE_FLASHCOMM` |
| Graph pass sequence parallel | 通过 `pass_config.enable_sp` 注册 `SequenceParallelismPass` / `SequenceParallelismMoePass` | `compilation_config.pass_config.enable_sp` |
| FlashComm dense POC pass | 本次新增的独立 POC，用于证明基础 dense pattern 可由 pass 表达 | `ascend_compilation_config.enable_flashcomm_pass_poc` |

关键文件：

- `vllm-ascend/vllm_ascend/ascend_forward_context.py`
- `vllm-ascend/vllm_ascend/utils.py`
- `vllm-ascend/vllm_ascend/ops/register_custom_ops.py`
- `vllm-ascend/vllm_ascend/ops/linear_op.py`
- `vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py`
- `vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism.py`
- `vllm-ascend/vllm_ascend/compilation/passes/sequence_parallelism_moe.py`
- `vllm-ascend/vllm_ascend/compilation/passes/flashcomm_pass.py`

## 4. shared_expert_dp 调研结论

### 4.1 开关入口

`enable_shared_expert_dp` 来自 `additional_config`，但不是只看用户配置。

真正生效条件在 `ascend_config.py` 中：

- `additional_config["enable_shared_expert_dp"] == true`
- `parallel_config.enable_expert_parallel == true`
- `parallel_config.tensor_parallel_size > 1`

当 `enable_shared_expert_dp` 真正为 True 时，会调用：

```python
enable_sp(vllm_config=vllm_config, enable_shared_expert_dp=True)
```

如果用户没有通过环境变量打开 FlashComm1，这里也会把内部 `_ENABLE_SP` 置为 True。

所以“shared_expert_dp 开启会默认开启 FlashComm”更精确的表述是：

```text
shared_expert_dp 开启会强制内部 enable_sp() 为 True，
但不会修改用户环境变量。
```

### 4.2 核心耦合点

`utils.py` 中的 helper：

```python
def shared_expert_dp_enabled() -> bool:
    return get_ascend_config().enable_shared_expert_dp or enable_sp() or enable_sp_by_pass()
```

这是当前 shared expert 与 FlashComm/SP 耦合不清的核心。

它导致：

- 用户没有显式开启 `enable_shared_expert_dp`，但只要开启 FlashComm/SP，`shared_expert_dp_enabled()` 也会返回 True。
- shared expert 是否 TP all-reduce、是否禁用 TP op、是否走某些 SED-like 行为，都会被 FlashComm/SP 间接改变。

### 4.3 运行时影响

主要影响点：

1. `linear_op.py`
   - shared expert 前缀命中 `shared_expert_dp_enabled()` 时，不挂 TP 自定义 op；
   - 等价 shared expert 更偏每卡复制，显存占用上升。

2. `fused_moe.py`
   - ALLTOALL / MC2 / FUSED_MC2 通信类型下，只有 `not shared_expert_dp_enabled()` 时才对 `shared_out` 做 `tensor_model_parallel_all_reduce`。

3. `prepare_finalize.py`
   - `enable_shared_expert_dp` 或 `replace_allreduce` 会影响 MoE 输入是否做 TP 维 pad/slice，以及 finalize 是否 all-gather 回完整 token。

### 4.4 显存劣化来源

`shared_expert_dp` 的显存劣化主要来自：

- shared expert 权重更偏复制，不再按 TP 切分；
- 跳过 TP token 切分后，单卡中间激活可能更大；
- 多流、AllGather、FC3 context 等缓冲可能增加峰值。

因此 FlashComm/SP 不应该默认打开 `enable_shared_expert_dp` 配置项，这一点是合理的。但当前 helper 会让 FlashComm/SP 间接影响 shared expert 行为，需要进一步解耦。

## 5. multistream_moe 调研结论

### 5.1 开关现状

当前生产逻辑中没有发现真正读取 `enable_multistream_moe` 的主路径。

实际生效的两个开关是：

- `multistream_overlap_shared_expert`
- `multistream_overlap_gate`

二者来自 `additional_config`，定义在 `ascend_config.py`。

### 5.2 multistream_overlap_shared_expert

`AscendSharedFusedMoE._forward_shared_experts` 中使用独立 shared expert stream。

执行方式：

- shared expert 的 gate_up / activation / down_proj 分段执行；
- 分别等待 routed experts 的 dispatch / combine event；
- 让 shared expert 计算与 routed experts 通信重叠；
- 最后 default stream wait shared expert stream。

### 5.3 multistream_overlap_gate

`multistream_overlap_gate` 使用 `FlashCommon3Context` 和 gate stream。

在 gate stream 上提前执行：

- `shared_experts(hidden_states)`
- `select_experts`
- 必要时 all-gather `topk_weights` / `topk_ids`

随后通过 `FlashCommon3Context` 把 `shared_out`、`topk_weights`、`topk_ids` 传回主流。

### 5.4 与 shared_expert_dp / FlashComm 的耦合

多流 MoE 本质是调度优化，但它产生的 `shared_out` 是否需要 TP all-reduce 仍然由 `shared_expert_dp_enabled()` 判断。

因此：

- 多流本身没有硬依赖 `enable_shared_expert_dp`；
- 但在 TP>1 + EP + shared expert 场景下，如果 SED 关闭但 FlashComm/SP 开启，`shared_expert_dp_enabled()` 仍会为 True；
- 这会让“SED 关、多流开”的语义不再纯粹。

### 5.5 当前组合判断

| SED | FlashComm/SP | multistream | 当前判断 |
|---|---|---|---|
| 开 | 开 | 开 | 可开，常见推荐组合，但显存和通信形态都变化 |
| 开 | 关 | 开 | 实际会被 SED 拉起 `enable_sp()`，不能视为纯关 FlashComm |
| 关 | 开 | 开 | 配置上可开，但 `shared_expert_dp_enabled()` 为 True，行为不是纯 SED off |
| 关 | 关 | 开 | 代码无硬禁止，但 TP>1 + EP + shared expert 下需要按模型实测 |

结论：多流 MoE 不应被当作一个纯 pass 问题。stream/event/context 调度更适合保留在 runtime；pass 只处理其中稳定的算子子图。

## 6. 多模态模型调研结论

### 6.1 核心问题

dense LLM 路径中，embedding 后存在可以被 FlashComm/SP 替换的通信点，使第一层 RMSNorm 前的 hidden_states 可以进入 token shard 状态。

多模态模型不同：

- 文本 embedding 和视觉 embedding 的路径不同；
- multimodal embeddings 通过 placeholder 写回 `inputs_embeds`；
- 视觉侧 merge 后的第一层输入没有完全复用 dense LLM 的 embedding 后通信结构；
- 第一层 RMSNorm 前没有稳定的 `all_reduce + norm` 子图可供 pass 匹配。

因此需求文档中提到的：

```text
多模态第一层缺少 embedding 后 allreduce
```

本质是第一层输入 layout 与 dense LLM 不同，不只是“少了一个算子”。

### 6.2 当前特判位置

关键文件：

- `ops/linear_op.py`
  - VL 第一层 attention 列并行时跳过普通 `maybe_all_gather`；

- `ops/mla.py`
  - `is_vl_first_layer` + FlashComm 下调整输出 buffer 和 `need_gather_q_kv`；

- `patch/worker/patch_qwen3vl.py`
  - FlashComm 下对 `deepstack_input_embeds` 做 `chunk(tp_size)[tp_rank]`；

- `patch/worker/patch_qwen3_5.py`
  - Qwen3.5 第 0 层 + FlashComm 下调整 self attention 输出 shape；

- `patch/worker/patch_multimodal_merge.py`
  - 覆盖多模态 embedding merge；

- `spec_decode/eagle_proposer.py`
  - Eagle / draft / multimodal 下对 `maybe_pad_and_reduce` 有额外分支；

- `ops/register_custom_ops.py`
  - draft + VL + 非 EP 时 `_maybe_pad_and_reduce_impl` 退回 all-reduce。

### 6.3 对 pass 化的影响

多模态第一层不适合直接复用 dense FlashComm pass，因为 pattern 不同。

可行策略：

1. 短期：
   - 明确跳过 VL 第一层 dense FlashComm pattern；
   - 保留当前 runtime 特判，避免错误匹配。

2. 中期：
   - 将文本 embedding、vision merge、placeholder 写回后的 layout 用 metadata 显式描述；
   - 统一进入第一层前的 token 分布状态。

3. 长期：
   - 如果数学上允许，构造与 dense 同构的第一层通信子图；
   - 否则新增 `FirstLayerVLPattern`，仅对白名单模型族启用。

多模态中间层 deepstack 已有 pass 思路：`sequence_parallelism.py` 中的 `Qwen3VLMiddleAllReduceRMSNormPattern` 会对 deepstack 输入做 chunk，可作为中间层模板。

## 7. 量化场景调研结论

### 7.1 Dense 量化

`linear_op.py` 中 `SequenceRowParallelOp.matmul_and_reduce`：

- 无量化时可走 `npu_mm_reduce_scatter_base`；
- W8A8 时先 quantize，再走 `npu_mm_reduce_scatter_base`；
- 否则普通 `quant_method.apply` 后再 `tensor_model_parallel_reduce_scatter`。

这说明 dense row 路径已经存在“量化后再通信”的优化思路。

### 7.2 norm_quant_fusion_pass

`norm_quant_fusion_pass.py` 中 SP 相关 pattern 会把：

```text
norm -> maybe_all_gather -> quantize
```

替换为：

```text
norm_quant -> maybe_all_gather
```

这等价于把 all-gather 放到量化之后，降低通信量。

这是需求中提到的：

```text
量化的话是要把 allgather 插在 quant 后面
```

的一个已有实现方向。

### 7.3 MoE 量化

`prepare_finalize.py` 中 AllGather+EP 路径：

- W8A8：在 EP all-gather 前做 `npu_dynamic_quant(hidden_states)`；
- MXFP8：当前有 TODO，不在 prepare 阶段预量化，而是保留在 MoE MLP 路径。

因此 W8A8 和 MXFP8 策略不一致，需要 SE 决策：

- 是否让 MXFP8 对齐 W8A8 的“预量化再 all-gather”；
- 还是明确 MXFP8 不支持该优化，并在文档和测试矩阵中列出限制。

### 7.4 multistream + quant

`w8a8_dynamic.py` 中，`multistream_overlap_gate` 开启时会从 `FlashCommon3Context` 读取 gate stream 上提前算好的 `topk_weights` / `topk_ids`。

这说明多流 + 量化的耦合包含 stream/context 依赖，不适合全部用 FX pass 表达。

## 8. FlashComm Dense Pass POC

本轮已实现一个 POC：

- 文件：`vllm-ascend/vllm_ascend/compilation/passes/flashcomm_pass.py`
- 类名：`FlashCommDensePass`
- 接入点：`vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py`
- 开关：`ascend_compilation_config.enable_flashcomm_pass_poc`

POC 覆盖：

```text
all_reduce + npu_add_rms_norm_bias
->
reduce_scatter + npu_add_rms_norm_bias + all_gather
```

POC 不覆盖：

- `shared_expert_dp`
- `multistream_moe`
- MoE all-gather epilogue
- 多模态第一层
- 量化专用 pattern

该 POC 通过独立开关接入，并在 `pass_config.enable_sp` 已开启时不注册，避免和现有 `SequenceParallelismPass` 重复注册同类 pattern。

已创建 commit：

```text
daf01dd7 [Misc] Add FlashComm dense pass POC
```

该 POC 适合作为与 SE 讨论的最小样例：证明基础 dense FlashComm 替换可以由 FX pass 表达，但不作为最终产品接口。

## 9. 已输出材料

当前已生成以下材料：

- `flashcomm_decomposition_materials/shared_expert_dp_flashcomm_coupling.pptx`
- `flashcomm_decomposition_materials/multistream_moe_flashcomm_coupling.pptx`
- `flashcomm_decomposition_materials/flashcomm_pass_poc_solution.pptx`
- `flashcomm_decomposition_materials/flashcomm_full_rectification_plan.pptx`
- `flashcomm_decomposition_materials/flashcomm_pass_poc_notes.md`
- `flashcomm_decomposition_materials/flashcomm_requirement_analysis.md`

生成脚本：

- `flashcomm_decomposition_materials/generate_flashcomm_ppts.py`
- `flashcomm_decomposition_materials/generate_flashcomm_pass_poc_ppt.py`
- `flashcomm_decomposition_materials/generate_flashcomm_full_rectification_ppt.py`

注意：PPT 生成时曾遇到 PowerShell 编码导致中文变成 `?` 的问题，已改为 UTF-8 `.py` 脚本生成，并验证 `question_marks=0`。

## 10. 完整改造建议

### 10.1 总体设计原则

不建议尝试用“一把大 pass”覆盖所有 FlashComm 场景。

更合理的架构是三条线并行：

1. **FX pass 层**
   - 处理稳定、可匹配的算子 pattern；
   - 如 dense `AR + Norm`、`Norm + Quant + AG`、MoE AG epilogue。

2. **metadata / runner 层**
   - 统一 padding、actual token length、first-layer layout；
   - 禁止图内根据 host int 做动态 slice。

3. **runtime 层**
   - 保留 stream/event、MoE dispatch/combine、EP/DP/TP 通信组、FC3 context；
   - 只负责调度和通信拓扑，不再承担稳定 pattern 替换。

### 10.2 分阶段方案

#### 阶段 0：统一术语和开关

目标：

- 明确 runtime FlashComm1、pass-based sequence parallel、FlashComm POC 三者区别；
- 讨论外部接口是否完全对齐主社区 `sequence_parallel`；
- 环境变量逐步迁移到 `additional_config` / `pass_config`。

建议：

- `FlashComm` 作为 Ascend 内部实现名逐步弱化；
- 外部配置尽可能对齐 `sequence_parallel`；
- `enable_flashcomm_pass_poc` 仅作为穿刺开关，后续应删除或合并。

#### 阶段 1：Dense 基线 pass 化

目标：

- 将 dense 基础 FlashComm 替换正式纳入 pass 体系；
- 用 POC 验证方向；
- 最终合并到 `SequenceParallelismPass` 命名体系。

动作：

- 补 dense TP>1 smoke test；
- 验证 matched_count 和输出一致性；
- 比较 pass 路径与 runtime FlashComm1 路径性能；
- 逐步日落 `linear_op.py` 中可由 pass 表达的 FC1 分支。

#### 阶段 2：padding 上移

目标：

- 删除自定义算子内部 pad / unpad / slice；
- 让图内 shape 更稳定；
- 解决 ACL graph capture 后 slice 长度固化问题。

涉及：

- `_maybe_all_gather_and_maybe_unpad_impl`
- `_maybe_pad_and_reduce_impl`
- `_maybe_chunk_residual_impl`
- `fc3_all_gather_and_maybe_unpad_impl`

建议：

- 在 attention metadata / runner 构造阶段统一 padding；
- 模型图内只处理已经规范化的 tensor；
- pass pattern 不再承担动态 unpad。

#### 阶段 3：shared_expert_dp 解耦

目标：

- 拆分 `shared_expert_dp_enabled()` 的混合语义；
- 区分用户显式 SED、runtime SP、pass SP。

建议拆成：

```text
explicit_shared_expert_dp_enabled
runtime_sequence_parallel_enabled
pass_sequence_parallel_enabled
shared_expert_reduce_strategy
```

重点：

- 保留正确性耦合；
- 性能策略显式配置；
- 历史胶水逐步日落。

#### 阶段 4：multistream_moe 保留 runtime 调度边界

目标：

- 明确多流是 runtime 调度优化，不是 pass 的主要职责；
- pass 只处理多流中稳定出现的算子片段。

建议：

- 建立多流依赖图：
  - gate stream
  - shared expert stream
  - quant stream
  - default stream
  - FlashCommon3Context
- 建立组合矩阵：
  - SED on/off
  - SP on/off
  - multistream gate/shared
  - MoECommType
  - quant type

#### 阶段 5：MoE pass 化策略

建议划分：

| MoE 子问题 | 建议机制 |
|---|---|
| routed expert dispatch/combine | runtime 保留 |
| shared expert 输出聚合 | runtime 保留并解耦 helper |
| MoE all-gather epilogue | pass + runtime 协同 |
| W8A8 prepare 预量化 | runtime 先保留，补测试 |
| MXFP8 prepare TODO | SE 决策 |

#### 阶段 6：多模态模型整改

目标：

- 不再把多模态第一层硬套 dense pattern；
- 先用 metadata 固化第一层输入 layout；
- 再决定是否新增专用 pass。

建议：

- 短期跳过 VL 第一层 dense FlashComm pass；
- 中期统一 vision/text merge 后的 layout 描述；
- 长期新增 `FirstLayerVLPattern` 或构造同构通信子图。

#### 阶段 7：量化场景整改

目标：

- 明确 quant 与 all-gather / reduce-scatter 的相对位置；
- 将稳定 pattern 放入 pass；
- 将 dtype、scale、pertoken_scale 等 runtime 依赖保留在合适层。

建议：

- dense norm+quant：纳入正式 pass 主线；
- row W8A8 + RS：先保留 runtime，后续考虑 GEMM+RS fusion；
- MoE W8A8 prepare：保留并补测试；
- MXFP8：需要 SE 决策；
- multistream gate + quant：保留 runtime context。

## 11. 推荐 pass 库分层

建议最终形成以下 pass 体系：

| Pass 层 | 负责 pattern | 备注 |
|---|---|---|
| DenseSequenceParallelPass | `AR + Norm -> RS + Norm + AG` | 对齐主社区 sequence_parallel |
| NormQuantSPPass | `Norm + AG + Quant -> NormQuant + AG` | 确保量化后通信 |
| MoeSequenceParallelPass | MoE all-gather epilogue / allgather+chunk no-op | 只处理稳定 epilogue |
| VLMiddleLayerPass | deepstack add + norm 中间层 pattern | 不处理第一层 |
| FirstLayerVLPass | 可选，模型族白名单 | 第一层 layout 固化后再做 |

## 12. 日落清单

建议按优先级逐步日落：

| 优先级 | 日落对象 | 前置条件 |
|---|---|---|
| P0 | 模型图内 pad/unpad/slice | metadata / runner 已统一 padding |
| P1 | dense linear 中可由 AR+Norm pass 表达的 FC1 分支 | dense pass e2e 和性能通过 |
| P1 | env 与 pass 双轨同步胶水 | 配置入口统一 |
| P2 | 多模态中间层 deepstack runtime chunk 特判 | VL middle pass 稳定覆盖 |
| P2 | norm_quant 中重复 AG/quant 分支 | NormQuantSPPass 覆盖主量化类型 |
| P3 | MoE prepare/finalize 中可图化的 epilogue | MoE pass 与通信矩阵验证完成 |

## 13. 测试矩阵建议

需要覆盖以下维度：

| 维度 | 建议覆盖 |
|---|---|
| 模型类型 | dense LLM、MoE、shared expert MoE、VL、VL+MoE 如有 |
| 并行方式 | TP=1/2/4、EP on/off、DP on/off、prefill/decode |
| 开关组合 | sequence_parallel/pass、runtime FC1、SED、multistream gate/shared |
| 通信类型 | ALLGATHER、ALLTOALL、MC2、FUSED_MC2 |
| 量化 | BF16、W8A8、MXFP8、动态量化 |
| 图模式 | eager、compile、ACL graph capture |
| 正确性 | 输出一致性、shape/layout invariants、matched_count、无图内动态 slice |
| 性能 | prefill 大 batch、decode 小 batch、通信量、显存峰值 |

## 14. 需要 SE 决策的问题

1. 外部接口是否完全对齐主社区 `sequence_parallel`？
2. FlashComm 是否只作为 Ascend 内部实现名，不再作为用户主开关？
3. `shared_expert_dp` 与 FlashComm/SP 的绑定是正确性要求，还是性能策略？
4. 如果只是性能策略，是否允许用户显式选择 SED 与 SP 的组合？
5. `shared_expert_dp_enabled()` 是否需要拆分，避免包含 `enable_sp()`？
6. `multistream_moe` 在 SED 关闭、SP 关闭时是否允许单开？
7. 多流单开需要哪些模型白名单、通信类型白名单和量化类型白名单？
8. 多模态第一层应构造同构通信子图，还是长期使用专用 FirstLayerVL pattern？
9. MXFP8 是否要对齐 W8A8 的“量化后 all-gather”策略？
10. padding 上移由哪个需求统一承接？完成前哪些 pass 只能作为实验能力？

## 15. 当前遗留问题

1. FlashComm POC 目前只提交了基础 pass 代码，未提交单测文件。
   - 当前工作区中 `tests/ut/compilation/test_flashcomm_pass.py` 是未跟踪文件。

2. 当前仓库还有未提交的文档改动。
   - `docs/source/user_guide/configuration/additional_config.md`
   - `docs/source/user_guide/feature_guide/sequence_parallelism.md`

3. POC 未做真实 NPU 环境 e2e。
   - 已做静态 py_compile；
   - 已做 linter 检查；
   - 仍需要 dense TP>1 compile/smoke 验证。

4. 多模态第一层方案仍需 SE 决策。
   - 当前建议是短期跳过，长期根据 layout 方案决定是否新增 pass。

5. MXFP8 与 W8A8 的量化前置策略未统一。

6. padding 上移是所有后续 pass 稳定性的前置条件，但目前尚未落代码。

## 16. 结论

本需求不适合直接用一个大 pass 一次性替换所有 FlashComm v1 逻辑。

推荐路线是：

```text
Dense 基线 pass 化
-> padding 上移
-> 量化稳定 pattern pass 化
-> shared_expert_dp 语义解耦
-> MoE epilogue pass 化
-> 多模态第一层 layout 专项
-> multistream 保留 runtime 调度边界
```

其中：

- FX pass 负责稳定算子 pattern；
- metadata / runner 负责 padding 和 layout；
- runtime 负责 stream、通信组、MoE 调度和上下文。

短期应拿 `FlashCommDensePass` POC 作为 SE 讨论样例，确认 pass 化方向；中长期应收敛到主社区 `sequence_parallel` 命名和 pass 体系，逐步日落 runtime FlashComm1 中可由图替换表达的分支。
