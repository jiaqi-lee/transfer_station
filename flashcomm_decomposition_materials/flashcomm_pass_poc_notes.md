# FlashComm Dense Pass POC Notes

## POC Scope

- Added a new pass named `FlashCommDensePass`.
- The POC only covers the dense baseline pattern:
  `all_reduce + npu_add_rms_norm_bias -> reduce_scatter + npu_add_rms_norm_bias + all_gather`.
- The POC intentionally excludes:
  - `shared_expert_dp`
  - `multistream_moe`
  - MoE all-gather epilogue movement
  - multimodal first-layer special cases
  - quantization-specific all-gather placement

## Entry Point

- New pass file:
  `vllm-ascend/vllm_ascend/compilation/passes/flashcomm_pass.py`
- Registration point:
  `vllm-ascend/vllm_ascend/compilation/graph_fusion_pass_manager.py`

Enable it through:

```json
{
  "ascend_compilation_config": {
    "enable_flashcomm_pass_poc": true
  }
}
```

The POC is not registered when `compilation_config.pass_config.enable_sp` is already true, so it does not double-register patterns with the existing `SequenceParallelismPass`.

## Validation Performed

- Static Python compilation passed:

```powershell
python -m py_compile "vllm_ascend\compilation\passes\flashcomm_pass.py" "vllm_ascend\compilation\graph_fusion_pass_manager.py"
```

- IDE diagnostics reported no linter errors for the modified files.

## SE Alignment Questions

- Should this POC stay under a `flashcomm` name, or should the final product align with upstream `sequence_parallel` naming?
- If the dense pass direction is accepted, should MoE be handled by additional FX patterns, or should MoE prepare/finalize keep specialized runtime logic?
- Should `shared_expert_dp_enabled()` be split into separate predicates for explicit SED, FlashComm/SP, and pass-based SP?
- Should multistream MoE remain a runtime scheduling optimization, rather than changing shared expert TP/DP semantics?
- How should pad/unpad be removed from FlashComm custom ops and moved to graph-external metadata handling?
