# GEMV Batch INT4 — Bit-Identical M=N Kernel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing GEMV OpenCL kernel (M=1 only) to support M>1 batch input using GEMV-identical per-row reduction, so INT4 DFlash batch-verify (M=N) produces bit-identical outputs to autoregressive M=1 inference.

**Architecture:** The GEMV kernel already computes one output row independently per work group; adding a third grid dimension `m = get_global_id(1)` and offsetting input/output by `m * K` / `m * N` extends it to batch without changing any arithmetic. The C++ kernel selector allows M>1, updates work-group dispatch, and gives the extended kernel higher priority than BF_TILED for compressed INT4 weights, so DFlash batch verify selects GEMV-batch instead of BF_TILED.

**Tech Stack:** OpenCL C (kernel), C++17 (kernel selector), CMake + Ninja (build), pytest (test).

---

## Background and key invariants

- **GEMV kernel** (`fully_connected_gpu_gemv.cl`): three variants selected by `FILTER_LAYOUT_OS_IS_YX_TYPE` (0/1/2):
  - `OSV16` (type 0): global `[N, 1, 16]`, each work group computes 1 output element per batch row.
  - `OSV32_ISV2` (type 1): global `[N/2, 1, 16]`, 2 outputs per work group.
  - `OSV64_ISV2` (type 2): global `[N/4, 1, 16]`, 4 outputs per work group.
- All variants share the same FP32 reduction tree: INT4 → half16 → `half_input * half_weight = float` (auto-promotion) → accumulate in `float8`. This is the tree we must preserve for bit-identity.
- DFlash INT4 batch verify selects **BF_TILED** for M=N because GEMV currently rejects `batch > 1`. BF_TILED uses a different reduction tree → hidden-state drift → argmax flip at token ~210.
- After this fix, GEMV-batch is selected for M>1 compressed INT4. The per-row arithmetic is unchanged → bit-identical to baseline M=1 GEMV.

## File structure

| File | Action | What changes |
|------|--------|-------------|
| `openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/fully_connected_gpu_gemv.cl` | **Modify** | Add `int m = get_global_id(1)` + input/output m-offsets in all 3 variants |
| `openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_gemv.cpp` | **Modify** | Allow M>1 in `Validate()`, set `gws[1] = batch` in `SetDefault()`, raise priority in `GetKernelsPriority()` |
| `openvino-explicit-modeling/scripts/test_dflash.py` | **Read-only** | Already exists; run to verify fix |

All paths below are relative to `D:\chuansheng\src_code\explicit_modeling\`.

---

## Chunk 1: OpenCL kernel — add batch (M) dimension

### Task 1: Add m-offset to OSV16 variant

**Files:**
- Modify: `openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/fully_connected_gpu_gemv.cl:117-244`

The OSV16 variant starts at line 117 with `#if KERNEL_LAYOUT_OS_IS_YX_OSV16`.

Currently:
```cl
int n = get_global_id(0);              // N
int thr_id = get_local_id(2);          // 0~15
int thr_num = get_local_size(2);       // 16
int wi_id = get_sub_group_local_id();  // 0~15
```

And input:
```cl
__global INPUT0_TYPE* A = input + gk * DECOMPRESSION_GROUP_SIZE;
```

And output (inside `if (wi_id == 0)` block, around line 228):
```cl
output[cur_n + i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
```

- [ ] **Step 1: Add `int m` declaration after `int n`**

Add immediately after `int n = get_global_id(0);`:
```cl
    int m = get_global_id(1);              // M (batch row)
```

- [ ] **Step 2: Offset input pointer by m × WEIGHTS_K**

Change the input line from:
```cl
        __global INPUT0_TYPE* A = input + gk * DECOMPRESSION_GROUP_SIZE;
```
To:
```cl
        __global INPUT0_TYPE* A = input + m * WEIGHTS_K + gk * DECOMPRESSION_GROUP_SIZE;
```

- [ ] **Step 3: Offset output writes by m × WEIGHTS_N**

Change the output write (non-fused-ops path) from:
```cl
            output[cur_n + i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
```
To:
```cl
            output[m * WEIGHTS_N + cur_n + i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
```

Also change the fused-ops path (the `output[cur_n + i] = FUSED_OPS_RESULT_VEC;` line in the `HAS_FUSED_OPS` block):
```cl
            output[m * WEIGHTS_N + cur_n + i] = FUSED_OPS_RESULT_VEC;
```

---

### Task 2: Add m-offset to OSV32_ISV2 variant

**Files:**
- Modify: `openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/fully_connected_gpu_gemv.cl:246-396`

The OSV32_ISV2 variant starts at line 246 with `#elif KERNEL_LAYOUT_OS_IS_YX_OSV32_ISV2`.

Currently:
```cl
    int n = get_global_id(0) * 2;          // N
    int thr_id = get_local_id(2);
    int thr_num = get_local_size(2);
    int wi_id = get_sub_group_local_id();
```

Input:
```cl
        __global INPUT0_TYPE* A = input + gk * DECOMPRESSION_GROUP_SIZE;
```

Output (inside `if (wi_id == 0)`, around line 388):
```cl
        output[cur_n + 16 * i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
```

- [ ] **Step 4: Add `int m` declaration after `int n`**

```cl
    int m = get_global_id(1);              // M (batch row)
```

- [ ] **Step 5: Offset input pointer**

```cl
        __global INPUT0_TYPE* A = input + m * WEIGHTS_K + gk * DECOMPRESSION_GROUP_SIZE;
```

- [ ] **Step 6: Offset output writes (loop runs i=0..1, 2 outputs)**

Non-fused-ops:
```cl
            output[m * WEIGHTS_N + cur_n + 16 * i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
```
Fused-ops:
```cl
            output[m * WEIGHTS_N + cur_n + 16 * i] = FUSED_OPS_RESULT_VEC;
```

---

### Task 3: Add m-offset to OSV64_ISV2 variant

**Files:**
- Modify: `openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/fully_connected_gpu_gemv.cl:397-593`

The OSV64_ISV2 variant starts at line 397 with `#elif KERNEL_LAYOUT_OS_IS_YX_OSV64_ISV2`.

Currently:
```cl
    int n = get_global_id(0) * 4;          // N
```

Input:
```cl
        __global INPUT0_TYPE* A = input + gk * DECOMPRESSION_GROUP_SIZE;
```

Output (loop runs i=0..3, 4 outputs):
```cl
        output[cur_n + 16 * i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
```

- [ ] **Step 7: Add `int m` declaration after `int n`**

```cl
    int m = get_global_id(1);              // M (batch row)
```

- [ ] **Step 8: Offset input pointer**

```cl
        __global INPUT0_TYPE* A = input + m * WEIGHTS_K + gk * DECOMPRESSION_GROUP_SIZE;
```

- [ ] **Step 9: Offset output writes (loop runs i=0..3, 4 outputs)**

Non-fused-ops:
```cl
        output[m * WEIGHTS_N + cur_n + 16 * i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
```
Fused-ops:
```cl
        output[m * WEIGHTS_N + cur_n + 16 * i] = FUSED_OPS_RESULT_VEC;
```

---

### Task 4: Commit kernel changes

- [ ] **Step 10: Commit**

```bash
cd /d/chuansheng/src_code/explicit_modeling/openvino
git add src/plugins/intel_gpu/src/kernel_selector/cl_kernels/fully_connected_gpu_gemv.cl
GIT_AUTHOR_NAME="Chuansheng Liu" GIT_AUTHOR_EMAIL="chuansheng.liu@intel.com" \
GIT_COMMITTER_NAME="Chuansheng Liu" GIT_COMMITTER_EMAIL="chuansheng.liu@intel.com" \
git commit -m "feat(gemv): add batch (M>1) support with m-offset in all 3 layout variants

Extend fully_connected_gpu_gemv.cl to process M>1 input rows in parallel.
Each variant now reads int m = get_global_id(1) and offsets the input
pointer by m*WEIGHTS_K and output pointer by m*WEIGHTS_N. The FP32
reduction tree is completely unchanged, preserving bit-identity with the
M=1 GEMV path."
```

---

## Chunk 2: C++ kernel selector — allow M>1, update dispatch, raise priority

### Task 5: Relax Validate() to allow batch > 1

**Files:**
- Modify: `openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_gemv.cpp:74-77`

Current code (lines 74-77):
```cpp
    // Only support vector data as input, the data size should be aligned by 16 elements
    auto input_size = get_input_bf_size(fc_params);
    if (input_size.first > 1 || input_size.second == 0 || input_size.second % 16 != 0 || weights.IFM().v % 16 != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
```

Change to allow any batch ≥ 1:
```cpp
    // Support batch ≥ 1; inner dimension must be 16-aligned
    auto input_size = get_input_bf_size(fc_params);
    if (input_size.second == 0 || input_size.second % 16 != 0 || weights.IFM().v % 16 != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
```

Also remove the dynamic-shapes batch=1 check (lines 87-96). The current code reads:
```cpp
    if (input_size.first != 0 && fc_input.is_dynamic()) {
        if (input_size.first != 1) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
        if (!(wl == WeightsLayout::os_is_yx_osv32_isv2 && wo % 32 == 0) &&
            !(wl == WeightsLayout::os_is_yx_osv64_isv2 && wo % 64 == 0) &&
            !(wl == WeightsLayout::os_iyx_osv16 && wo % 16 == 0)) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
    }
```

Replace the **entire block** (both the outer condition and both inner checks) with:
```cpp
    if (fc_input.is_dynamic()) {
        // Allow any batch size for compressed INT4 (GEMV-batch supports M>1).
        if (!(wl == WeightsLayout::os_is_yx_osv32_isv2 && wo % 32 == 0) &&
            !(wl == WeightsLayout::os_is_yx_osv64_isv2 && wo % 64 == 0) &&
            !(wl == WeightsLayout::os_iyx_osv16 && wo % 16 == 0)) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
    }
```

Two changes: (a) remove `input_size.first != 0 &&` from the outer `if`; (b) remove the `if (input_size.first != 1)` guard entirely.

- [ ] **Step 11: Apply Validate() changes**

---

### Task 6: Update SetDefault() to set gws[1] = actual batch size

**Files:**
- Modify: `openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_gemv.cpp:113-131`

Current `SetDefault()`:
```cpp
FullyConnected_GEMV::DispatchData FullyConnected_GEMV::SetDefault(const fully_connected_params& params,
                                                                  int,
                                                                  int /*kernel_number*/) const {
    auto dispatchData = Parent::SetDefault(params);

    std::vector<size_t> global = {params.weights.OFM().v, 1, 16};
    if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
        global[0] = params.weights.OFM().v;
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
        global[0] = params.weights.OFM().v / 2;
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
        global[0] = params.weights.OFM().v / 4;
    }

    dispatchData.gws = global;
    dispatchData.lws = {16, 1, 16};

    return dispatchData;
}
```

Change to:
```cpp
FullyConnected_GEMV::DispatchData FullyConnected_GEMV::SetDefault(const fully_connected_params& params,
                                                                  int,
                                                                  int /*kernel_number*/) const {
    auto dispatchData = Parent::SetDefault(params);

    auto input_size = get_input_bf_size(params);
    // For dynamic shapes, input_size.first is 0 at compile time but resolved at runtime.
    // Use max(1, batch) so the static compile-time path still generates valid gws.
    size_t batch = (input_size.first > 0) ? input_size.first : 1;

    std::vector<size_t> global = {params.weights.OFM().v, batch, 16};
    if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
        global[0] = params.weights.OFM().v;
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
        global[0] = params.weights.OFM().v / 2;
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
        global[0] = params.weights.OFM().v / 4;
    }

    dispatchData.gws = global;
    dispatchData.lws = {16, 1, 16};

    return dispatchData;
}
```

- [ ] **Step 12: Apply SetDefault() changes**

---

### Task 6b: Add GetUpdateDispatchDataFunc override for dynamic-shape runtime batch

**Files:**
- Modify: `openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_gemv.cpp`
- Modify: `openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_gemv.h`

**Why this is critical:** For shape-agnostic kernels (`is_shape_agnostic = true`), the GWS is compiled with static (placeholder) values. At inference time, `update_dispatch_data_func` is called with the actual tensor shapes to update GWS before kernel launch. The base class `GetUpdateDispatchDataFunc` calls `SetDefault(prim_params)` which uses `get_input_bf_size` — this works correctly ONLY if `get_input_bf_size` returns the runtime batch size. For dynamic shapes where batch varies per call (like DFlash with M=N draft tokens), the runtime batch is available in `prim_params.inputs[0].Batch().v`. Verify that `get_input_bf_size` reads from the resolved tensor at runtime, not a compile-time placeholder.

- [ ] **Step 12b: Verify base-class update_dispatch_data_func is sufficient**

Check if `FullyConnectedKernelBase::GetUpdateDispatchDataFunc` (in `fully_connected_kernel_base.cpp`) calls `SetDefault(prim_params)` with the actual runtime params:

```bash
grep -n "GetUpdateDispatchDataFunc\|update_dispatch_data\|SetDefault" \
  /d/chuansheng/src_code/explicit_modeling/openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_base.cpp
```

The base class (line 74-83 in `fully_connected_kernel_base.cpp`) already does:
```cpp
kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
    const auto& prim_params = static_cast<const fully_connected_params&>(params);
    auto dispatchData = SetDefault(prim_params);
    kd.kernels[0].params.workGroups.global = dispatchData.gws;
    kd.kernels[0].params.workGroups.local = dispatchData.lws;
    ...
};
```

`SetDefault(prim_params)` will call the GEMV-overridden `SetDefault`, which calls `get_input_bf_size(params)`. The key question: does `get_input_bf_size` return the actual batch from `prim_params.inputs[0].Batch().v` at runtime?

```bash
grep -n "get_input_bf_size\|Batch().v" \
  /d/chuansheng/src_code/explicit_modeling/openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_bf_tiled.cpp \
  | head -10
```

If `Batch().v` is the actual runtime batch, no override is needed. If it's 0 (unresolved dynamic), add an override:

- [ ] **Step 12c: Add override if needed**

If `get_input_bf_size` returns 0 for dynamic batch at runtime, add to `fully_connected_kernel_gemv.cpp`:
```cpp
void FullyConnected_GEMV::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const fully_connected_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        // SetDefault may return batch=1 if get_input_bf_size returns 0 for dynamic shapes.
        // Override with actual runtime batch from the input tensor.
        size_t actual_batch = prim_params.inputs[0].Batch().v;
        if (actual_batch > 1) {
            dispatchData.gws[1] = actual_batch;
        }
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}
```

And declare it in the header:
```cpp
void GetUpdateDispatchDataFunc(KernelData& kd) const override;
```

---

### Task 7: Raise priority for compressed INT4 + M>1

**Files:**
- Modify: `openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_gemv.cpp:133-144`

**Known BF_TILED priority facts** (from `fully_connected_kernel_bf_tiled.cpp::GetKernelsPriority`):
- BF_TILED returns `FORCE_PRIORITY_4` for F16 input + M>1 (DFlash batch verify path)
- BF_TILED returns `FORCE_PRIORITY_5` for M=1

So GEMV-batch must use `FORCE_PRIORITY_2` (lower number = higher priority) to beat BF_TILED's `FORCE_PRIORITY_4`.

Current `GetKernelsPriority()`:
```cpp
KernelsPriority FullyConnected_GEMV::GetKernelsPriority(const Params& params) const {
    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    auto priority = FORCE_PRIORITY_9;
    if (!params.is_shape_agnostic) {
        auto output_size = get_output_aligned_bf_size(fc_params, false);
        if (output_size.first == 1 && output_size.second % 16 == 0) {
            priority = FORCE_PRIORITY_2;
        }
    }
    return priority;
}
```

Change to:
```cpp
KernelsPriority FullyConnected_GEMV::GetKernelsPriority(const Params& params) const {
    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    auto priority = FORCE_PRIORITY_9;
    if (!params.is_shape_agnostic) {
        auto output_size = get_output_aligned_bf_size(fc_params, false);
        if (output_size.second % 16 == 0) {
            // M=1 (original GEMV) and M>1 (GEMV-batch): FORCE_PRIORITY_2 beats
            // BF_TILED's FORCE_PRIORITY_4 for F16+M>1 and FORCE_PRIORITY_5 for M=1.
            priority = FORCE_PRIORITY_2;
        }
    } else {
        // Shape-agnostic (dynamic shapes, used by DFlash): high priority for compressed INT4.
        // BF_TILED returns FORCE_PRIORITY_4 for this case (F16 input, M>1).
        if (fc_params.compressed) {
            priority = FORCE_PRIORITY_2;
        }
    }
    return priority;
}
```

**Note on fused-ops limitation:** `GetJitConstants()` hardcodes batch index `"0"` in the fused-ops index order. For M>1, fused ops will use batch=0 for all rows. For DFlash (no fused ops in the verify path) this is harmless. Do NOT change the fused-ops index order in this task — it requires a separate, more complex fix.

- [ ] **Step 13: Apply GetKernelsPriority() changes**

---

### Task 8: Commit C++ selector changes

- [ ] **Step 15: Commit**

```bash
cd /d/chuansheng/src_code/explicit_modeling/openvino
git add src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/fully_connected_kernel_gemv.cpp
GIT_AUTHOR_NAME="Chuansheng Liu" GIT_AUTHOR_EMAIL="chuansheng.liu@intel.com" \
GIT_COMMITTER_NAME="Chuansheng Liu" GIT_COMMITTER_EMAIL="chuansheng.liu@intel.com" \
git commit -m "feat(gemv): extend selector to support M>1 batch for compressed INT4

- Validate(): allow batch>1 (remove the batch==1 restriction)
- SetDefault(): set gws[1] = actual batch size (was hardcoded 1)
- GetKernelsPriority(): FORCE_PRIORITY_2 for compressed INT4 + shape-agnostic,
  so GEMV-batch beats BF_TILED for DFlash batch verify

For M=1, gws[1]=1 and get_global_id(1)=0, so m*WEIGHTS_K=0 — fully
backward compatible with existing M=1 behavior."
```

---

## Chunk 3: Build, test, and document

### Task 9: Build openvino GPU plugin

- [ ] **Step 16: Build**

```bash
cd /d/chuansheng/src_code/explicit_modeling/openvino/build
cmake --build . --target openvino_intel_gpu_plugin --config Release -j12
```

Expected: `[100%] Built target openvino_intel_gpu_plugin`. Fix any compile errors (type mismatches from `m * WEIGHTS_K` if WEIGHTS_K is `size_t` and m is `int` — cast to `int` or `ptrdiff_t` as needed).

- [ ] **Step 17: Clear NEO compiler cache**

```bash
rm -rf /c/Users/PET/AppData/Local/NEO/neo_compiler_cache/
```

---

### Task 10: Run accuracy tests

- [ ] **Step 18: Run accuracy-only tests**

```bash
cd /d/chuansheng/src_code/explicit_modeling/openvino-explicit-modeling
python -m pytest scripts/test_dflash.py -v -s -k "accuracy" 2>&1 | tail -20
```

**Expected outcome (all 6 pass):**
```
PASSED scripts/test_dflash.py::TestDFlashAccuracy::test_fp16_exact_match[Joy can read...]
PASSED scripts/test_dflash.py::TestDFlashAccuracy::test_fp16_exact_match[Write a Python function...]
PASSED scripts/test_dflash.py::TestDFlashAccuracy::test_fp16_exact_match[What is 2 + 2?]
PASSED scripts/test_dflash.py::TestDFlashAccuracy::test_int4_exact_match[Joy can read...]
PASSED scripts/test_dflash.py::TestDFlashAccuracy::test_int4_exact_match[Write a Python function...]
PASSED scripts/test_dflash.py::TestDFlashAccuracy::test_int4_exact_match[What is 2 + 2?]
6 passed in ...
```

If `test_int4_exact_match[What is 2 + 2?]` still fails:
- Check that GEMV-batch is actually being selected (add a log or check priority values)
- If BF_TILED still wins, lower GEMV priority to FORCE_PRIORITY_1
- If GEMV-batch is selected but outputs differ, verify the m-offset arithmetic in the OCL kernel

---

### Task 11: Run performance test and update report

- [ ] **Step 19: Run full perf test**

```bash
cd /d/chuansheng/src_code/explicit_modeling/openvino-explicit-modeling
python -m pytest scripts/test_dflash.py -v -s -k "perf" 2>&1 | tail -40
```

Record the new INT4/FP16 and INT4/INT4 TPOT values. The DFlash INT4 speedup may change:
- If GEMV-batch runs at ~same wall time as M=1 GEMV (expected, since weights dominate bandwidth), INT4/FP16 speedup stays ~1.35×.
- If GEMV-batch is slower (because N=16 work groups serialize on weight bandwidth), speedup drops slightly. Either outcome is acceptable — accuracy is the primary goal.

- [ ] **Step 20: Update perf_summary.md**

Add a Phase 3 section to `dflash优化报告/perf_summary.md` with the new benchmark results.

- [ ] **Step 21: Commit report and test artifacts**

```bash
cd /d/chuansheng/src_code/explicit_modeling/openvino-explicit-modeling
git add dflash优化报告/perf_summary.md
GIT_AUTHOR_NAME="Chuansheng Liu" GIT_AUTHOR_EMAIL="chuansheng.liu@intel.com" \
GIT_COMMITTER_NAME="Chuansheng Liu" GIT_COMMITTER_EMAIL="chuansheng.liu@intel.com" \
git commit -m "docs(dflash): update perf summary with Phase 3 GEMV-batch results (6/6 accuracy)"
```

---

## Debugging guide

**If GEMV-batch is not selected for M>1:**
1. Add a temporary `printf` or check kernel name at runtime
2. Verify `GetKernelsPriority()` returns a lower number than BF_TILED for compressed INT4 shape-agnostic
3. Check that `Validate()` returns `true` for the M>1 case

**If outputs differ from M=1 GEMV:**
1. Verify the input offset: `m * WEIGHTS_K` — confirm `WEIGHTS_K` is the inner dim (= IFM = FC input size)
2. Verify the output offset: `m * WEIGHTS_N` — confirm `WEIGHTS_N` is the output dim (= OFM)
3. Check all three variants were updated (OSV16, OSV32_ISV2, OSV64_ISV2)
4. The layout used for Qwen3.5-4B INT4 with group_size=128 is typically `os_is_yx_osv32_isv2` — verify by checking which variant is hit

**If `m * WEIGHTS_K` produces wrong offset for non-`bf` input layouts:**
- The `bfyx` layout has 4D structure. For FC with bfyx input where X=Y=1 (typical), element [m, k, 0, 0] is still at offset `m * K`. Check with a breakpoint or add an assert.

**Type mismatch compile error:**
If `m * WEIGHTS_K` causes a compile warning (int × size_t), cast:
```cl
__global INPUT0_TYPE* A = input + (int)(m * WEIGHTS_K) + gk * DECOMPRESSION_GROUP_SIZE;
```
Or in OCL, `m` is `int` from `get_global_id`, and `WEIGHTS_K` is a macro expanding to `size_t`. The expression `m * WEIGHTS_K` promotes m to size_t automatically in C99/OCL. No cast needed, but be careful of negative m (get_global_id always returns ≥ 0, so no issue).
