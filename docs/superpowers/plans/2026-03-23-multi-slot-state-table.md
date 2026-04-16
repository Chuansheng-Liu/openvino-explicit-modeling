# Multi-Slot State Table Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add indexed multi-slot state tables to the LinearAttention kernel so MTP batch verify writes per-token GDA states to dedicated slots, eliminating save/restore/replay overhead.

**Architecture:** Enlarge recurrent Variable from `[1, ...]` to `[max_slots, ...]`. Add `ssm_state_indices` input to the LA kernel. The kernel reads initial state from `ssm_state_indices[0]` and writes each token's recurrent state to its indexed slot. After batch verify, the accepted recurrent state is already in the correct slot — zero fixup. Conv state (18 KB/layer, trivial) is handled by existing FusedConv + save/restore (~0.2 ms total).

**Phased approach**: This plan covers recurrent state multi-slot only (eliminates the expensive 2 MB × 30 layer save/restore). Conv state fusion into the LA kernel is deferred to a future iteration — the cost is only ~0.2 ms for save/restore of 30 × 18 KB = 540 KB.

**Tech Stack:** OpenVINO C++ IR nodes, cldnn primitives, OpenCL kernel, openvino.genai model builder C++

**Spec:** `docs/superpowers/specs/2026-03-23-multi-slot-state-table-batch-verify-design.md`

**Repos:**
- GPU plugin: `D:\chuansheng\src_code\explicit_modeling\openvino` — branch `batch_verify_mtp` (HEAD: `25c5bb9936`)
- GenAI app: `D:\chuansheng\src_code\explicit_modeling\openvino.genai` — branch `batch_verify_mtp` (HEAD: `1526867c`)

**Build:**
- GPU plugin: `cd D:\chuansheng\src_code\explicit_modeling\openvino\build && cmake --build . --target openvino_intel_gpu_plugin --config Release -j12`
- GenAI: rebuild after plugin changes are in place

**Important notes:**
- The OCL kernel parameter order swaps g and beta: `input[3]=g, input[4]=beta` (see ops.cpp line 117-119)
- Variable memory is bound at runtime via `get_arguments()` — `args.inputs[5]` and `args.outputs[1]` point to the same GPU buffer
- `init_base` formula for state indexing: `b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS`
- Conv state uses channels-first layout: `[B, conv_dim, conv_kernel-1]`
- After enabling `use_state_table`, cached IR XML/bin files must be deleted

---

## Chunk 1: GPU Plugin — IR Node + Primitive

### Task 1: LinearAttention IR Node — `use_state_table` attribute and new constructor

**Files:**
- Modify: `src/core/dev_api/openvino/op/linear_attn.hpp` (42 lines → ~60 lines)
- Modify: `src/core/src/op/linear_attn.cpp` (131 lines → ~185 lines)

- [ ] **Step 1: Add `use_state_table` attribute to header**

In `linear_attn.hpp`, add:

```cpp
// After the existing constructor declarations (line 22):
LinearAttention(const ov::OutputVector& args,
                const std::shared_ptr<ov::op::util::Variable>& variable,
                bool use_state_table);

// Add getter:
bool get_use_state_table() const { return m_use_state_table; }

// In protected section (after m_output_type, line 37):
bool m_use_state_table = false;
```

- [ ] **Step 2: Implement the new constructor in linear_attn.cpp**

After the existing constructors (line 71):

```cpp
LinearAttention::LinearAttention(const ov::OutputVector& args,
                                 const std::shared_ptr<ov::op::util::Variable>& variable,
                                 bool use_state_table)
    : ov::op::Op(args), m_use_state_table(use_state_table) {
    m_variable = variable;
    constructor_validate_and_infer_types();
}
```

- [ ] **Step 3: Update `visit_attributes` to serialize `use_state_table`**

In `visit_attributes()` (line 73), add after the existing variable serialization:

```cpp
visitor.on_attribute("use_state_table", m_use_state_table);
```

- [ ] **Step 4: Update `validate_and_infer_types` for 7-input case**

Replace the current validation (lines 87-115) with:

```cpp
void LinearAttention::validate_and_infer_types() {
    OV_OP_SCOPE(LinearAttention_validate_and_infer_types);

    if (m_use_state_table) {
        NODE_VALIDATION_CHECK(this, get_input_size() == 7,
            "LinearAttention (state_table) expects 7 inputs, but has ", get_input_size());
        input_check(this, 0, "query", {4}, {});
        input_check(this, 1, "key", {4}, {});
        input_check(this, 2, "value", {4}, {});
        input_check(this, 3, "g", {3}, {});
        input_check(this, 4, "beta", {3}, {});
        input_check(this, 5, "initial_states", {4}, {});
        input_check(this, 6, "ssm_state_indices", {1}, {ov::element::i32});
    } else {
        NODE_VALIDATION_CHECK(this, get_input_size() == 6,
            "LinearAttention expects 6 inputs, but has ", get_input_size());
        input_check(this, 0, "query", {4}, {});
        input_check(this, 1, "key", {4}, {});
        input_check(this, 2, "value", {4}, {});
        input_check(this, 3, "g", {3}, {});         // Note: swapped in OCL kernel
        input_check(this, 4, "beta", {3}, {});
        input_check(this, 5, "initial_states", {4}, {});
    }

    const auto& q_ps = get_input_partial_shape(0);
    const auto& v_ps = get_input_partial_shape(2);
    const auto& h_ps = get_input_partial_shape(5);

    ov::PartialShape out_ps = v_ps;
    if (out_ps.rank().is_static() && q_ps.rank().is_static() &&
        out_ps.rank().get_length() == 4 && q_ps.rank().get_length() == 4) {
        out_ps[0] = q_ps[0];
        out_ps[1] = q_ps[1];
    }
    set_output_type(0, get_input_element_type(0), out_ps);
    set_output_type(1, get_input_element_type(5), h_ps);
}
```

- [ ] **Step 5: Update `clone_with_new_inputs`**

Replace lines 117-122:

```cpp
std::shared_ptr<ov::Node> LinearAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    if (m_use_state_table && m_variable) {
        return std::make_shared<LinearAttention>(new_args, m_variable, true);
    }
    if (m_variable) {
        return std::make_shared<LinearAttention>(new_args, m_variable);
    }
    return std::make_shared<LinearAttention>(new_args);
}
```

- [ ] **Step 6: Build GPU plugin to verify compilation**

Run: `cd D:\chuansheng\src_code\explicit_modeling\openvino\build && cmake --build . --target openvino_intel_gpu_plugin --config Release -j12`
Expected: Compiles without errors

- [ ] **Step 7: Commit**

```
feat(linear_attn): add use_state_table attribute and 7-input constructor
```

### Task 2: cldnn Primitive — `use_state_table` fields

**Files:**
- Modify: `src/plugins/intel_gpu/include/intel_gpu/primitives/linear_attention.hpp` (81 lines → ~105 lines)

- [ ] **Step 1: Add `use_state_table` field to the primitive struct**

After `variable_info` (line 45), add:

```cpp
bool use_state_table = false;
```

- [ ] **Step 2: Add new constructor overload**

After the existing variable-aware constructor (line 43):

```cpp
/// @brief Constructs linear_attention with state table support.
linear_attention(const primitive_id& id,
        const std::vector<input_info>& inputs,
        const ov::op::util::VariableInfo& variable_info,
        bool use_state_table)
    : primitive_base(id, inputs),
      variable_info(variable_info),
      use_state_table(use_state_table) {
}
```

- [ ] **Step 3: Update `hash()` to include `use_state_table`**

Replace the hash function (lines 47-53):

```cpp
size_t hash() const override {
    size_t seed = primitive::hash();
    seed = hash_combine(seed, use_state_table);
    return seed;
}
```

- [ ] **Step 4: Update `operator==`**

```cpp
bool operator==(const primitive& rhs) const override {
    if (!compare_common_params(rhs)) return false;
    auto rhs_casted = downcast<const linear_attention>(rhs);
    return use_state_table == rhs_casted.use_state_table;
}
```

- [ ] **Step 5: Update `save()`/`load()` for serialization**

Extend save/load (lines 60-77) to include the new field:

```cpp
void save(BinaryOutputBuffer& ob) const override {
    primitive_base<linear_attention>::save(ob);
    ov::element::Type_t data_type = variable_info.data_type;
    ob << variable_info.variable_id;
    ob << variable_info.data_shape;
    ob << make_data(&data_type, sizeof(ov::element::Type_t));
    ob << use_state_table;
}

void load(BinaryInputBuffer& ib) override {
    primitive_base<linear_attention>::load(ib);
    ov::PartialShape data_shape;
    ov::element::Type_t data_type = ov::element::Type_t::dynamic;
    std::string variable_id;
    ib >> variable_id;
    ib >> data_shape;
    ib >> make_data(&data_type, sizeof(ov::element::Type_t));
    variable_info = {data_shape, data_type, variable_id};
    ib >> use_state_table;
}
```

- [ ] **Step 6: Build and verify**

Run: `cd D:\chuansheng\src_code\explicit_modeling\openvino\build && cmake --build . --target openvino_intel_gpu_plugin --config Release -j12`

- [ ] **Step 7: Commit**

```
feat(linear_attn): add use_state_table/conv fields to cldnn primitive
```

### Task 3: Plugin Ops Registration — handle 7-input state table case

**Files:**
- Modify: `src/plugins/intel_gpu/src/plugin/ops/linear_attention.cpp` (42 lines → ~60 lines)

- [ ] **Step 1: Update `CreateLinearAttentionOp` for state table mode**

Replace `validate_inputs_count(op, {6});` and the body (lines 22-36):

```cpp
static void CreateLinearAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::LinearAttention>& op) {
    if (op->get_use_state_table()) {
        validate_inputs_count(op, {7});
    } else {
        validate_inputs_count(op, {6});
    }

    auto inputs = p.GetInputInfo(op);
    const std::string layerName = layer_type_name_ID(op);

    if (op->get_use_state_table() && op->get_variable()) {
        cldnn::linear_attention prim(layerName, inputs, op->get_variable()->get_info(), true);
        prim.num_outputs = op->get_output_size();
        p.add_primitive(*op, prim);
    } else if (op->get_variable()) {
        cldnn::linear_attention prim(layerName, inputs, op->get_variable()->get_info());
        prim.num_outputs = op->get_output_size();
        p.add_primitive(*op, prim);
    } else {
        cldnn::linear_attention prim(layerName, inputs);
        prim.num_outputs = op->get_output_size();
        p.add_primitive(*op, prim);
    }
}
```

- [ ] **Step 2: Build and verify**

- [ ] **Step 3: Commit**

```
feat(linear_attn): handle 7-input state table mode in plugin ops
```

### Task 4: Graph-level — `calc_output_layouts` for 7 inputs

**Files:**
- Modify: `src/plugins/intel_gpu/src/graph/linear_attention.cpp` (66 lines → ~75 lines)

- [ ] **Step 1: Update `calc_output_layouts` to handle 7-input case**

Replace the input size check (lines 24-40):

```cpp
template<typename ShapeType>
std::vector<layout> linear_attention_inst::calc_output_layouts(linear_attention_node const& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<linear_attention>();
    const auto& all_inputs = node.get_input_layouts();
    const auto num_outputs = desc->output_size();

    if (desc->use_state_table) {
        if (all_inputs.size() != 7)
            OPENVINO_THROW("linear_attention (state_table) must have 7 inputs, got ", all_inputs.size());
    } else {
        if (all_inputs.size() != 6)
            OPENVINO_THROW("linear_attention must have 6 inputs, got ", all_inputs.size());
    }

    auto query_layout = impl_param.get_input_layout(0);
    auto value_layout = impl_param.get_input_layout(2);
    auto out_ps = value_layout.get_partial_shape();
    const auto& q_ps = query_layout.get_partial_shape();
    if (out_ps.rank().is_static() && q_ps.rank().is_static() &&
        out_ps.rank().get_length() == 4 && q_ps.rank().get_length() == 4) {
        out_ps[0] = q_ps[0];
        out_ps[1] = q_ps[1];
    }

    std::vector<layout> output_layouts;
    output_layouts.emplace_back(out_ps, value_layout.data_type, value_layout.format);
    if (num_outputs >= 2) {
        // Output[1]: recurrent state — same layout as input[5]
        output_layouts.push_back(impl_param.get_input_layout(5));
    }
    return output_layouts;
}
```

- [ ] **Step 2: Build and verify**

- [ ] **Step 3: Commit**

```
feat(linear_attn): calc_output_layouts supports 7-input state table mode
```

---

## Chunk 2: GPU Plugin — OpenCL Kernel + Implementation

### Task 5: Kernel implementation (`linear_attention_ref.cpp`) — JIT constants for state table

**Files:**
- Modify: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/linear_attention_ref.cpp` (191 lines → ~200 lines)

- [ ] **Step 1: Add JIT constants for state table mode**

In `get_jit_constants()` (after line 55), add:

```cpp
const auto& desc = params.typed_desc<linear_attention>();
jit.make("USE_STATE_TABLE", desc && desc->use_state_table ? 1 : 0);
if (desc && desc->use_state_table) {
    jit.make("STATE_STRIDE", v_head_nums * k_head_dims * k_head_dims);  // HV * V * K
}
```

No changes to `get_arguments_desc()` — the generic input/output loop already handles the 7th input (`ssm_state_indices` at index 6).

No changes to `get_arguments()` or `execute()` — the recurrent Variable binding (input[5]/output[1]) is already correct. The enlarged `[max_slots, ...]` Variable memory is bound to input[5] and output[1] as before.

- [ ] **Step 2: Build and verify**

Run: `cd D:\chuansheng\src_code\explicit_modeling\openvino\build && cmake --build . --target openvino_intel_gpu_plugin --config Release -j12`

- [ ] **Step 3: Commit**

```
feat(linear_attn): add USE_STATE_TABLE/STATE_STRIDE JIT constants
```

### Task 6: OpenCL kernel — multi-slot state reads/writes

**Files:**
- Modify: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/linear_attention_ref.cl` (562 lines → ~630 lines)

This is the most complex task. The kernel has 4 specialization paths based on `K_HEAD_DIMS` and `SUBGROUP_SIZE`. The multi-slot changes wrap around the existing state load/store logic.

**Argument ordering**: The generic arg loop in `get_arguments_desc()` emits all inputs (0..N-1), then all outputs (0..M-1), then scalars. For 7-input mode, `ssm_state_indices` (input[6]) comes BEFORE outputs in the kernel signature.

- [ ] **Step 1: Add `ssm_state_indices` to kernel signature**

Replace the kernel signature (lines 264-277). Insert `ssm_state_indices` between `initial_state` and `output` to match argument descriptor ordering:

```c
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(linear_attention_ref)
(__global INPUT0_TYPE* q,
 __global INPUT1_TYPE* k,
 __global INPUT2_TYPE* v,
 __global INPUT3_TYPE* g,
 __global INPUT4_TYPE* beta,
 __global INPUT5_TYPE* initial_state,
#if USE_STATE_TABLE
 __global int* ssm_state_indices,
#endif
 __global OUTPUT_TYPE* output,
#if OUTPUT_STATE
 __global OUTPUT1_TYPE* output_state,
#endif
 int seq_len,
 int key_offset,
 int value_offset) {
```

- [ ] **Step 2: Add zero-initialization for K_HEAD_DIMS==128 paths**

The K_HEAD_DIMS==128/SG8 and K_HEAD_DIMS==128/SG16 paths declare `init_state` arrays without zero-initialization. When `init_slot < 0` (PAD_SLOT_ID), the load is skipped — leaving uninitialized data in registers.

For each K_HEAD_DIMS==128 path, after the `init_state` array declaration, add:

```c
#if USE_STATE_TABLE
    // Zero-init required: K_HEAD_DIMS==128 paths don't use `= {0}` initializer.
    // When PAD_SLOT_ID (init_slot < 0) we skip the load, so must start from zeros.
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++)
        for (int ik = 0; ik < K_HEAD_DIMS / SUBGROUP_SIZE; ik++)
            init_state[iv][ik] = 0;
#endif
```

Apply to both the SG8 and SG16 specialization blocks. (The generic and K_HEAD_DIMS%32 paths already zero-initialize.)

- [ ] **Step 3: Add multi-slot initial state load**

Replace the state load block. Wrap existing load in a slot-index guard:

```c
    // load initial state
#if USE_STATE_TABLE
    int init_slot = ssm_state_indices[0];
#endif
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
#if USE_STATE_TABLE
        if (init_slot >= 0) {
            int init_base = init_slot * STATE_STRIDE
                          + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#else
        {
            int init_base = b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS
                          + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#endif
            // === existing load code (unchanged per-specialization) ===
        }
    }
```

Key changes:
- `init_base` uses `init_slot * STATE_STRIDE` instead of `b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS`
- When `init_slot < 0` (PAD_SLOT_ID), load is skipped — state starts from zeros (Step 2)

- [ ] **Step 4: Add per-token state write inside the token loop**

After the inner V_BLOCK loop, before the closing `}` of the token loop, add:

```c
#if USE_STATE_TABLE
        // Per-token state write to indexed slot
        {
            int slot = ssm_state_indices[i];
            if (slot >= 0) {
                for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
                    int i_v = i_v_base + iv;
                    int store_base = slot * STATE_STRIDE
                                   + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
                    // Reuse existing store helper for current specialization
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
                    store_init_state_128_sg8(init_state[iv], initial_state, store_base);
#    else
                    store_init_state_128(&init_state[iv], initial_state, store_base);
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
                    store_init_state_32_sg16(init_state[iv], initial_state, store_base, id_sg_local);
#    else
                    store_init_state_32(init_state[iv], initial_state, store_base, id_sg_local);
#    endif
#else
                    store_init_state_generic(init_state[iv], initial_state, store_base, id_sg_local);
#endif
                }
            }
        }
#endif
```

- [ ] **Step 5: Wrap the existing final state write in `#if !USE_STATE_TABLE`**

The existing final state write (lines 537-560) should only run when NOT using state table (per-token writes already handle state output):

```c
#if !USE_STATE_TABLE
    // store final state (existing code — unchanged)
    __global INPUT5_TYPE* state_out = initial_state;
#if OUTPUT_STATE
    state_out = (__global INPUT5_TYPE*)output_state;
#endif
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        // ... existing store code unchanged ...
    }
#endif
```

- [ ] **Step 6: Build and verify kernel compiles**

Run: `cd D:\chuansheng\src_code\explicit_modeling\openvino\build && cmake --build . --target openvino_intel_gpu_plugin --config Release -j12`

- [ ] **Step 7: Commit**

```
feat(linear_attn): multi-slot indexed state read/write in OCL kernel
```

### Task 7: Instance header — verify build

**Files:**
- Check: `src/plugins/intel_gpu/src/graph/include/linear_attention_inst.h` (43 lines — likely no changes needed)

No changes expected for the 7-input approach. The inst header accesses the primitive descriptor for variable info, which is unchanged. The only new input (ssm_state_indices) is handled generically.

- [ ] **Step 1: Verify build passes from Task 6**

If the build from Task 6 passed, skip this task entirely. If compilation fails, inspect error messages and add any missing declarations.

- [ ] **Step 2: Commit (only if changes were needed)**

```
fix(linear_attn): inst header adjustments for state table mode
```

---

## Chunk 3: GenAI Model Builder

### Task 8: ops.hpp/cpp — `linear_attention_with_state_table` function

**Files:**
- Modify: `src/cpp/src/modeling/ops/ops.hpp` (add declaration after line 41)
- Modify: `src/cpp/src/modeling/ops/ops.cpp` (add implementation after line 148)

- [ ] **Step 1: Add declaration to ops.hpp**

After the existing `linear_attention` declarations (line 41):

```cpp
Tensor linear_attention_with_state_table(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& beta, const Tensor& g,
    const Tensor& initial_state,
    const std::shared_ptr<ov::op::util::Variable>& recurrent_var,
    const Tensor& ssm_state_indices);
```

Note: no `raw_input`, `conv_state`, or `conv_var` — conv is handled by existing FusedConv path.

- [ ] **Step 2: Add implementation to ops.cpp**

After the existing `linear_attention` implementations (line 148):

```cpp
Tensor linear_attention_with_state_table(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& beta, const Tensor& g,
    const Tensor& initial_state,
    const std::shared_ptr<ov::op::util::Variable>& recurrent_var,
    const Tensor& ssm_state_indices) {
    auto* ctx = q.context();

    // Note: OCL kernel expects input[3]=g, input[4]=beta (swapped vs API order)
    ov::OutputVector args = {
        q.output(), k.output(), v.output(),
        g.output(), beta.output(), initial_state.output(),
        ssm_state_indices.output()
    };
    auto node = std::make_shared<ov::op::LinearAttention>(args, recurrent_var, true);
    return Tensor(node->output(0), ctx);
}
```

- [ ] **Step 3: Build genai to verify**

- [ ] **Step 4: Commit**

```
feat(ops): add linear_attention_with_state_table function
```

### Task 9: Qwen3_5TextModelConfig — `use_state_table` and `max_slots`

**Files:**
- Modify: `src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_text.hpp` (add fields after line 76)

- [ ] **Step 1: Add config fields**

After `mtp_num_hidden_layers` (line 76):

```cpp
bool use_state_table = false;
int max_slots = 8;
```

- [ ] **Step 2: Commit**

```
feat(qwen3_5): add use_state_table/max_slots config fields
```

### Task 10: Qwen3_5GatedDeltaNet — conditional wiring with state table

**Files:**
- Modify: `src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_text.hpp` (add `ssm_state_indices` member to class)
- Modify: `src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_text.cpp` (~lines 440-545, plus Qwen3_5Model and Qwen3_5DecoderLayer)

This is the main model builder change. When `use_state_table=true`:
1. Recurrent Variable gets `[max_slots, HV, K, V]` shape (enlarged from `[-1, HV, K, V]`)
2. LA call gets `ssm_state_indices` as 7th input via new ops function
3. No Assign for recurrent variable (kernel writes per-token state to indexed slots)
4. Conv state: **completely unchanged** — existing FusedConv + Assign path handles it

**Key**: `ssm_state_indices` is a single model-level Parameter shared by all 30 GDA layers. It must be threaded from `Qwen3_5Model` → `Qwen3_5DecoderLayer` → `Qwen3_5GatedDeltaNet`.

- [ ] **Step 1: Add `ssm_state_indices` model-level Parameter in Qwen3_5Model**

In `Qwen3_5Model::forward()` (or constructor), when `cfg_.use_state_table`:

```cpp
Tensor ssm_state_indices;
if (cfg_.use_state_table) {
    ssm_state_indices = ctx().add_parameter("ssm_state_indices",
        ov::element::i32, ov::PartialShape{ov::Dimension::dynamic()});
}
```

This creates a single Parameter node in the IR graph. All GDA layers share it.

- [ ] **Step 2: Thread `ssm_state_indices` through Qwen3_5DecoderLayer**

Update `Qwen3_5DecoderLayer::forward()` signature to accept `ssm_state_indices`:

```cpp
// In the header (modeling_qwen3_5_text.hpp):
Tensor forward(const Tensor& hidden_states, const Tensor& attention_mask,
               const Tensor& position_ids, const Tensor& ssm_state_indices = {});
```

In the implementation, pass it to the GatedDeltaNet submodule's forward call. For decoder layers that use SDPA (not GDA), `ssm_state_indices` is simply ignored.

In `Qwen3_5Model::forward()`, pass the tensor in the decoder layer loop:

```cpp
for (auto& layer : layers_) {
    hidden = layer.forward(hidden, attention_mask, position_ids, ssm_state_indices);
}
```

- [ ] **Step 3: Update Qwen3_5GatedDeltaNet::forward() for state table path**

In `forward()`, add `ssm_state_indices` parameter and branch on `cfg_.use_state_table`:

```cpp
if (use_linear_attention_op()) {
    if (cfg_.use_state_table && ssm_state_indices.output().get_node()) {
        // ── Multi-slot recurrent state table path ──
        // Recurrent Variable with enlarged shape [max_slots, HV, K, V]
        ov::op::util::VariableInfo recurrent_info{
            ov::PartialShape{cfg_.max_slots, num_v_heads_, head_k_dim_, head_v_dim_},
            ov::element::f32,
            state_prefix + ".recurrent"};
        auto recurrent_var = std::make_shared<ov::op::util::Variable>(recurrent_info);
        auto recurrent_read = std::make_shared<ov::op::v6::ReadValue>(recurrent_init.output(), recurrent_var);
        auto recurrent_state = Tensor(recurrent_read->output(0), op_ctx);

        auto la_out = ops::linear_attention_with_state_table(
            q_f32, k_f32, v_f32, beta, g,
            recurrent_state, recurrent_var,
            ssm_state_indices);
        core_attn_tensor = la_out;
        // NO Assign for recurrent_var — kernel writes per-token state to indexed slots
        // Conv state: existing FusedConv + Assign path is UNCHANGED
    } else {
        // Original fused LA path (unchanged)
        auto la_result = ops::linear_attention(q_f32, k_f32, v_f32, beta, g,
                                                recurrent_init, recurrent_var);
        core_attn_tensor = la_result.first;
    }
}
```

Conv Variable shape stays unchanged (`[-1, conv_dim, conv_kernel-1]`). Conv Assign remains. Only the recurrent path changes.

- [ ] **Step 4: Build genai to verify**

- [ ] **Step 5: Commit**

```
feat(qwen3_5): conditional state table wiring in GatedDeltaNet
```

---

## Chunk 4: Sample Code + Integration Testing

### Task 11: Sample code — slot management for batch verify

**Files:**
- Modify: `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` (~1525 lines)

- [ ] **Step 1: Add slot management state**

In the MTP-related state section, add:

```cpp
int current_slot = 0;
const int max_slots = 8;
ov::Tensor ssm_state_indices_tensor;
// Initialize:
ssm_state_indices_tensor = ov::Tensor(ov::element::i32, {1});
ssm_state_indices_tensor.data<int32_t>()[0] = 0;
```

- [ ] **Step 2: Add `copy_slot_to_zero` helper for periodic reset**

```cpp
void copy_slot_to_zero(ov::InferRequest& request, int src_slot) {
    if (src_slot == 0) return;
    for (auto& vs : request.query_state()) {
        if (vs.get_name().find(".recurrent") == std::string::npos) continue;

        ov::Tensor full_state = vs.get_state();
        auto shape = full_state.get_shape();
        size_t slot_bytes = full_state.get_element_type().size();
        for (size_t d = 1; d < shape.size(); d++) slot_bytes *= shape[d];

        auto* base = static_cast<char*>(full_state.data());
        std::memcpy(base, base + src_slot * slot_bytes, slot_bytes);
        vs.set_state(full_state);
    }
}
```

Note: filter by `.recurrent` (not `linear_states`) to match the Variable naming in the model builder.

- [ ] **Step 3: Update prefill to set ssm_state_indices**

In the prefill path:

```cpp
ssm_state_indices_tensor.set_shape({static_cast<size_t>(prompt_length)});
for (int t = 0; t < prompt_length - 1; t++)
    ssm_state_indices_tensor.data<int32_t>()[t] = -1;  // PAD_SLOT_ID
ssm_state_indices_tensor.data<int32_t>()[prompt_length - 1] = 0;
text_request.set_tensor("ssm_state_indices", ssm_state_indices_tensor);
```

- [ ] **Step 4: Update normal decode (M=1) to set ssm_state_indices**

```cpp
ssm_state_indices_tensor.set_shape({1});
ssm_state_indices_tensor.data<int32_t>()[0] = current_slot;
text_request.set_tensor("ssm_state_indices", ssm_state_indices_tensor);
```

- [ ] **Step 5: Add batch verify path**

Replace the sequential verify loop with the batch verify path:

```cpp
// Set up indices for batch verify: [current_slot, current_slot+1, ..., current_slot+N]
ssm_state_indices_tensor.set_shape({static_cast<size_t>(N + 1)});
for (int t = 0; t <= N; t++)
    ssm_state_indices_tensor.data<int32_t>()[t] = current_slot + t;
text_request.set_tensor("ssm_state_indices", ssm_state_indices_tensor);

// Run batch verify (M=N+1 tokens in one infer)
// ... set input_ids to [next_id, drafts[0..N-1]] ...
text_request.infer();

// Determine accept count
auto refs = get_logits_argmax();
int j = N;
for (int k = 0; k < N; ++k)
    if (refs[k] != drafts[k]) { j = k; break; }

// Advance slot — recurrent state already in correct slot, zero fixup
current_slot += j;

// Periodic reset when slots are running out
if (current_slot + N + 1 > max_slots) {
    copy_slot_to_zero(text_request, current_slot);
    current_slot = 0;
}

// Trim KV cache for rejected draft tokens
if (j < N) trim_main_kv_cache(N - j);

// Emit accepted tokens + bonus token
// ...
```

**Conv state contamination (known limitation of phased approach):**
During batch verify, FusedConv processes all M=N+1 tokens and Assign commits the final conv state. If `j < N` tokens are accepted, the conv state is `(N-j)` positions ahead — it contains `(N-j)` rejected draft tokens in the sliding window. This contamination:
- Does **NOT** affect output correctness (accepted tokens were already verified)
- Only affects draft quality (slightly lower accept rate for subsequent drafts)
- Self-corrects after `conv_kernel-1 = 3` M=1 decode steps (sliding window flushes)
- For N=1 with full accept (j=1), there is zero contamination

This is acceptable for the initial implementation. A future iteration can add multi-slot conv state or save/restore (~0.2 ms) to eliminate contamination entirely.

- [ ] **Step 6: Thread `use_state_table` flag from CLI**

```cpp
text_cfg.use_state_table = opts.mtp;  // only in MTP mode
text_cfg.max_slots = std::max(opts.mtp_draft_n + 2, 8);  // N+1 for verify + headroom
```

- [ ] **Step 7: Build and first smoke test**

Build genai, then run a quick sanity check:
```
python scripts/auto_tests.py --tests 53 --models-root "C:\data\models"
```

Test [53] = 2B MTP N=3 greedy no-think. Verify output is coherent and no crashes.

- [ ] **Step 8: Commit**

```
feat(qwen3_5): multi-slot batch verify in sample code
```

### Task 12: End-to-end testing

- [ ] **Step 1: Test 2B MTP N=3 — correctness baseline**

Run: `python scripts/auto_tests.py --tests 53 --models-root "C:\data\models"`

Test [53] = Qwen3.5-2B MTP N=3 greedy no-think. Verify:
- Output is coherent and matches Phase-2 sequential verify output
- No crashes or OOM
- Accept rate and throughput are in expected range

- [ ] **Step 2: Test 4B MTP N=3 — correctness + performance**

Run: `python scripts/auto_tests.py --tests 55 --models-root "C:\data\models"`

Test [55] = Qwen3.5-4B MTP N=3 greedy template. Verify:
- Output is coherent
- Compare throughput with Phase-2 sequential baseline
- Check accept rate (batch verify should match sequential since acceptance logic is identical)

- [ ] **Step 3: Benchmark copy_slot_to_zero**

Add timing instrumentation around `copy_slot_to_zero` to validate the 1-2 ms estimate for periodic reset.

- [ ] **Step 4: Test with larger N (if results are good)**

If N=3 works well, optionally test with N=5 or N=7 to stress the slot table (max_slots=8).

- [ ] **Step 5: Record results and commit**

```
test(mtp): multi-slot state table batch verify results
```

---

## Task Dependencies

```
Chunk 1 (GPU Plugin — IR + Primitive):
  Task 1 (IR node) → Task 2 (primitive) → Task 3 (plugin ops) → Task 4 (graph layouts)

Chunk 2 (GPU Plugin — Kernel):
  Task 5 (ref.cpp JIT) → Task 6 (kernel .cl) → Task 7 (verify build)
  Depends on: Task 2 (primitive must exist for desc->use_state_table)

Chunk 3 (GenAI — Model Builder):
  Task 8 (genai ops) → Task 9 (config) → Task 10 (model builder)
  Depends on: Task 1 (IR node must support 7-input constructor)

Chunk 4 (Integration):
  Task 11 (sample code) → Task 12 (end-to-end testing)
  Depends on: ALL of Chunks 1-3 (full plugin + model builder must be in place)
```

Within each chunk, tasks are sequential. Cross-chunk dependencies:
- **Chunk 2 starts after Task 2** (needs `use_state_table` field in primitive)
- **Chunk 3 starts after Task 1** (needs 7-input LinearAttention IR constructor)
- **Chunk 4 starts after all 3 chunks complete** (needs working plugin + model)
- Chunks 1, 2, and 3 can partially overlap given the above constraints
