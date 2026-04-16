# Multi-Slot State Table for MTP Batch Verify

**Date**: 2026-03-23
**Status**: Approved (spec review iteration 2)
**Repo**: openvino GPU plugin + openvino.genai
**Supersedes**: 2026-03-20-chunk-gda-batch-verify-design.md (Phase 3 intermediate output approach)

---

## Background

### Problem

MTP speculative decoding generates N draft tokens, then verifies them against the main model. Phase-2 uses sequential verify (M=1 per token) which cannot exploit FFN batch matmul gains. The earlier batched verify (commit `fd3d2f03`) was 5x slower due to:

1. SDPA M>1 slow kernel path (attention_mask `{1, past_len+M}` triggers SDPA fallback)
2. GDA recurrent state contamination on reject requiring save/restore + replay

### Inspiration: vllm's Multi-Slot Architecture

vllm (at `D:\chuansheng\src_code\vllm`) solves GDA state contamination with a fundamentally different approach: **indexed state slot tables**. Instead of saving/restoring states, the fused recurrent kernel writes each token's state to a dedicated pre-allocated slot. After verification, the correct state is already in the table — zero fixup needed.

Key vllm files studied:
- `vllm/model_executor/layers/fla/ops/fused_recurrent.py` — Triton kernel with `ssm_state_indices` per-token slot writes
- `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` — Same pattern with sigmoid gating
- `vllm/model_executor/layers/fla/ops/chunk.py` — Chunked prefill kernel
- `vllm/model_executor/models/qwen3_next.py` — GDA layer with spec decode metadata
- `vllm/v1/attention/backends/gdn_attn.py` — GDN attention metadata builder with `num_accepted_tokens`
- `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` — Multi-slot conv state with sliding window offset

### vllm Design Pattern

```
ssm_state:            [max_slots, HV, V, K]     ← GPU state pool
ssm_state_indices:    [T] int32                  ← maps token position → slot
num_accepted_tokens:  [1] int32                  ← last known-good slot offset (vllm)

vllm kernel loop:
  initial_slot = ssm_state_indices[num_accepted_tokens - 1]
  state = ssm_state[initial_slot]
  for t in 0..T:
      state = recurrent_update(state, q[t], k[t], v[t], ...)
      slot = ssm_state_indices[t]
      if slot >= 0:
          ssm_state[slot] = state        ← write to dedicated slot
  // After verify: correct state already in table. Zero fixup.

OpenVINO simplification (single-sequence, num_accepted_tokens always 1):
  initial_slot = ssm_state_indices[0]    ← always first element
  // num_accepted_tokens eliminated as kernel input
```

---

## Design: Multi-Slot State Table for OpenVINO

### Core Idea

Enlarge the existing LinearAttention Variables from single-slot `[1, HV, V, K]` to multi-slot `[max_slots, HV, V, K]`, and add `ssm_state_indices` as a new kernel input. The LA kernel writes both recurrent AND conv state to indexed slots during the token loop. After batch verify, the accepted state is already in the correct slot — no save/restore, no replay, no intermediate output copy.

**Simplification vs vllm**: vllm passes both `ssm_state_indices` and `num_accepted_tokens` to the kernel to determine the initial state slot. In our single-sequence design, `num_accepted_tokens` is always 1, making it a constant. We eliminate it as a kernel input and always load initial state from `ssm_state_indices[0]`. The sample code places the correct current slot index at position 0 of the indices array.

Conv state tracking is fused into the LA kernel: the kernel maintains a sliding window of raw (pre-conv) input values in `__local` memory and writes conv state snapshots to indexed slots alongside recurrent state.

### Key Invariants

- **Zero fixup on accept/reject** — states already in slots
- **Backward compatible** — when `use_state_table=false`, kernel uses single-slot Variable (current behavior)
- **Periodic reset** — when slot indices approach max_slots, copy current slot to slot 0 (~1-2 ms every ~2 verify rounds)
- **Conv state fused** — no separate save/restore for conv state
- **Prefill safe** — during prefill, `ssm_state_indices` uses PAD_SLOT_ID (-1) for all positions except the last, so only the final state is written to slot 0
- **Cache invalidation** — enabling `use_state_table` changes the IR topology; cached XML/bin files must be deleted when switching modes

---

## Section 1: GPU Plugin Changes

### 1.1 LinearAttention IR Node (`linear_attn.hpp`)

**New attribute**:
```cpp
bool m_use_state_table = false;
```

**New constructor overload** (when `use_state_table=true`):
```cpp
LinearAttention(const ov::OutputVector& args,
                const std::shared_ptr<ov::op::util::Variable>& recurrent_variable,
                const std::shared_ptr<ov::op::util::Variable>& conv_variable,
                bool use_state_table);
```

**New inputs** (inputs 6-8, conditional on `use_state_table`):
```
input[6]: raw_input           [B, T, conv_dim]  — pre-conv projected values
input[7]: conv_initial_state  from ReadValue of conv Variable [max_slots, conv_dim, conv_kernel-1]
input[8]: ssm_state_indices   [T] int32
```
Note: `num_accepted_tokens` is NOT a kernel input — the kernel always loads initial state from `ssm_state_indices[0]` (see Core Idea simplification).

### 1.2 LinearAttention IR Node (`linear_attn.cpp`)

**`visit_attributes`**:
```cpp
visitor.on_attribute("use_state_table", m_use_state_table);
```

**`validate_and_infer_types`**:
```cpp
if (m_use_state_table) {
    NODE_VALIDATION_CHECK(this, get_input_size() == 9, ...);
    // input[6]: raw_input — rank 3, matches seq_len of input[0]
    // input[7]: conv state — rank 3 [max_slots, conv_dim, conv_kernel-1]
    // input[8]: ssm_state_indices — rank 1, int32
} else {
    NODE_VALIDATION_CHECK(this, get_input_size() == 6, ...);
}
// Output[0]: hidden_out (unchanged)
// Output[1]: final recurrent state (unchanged shape = input[5] shape)
// Output[2] (new, conditional): final conv state (shape = input[7] shape)
```

**`output_size()`**: returns 3 when `use_state_table`, 2 otherwise.

**`set_out_type`**: Current code asserts `index < 2`. Update to `index < output_size()` to support output[2] (conv state).

**`clone_with_new_inputs`**: propagate `use_state_table` and both Variable refs.

### 1.3 `cldnn::linear_attention` Primitive (`linear_attention.hpp`)

```cpp
struct linear_attention : public primitive_base<linear_attention> {
    // existing fields ...
    bool use_state_table = false;
    ov::op::util::VariableInfo conv_variable_info;  // when use_state_table
    int conv_kernel_size = 4;                        // for conv state tracking
};
```

**Hash considerations**: `use_state_table` and `conv_kernel_size` must be included in `hash()` — they change kernel compilation. `conv_variable_info` is excluded from `hash()` (runtime state binding, not kernel shape).

### 1.4 Plugin Ops Registration (`plugin/ops/linear_attention.cpp`)

Read `use_state_table` from the IR node. When true:
- Read conv Variable info
- Read conv_kernel_size from model config (or infer from input[7] shape)
- Set on primitive

### 1.5 `linear_attention_ref.cpp` — Implementation Changes

**JIT constants**:
```cpp
jit.make("USE_STATE_TABLE", desc->use_state_table ? 1 : 0);
jit.make("CONV_KERNEL_SIZE", desc->conv_kernel_size);
jit.make("CONV_DIM", desc->conv_dim);               // from input[6] layout
jit.make("STATE_STRIDE", HV * V * K);               // recurrent slot stride (elements)
jit.make("CONV_STATE_STRIDE", conv_dim * (conv_kernel_size - 1));  // conv slot stride (elements)
```

**Argument descriptors** — add input slots 6-8 when USE_STATE_TABLE:
```cpp
if (desc->use_state_table) {
    args.push_back({ArgumentDescriptor::Types::INPUT, 6});  // raw_input
    args.push_back({ArgumentDescriptor::Types::INPUT, 7});  // conv state (from conv variable)
    args.push_back({ArgumentDescriptor::Types::INPUT, 8});  // ssm_state_indices
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 2}); // conv state output
}
```

**Variable binding** (modified):
```cpp
// Recurrent Variable (input[5] / output[1]):
if (recurrent_variable.is_set())
    args.inputs[5] = recurrent_variable.get_memory();  // [max_slots, HV, V, K]
args.outputs[1] = recurrent_variable.get_memory();      // same buffer, in-place writes

// Conv Variable (input[7] / output[2]) — only when use_state_table:
if (desc->use_state_table) {
    if (conv_variable.is_set())
        args.inputs[7] = conv_variable.get_memory();    // [max_slots, conv_dim, conv_kernel-1]
    args.outputs[2] = conv_variable.get_memory();        // same buffer, in-place writes
}
```

### 1.6 `linear_attention_ref.cl` — Kernel Changes

**Stride constants** (set via JIT from Variable shape):
```c
// STATE_STRIDE = HV * V * K  (elements per slot in recurrent state table)
// For 35B-A3B: 64 × 128 × 128 = 1,048,576 elements = 2 MB (bf16)
// CONV_STATE_STRIDE = conv_dim × (conv_kernel - 1)  (elements per slot in conv table)
// For 35B-A3B: 3072 × 3 = 9,216 elements ≈ 18 KB (bf16)
```

**Initial state load (multi-slot aware)**:
```c
#if USE_STATE_TABLE
    // Read initial slot from ssm_state_indices[0] (always current_slot)
    int init_slot = ssm_state_indices[0];
    if (init_slot < 0) {
        // PAD_SLOT_ID: skip initial state load (prefill start — use zeros)
        // init_state[] already zero-initialized
    } else {
        // Index into recurrent state table
        __global INPUT5_TYPE* state_ptr = initial_state + init_slot * STATE_STRIDE;
        // Load into registers (existing code, unchanged)
        for (int iv = 0; iv < V_BLOCK_SIZE; iv++)
            init_state[iv] = load_state(state_ptr, head_idx, iv);
    }
#else
    __global INPUT5_TYPE* state_ptr = initial_state;  // current single-slot
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++)
        init_state[iv] = load_state(state_ptr, head_idx, iv);
#endif
```

**Conv state tracking (new, inside token loop)**:
```c
#if USE_STATE_TABLE
    // Conv sliding window in __local memory (shared across subgroup)
    // Size: conv_dim × (conv_kernel-1) elements
    // For 35B-A3B: 3072 × 3 = 9,216 bf16 = ~18 KB
    // Fits in SLM (64 KB per subgroup on Arc)
    __local INPUT7_TYPE conv_window[CONV_DIM * (CONV_KERNEL_SIZE - 1)];

    // Load initial conv window from indexed slot (cooperative across work items)
    int init_conv_slot = ssm_state_indices[0];  // current_slot
    if (init_conv_slot >= 0) {
        __global INPUT7_TYPE* conv_init_ptr = conv_state + init_conv_slot * CONV_STATE_STRIDE;
        for (int c = get_local_id(0); c < CONV_DIM * (CONV_KERNEL_SIZE - 1); c += get_local_size(0))
            conv_window[c] = conv_init_ptr[c];
    } else {
        for (int c = get_local_id(0); c < CONV_DIM * (CONV_KERNEL_SIZE - 1); c += get_local_size(0))
            conv_window[c] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

for (int i = 0; i < seq_len; i++) {
    // === Existing: load q,k,v,g,beta; update recurrent state; compute output ===
    // ... (unchanged) ...

#if USE_STATE_TABLE
    // === NEW: write recurrent state to indexed slot ===
    int slot = ssm_state_indices[i];
    if (slot >= 0) {
        __global INPUT5_TYPE* slot_ptr = initial_state + slot * STATE_STRIDE;
        for (int iv = 0; iv < V_BLOCK_SIZE; iv++)
            store_state(slot_ptr, head_idx, iv, init_state[iv]);
    }

    // === NEW: update conv sliding window in __local memory ===
    // Shift left: drop oldest column, insert raw_input[i]
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int d = get_local_id(0); d < CONV_DIM; d += get_local_size(0)) {
        for (int c = 0; c < CONV_KERNEL_SIZE - 2; c++)
            conv_window[c * CONV_DIM + d] = conv_window[(c + 1) * CONV_DIM + d];
        conv_window[(CONV_KERNEL_SIZE - 2) * CONV_DIM + d] = raw_input[i * CONV_DIM + d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // === NEW: write conv state to indexed slot ===
    if (slot >= 0) {
        __global INPUT7_TYPE* conv_slot_ptr = conv_state + slot * CONV_STATE_STRIDE;
        for (int c = get_local_id(0); c < CONV_DIM * (CONV_KERNEL_SIZE - 1); c += get_local_size(0))
            conv_slot_ptr[c] = conv_window[c];
    }
#endif
}

#if !USE_STATE_TABLE
    // Current: write final state once (unchanged)
    store_final_state(state_out, head_idx, init_state);
#endif
```

**Conv state memory**: The conv window uses `__local` (SLM) memory. For 35B-A3B: `3072 × 3 = 9,216` bf16 elements ≈ 18 KB. Arc 140T provides 64 KB SLM per subgroup — comfortably fits. Load/store is cooperative across work items in the subgroup for full bandwidth.

### 1.7 Memory Cost

```
Recurrent state table: max_slots × 30 layers × 2 MB
  max_slots=8: 8 × 60 MB = 480 MB (was 60 MB → +420 MB)

Conv state table: max_slots × num_gda_layers × conv_dim × (conv_kernel-1) × 2 bytes
  35B-A3B: ~18 KB/slot/layer; max_slots=8, 30 layers: 8 × 30 × 18 KB ≈ 4.2 MB (negligible)

Total additional memory: ~424 MB for max_slots=8
  On 32 GB system with ~18 GB model: acceptable (~2.6 GB headroom remaining)
```

### 1.8 Write Bandwidth Overhead

Per-token state writes are the main cost. With multi-slot, the kernel writes state after EVERY token (not just the final one), resulting in T× more state bandwidth:

```
Per-token state write per GDA layer: 2 MB (recurrent) + 18 KB (conv) ≈ 2 MB
Current decode (T=1):  1 write × 2 MB × 30 layers =  60 MB
Batch verify (T=4):    4 writes × 2 MB × 30 layers = 240 MB  ← 4× state bandwidth
Extra bandwidth: 180 MB @ 68 GB/s ≈ 2.6 ms

Comparison with Phase 3 (intermediate output approach):
  Phase 3:    2.6 ms (kernel writes) + ~1-2 ms (CPU fixup copy) = ~3.6-4.6 ms
  Multi-slot: 2.6 ms (kernel writes) + 0 ms (zero fixup)        = 2.6 ms
  → Multi-slot saves ~1-2 ms per batch verify pass
```

The 4× bandwidth increase is specific to GDA state writes only — FFN and SDPA bandwidth is unchanged. Since GDA state writes are a small fraction of total inference time (~3 ms out of ~35 ms), the 4× multiplier translates to <3% total overhead.

---

## Section 2: Model Builder Changes (`openvino.genai`)

### 2.1 Config

```cpp
struct Qwen3_5TextModelConfig {
    // existing fields ...
    bool use_state_table = false;   // enables multi-slot indexing
    int max_slots = 8;              // compile-time max (supports up to N=7)
};
```

### 2.2 `ops.hpp/cpp` — New LA Function

```cpp
Tensor linear_attention_with_state_table(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& beta, const Tensor& gate,
    std::shared_ptr<Variable> recurrent_var,     // [max_slots, HV, V, K]
    const Tensor& raw_input,                     // [B, T, conv_dim]
    std::shared_ptr<Variable> conv_var,          // [max_slots, conv_dim, conv_kernel-1]
    const Tensor& ssm_state_indices);            // [T] int32
// Returns: hidden_out [B, T, HV, V]
```

### 2.3 `Qwen3_5GatedDeltaNet::forward()` — Conditional Wiring

```cpp
auto mixed_qkv = ops::linear(x, in_proj_weight, in_proj_bias);

if (cfg_.use_state_table) {
    // Extract current conv state from multi-slot table via Gather
    auto conv_full = ReadValue(conv_var_);   // [max_slots, conv_dim, conv_kernel-1]
    auto zero = Constant::create(i32, {}, {0});
    auto current_slot_idx = ops::gather(ssm_state_indices_, zero, zero);  // scalar
    auto current_conv = ops::gather(conv_full, current_slot_idx, zero);   // [1, conv_dim, k-1]

    // Conv path — uses extracted single-slot conv state
    auto concat = ops::concat({current_conv, mixed_qkv}, seq_dim);
    auto conv_out = ops::group_conv(concat, conv_weight, conv_bias);
    auto activated = ops::silu(conv_out);
    auto [q, k, v, beta, gate] = split_qkvbg(activated);

    // LA handles both recurrent and conv state writes via indexed slots
    auto out = ops::linear_attention_with_state_table(
        q, k, v, beta, gate,
        recurrent_var_, mixed_qkv, conv_var_,
        ssm_state_indices_);
    // NO Assign nodes for recurrent_var_ or conv_var_
    // The kernel writes directly to Variable memory at indexed slots
} else {
    // Current path (backward compatible)
    auto concat = ops::concat({ReadValue(conv_var_), mixed_qkv}, seq_dim);
    auto conv_out = ops::group_conv(concat, conv_weight, conv_bias);
    auto activated = ops::silu(conv_out);
    auto [q, k, v, beta, gate] = split_qkvbg(activated);

    auto [out, final_state] = ops::linear_attention(q, k, v, beta, gate,
                                                     init_state, recurrent_var_);
    // Assign(conv_var_, new_conv_state);
    // Assign(recurrent_var_, final_state);
}
```

**Graph structure when `use_state_table=true`**:
```
mixed_qkv = input_proj(x)
     │
     ├─→ Gather(ReadValue(conv_var), Gather(ssm_state_indices, 0), axis=0)
     │       ↓   [1, conv_dim, conv_kernel-1]  (single-slot extract)
     ├─→ Concat(current_conv, mixed_qkv) → GroupConv → SiLU → split → q,k,v,β,g
     │                                                                     ↓
     └──────────────────────────────────────────────────→ LinearAttention(
                                                            q, k, v, β, g,
                                                            ReadValue(recurrent_var),
                                                            mixed_qkv,
                                                            ReadValue(conv_var),
                                                            ssm_state_indices,
                                                            num_accepted_tokens
                                                          )
Note: GroupConvolution still runs to produce correct q,k,v.
      Conv state is read via Gather (single slot) for GroupConv input.
      Both Assign nodes are REMOVED — LA kernel handles state writes via indexed slots.
```

### 2.4 Model-Level Parameters

One new Parameter shared across all 30 GDA layers:
```cpp
auto ssm_state_indices = ctx().add_parameter("ssm_state_indices",
    ov::element::i32, ov::PartialShape{ov::Dimension::dynamic()});
```

### 2.5 Variable Shape Changes

```cpp
// Recurrent Variable (per GDA layer):
// Before: [1, num_v_heads, head_v_dim, head_k_dim]
// After:  [max_slots, num_v_heads, head_v_dim, head_k_dim]

// Conv Variable (per GDA layer):
// Before: [1, conv_dim, conv_kernel-1]
// After:  [max_slots, conv_dim, conv_kernel-1]
```

### 2.6 Flag Threading

```cpp
// In modeling_qwen3_5.cpp, at model construction:
text_cfg.use_state_table = opts.mtp;  // only in MTP mode
text_cfg.max_slots = opts.mtp_draft_n + 1;  // minimum: N+1 slots
// Round up to next power-of-2 or use fixed 8 for headroom
text_cfg.max_slots = std::max(text_cfg.max_slots, 8);
```

---

## Section 3: Sample Code Changes (`modeling_qwen3_5.cpp`)

### 3.1 Slot Management State

```cpp
int current_slot = 0;                   // which slot holds "current" state
const int max_slots = 8;                // from model config
ov::Tensor ssm_state_indices_tensor;    // [max T]

// Initialization:
ssm_state_indices_tensor = ov::Tensor(ov::element::i32, {1});
ssm_state_indices_tensor.data<int32_t>()[0] = 0;
```

### 3.2 Normal Decode (M=1)

```cpp
ssm_state_indices_tensor.set_shape({1});
ssm_state_indices_tensor.data<int32_t>()[0] = current_slot;
text_request.set_tensor("ssm_state_indices", ssm_state_indices_tensor);
text_request.infer();
// Kernel reads from slot current_slot, writes to slot current_slot
```

### 3.3 Batch Verify (M = N+1)

```cpp
// [1] SET UP INDICES
ssm_state_indices_tensor.set_shape({N + 1});
for (int t = 0; t <= N; t++)
    ssm_state_indices_tensor.data<int32_t>()[t] = current_slot + t;

// [2] RUN BATCH VERIFY
set_input_ids({next_id, drafts[0], ..., drafts[N-1]});
text_request.set_tensor("ssm_state_indices", ssm_state_indices_tensor);
text_request.infer();
// Kernel writes recurrent+conv state to slots [current_slot..current_slot+N]

// [3] DETERMINE ACCEPT INDEX
auto refs = get_logits_argmax();  // [N+1] reference tokens
int j = N;
for (int k = 0; k < N; ++k)
    if (refs[k] != drafts[k]) { j = k; break; }

// [4] ADVANCE SLOT — ZERO FIXUP
current_slot += j;
// Correct recurrent AND conv states are ALREADY in slot current_slot

// [5] PERIODIC RESET
if (current_slot + N + 1 > max_slots) {
    copy_slot_to_zero(current_slot);  // ~1-2 ms (CPU memcpy), every ~2 verify rounds
    current_slot = 0;
}

// [6] TRIM KV CACHE (existing, unchanged)
if (j < N) trim_main_kv_cache(N - j);

// [7] EMIT TOKENS
emit(next_id_corrected);
for (int k = 0; k < j; k++) emit(drafts[k]);
emit(refs[j]);  // bonus token
```

### 3.4 Slot Copy Helper

```cpp
void copy_slot_to_zero(int src_slot) {
    if (src_slot == 0) return;
    auto var_states = text_request.query_state();
    for (auto& vs : var_states) {
        const auto& name = vs.get_name();
        if (name.find("linear_states") == std::string::npos) continue;
        // Covers both ".recurrent" and ".conv" variables

        ov::Tensor full_state = vs.get_state();  // [max_slots, ...]
        auto shape = full_state.get_shape();
        size_t slot_bytes = full_state.get_element_type().size();
        for (size_t d = 1; d < shape.size(); d++) slot_bytes *= shape[d];

        auto* base = static_cast<char*>(full_state.data());
        std::memcpy(base, base + src_slot * slot_bytes, slot_bytes);
        vs.set_state(full_state);
    }
}
```

**Reset frequency**: with max_slots=8, N=3: reset every ~2 verify rounds.
At ~35 ms per verify, the ~1-2 ms copy (CPU memcpy via `get_state()`/`set_state()` with GPU sync) is ~2-3% overhead.

### 3.5 Prefill Handling

During prefill, the prompt has T tokens but we only want to write the final recurrent/conv state (after processing the whole prompt) to slot 0. Intermediate positions use PAD_SLOT_ID (-1) to suppress writes.

```cpp
// Prefill: T = prompt_length
ssm_state_indices_tensor.set_shape({prompt_length});
for (int t = 0; t < prompt_length - 1; t++)
    ssm_state_indices_tensor.data<int32_t>()[t] = -1;  // PAD_SLOT_ID
ssm_state_indices_tensor.data<int32_t>()[prompt_length - 1] = 0;  // write final state to slot 0

text_request.set_tensor("ssm_state_indices", ssm_state_indices_tensor);
text_request.infer();
current_slot = 0;  // state is now in slot 0
```

**Why PAD_SLOT_ID for intermediate positions**: Writing per-token states during prefill would be wasteful (T could be 1K+ tokens) and would exhaust `max_slots`. The kernel checks `if (slot >= 0)` before each write, so PAD_SLOT_ID positions are free.

**Initial state during prefill**: For the first infer() call, `ssm_state_indices[0] = -1` (PAD_SLOT_ID). The kernel detects this and skips the initial state load, starting from zeros. This matches current prefill behavior where the Variable is zero-initialized.

### 3.6 MTP Draft — No Change

The MTP draft runner has its own separate model and state. `ssm_state_indices` and `num_accepted_tokens` only affect the main model. MTP draft generation is unchanged.

---

## Section 4: Data Flow (N=3 Example)

```
Input: [next_id, draft[0], draft[1], draft[2]]  T=4
Slots: ssm_state_indices = [0, 1, 2, 3]
Load from: ssm_state_indices[0] = slot 0

                    ONE infer() call
     ┌────────────────────────────────────────────────────────────┐
     │ GroupConv: 4 tokens → correct q,k,v,β,g (unchanged)       │
     │                                                            │
     │ LA kernel token loop:                                      │
     │   Token 0 (next_id):                                       │
     │     recurrent: S0 → S1    conv: C0 → C1                   │
     │     Write: recurrent_var[0]=S1, conv_var[0]=C1             │
     │                                                            │
     │   Token 1 (draft[0]):                                      │
     │     recurrent: S1 → S2    conv: C1 → C2                   │
     │     Write: recurrent_var[1]=S2, conv_var[1]=C2             │
     │                                                            │
     │   Token 2 (draft[1]):                                      │
     │     recurrent: S2 → S3    conv: C2 → C3                   │
     │     Write: recurrent_var[2]=S3, conv_var[2]=C3             │
     │                                                            │
     │   Token 3 (draft[2]):                                      │
     │     recurrent: S3 → S4    conv: C3 → C4                   │
     │     Write: recurrent_var[3]=S4, conv_var[3]=C4             │
     │                                                            │
     │ KV cache: past_len → past_len+4                            │
     │ Output: refs[0..3] = logits argmax                         │
     └────────────────────────────────────────────────────────────┘

j=0 (all reject):
  current_slot = 0
  recurrent_var[0]=S1 ✓  conv_var[0]=C1 ✓
  KV trim 3, emit: refs[0]

j=1 (accept draft[0]):
  current_slot = 1
  recurrent_var[1]=S2 ✓  conv_var[1]=C2 ✓
  KV trim 2, emit: draft[0], refs[1]

j=2 (accept draft[0,1]):
  current_slot = 2
  recurrent_var[2]=S3 ✓  conv_var[2]=C3 ✓
  KV trim 1, emit: draft[0], draft[1], refs[2]

j=3 (full accept):
  current_slot = 3
  recurrent_var[3]=S4 ✓  conv_var[3]=C4 ✓
  KV trim 0, emit: draft[0], draft[1], draft[2], refs[3]
```

---

## Section 5: File Change Summary

### openvino GPU plugin repo

| File | Lines | Description |
|---|---|---|
| `src/core/dev_api/openvino/op/linear_attn.hpp` | +15 | `use_state_table` attr, conv variable ref, new constructor |
| `src/core/src/op/linear_attn.cpp` | +50 | `visit_attributes`, `validate_and_infer_types` for 4 new inputs, output[2], `set_out_type` bounds fix |
| `src/plugins/intel_gpu/include/intel_gpu/primitives/linear_attention.hpp` | +8 | `use_state_table`, conv variable info, conv_kernel_size |
| `src/plugins/intel_gpu/src/plugin/ops/linear_attention.cpp` | +15 | Read new attrs, propagate, `calc_output_layouts` for 9-input + 3-output case |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/linear_attention_ref.cpp` | +35 | JIT constants, arg descriptors 6-9, conv variable binding |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/linear_attention_ref.cl` | +50 | Indexed initial load, per-token dual-state write, conv window tracking |
| `src/plugins/intel_gpu/src/graph/linear_attention.cpp` | +5 | Conv variable dependency registration |
| `src/plugins/intel_gpu/src/graph/include/linear_attention_inst.h` | +3 | Conv variable storage |

**GPU plugin total**: ~178 lines across 8 files

### openvino.genai repo

| File | Lines | Description |
|---|---|---|
| `src/cpp/src/modeling/ops/ops.hpp` | +12 | `linear_attention_with_state_table` declaration |
| `src/cpp/src/modeling/ops/ops.cpp` | +30 | Implementation: construct LA node with 10 inputs, 3 outputs |
| `src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_text.hpp` | +3 | Config flags |
| `src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_text.cpp` | +45 | Conditional wiring: Gather for conv slot extract, raw_input routing, skip Assign, variable shapes |
| `src/cpp/src/modeling/samples/modeling_qwen3_5.cpp` | +55 | Slot management, batch verify indices, copy_slot_to_zero, reset |

**genai total**: ~145 lines across 5 files

**Grand total**: ~323 lines across 13 files in 2 repos

---

## Section 6: Performance Projection

### 35B-A3B, N=3, greedy (Arc 140T)

```
Current sequential verify (Phase 2):
  avg 2.0 main steps/super-step (32% per-token accept)
  throughput: ~20.9 tok/s

Batch verify with multi-slot state table (this design):
  1× batch infer (M=4, FFN batching gain ~1.6×)
  + per-token state writes: ~2.6 ms (180 MB @ 68 GB/s)
  + periodic reset: ~0.45 ms amortized
  + zero fixup overhead
  Net projection: ~25-33% improvement → ~26-28 tok/s

Comparison with Phase 3 (intermediate output approach):
  Phase 3: 2.6 ms (kernel writes) + 0.9 ms (fixup copy) = 3.5 ms overhead
  Multi-slot: 2.6 ms (kernel writes) + 0 ms = 2.6 ms overhead
  → Multi-slot saves ~0.9 ms per verify pass

Risk: SDPA M>1 slow path
  micro-SDPA for M>1 still falls back to slow kernel (CAUSAL_KV_OFFSET not supported)
  But FFN matmul batching gain is the primary speedup on weight-bound iGPU
  SDPA fix is a separate work item (independent of this design)
```

---

## Section 7: Out of Scope

- TensorIterator path support (Arc 140T uses fused LA op)
- MTP module state table (draft model has its own state, not shared)
- N>1 batched MTP draft generation
- micro-SDPA M>1 CAUSAL_KV_OFFSET support (separate work item)
- Discrete GPU support (USM host allocation differences)
- Prefix caching / multi-sequence batching (single sequence only)
