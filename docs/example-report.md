# Example Report using command `diffinator -c llamacpp --output markdown b4273 b4418`

Analyzing changes between b4273 and b4418:

## Summary
- Total commits: 145
- Files changed: 300
- Important files modified: 5

## Commits by Type

### LLAMA-RUN
- llama-run: improve progress bar (#10821)
- llama-run: include temperature option (#10899)

### LLAMA
- llama: add 128k yarn context for Qwen (#10698)
- llama: use cmake for swift build (#10525)
- llama: add Qwen2VL support + multimodal RoPE (#10361)
- llama: add Deepseek MoE v1 & GigaChat models (#10827)
- llama: add Falcon3 support (#10864)
- llama: fix Roberta embeddings (#10856)
- llama: minor grammar refactor (#10897)
- llama: add Falcon3 support (#10883)
- llama: support for Llama-3_1-Nemotron-51B (#10669)
- llama: support InfiniAI Megrez 3b (#10893)
- llama: the WPM vocabs use the CLS token as BOS (#10930)
- llama: refactor `src/llama.cpp` (#10902)
- llama: add support for the cohere2 model architecture (#10900)
- llama: Add support for DeepSeek V3 (#11049)

### COMMON
- common: bring back --no-warmup to server (#10686)
- common: add missing env var for speculative (#10801)
- common: improve -ctv -ctk CLI arguments (#10806)
- common, examples, ggml: fix MSYS2 GCC compiler errors and warnings when building with LLAMA_CURL=ON and GGML_OPENCL=ON (#11013)
- common: disable KV cache shifting automatically for unsupported models (#11053)

### CONVERT
- convert: add custom attention mapping
- convert: add support for Roberta embeddings (#10695)
- convert: Add support for Microsoft Phi-4 model  (#10817)
- convert: fix RWKV v6 model conversion (#10913)
- convert: add BertForMaskedLM (#10919)
- convert: fix Llama-3_1-Nemotron-51B rope settings (#11008)

### SERVER
- server: (refactoring) do not rely on JSON internally (#10643)
- server: fix free of spec context and batch (#10651)
- server: various fixes (#10704)
- server: (refactor) no more json in server_task input (#10691)
- server: bring back info of final chunk in stream mode (#10722)
- server: fix format_infill (#10724)
- server: add flag to disable the web-ui (#10762) (#10751)
- server: (UI) add tok/s, get rid of completion.js (#10786)
- server: Fix `has_next_line` in JSON response (#10818)
- server: (UI) add syntax highlighting and latex math rendering (#10808)
- server: (UI) fix missing async generator on safari (#10857)
- server: fill usage info in embeddings and rerank responses (#10852)
- server: (embeddings) using same format for "input" and "content" (#10872)
- server: add "tokens" output (#10853)
- server: output embeddings for all tokens when pooling = none (#10861)
- server: avoid overwriting Authorization header (#10878)
- server: fix logprobs, make it OAI-compatible (#10783)
- server: (UI) fix copy to clipboard function (#10916)
- server: add system_fingerprint to chat/completion (#10917)
- server: fix missing model id in /model endpoint (#10957)
- server: allow filtering llama server response fields (#10940)
- server: add support for "encoding_format": "base64" to the */embeddings endpoints (#10967)
- server: fix token duplication when streaming with stop strings (#10997)
- server: added more docs for response_fields field (#10995)
- server: add OAI compat for /v1/completions (#10974)
- server: clean up built-in template detection (#11026)
- server: allow using LoRA adapters per-request (#10994)
- server: bench: minor fixes (#10765)

### RPC-SERVER
- rpc-server: add support for the SYCL backend (#10934)

### GGUF
- gguf-py: bump version to 0.11.0
- gguf-py: numpy 2 newbyteorder fix (#9772)
- gguf-py: bump to v0.13.0

### GGML
- ggml: refactor online repacking (#10446)
- ggml: disable iq4_nl interleave size 8 (#10709)
- ggml: load all backends from a user-provided search path (#10699)
- ggml: Fix compilation issues on ARM platform when building without fp16 (#10811)
- ggml: update ggml_backend_cpu_device_supports_op (#10867)
- ggml: add check for grad_accs (ggml/1046)
- ggml: remove return from ggml_gallocr_allocate_node (ggml/1048)
- ggml: fix arm build (#10890)
- ggml: fix arm build with gcc (#10895)
- ggml: add test for SVE and disable when it fails (#10906)
- ggml-cpu: replace NEON asm with intrinsics in ggml_gemv_q4_0_4x8_q8_0() (#10874)
- ggml: fix run-time on FreeBSD in get_executable_path() (#10948)
- ggml: fix const usage in SSE path (#10962)
- ggml: fix arm enabled features check (#10961)
- ggml: use wstring for backend search paths (#10960)
- ggml: more perfo with llamafile tinyblas on x86_64 (#10714)
- ggml: fixes for AVXVNNI instruction set with MSVC and Clang (#11027)
- ggml: improve inputs log sched_print_assignments (ggml/1053)
- ggml: do not install metal source when embed library (ggml/1054)

### BUG FIXES
- fix(server): not show alert when DONE is received (#10674)
- bug-fix: snprintf prints NULL in place of the last character (#10419)
- Fix crash caused by ggml_backend_load_all when launching on Android Activity (#10812)
- fix: graceful shutdown for Docker images (#10815)
- fix: Vulkan shader gen binary path (#11037)

### VULKAN
- Vulkan: VK_KHR_cooperative_matrix support to speed up prompt processing (#10597)
- vulkan: compile a test shader in cmake to check for coopmat2 support (#10713)
- Vulkan: fix NaN in tanh.comp with AMD proprietary driver on Windows (#10723)
- vulkan: fix compile warnings (#10731)
- vulkan: disable spirv-opt for coopmat shaders (#10763)
- vulkan: dynamic subgroup size for the remaining k quants (#10745)
- vulkan: request round-to-even for fp16 in im2col/rope_head (#10767)
- Vulkan: Add VK_EXT_subgroup_size_control support to ensure full subgroups for coopmats (#10721)
- Vulkan: Use improved q4_k and q5_k dequant code in dequant shaders (#10798)
- vulkan: small mul_mat_vec optimizations (#10665)
- vulkan: bugfixes for small subgroup size systems + llvmpipe test (#10809)
- vulkan: fix soft_max.comp division by zero (whisper/2633)
- vulkan: optimize coopmat2 dequant functions (#10855)
- vulkan: build fixes for 32b (#10927)
- vulkan: multi-row k quants (#10846)
- vulkan: Use push constant offset to handle misaligned descriptors (#10987)
- vulkan: im2col and matmul optimizations for stable diffusion (#10942)
- vulkan: optimize mul_mat for small values of N (#10991)
- Vulkan: Add device-specific blacklist for coopmat for the AMD proprietary driver (#11074)

### SYSCL
- SYCL: Reduce most of the compiler warnings (#10748)
- SYCL: Migrate away from deprecated ggml_tensor->backend (#10840)

### CUDA
- CUDA: fix shared memory access condition for mmv (#10740)
- CUDA: rename macros to avoid conflicts with WinAPI (#10736)
- CUDA: faster non-contiguous concat (#10760)

### METAL
- metal: Extend how Llama.cpp locates metal resources (#10676)
- metal: avoid uint (#11019)

### SCRIPTS
- scripts: change build path to "build-bench" for compare-commits.sh (#10836)

### LLAVA
- llava: Allow locally downloaded models for QwenVL (#10833)

### BUILD SYSTEM
- cmake: simplify msvc charsets (#10672)
- ci: pin nodejs to 22.11.0 (#10779)
- nix: allow to override rocm gpu targets (#10794)
- cmake: fix "amd64" processor string (whisper/2638)

### TESTS
- tests: add tests for GGUF (#10830)
- tests: disable GGUF test for bad value size (#10886)

### DOCUMENTATION
- docs: fix server documentation formatting (#10776)
- docs: update server streaming mode documentation (#9519)
- readme: update typos (#10863)
- docs: Fix HIP (née hipBLAS) in README (#10880)
- readme: add llama-swap to infrastructure section (#11032)

### OTHER
- Changes to CMakePresets.json to add ninja clang target on windows (#10668)
- imatrix: Add imatrix to --no-context-shift (#10766)
- Update README.md (#10772)
- Merge pull request #10788 from ggerganov/gg/gguf-py-0.11.0
- remove CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS (#10797)
- contrib: add ngxson as codeowner (#10804)
- Opt class for positional argument handling (#10508)
- Introducing experimental OpenCL backend with support for Qualcomm Adreno GPUs (#10693)
- Removes spurious \r in output that causes logging in journalctl to treat lines as binary and therefore hidden by default (#10771)
- sampling: refactor + optimize penalties sampler (#10803)
- unicode: improve naming style (#10838)
- rwkv6: add wkv6 support for Vulkan backend (#10829)
- sync: ggml
- Use model->gguf_kv for loading the template instead of using the C API. (#10868)
- Revert "llama: add Falcon3 support (#10864)" (#10876)
- tts: add OuteTTS support (#10784)
- tts: small QoL for easy model fetch (#10903)
- clip: disable GPU support (#10896)
- devops: add docker-multi-stage builds (#10832)
- examples, ggml: fix GCC compiler warnings (#10983)
- android: fix llama_batch free (#11014)
- sync: ggml
- [GGML][RPC] Support for models with non-512-aligned tensors over RPC. (#11047)

## Important File Changes

### CMakeLists.txt
Status: modified | +3/-5

```diff
@@ -46,11 +46,9 @@ if (WIN32)
     add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
 endif()
 
-if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
-    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/source-charset:utf-8>")
-    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/source-charset:utf-8>")
-    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/execution-charset:utf-8>")
-    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/execution-charset:utf-8>")
+if (MSVC)
+    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
+    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
 endif()
 
 #
```


### CMakePresets.json
Status: modified | +12/-0

```diff
@@ -31,6 +31,13 @@
     { "name": "sycl_f16", "hidden": true, "cacheVariables": { "GGML_SYCL_F16":    "ON" } },
     { "name": "vulkan",   "hidden": true, "cacheVariables": { "GGML_VULKAN":      "ON" } },
 
+    {
+        "name": "x64-windows-llvm", "hidden": true,
+        "cacheVariables": {
+            "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/cmake/x64-windows-llvm.cmake"
+        }
+    },
+
     {
         "name": "arm64-windows-msvc", "hidden": true,
         "architecture": { "value": "arm64",    "strategy": "external" },
@@ -70,6 +77,11 @@
     { "name": "arm64-windows-msvc-release", "inherits": [ "base", "arm64-windows-msvc",  "reldbg" ] },
     { "name": "arm64-windows-msvc+static-release", "inherits": [ "base", "arm64-windows-msvc",  "reldbg", "static" ] },
 
+    { "name": "x64-windows-llvm-debug", "inherits": [ "base", "x64-windows-llvm", "debug" ] },
+    { "name": "x64-windows-llvm-release", "inherits": [ "base", "x64-windows-llvm", "release" ] },
+    { "name": "x64-windows-llvm-reldbg", "inherits": [ "base", "x64-windows-llvm", "reldbg" ] },
+    { "name": "x64-windows-llvm+static-release", "inherits": [ "base", "x64-windows-llvm", "reldbg", "static" ] },
+
     { "name": "x64-windows-msvc-debug", "inherits": [ "base", "debug" ] },
     { "name": "x64-windows-msvc-release", "inherits": [ "base", "reldbg" ] },
     { "name": "x64-windows-msvc+static-release", "inherits": [ "base", "reldbg", "static" ] },
```


### ggml/src/ggml-cpu/ggml-cpu.c
Status: modified | +383/-348

```diff
@@ -3,7 +3,7 @@
 
 #include "ggml-backend-impl.h"
 #include "ggml-backend.h"
-#include "ggml-cpu-aarch64.h"
+#include "ggml-cpu-traits.h"
 #include "ggml-cpu-impl.h"
 #include "ggml-cpu.h"
 #include "ggml-impl.h"
@@ -126,8 +126,7 @@ struct ggml_arm_arch_features_type {
 #endif
 #include <windows.h>
 
-
-#if !defined(__clang__)
+#if defined(_MSC_VER) && !defined(__clang__)
 #define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))
 
 typedef volatile LONG atomic_int;
@@ -224,10 +223,6 @@ typedef void * thread_ret_t;
 
 typedef pthread_t ggml_thread_t;
 
-#ifdef GGML_USE_CPU_HBM
-#include <hbwmalloc.h>
-#endif
-
 #if defined(__APPLE__)
 #include <unistd.h>
 #include <mach/mach.h>
@@ -301,7 +296,6 @@ static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
     },
     [GGML_TYPE_Q8_0] = {
         .from_float               = quantize_row_q8_0,
-        .from_float_to_mat        = quantize_mat_q8_0,
         .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
         .vec_dot_type             = GGML_TYPE_Q8_0,
 #if defined (__ARM_FEATURE_MATMUL_INT8)
@@ -409,33 +403,6 @@ static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
         .vec_dot_type             = GGML_TYPE_BF16,
         .nrows                    = 1,
     },
-    [GGML_TYPE_Q4_0_4_4] = {
-        .from_float               = NULL,
-        .vec_dot                  = NULL,
-        .vec_dot_type             = GGML_TYPE_Q8_0,
-        .nrows                    = 1,
-        .ncols                    = 4,
-        .gemv                     = ggml_gemv_q4_0_4x4_q8_0,
-        .gemm                     = ggml_gemm_q4_0_4x4_q8_0,
-    },
-    [GGML_TYPE_Q4_0_4_8] = {
-        .from_float               = NULL,
-        .vec_dot                  = NULL,
-        .vec_dot_type             = GGML_TYPE_Q8_0,
-        .nrows                    = 1,
-        .ncols                    = 4,
-        .gemv                     = ggml_gemv_q4_0_4x8_q8_0,
-        .gemm                     = ggml_gemm_q4_0_4x8_q8_0,
-    },
-    [GGML_TYPE_Q4_0_8_8] = {
-        .from_float               = NULL,
-        .vec_dot                  = NULL,
-        .vec_dot_type             = GGML_TYPE_Q8_0,
-        .nrows                    = 1,
-        .ncols                    = 8,
-        .gemv                     = ggml_gemv_q4_0_8x8_q8_0,
-        .gemm                     = ggml_gemm_q4_0_8x8_q8_0,
-    },
     [GGML_TYPE_TQ1_0] = {
         .from_float               = quantize_row_tq1_0,
         .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
@@ -448,15 +415,6 @@ static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
         .vec_dot_type             = GGML_TYPE_Q8_K,
         .nrows                    = 1,
     },
-    [GGML_TYPE_IQ4_NL_4_4] = {
-        .from_float               = NULL,
-        .vec_dot                  = NULL,
-        .vec_dot_type             = GGML_TYPE_Q8_0,
-        .nrows                    = 1,
-        .ncols                    = 4,
-        .gemv                     = ggml_gemv_iq4_nl_4x4_q8_0,
-        .gemm                     = ggml_gemm_iq4_nl_4x4_q8_0,
-    },
 };
 
 const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
@@ -496,21 +454,21 @@ const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type
 #define GGML_F32x4_ADD          vaddq_f32
 #define GGML_F32x4_MUL          vmulq_f32
 #define GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
-#define GGML_F32x4_REDUCE(res, x)                  \
-{                                                  \
-    int offset = GGML_F32_ARR >> 1;                \
-    for (int i = 0; i < offset; ++i) {             \
-        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]); \
-    }                                              \
-    offset >>= 1;                                  \
-    for (int i = 0; i < offset; ++i) {             \
-        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]); \
-    }                                              \
-    offset >>= 1;                                  \
-    for (int i = 0; i < offset; ++i) {             \
-        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]); \
-    }                                              \
-    (res) = GGML_F32x4_REDUCE_ONE((x)[0]);         \
+#define GGML_F32x4_REDUCE(res, x)                       \
+{                                                       \
+    int offset = GGML_F32_ARR >> 1;                     \
+    for (int i = 0; i < offset; ++i) {                  \
+        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
+    }                                                   \
+    offset >>= 1;                                       \
+    for (int i = 0; i < offset; ++i) {                  \
+        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
+    }                                                   \
+    offset >>= 1;                                       \
+    for (int i = 0; i < offset; ++i) {                  \
+        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
+    }                                                   \
+    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
 }
 
 #define GGML_F32_VEC        GGML_F32x4
@@ -1028,7 +986,7 @@ inline static void __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
 #define GGML_F16_STEP 32
 #define GGML_F16_EPR  4
 
-static inline __m128 __sse_f16x4_load(ggml_fp16_t *x) {
+static inline __m128 __sse_f16x4_load(const ggml_fp16_t * x) {
     float tmp[4];
 
     tmp[0] = GGML_FP16_TO_FP32(x[0]);
@@ -1039,7 +997,7 @@ static inline __m128 __sse_f16x4_load(ggml_fp16_t *x) {
     return _mm_loadu_ps(tmp);
 }
 
-static inline void __sse_f16x4_store(ggml_fp16_t *x, __m128 y) {
+static inline void __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
     float arr[4];
 
     _mm_storeu_ps(arr, y);
@@ -2437,7 +2395,7 @@ static void ggml_init_arm_arch_features(void) {
     uint32_t hwcap2 = getauxval(AT_HWCAP2);
 
     ggml_arm_arch_features.has_neon = !!(hwcap & HWCAP_ASIMD);
-    ggml_arm_arch_features.has_dotprod = !!(hwcap && HWCAP_ASIMDDP);
+    ggml_arm_arch_features.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
     ggml_arm_arch_features.has_i8mm = !!(hwcap2 & HWCAP2_I8MM);
     ggml_arm_arch_features.has_sve  = !!(hwcap & HWCAP_SVE);
 
@@ -4509,9 +4467,6 @@ static void ggml_compute_forward_add(
         case GGML_TYPE_IQ4_XS:
         case GGML_TYPE_IQ3_S:
         case GGML_TYPE_IQ2_S:
-        case GGML_TYPE_Q4_0_4_4:
-        case GGML_TYPE_Q4_0_4_8:
-        case GGML_TYPE_Q4_0_8_8:
             {
                 ggml_compute_forward_add_q_f32(params, dst);
             } break;
@@ -4889,9 +4844,6 @@ static void ggml_compute_forward_add1(
         case GGML_TYPE_IQ4_XS:
         case GGML_TYPE_IQ3_S:
         case GGML_TYPE_IQ2_S:
-        case GGML_TYPE_Q4_0_4_4:
-        case GGML_TYPE_Q4_0_4_8:
-        case GGML_TYPE_Q4_0_8_8:
             {
                 ggml_compute_forward_add1_q_f32(params, dst);
             } break;
@@ -5019,9 +4971,6 @@ static void ggml_compute_forward_acc(
         case GGML_TYPE_IQ4_XS:
         case GGML_TYPE_IQ3_S:
         case GGML_TYPE_IQ2_S:
-        case GGML_TYPE_Q4_0_4_4:
-        case GGML_TYPE_Q4_0_4_8:
-        case GGML_TYPE_Q4_0_8_8:
         default:
             {
                 GGML_ABORT("fatal error");
@@ -7437,35 +7386,17 @@ static void ggml_compute_forward_mul_mat(
     const int ith = params->ith;
     const int nth = params->nth;
 
-    enum ggml_type type = src0->type;
-
-    if (src0->buffer && ggml_backend_cpu_buft_is_aarch64(src0->buffer->buft)) {
-        type = (enum ggml_type)(intptr_t)src0->extra;
-    }
-
-#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
-    if (src0->buffer && ggml_backend_amx_buft_is_amx(src0->buffer->buft)) {
-        ggml_backend_amx_mul_mat(params, dst);
-        return;
-    }
-#endif
-
-    enum ggml_type           const vec_dot_type         = type_traits_cpu[type].vec_dot_type;
+    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
     ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
-    ggml_from_float_to_mat_t const from_float_to_mat    = type_traits_cpu[vec_dot_type].from_float_to_mat;
-    int64_t                  const vec_dot_num_rows     = type_traits_cpu[type].nrows;
-    int64_t                  const matmul_num_cols      = type_traits_cpu[type].ncols;
-    int64_t                  const blck_size_interleave = ggml_get_type_traits(type)->blck_size_interleave;
-    ggml_gemv_t              const gemv                 = type_traits_cpu[type].gemv;
-    ggml_gemm_t              const gemm                 = type_traits_cpu[type].gemm;
+    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;
 
     GGML_ASSERT(ne0 == ne01);
     GGML_ASSERT(ne1 == ne11);
     GGML_ASSERT(ne2 == ne12);
     GGML_ASSERT(ne3 == ne13);
 
     // we don't support permuted src0 or src1
-    GGML_ASSERT(nb00 == ggml_type_size(type));
+    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
     GGML_ASSERT(nb10 == ggml_type_size(src1->type));
 
     // dst cannot be transposed or permuted
@@ -7477,6 +7408,7 @@ static void ggml_compute_forward_mul_mat(
     // nb01 >= nb00 - src0 is not transposed
     //   compute by src0 rows
 
+    // TODO: extract to "extra_op"
 #if GGML_USE_LLAMAFILE
     // broadcast factors
     const int64_t r2 = ne12 / ne02;
@@ -7487,15 +7419,15 @@ static void ggml_compute_forward_mul_mat(
     if (src1_cont) {
         for (int64_t i13 = 0; i13 < ne13; i13++)
             for (int64_t i12 = 0; i12 < ne12; i12++)
-                if (!llamafile_sgemm(ne01, ne11, ne00/ggml_blck_size(type),
+                if (!llamafile_sgemm(params,
+                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                      (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
-                                     nb01/ggml_type_size(type),
+                                     nb01/ggml_type_size(src0->type),
                                      (const char *)src1->data + i12*nb12 + i13*nb13,
                                      nb11/ggml_type_size(src1->type),
                                      (char *)dst->data + i12*nb2 + i13*nb3,
                                      nb1/ggml_type_size(dst->type),
-                                     ith, nth,
-                                     type,
+                                     src0->type,
                                      src1->type,
                                      dst->type))
                     goto UseGgmlGemm1;
@@ -7516,19 +7448,10 @@ UseGgmlGemm1:;
 
         for (int64_t i13 = 0; i13 < ne13; ++i13) {
             for (int64_t i12 = 0; i12 < ne12; ++i12) {
-                int64_t i11_processed = 0;
-                if ((ggml_n_dims(src1) == 2) && from_float_to_mat && gemm) {
-                    for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
-                        from_float_to_mat((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
-                                          (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
-                                          4, ne10, blck_size_interleave);
-                    }
-                    i11_processed = ne11 - ne11 % 4;
-                }
-                for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
+                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                     from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
-                           (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
-                           ne10);
+                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
+                                ne10);
                 }
             }
         }
@@ -7548,15 +7471,15 @@ UseGgmlGemm1:;
 
         for (int64_t i13 = 0; i13 < ne13; i13++)
             for (int64_t i12 = 0; i12 < ne12; i12++)
-                if (!llamafile_sgemm(ne01, ne11, ne00/ggml_blck_size(type),
+                if (!llamafile_sgemm(params,
+                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                      (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
-                                     nb01/ggml_type_size(type),
+                                     nb01/ggml_type_size(src0->type),
                                      (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                      row_size/ggml_type_size(vec_dot_type),
                                      (char *)dst->data + i12*nb2 + i13*nb3,
                                      nb1/ggml_type_size(dst->type),
-                                     ith, nth,
-                                     type,
+                                     src0->type,
                                      vec_dot_type,
                                      dst->type))
                     goto UseGgmlGemm2;
@@ -7598,28 +7521,6 @@ UseGgmlGemm2:;
     const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
     const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;
 
-    if ((ggml_n_dims(src0) == 2) && gemv) {
-        const void * src1_wdata      = (src1->type == vec_dot_type) ? src1->data : params->wdata;
-        const size_t src1_col_stride = ggml_is_contiguous(src1) || src1->type != vec_dot_type ? ggml_row_size(vec_dot_type, ne10) : nb11;
-        int64_t src0_start = (ith * ne01) / nth;
-        int64_t src0_end   = ((ith + 1) * ne01) / nth;
-        src0_start = (src0_start % matmul_num_cols) ? src0_start + matmul_num_cols - (src0_start % matmul_num_cols): src0_start;
-        src0_end   = (src0_end   % matmul_num_cols) ? src0_end   + matmul_num_cols - (src0_end   % matmul_num_cols): src0_end;
-        if (src0_start >= src0_end) return;
-
-        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
-        if (gemm && (ne11 > 3)) {
-            gemm(ne00, (float *)((char *) dst->data) + src0_start, ne01, (const char *) src0->data + src0_start * nb01,
-                 (const char *) src1_wdata, ne11 - ne11 % 4, src0_end - src0_start);
-        }
-        for (int iter = gemm ? ne11 - ne11 % 4 : 0; iter < ne11; iter++) {
-            gemv(ne00, (float *)((char *) dst->data + (iter * nb1)) + src0_start, ne01,
-                 (const char *) src0->data + src0_start * nb01, (const char *) src1_wdata + (src1_col_stride * iter), 1,
-                 src0_end - src0_start);
-        }
-        return;
-    }
-
     // The first chunk comes from our thread_id, the rest will get auto-assigned.
     int current_chunk = ith;
 
@@ -7642,7 +7543,7 @@ UseGgmlGemm2:;
             num_rows_per_vec_dot = 1;
         }
 
-        ggml_compute_forward_mul_mat_one_chunk(params, dst, type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);
+        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);
 
         if (nth >= nchunk0 * nchunk1) {
             break;
@@ -7674,8 +7575,6 @@ static void ggml_compute_forward_mul_mat_id(
     ggml_vec_dot_t    const vec_dot         = type_traits_cpu[type].vec_dot;
     enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
     ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;
-    int64_t           const matmul_num_cols = type_traits_cpu[type].ncols;
-    ggml_gemv_t       const gemv            = type_traits_cpu[type].gemv;
 
     // we don't support permuted src0 or src1
     GGML_ASSERT(nb00 == ggml_type_size(type));
@@ -7761,34 +7660,6 @@ static void ggml_compute_forward_mul_mat_id(
         const int64_t nr0 = ne01; // src0 rows
         const int64_t nr1 = cne1; // src1 rows
 
-        if (((ggml_n_dims(src0) - 1) == 2) && gemv) {
-            int64_t src0_cur_start = (ith * ne01) / nth;
-            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;
-            src0_cur_start = (src0_cur_start % matmul_num_cols) ? src0_cur_start + matmul_num_cols - (src0_cur_start % matmul_num_cols): src0_cur_start;
-            src0_cur_end   = (src0_cur_end % matmul_num_cols) ? src0_cur_end + matmul_num_cols - (src0_cur_end % matmul_num_cols): src0_cur_end;
-            if (src0_cur_start >= src0_cur_end) return;
-
-            for (int ir1 = 0; ir1 < nr1; ir1++) {
-                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);
-                const int id       = row_mapping.i1; // selected expert index
-
-                const int64_t  i11 = id % ne11;
-                const int64_t  i12 = row_mapping.i2; // row index in src1
-
-                const int64_t  i1 = id;  // selected expert index
-                const int64_t  i2 = i12; // row
-
-                const char * src1_col = (const char *) wdata +
-                    (src1_cont || src1->type != vec_dot_type
-                    ? (i11        + i12 * ne11) * row_size
-                    : (i11 * nb11 + i12 * nb12));
-
-                gemv(ne00, (float *)((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
-                     (const char *) src0_cur + src0_cur_start * nb01, src1_col, 1, src0_cur_end - src0_cur_start);
-            }
-            continue;
-        }
-
         // distribute the thread work across the inner or outer loop based on which one is larger
 
         const int64_t nth0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
@@ -8096,9 +7967,6 @@ static void ggml_compute_forward_out_prod(
         case GGML_TYPE_IQ4_XS:
         case GGML_TYPE_IQ3_S:
         case GGML_TYPE_IQ2_S:
-        case GGML_TYPE_Q4_0_4_4:
-        case GGML_TYPE_Q4_0_4_8:
-        case GGML_TYPE_Q4_0_8_8:
             {
                 ggml_compute_forward_out_prod_q_f32(params, dst);
             } break;
@@ -8361,9 +8229,6 @@ static void ggml_compute_forward_set(
         case GGML_TYPE_IQ4_XS:
         case GGML_TYPE_IQ3_S:
         case GGML_TYPE_IQ2_S:
-        case GGML_TYPE_Q4_0_4_4:
-        case GGML_TYPE_Q4_0_4_8:
-        case GGML_TYPE_Q4_0_8_8:
         default:
             {
                 GGML_ABORT("fatal error");
@@ -8625,9 +8490,6 @@ static void ggml_compute_forward_get_rows(
         case GGML_TYPE_IQ4_XS:
         case GGML_TYPE_IQ3_S:
         case GGML_TYPE_IQ2_S:
-        case GGML_TYPE_Q4_0_4_4:
-        case GGML_TYPE_Q4_0_4_8:
-        case GGML_TYPE_Q4_0_8_8:
             {
                 ggml_compute_forward_get_rows_q(params, dst);
             } break;
@@ -9217,10 +9079,6 @@ static void ggml_compute_forward_clamp(
         case GGML_TYPE_IQ3_S:
         case GGML_TYPE_IQ2_S:
         case GGML_TYPE_Q8_K:
-        case GGML_TYPE_Q4_0_4_4:
-        case GGML_TYPE_Q4_0_4_8:
-        case GGML_TYPE_Q4_0_8_8:
-        case GGML_TYPE_IQ4_NL_4_4:
         case GGML_TYPE_I8:
         case GGML_TYPE_I16:
         case GGML_TYPE_I32:
@@ -9275,6 +9133,64 @@ static void ggml_rope_cache_init(
     }
 }
 
+static void ggml_mrope_cache_init(
+     float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool indep_sects,
+     float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
+     float * cache, float sin_sign, float theta_scale) {
+    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
+    float theta_t = theta_base_t;
+    float theta_h = theta_base_h;
+    float theta_w = theta_base_w;
+    float theta_e = theta_base_e;  // extra position id for vision encoder
+    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
+    int sec_w = sections[1] + sections[0];
+    int sec_e = sections[2] + sec_w;
+    GGML_ASSERT(sect_dims <= ne0);
+
+    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
+        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
+
+        int sector = (i0 / 2) % sect_dims;
+        if (indep_sects) {
+            // compute theta independently for each dim sections
+            // (i.e. reset corresponding theta when `i0` go from one section to another)
+            if (sector == 0) {
+                theta_t = theta_base_t;
+            }
+            else if (sector == sections[0]) {
+                theta_h = theta_base_h;;
+            }
+            else if (sector == sec_w) {
+                theta_w = theta_base_w;
+            }
+            else if (sector == sec_e) {
+                theta_e = theta_base_e;
+            }
+        }
+
+        float theta = theta_t;
+        if (sector >= sections[0] && sector < sec_w) {
+            theta = theta_h;
+        }
+        else if (sector >= sec_w && sector < sec_w + sections[2]) {
+            theta = theta_w;
+        }
+        else if (sector >= sec_w + sections[2]) {
+            theta = theta_e;
+        }
+
+        rope_yarn(
+            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
+        );
+        cache[i0 + 1] *= sin_sign;
+
+        theta_t *= theta_scale;
+        theta_w *= theta_scale;
+        theta_h *= theta_scale;
+        theta_e *= theta_scale;
+    }
+}
+
 static void ggml_compute_forward_rope_f32(
         const struct ggml_compute_params * params,
         struct ggml_tensor * dst,
@@ -9285,6 +9201,7 @@ static void ggml_compute_forward_rope_f32(
     const struct ggml_tensor * src2 = dst->src[2];
 
     float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
+    int sections[4];
 
     //const int n_past     = ((int32_t *) dst->op_params)[0];
     const int n_dims     = ((int32_t *) dst->op_params)[1];
@@ -9298,6 +9215,7 @@ static void ggml_compute_forward_rope_f32(
     memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
     memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
     memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
+    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);
 
     GGML_TENSOR_UNARY_OP_LOCALS
 
@@ -9330,6 +9248,16 @@ static void ggml_compute_forward_rope_f32(
     ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
 
     const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
+    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
+    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
+
+    if (is_mrope) {
+        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
+    }
+
+    if (is_vision) {
+        GGML_ASSERT(n_dims == ne0/2);
+    }
 
     const float * freq_factors = NULL;
     if (src2 != NULL) {
@@ -9345,18 +9273,63 @@ static void ggml_compute_forward_rope_f32(
 
     const int32_t * pos = (const int32_t *) src1->data;
 
-    for (int64_t i3 = 0; i3 < ne3; i3++) {
-        for (int64_t i2 = 0; i2 < ne2; i2++) {
-            const int64_t p = pos[i2];
+    for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
+        for (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
 
             float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
-            ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
+            if (!is_mrope) {
+                const int64_t p = pos[i2];
+                ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
+            }
+            else {
+                const int64_t p_t = pos[i2];
+                const int64_t p_h = pos[i2 + ne2];
+                const int64_t p_w = pos[i2 + ne2 * 2];
+                const int64_t p_e = pos[i2 + ne2 * 3];
+                ggml_mrope_cache_init(
+                    p_t, p_h, p_w, p_e, sections, is_vision,
+                    freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
+            }
 
-            for (int64_t i1 = 0; i1 < ne1; i1++) {
+            for (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                 if (ir++ < ir0) continue;
                 if (ir   > ir1) break;
 
-                if (!is_neox) {
+                if (is_neox || is_mrope) {
+                    if (is_vision){
+                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
+                            const int64_t ic = i0/2;
+
+                            const float cos_theta = cache[i0 + 0];
+                            const float sin_theta = cache[i0 + 1];
+
+                            const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
+                            float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
+
+                            const float x0 = src[0];
+                            const float x1 = src[n_dims];
+
+                            dst_data[0]      = x0*cos_theta - x1*sin_theta;
+                            dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
+                        }
+                    } else {
+                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
+                            const int64_t ic = i0/2;
+
+                            const float cos_theta = cache[i0 + 0];
+                            const float sin_theta = cache[i0 + 1];
+
+                            const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
+                            float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
+
+                            const float x0 = src[0];
+                            const float x1 = src[n_dims/2];
+
+                            dst_data[0]        = x0*cos_theta - x1*sin_theta;
+                            dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
+                        }
+                    }
+                } else {
                     for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                         const float cos_theta = cache[i0 + 0];
                         const float sin_theta = cache[i0 + 1];
@@ -9370,8 +9343,10 @@ static void ggml_compute_forward_rope_f32(
                         dst_data[0] = x0*cos_theta - x1*sin_theta;
                         dst_data[1] = x0*sin_theta + x1*cos_theta;
                     }
-                } else {
-                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
+                }
+
+                if (is_vision) {
+                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                         const int64_t ic = i0/2;
 
                         const float cos_theta = cache[i0 + 0];
@@ -9381,19 +9356,20 @@ static void ggml_compute_forward_rope_f32(
                         float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
 
                         const float x0 = src[0];
-                        const float x1 = src[n_dims/2];
+                        const float x1 = src[n_dims];
 
-                        dst_data[0]        = x0*cos_theta - x1*sin_theta;
-                        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
+                        dst_data[0]      = x0*cos_theta - x1*sin_theta;
+                        dst_data[n_dims] = x0*sin_theta + x1*cos_theta;
                     }
-                }
-
-                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
-                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
-                    float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
+                } else {
+                    // fill the remain channels with data from src tensor
+                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
+                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
+                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
 
-                    dst_data[0] = src[0];
-                    dst_data[1] = src[1];
+                        dst_data[0] = src[0];
+                        dst_data[1] = src[1];
+                    }
                 }
             }
         }
@@ -9411,6 +9387,7 @@ static void ggml_compute_forward_rope_f16(
     const struct ggml_tensor * src2 = dst->src[2];
 
     float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
+    int sections[4];
 
     //const int n_past     = ((int32_t *) dst->op_params)[0];
     const int n_dims     = ((int32_t *) dst->op_params)[1];
@@ -9423,6 +9400,8 @@ static void ggml_compute_forward_rope_f16(
     memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
     memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
     memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
+    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);
+
 
     GGML_TENSOR_UNARY_OP_LOCALS
 
@@ -9455,6 +9434,16 @@ static void ggml_compute_forward_rope_f16(
     ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
 
     const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
+    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
+    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
+
+    if (is_mrope) {
+        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
+    }
+
+    if (is_vision) {
+        GGML_ASSERT(n_dims == ne0/2);
+    }
 
     const float * freq_factors = NULL;
     if (src2 != NULL) {
@@ -9472,16 +9461,61 @@ static void ggml_compute_forward_rope_f16(
 
     for (int64_t i3 = 0; i3 < ne3; i3++) {
         for (int64_t i2 = 0; i2 < ne2; i2++) {
-            const int64_t p = pos[i2];
 
             float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
-            ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
+            if (!is_mrope) {
+                const int64_t p = pos[i2];
+                ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
+            }
+            else {
+                const int64_t p_t = pos[i2];
+                const int64_t p_h = pos[i2 + ne2];
+                const int64_t p_w = pos[i2 + ne2 * 2];
+                const int64_t p_e = pos[i2 + ne2 * 3];
+                ggml_mrope_cache_init(
+                    p_t, p_h, p_w, p_e, sections, is_vision,
+                    freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
+            }
 
             for (int64_t i1 = 0; i1 < ne1; i1++) {
                 if (ir++ < ir0) continue;
                 if (ir   > ir1) break;
 
-                if (!is_neox) {
+                if (is_neox || is_mrope) {
+                    if (is_vision) {
+                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
+                            const int64_t ic = i0/2;
+
+                            const float cos_theta = cache[i0 + 0];
+                            const float sin_theta = cache[i0 + 1];
+
+                            const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
+                            ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
+
+                            const float x0 = GGML_FP16_TO_FP32(src[0]);
+                            const float x1 = GGML_FP16_TO_FP32(src[n_dims]);
+
+                            dst_data[0]      = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
+                            dst_data[n_dims] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
+                        }
+                    } else {
+                        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
+                            const int64_t ic = i0/2;
+
+                            const float cos_theta = cache[i0 + 0];
+                            const float sin_theta = cache[i0 + 1];
+
+                            const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
+                            ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
+
+                            const float x0 = GGML_FP16_TO_FP32(src[0]);
+                            const float x1 = GGML_FP16_TO_FP32(src[n_dims/2]);
+
+                            dst_data[0]        = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
+                            dst_data[n_dims/2] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
+                        }
+                    }
+                } else {
                     for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                         const float cos_theta = cache[i0 + 0];
                         const float sin_theta = cache[i0 + 1];
@@ -9495,8 +9529,10 @@ static void ggml_compute_forward_rope_f16(
                         dst_data[0] = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                         dst_data[1] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                     }
-                } else {
-                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
+                }
+
+                if (is_vision) {
+                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                         const int64_t ic = i0/2;
 
                         const float cos_theta = cache[i0 + 0];
@@ -9506,19 +9542,19 @@ static void ggml_compute_forward_rope_f16(
                         ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);
 
                         const float x0 = GGML_FP16_TO_FP32(src[0]);
-                        const float x1 = GGML_FP16_TO_FP32(src[n_dims/2]);
+                        const float x1 = GGML_FP16_TO_FP32(src[n_dims]);
 
-                        dst_data[0]        = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
-                        dst_data[n_dims/2] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
+                        dst_data[0]      = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
+                        dst_data[n_dims] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                     }
-                }
-
-                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
-                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
-                    ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
+                } else {
+                    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
+                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
+                        ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);
 
-                    dst_data[0] = src[0];
-                    dst_data[1] = src[1];
+                        dst_data[0] = src[0];
+                        dst_data[1] = src[1];
+                    }
                 }
             }
         }
@@ -12426,6 +12462,9 @@ static void ggml_compute_forward(struct ggml_compute_params * params, struct ggm
         return;
     }
 
+    // extra_buffer op?
+    if (ggml_cpu_extra_compute_forward(params, tensor)) return;
+
     switch (tensor->op) {
         case GGML_OP_DUP:
             {
@@ -13083,7 +13122,7 @@ static thread_ret_t ggml_graph_compute_secondary_thread(void* data);
 #include "windows.h"
 
 // TODO: support > 64 CPUs
-bool ggml_thread_apply_affinity(bool * mask) {
+static bool ggml_thread_apply_affinity(bool * mask) {
     HANDLE    h = GetCurrentThread();
     uint64_t  bitmask = 0ULL;
 
@@ -13373,146 +13412,142 @@ struct ggml_cplan ggml_graph_plan(
 
         size_t cur = 0;
 
-        switch (node->op) {
-            case GGML_OP_CPY:
-            case GGML_OP_DUP:
-                {
-                    if (ggml_is_quantized(node->type) ||
-                        // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
-                        (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
-                        (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16)) {
+        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
+
+            switch (node->op) {
+                case GGML_OP_CPY:
+                case GGML_OP_DUP:
+                    {
+                        if (ggml_is_quantized(node->type) ||
+                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
+                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
+                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16)) {
+                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
+                        }
+                    } break;
+                case GGML_OP_ADD:
+                case GGML_OP_ADD1:
+                    {
+                        if (ggml_is_quantized(node->src[0]->type)) {
+                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
+                        }
+                    } break;
+                case GGML_OP_ACC:
+                    {
+                        if (ggml_is_quantized(node->src[0]->type)) {
+                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
+                        }
+                    } break;
+                case GGML_OP_COUNT_EQUAL:
+                    {
+                        cur = ggml_type_size(node->type)*n_tasks;
+                    } break;
+                case GGML_OP_MUL_MAT:
+                    {
+                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;
+
+                        if (node->src[1]->type != vec_dot_type) {
+                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
+                        }
+                    } break;
+                case GGML_OP_MUL_MAT_ID:
+                    {
+                        cur = 0;
+                        const struct ggml_tensor * src0 = node->src[0];
+                        const struct ggml_tensor * src1 = node->src[1];
+                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
+                        if (src1->type != vec_dot_type) {
+                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1));
+                        }
+                        const int n_as = src0->ne[2];
+                        cur += GGML_PAD(cur, sizeof(int64_t));       // align
+                        cur += n_as * sizeof(int64_t);               // matrix_row_counts
+                        cur += n_as * src1->ne[2] * sizeof(int64_t); // matrix_rows
+                    } break;
+                case GGML_OP_OUT_PROD:
+                    {
+                        if (ggml_is_quantized(node->src[0]->type)) {
+                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
+                        }
+                    } break;
+                case GGML_OP_SOFT_MAX:
+                case GGML_OP_ROPE:
+                    {
                         cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
-                    }
-                } break;
-            case GGML_OP_ADD:
-            case GGML_OP_ADD1:
-                {
-                    if (ggml_is_quantized(node->src[0]->type)) {
-                        cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
-                    }
-                } break;
-            case GGML_OP_ACC:
-                {
-                    if (ggml_is_quantized(node->src[0]->type)) {
-                        cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
-                    }
-                } break;
-            case GGML_OP_COUNT_EQUAL:
-                {
-                    cur = ggml_type_size(node->type)*n_tasks;
-                } break;
-            case GGML_OP_MUL_MAT:
-                {
-#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
-                    if (node->src[0]->buffer && ggml_backend_amx_buft_is_amx(node->src[0]->buffer->buft)) {
-                        cur = ggml_backend_amx_desired_wsize(node);
-                    }
-#endif
-                    const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;
+                    } break;
+                case GGML_OP_CONV_TRANSPOSE_1D:
+                    {
+                        GGML_ASSERT(node->src[0]->ne[3] == 1);
+                        GGML_ASSERT(node->src[1]->ne[2] == 1);
+                        GGML_ASSERT(node->src[1]->ne[3] == 1);
+
+                        const int64_t ne00 = node->src[0]->ne[0];  // K
+                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
+                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
+                        const int64_t ne10 = node->src[1]->ne[0];  // L
+                        const int64_t ne11 = node->src[1]->ne[1];  // Cin
+
+                        if ((node->src[0]->type == GGML_TYPE_F16 ||
+                             node->src[0]->type == GGML_TYPE_BF16) &&
+                            node->src[1]->type == GGML_TYPE_F32) {
+                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
+                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
+                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
+                                   node->src[1]->type == GGML_TYPE_F32) {
+                            cur += sizeof(float)*ne00*ne01*ne02;
+                            cur += sizeof(float)*ne10*ne11;
+                        } else {
+                            GGML_ABORT("fatal error");
+                        }
+                    } break;
+                case GGML_OP_CONV_TRANSPOSE_2D:
+                    {
+                        const int64_t ne00 = node->src[0]->ne[0]; // W
+                        const int64_t ne01 = node->src[0]->ne[1]; // H
+                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
+                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In
 
-                    if (node->src[1]->type != vec_dot_type) {
-                        size_t cur2 = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
-                        cur = MAX(cur, cur2);
-                    }
-                } break;
-            case GGML_OP_MUL_MAT_ID:
-                {
-                    cur = 0;
-                    const struct ggml_tensor * src0 = node->src[0];
-                    const struct ggml_tensor * src1 = node->src[1];
-                    const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
-                    if (src1->type != vec_dot_type) {
-                        cur += ggml_row_size(vec_dot_type, ggml_nelements(src1));
-                    }
-                    const int n_as = src0->ne[2];
-                    cur += GGML_PAD(cur, sizeof(int64_t));       // align
-                    cur += n_as * sizeof(int64_t);               // matrix_row_counts
-                    cur += n_as * src1->ne[2] * sizeof(int64_t); // matrix_rows
-                } break;
-            case GGML_OP_OUT_PROD:
-                {
-                    if (ggml_is_quantized(node->src[0]->type)) {
-                        cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
-                    }
-                } break;
-            case GGML_OP_SOFT_MAX:
-            case GGML_OP_ROPE:
-                {
-                    cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
-                } break;
-            case GGML_OP_CONV_TRANSPOSE_1D:
-                {
-                    GGML_ASSERT(node->src[0]->ne[3] == 1);
-                    GGML_ASSERT(node->src[1]->ne[2] == 1);
-                    GGML_ASSERT(node->src[1]->ne[3] == 1);
-
-                    const int64_t ne00 = node->src[0]->ne[0];  // K
-                    const int64_t ne01 = node->src[0]->ne[1];  // Cout
-                    const int64_t ne02 = node->src[0]->ne[2];  // Cin
-
-                    const int64_t ne10 = node->src[1]->ne[0];  // L
-                    const int64_t ne11 = node->src[1]->ne[1];  // Cin
-
-                    if ((node->src[0]->type == GGML_TYPE_F16 ||
-                         node->src[0]->type == GGML_TYPE_BF16) &&
-                        node->src[1]->type == GGML_TYPE_F32) {
-                        cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
-                        cur += sizeof(ggml_fp16_t)*ne10*ne11;
-                    } else if (node->src[0]->type == GGML_TYPE_F32 &&
-                               node->src[1]->type == GGML_TYPE_F32) {
-                        cur += sizeof(float)*ne00*ne01*ne02;
-                        cur += sizeof(float)*ne10*ne11;
-                    } else {
-                        GGML_ABORT("fatal error");
-                    }
-                } break;
-            case GGML_OP_CONV_TRANSPOSE_2D:
-                {
-                    const int64_t ne00 = node->src[0]->ne[0]; // W
-                    const int64_t ne01 = node->src[0]->ne[1]; // H
-                    const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
-                    const int64_t ne03 = node->src[0]->ne[3]; // Channels In
-
-                    const int64_t ne10 = node->src[1]->ne[0]; // W
-                    const int64_t ne11 = node->src[1]->ne[1]; // H
-                    const int64_t ne12 = node->src[1]->ne[2]; // Channels In
-
-                    cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02*ne03;
-                    cur += sizeof(ggml_fp16_t)*ne10*ne11*ne12;
-                } break;
-            case GGML_OP_FLASH_ATTN_EXT:
-                {
-                    const int64_t ne00 = node->src[0]->ne[0]; // D
+                        const int64_t ne10 = node->src[1]->ne[0]; // W
+                        const int64_t ne11 = node->src[1]->ne[1]; // H
+                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In
 
-                    cur = 3*sizeof(float)*ne00*n_tasks; // 3x head size/thread
-                } break;
-            case GGML_OP_FLASH_ATTN_BACK:
-                {
-                    const int64_t    D = node->src[0]->ne[0];
-                    const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
-                    const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
-                    if (node->src[1]->type == GGML_TYPE_F32) {
-                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
-                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
-                    } else if (node->src[1]->type == GGML_TYPE_F16) {
-                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
-                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
-                    } else if (node->src[1]->type == GGML_TYPE_BF16) {
-                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
-                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
-                    }
-                } break;
+                        cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02*ne03;
+                        cur += sizeof(ggml_fp16_t)*ne10*ne11*ne12;
+                    } break;
+                case GGML_OP_FLASH_ATTN_EXT:
+                    {
+                        const int64_t ne00 = node->src[0]->ne[0]; // D
 
-            case GGML_OP_CROSS_ENTROPY_LOSS:
-                {
-                    cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
-                } break;
-            case GGML_OP_COUNT:
-                {
-                    GGML_ABORT("fatal error");
-                }
-            default:
-                break;
+                        cur = 3*sizeof(float)*ne00*n_tasks; // 3x head size/thread
+                    } break;
+                case GGML_OP_FLASH_ATTN_BACK:
+                    {
+                        const int64_t    D = node->src[0]->ne[0];
+                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
+                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
+                        if (node->src[1]->type == GGML_TYPE_F32) {
+                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
+                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
+                        } else if (node->src[1]->type == GGML_TYPE_F16) {
+                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
+                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
+                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
+                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
+                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
+                        }
+                    } break;
+
+                case GGML_OP_CROSS_ENTROPY_LOSS:
+                    {
+                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
+                    } break;
+                case GGML_OP_COUNT:
+                    {
+                        GGML_ABORT("fatal error");
+                    }
+                default:
+                    break;
+            }
         }
 
         work_size = MAX(work_size, cur);
```


### ggml/src/ggml-metal/ggml-metal.m
Status: modified | +45/-4

```diff
@@ -510,6 +510,35 @@ @implementation GGMLMetalClass
 #endif
 
         NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
+        if (path_lib == nil) {
+            // Try to find the resource in the directory where the current binary located.
+            NSString * current_binary = [[NSProcessInfo processInfo] arguments][0];
+            NSString * bin_dir = [current_binary stringByDeletingLastPathComponent];
+            NSString * default_metallib_path = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
+            if ([[NSFileManager defaultManager] isReadableFileAtPath:default_metallib_path]) {
+                GGML_LOG_INFO("%s: found '%s'\n", __func__, [default_metallib_path UTF8String]);
+                NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:default_metallib_path error:&error];
+                if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
+                    // Optionally, if this is a symlink, try to resolve it.
+                    default_metallib_path = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:default_metallib_path error:&error];
+                    if (default_metallib_path && [default_metallib_path length] > 0 && ![[default_metallib_path substringToIndex:1] isEqualToString:@"/"]) {
+                        // It is a relative path, adding the binary directory as directory prefix.
+                        default_metallib_path = [NSString pathWithComponents:@[bin_dir, default_metallib_path]];
+                    }
+                    if (!default_metallib_path || ![[NSFileManager defaultManager] isReadableFileAtPath:default_metallib_path]) {
+                        // Link to the resource could not be resolved.
+                        default_metallib_path = nil;
+                    } else {
+                        GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [default_metallib_path UTF8String]);
+                    }
+                }
+            } else {
+                // The resource couldn't be found in the binary's directory.
+                default_metallib_path = nil;
+            }
+            path_lib = default_metallib_path;
+        }
+
         if (try_metallib && path_lib != nil) {
             // pre-compiled library found
             NSURL * libURL = [NSURL fileURLWithPath:path_lib];
@@ -1096,8 +1125,18 @@ static bool ggml_metal_supports_op(const struct ggml_backend_metal_device_contex
             return has_simdgroup_reduction && (op->ne[0] % 4 == 0);
         case GGML_OP_ARGMAX:
         case GGML_OP_NORM:
-        case GGML_OP_ROPE:
             return true;
+        case GGML_OP_ROPE:
+            {
+                const int mode = ((const int32_t *) op->op_params)[2];
+                if (mode & GGML_ROPE_TYPE_MROPE) {
+                    return false;
+                }
+                if (mode & GGML_ROPE_TYPE_VISION) {
+                    return false;
+                }
+                return true;
+            }
         case GGML_OP_IM2COL:
             return op->src[0]->type == GGML_TYPE_F16;
         case GGML_OP_POOL_1D:
@@ -2028,8 +2067,8 @@ static void ggml_metal_encode_node(
                 GGML_ASSERT(ne12 % ne02 == 0);
                 GGML_ASSERT(ne13 % ne03 == 0);
 
-                const uint r2 = ne12/ne02;
-                const uint r3 = ne13/ne03;
+                const uint32_t r2 = ne12/ne02;
+                const uint32_t r3 = ne13/ne03;
 
                 // find the break-even point where the matrix-matrix kernel becomes more efficient compared
                 // to the matrix-vector kernel
@@ -2997,7 +3036,9 @@ static void ggml_metal_encode_node(
             } break;
         case GGML_OP_ROPE:
             {
-                GGML_ASSERT(ne10 == ne02);
+                // make sure we have one or more position id(ne10) per token(ne02)
+                GGML_ASSERT(ne10 % ne02 == 0);
+                GGML_ASSERT(ne10 >= ne02);
 
                 const int nth = MIN(1024, ne00);
 
```


### gguf-py/pyproject.toml
Status: modified | +0/-0

```diff
--- a/gguf-py/pyproject.toml
+++ b/gguf-py/pyproject.toml
@@ -1,6 +1,6 @@
 [tool.poetry]
 name = "gguf"
-version = "0.10.0"
+version = "0.13.0"
 description = "Read and write ML models in GGUF for GGML"
 authors = ["GGML <ggml@ggml.ai>"]
 packages = [

```

