# Example Report (PyTorch)

Command: `diffinator -c pytorch v2.4.0 v2.5.1 --output markdown`

Analyzing changes between v2.4.0 and v2.5.1:

## Summary
- Total commits: 250
- Files changed: 300
- Important files modified: 1

## Changes by Category

### Inductor
- [Inductor] support masked vectorization for the tail_loop for dynamic shapes (#131745)
- [inductor] [cpp] generate reindexer for each epilogue_node (#134984)
- [inductor] [cpp] use_local_acc if template_buffer_has_other_users (#135081)
- [inductor][test] in test_unbacked_symints, replace inductor's skipCUDAIf with common device type's skipcudaif (#133936)
- [Inductor][CPP] Leverage full bits for BF16/FP16 vectorization (#126502)
- [inductor] Improve compile time regression from MemoryDep.normalize (#135070)
- [Inductor] Fix AOT weight alignment issue on CPU (#135205)
- [Inductor][CPP] Avoid mistake wgt tensor delete (#135100)
- [inductor] check intel compiler minimal version (#135209)
- [inductor][debug] fix draw_buffers (#135266)
- [inductor] Skip retracing an existing LoopBody (#135235)
- [inductor] Remove LoopBody.reads,writes,other (#135256)
- [Inductor][CPP] Select tiling factor for lower precision data types (#133830)
- [inductor] Fix gen_transposed_tile_load_store (#135307)
- [inductor][cpp][gemm] fix autotune runtime error from linear_binary fusion (#135275)
- [inductor] [cpp] improve cache blocking for is_dynamic_M (#131306)
- [inductor][triton] mark workspace args as mutated (#134648)
- [Inductor] Use argument names as the key for the `constants` dict and the `signature` dict (#135170)
- [inductor] Fix loop split optimization (#135303)
- [Inductor][CPP] Fix the issue of view dtype (#135301)
- [Inductor] Optionally allow padding on non-GPU devices (#135280)
- [inductor][cpp][gemm] reduce memory alloc overhead by allocating local acc once per thread (#135277)
- [inductor][cpp][gemm] enable dynamic M for k-slicing (#133447)
- [inductor][cpp][gemm] cache blocking config for dynamic shapes (#133538)
- [inductor] Move LoopBody to its own file (#135257)
- [inductor] Catch BrokenProcessPool and print a more helpful message. (#135120)
- [inductor] Refactor LoopBody.memory_usage (#135286)
- [inductor] Remove ReadWrites.op_counts (#135306)
- [inductor] Fast path for extract_read_writes without tracing (#135377)
- [inductor] Refactor BaseSchedulerNode.__init__ (#135400)
- [inductor] Cleanup analysis done at lowering time (#135412)
- [inductor] calibration inductor windows uts (18/N) (#135449)
- [Inductor] simplify indexing_exprs in LoopBody._init_with_copy (#135574) (#135935)
- [inductor] [cpp] fix the input contiguous check in max-autotune (#135561)
- [Inductor] Increase multiplier to 3 for Inductor AMP FP16 benchmark correctness check (#135932) (#136262)

### Documentation
- docs: `torch.nn.utils.rnn.pack_padded_sequence`: docs improve (#135417)

### Other
- Fix: use clone_preserve_strides in auto_functionalized_v2 (#135142)
- Fix binary builds artifact download (#135139)
- [FR] Add version based logic to FR script and make traces print can be filtered (#135154)
- [dynamo] Retire CompileProfiler (#135133)
- Remove redundant code (#134955)
- [FlexAttention] Fix mismatched backward strides for eager impl (#135152)
- Update current scripts used for setting up s390x runners (#129866)
- [FSDP] casting input args with dataclass(frozen=True) (#135067)
- Fix typo in comment (#135111)
- [Intel GPU]device guard codegen for XPU (#133980)
- [Traceable FSDP2] Skip _backward_prefetch under compile, and rely on compiler pass to have prefetching (#135163)
- [executorch hash update] update the pinned executorch hash (#135162)
- [export] dynamic_shapes serialization, load/dump (#134718)
- Fix decomp behaviour in export training IR (#134801)
- [Intel GPU] Customized XPU behaviour in indexing, group norm (#134453)
- restore CSE'd node metadata in runtime asserts pass (#134516)
- [c10d] Change collective to take in a list of tensors so it work fully for all collectives (#135049)
- Enhance the stability of the complex divide code (#134647)
- Update unbacked symints in masked_select more precisely (#134899)
- Update torch-xpu-ops pin (ATen XPU implementation) (#135185)
- [AOTI] Fix a unbacked symint retrieve bug (#134670)
- Revise CPU vectorization ISA support API (#135075)
- [Dynamo] Support builtin function frozenset (#134563)
- [Intel GPU][Windows] Fix overriding default CMAKE_CXX_FLAGS (#135093)
- [CI] Use larger instance for building triton whl (#135201)
- Upgrade expecttest to 0.2.1 (#135136)
- Switch torch pt2e xnnpack tests to use export_for_training (#134788)
- Implement VariableTracker.python_type() (#134215)
- Gradient scaler for DTensor (#132816)
- Render log filepaths that are not anchored in torch's directory in a reasonable way (#135165)
- Add torch.serialization.skip_data context manager (#134504)
- [fake_tensor] Move unrecognized_type NotImplemented before ConstProp (#135033)
- [PT2] Directly set meta.val in group_batch_fusion_aten (#135078)
- Update Resize.cpp with new device type (#135117)
- Allow cross-device copies for cpu scalars in refs (#135140)
- Use actions/upload-artifact@v4.4.0 for triton builds (#135263)
- [export] Add ability to run eagerly on UnflattenedModule (#133996)
- [ROCm] remove triton-rocm commit pin and merge pins with triton.txt (#133438)
- Use actions/upload-artifact@v4.4.0 for rest of workflows (#135264)
- Revert "Remove Caffe2 code from tool scripts (#134941)"
- Revert "[Reland] Refactor caching device allocator utils (#130923)"
- [training ir migration] Fix quantization tests (#135184)
- [Traceable FSDP2][Dynamo] allow tracing through auto_functionalized HOP (#135169)
- Run inductor micro benchmark on x86 metal runner (#135042)
- [hop] preserve metadata in re-tracing hop subgraph by running with interpreter (#135159)
- Revert "Use actions/upload-artifact@v4.4.0 for rest of workflows (#135264)"
- [PyTorch] Fix -Wshadow -Werror build in BFloat16-inl.h (#135031)
- [PyTorch] Add isfinite to BFloat16-math.h (#135052)
- [export] Expand coverage to more copied sym ops for unflattener. (#135119)
- Tune int8 AMX WoQ micro-kernel for CPU (#134832)
- [fbcode][dynamo] Turn on guard_nn_modules using justknobs_check (#134928)
- Support rolling over a percentage of workflows (#134816)
- [cuDNN][64-bit indexing] cuDNN v9.3+ supports non-batch-splittable convolutions with > 2**31 elements (#134890)
- Add torch._logging.scribe (#135224)
- [cond] fix typo in cond codegen (#134708)
- [Intel Triton] Update Intel Triton to release/2.5.0 (#134074)
- [rfc] scuba for flight recorder (#134794)
- fix fake tensor tolist implementation (#135131)
- [MPS] Add support for autocast in MPS  (#99272)
- Revert "Support rolling over a percentage of workflows (#134816)"
- [fx] Compile time optimization in Node.__update_args_kwargs (#135076)
- [fx] Don't use generators in map_aggregate (#135082)
- [debug] Add helper to run cProfile on a function (#135084)
- [ONNX] Delete ONNXProgramSerializer (#135261)
- [DTensor] Fix view op replicating on tensor dim when the size of the tensor dim = 1 (#135054)
- [Submodule] Bump pybind11 to v2.13.5 (#135202)
- Track base of FunctionalTensor in inference mode.  (#135141)
- Add Percentages to Function Events (#135155)
- [AOTI] Support MKLDNN conv ops in cpp wrapper (#134475)
- [AOTI] Support MKLDNN qlinear ops in cpp wrapper (#134783)
- [AOTI] Support MKLDNN qconv ops in cpp wrapper (#134795)
- [ONNX] Enable experimental exporter logic to dynamo_export and support refine dynamic_shapes (#134976)
- aarch64: extend matmul heuristic checks to all neoverse platforms (#134548)
- Use Python 3.9 on all libtorch jobs (#135245)
- Improve test_public_bindings import module error reporting (#135258)
- add instrumentation of CCA stats for reserved and allocated memory size (#135231)
- Consolidate raise and rewrap raise error branches (#135148)
- Include exception type qualname when rewrapping InternalTorchDynamoError (#135145)
- Report qualname of exception type rather than <class 'RuntimeError'> (#135146)
- Remove dead expect_rational (#135105)
- [Distributed] Change function call in test to non-deprecated to eliminate warning (#134938)
- Add randomness checking for sdpa vmap (#135176)
- [RDP] Fix "No module named 'libfb’" (#135244)
- [fx] Bypass custom __setattr__ in Node.__init__ (#135079)
- [Fix][FR][ez] Remove debugging logs (#135308)
- [DeviceMesh][Easy] Make RuntimeError a bit more descriptive by including the actual world_size (#135271)
- [FlexAttention] Specify padding_value for boundary checked loads (#134573)
- [export][training ir migration] quantized_decomposed.quantize_per_tensor decomposition (#134525)
- Update torch-xpu-ops pin (ATen XPU implementation) (#135300)
- [Docs] Update FileCheck doc (#135199)
- [Dynamo] Automatically in-graph traceable tensor subclass ctors (#135151)
- check compilation status before query cudnn version in conv (#135332)
- Ignore fresh unbacked when doing recursive make_fx inside HOPs (#135053)
- Also handle compiler collective when input variable doesn't exist on all ranks (#135147)
- [CD] Update binary_linux_test.sh to include calling builder smoke test (#133869)
- AOTDispatcher: limit cases when we detach() graph inputs to non-leaves (#134193)
- [torchelastic] Don't do signal handling when off the main thread (#135088)
- Add Inductor config for default stride behavior (#135238)
- error on exporting ScriptModule (#135302)
- [AOTI][Tooling][6/n] Fix long dtype input tensors calling `mean()` in `aoti_torch_print_tensor_handle` (#135072)
- Porting to GCC 15 (#135188)
- Update submodule ideep to include aarch64 change (#134897)
- [torch][fx] Set maximum warning count during fx.Graph.lint (#135069)
- [export] Record the global torch version in serialization. (#135243)
- [aoti][easy] remove breakpoint() in wrapper.py (#134807)
- remove _check call on item() for torch.istft (#135234)
- [elastic] support local_addr across all rendezvous impls (#135262)
- Fix incorrect trace of post-accumulate grad hook on tensor with zero dims (#135226)
- Run bypassed graph compile outside the except block to avoid chaining of exceptions (#135175)
- Add MaskedTensor passthrough: unfold, F.Unfold, F.Fold, stack (#125262)
- [ONNX] Refactor exporter errors (#135180)
- [ONNX] Properly handle Attributes in traceable functions (#135367)
- Run all autograd node post hooks (#134728)
- [TCPStore] use wait counters (#135283)
- [ONNX] Clean up the missed lines from previous PRs (#135368)
- [test][easy] Add debug utils for cpu select algorithm test (#135038)
- [dynamo] reland map/zip iterator related changes (#135074)
- Revert expectFailureIf condition on tests with torch.compile on Windows (#134759)
- [ez][TD] Fix request for issue body returns None (#135389)
- [dynamo] recursively skip frames when Dynamo cache limit is hit (#135144)
- Revert "[ONNX] Refactor exporter errors (#135180)"
- [FR] Make trace_dir a required argument (#135157)
- [FR] Automatically infer a common filename prefix (#135158)
- [BE] Clarify defaulting behavior in optimizer (#135384)
- Require tlparse for failing tests in test_structured_trace.py (#135376)
- [export] fix placeholder name collision tests by removing map call (#135366)
- [aoti test] Disable FP8 funz dtypes in fp8 runtime check test (#135373)
- [Split Build] Refactor split build binary builds into their own workflows and move split build binary builds to periodic (#134624)
- Change test_constant_prop_preserve_metadata (#135268)
- [Dynamo][DTensor] Fixes SymNodeVariable() is not a constant error in Compiled DDP + TP unit test (#135315)
- Add release matrix for 2.5 (#135383)
- [ONNX] Refactor exporter errors (#135180)
- [quant][pt2e] fix placeholder typo and related quantization tests (#135379)
- [ONNX] Handle mixed sequence inputs properly (#135378)
- [FlexAttention] Skip very small block size unit tests on H100 due to Triton bug (#135393)
- [ONNX] Support FakeTensor in ONNXProgram (#135399)
- [Reland] Refactor caching device allocator utils (#130923)
- [Intel GPU] Add XPU memory-related APIs (#129919)
- Add oneDNN BRGEMM support on CPU (#131878)
- [Doc] update max-autotune for CPU (#134986)
- [FlexAttention] Align the matmul tensorcore usage (#135168)
- [Dynamo] Fix Huggingface PretrainedConfig get non const attr (#135413)
- remove commented out breakpoints (#135363)
- Change wrapped_linear_prepack and wrapped_quantized_linear_prepacked to private by adding _ as prefix (#135401)
- [ONNX] Re-raise the exception if the dynamic shapes cannot be refined (#135418)
- Use BRGEMM for Half flash attention forward kernel (#131879)
- [dynamo] Remove skip from jit freeze tests (#135281)
- [reland][dtensor] move DTensor to public namespace (#134203)
- [22/N] Fix clang-tidy warnings in jit  (#135319)
- [NJT]Add permute ops support (#135336)
- [RELEASE-ONLY CHANGES] Branch Cut for Release 2.5 (#135506)
- [RELEASE-ONLY CHANGES] Temp changes to build triton from pin for first RC (#135517)
- Use upload-artifact@v4.4.0 for create_release.yml (#135534)
- [Cherry-Pick] Bump triton pin and release version, revert temporary changes to build from pin (#135613)
- [ONNX] Improves documentation of ONNX exporter (#135526)
- [ONNX] Fix scaled_dot_product_attention with float scale (#135710)
- [Release only] Temporary disable triton xpu build (#136206)
- Revert "[Release only] Temporary disable triton xpu build" (#136276)
- Revert "[fx] Bypass custom __setattr__ in Node.__init__ (#135079)" (#… (#135625)
- [ONNX] Fix symbolic values and numpy implementation (#135786) (#135868)
- [DCP] Fixes the stateless optimizer issue of distributed state_dict (… (#136000)
- [Cherry-pick] [ONNX] Update fake mode usage in onnx docs (#135512) and Drop final None values as inputs for nodes in exporter graph (#135520) (#136005)
- Update document for autocast on CPU (#136082)
- Update torch-xpu-ops pin (ATen XPU implementation)  (#135833)
- [ONNX] Fix numpy method to return the correct type (#136162)  (#136203)
- Fix dynamo benchmark skip logic for cpu device (#135193) (#135793)
- Fix xpu memory stats error (#135818) (#136420)
- [ROCm] [BUGFIX] Re-enable rocm-specific tuning parameters v2 (#133852) (#136139)
- fix stride compare failed when size value equal to one in ForeachUtils.h (#136426)
- [ROCm] Cherry-pick unit test fixes to release/2.5 (#136557)
- [ROCm][CI] upgrade CI to ROCm 6.2 (#132555) (#136467)
- [ROCm] upgrade ROCm CI builds to py3.10 (#134108) (#136696)
- Fix hardcoded ROCm paths in `Caffe2Targets.cmake` (#136700)
- [RELEASE-ONLY CHANGES] Don't push to https: //ghcr.io/ (#136703)
- Disable iOS workflow (#136706)
- Make test_skip_data_serialization regex more flexible (#136710)
- Revert "Trace fwd graph under no_grad mode #134872" (#136734)
- Constraint setuptools to 72.1.0 or older in requirements.txt (#136729)
- Update current maintainers (#136769)
- Fix ROCm skip decorator for test_ddp_tp and multiprocess UTs (#136161) (#136801)
- [Update] Update note for Getting Started with PyTorch on Intel GPUs (#136731)
- [Docs] fix inconsistent docs in conv1d, conv2d, and conv3d (#136813)
- SDPA regression fix to work around high-precision by default (#136536)
- [RELEASE ONLY CHANGES] Revert XNNPACK Update (#136522)
- [ROCm] Update to AOTriton 0.7b (Cherry-picked) (#135869)
- [cuDNN][SDPA] cherrypick Support `attn_bias` in cuDNN (#130482) (#136885)
- Fix lint (#137052)
- fix requirements.txt installation failure issue on Windows (#136893)
- [SymmetricMemory] improve multicast initialization/fallback logic (#136894)
- [Cherry-pick][DSD] Fix distributed state dict full_state_dict option hang during set_state_dict (#135725) and Fix loading uneven full tensor into sharded state dict (#136365) (#136903)
- [FlexAttention] Fix output layout (#135882) (#136905)
- [dynamo] Do not treat user defined nn module attributes static for dynamic shape infra (#137025)
- [RELEASE-ONLY CHANGES] Disable slow workflows (#136805)
- [RELEASE-ONLY Change] Push ROCm images on RC (#137148)
- [Release only] Set WITH_PUSH when WITH_PUSH_ROCM is set (#137177)
- Clarify that `libtorch` API is C++17 compatible (#137206)
- [ONNX] Add assertion nodes to ignoring list (#137214)
- Fix addmm silent correctness on aarch64 (#137208)
- [NCCL] Don't override `waitUntilInitialized`'s setting of `comm->initialized_` (#137210)
- [MPS] Fix 5D+ reductions over negative dimentions (#137211)
- [MPS] Add missing dispatch to rshift.Tensor (#137212)
- [MPS] Add regression test for `fft.fftfreq` (#137215)
- [RELEASE-ONLY CHANGES] Fix dependency on filesystem on Linux (#137242)
- [split build] move periodic split builds into own concurrency group (#135510) (#137265)
- [Release only] use triton 3.1.x from pypi (#137895)
- update getting started xpu (#138090)
- [Cherry-Pick] Use cuda 12.4 pytorch_extra_install_requirements as default (#138526)
- Don't try to load cufile (#138539)
- Add link to torch.compile the missing manual in troubleshooting (#137369)
- Update cpuinfo submodule (#138600)
- Update doc copyrights to 2024 (#138650)
- [SDPA-CUDNN] Make CuDNN Attention Opt in (#138587)
- [MPS] Fix sliced cast (#138535)
- Disabling amp context when invoking compiler (#138659)

## Important File Changes

### CMakeLists.txt
Status: modified | +0/-0

```diff
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -208,7 +208,6 @@
 include(CMakeDependentOption)
 option(ATEN_NO_TEST "Do not build ATen test binaries" OFF)
 option(BUILD_BINARY "Build C++ binaries" OFF)
-option(BUILD_DOCS "Build Caffe2 documentation" OFF)
 option(BUILD_CUSTOM_PROTOBUF
        "Build and use Caffe2's own protobuf under third_party" ON)
 option(BUILD_PYTHON "Build Python binaries" ON)
@@ -252,6 +251,16 @@
 cmake_dependent_option(USE_STATIC_CUDNN "Use cuDNN static libraries" OFF
                        "USE_CUDNN" OFF)
 cmake_dependent_option(USE_CUSPARSELT "Use cuSPARSELt" ON "USE_CUDA" OFF)
+cmake_dependent_option(USE_CUDSS "Use cuDSS" ON "USE_CUDA" OFF)
+# Binary builds will fail for cufile due to https://github.com/pytorch/builder/issues/1924
+# Using TH_BINARY_BUILD to check whether is binary build.
+# USE_ROCM is guarded against in Dependencies.cmake because USE_ROCM is not properly defined here
+if(DEFINED ENV{TH_BINARY_BUILD})
+  cmake_dependent_option(USE_CUFILE "Use cuFile" OFF
+                         "USE_CUDA AND NOT $ENV{TH_BINARY_BUILD} AND NOT WIN32" OFF)
+else()
+  cmake_dependent_option(USE_CUFILE "Use cuFile" OFF "USE_CUDA AND NOT WIN32" OFF)
+endif()
 option(USE_FBGEMM "Use FBGEMM (quantized 8-bit server operators)" ON)
 option(USE_KINETO "Use Kineto profiling library" ON)
 option(USE_CUPTI_SO "Use CUPTI as a shared library" ON)
@@ -481,10 +490,6 @@
   endif()
 endif()
 
-# Used when building Caffe2 through setup.py
-option(BUILDING_WITH_TORCH_LIBS
-       "Tell cmake if Caffe2 is being built alongside torch libs" ON)
-
 # /Z7 override option When generating debug symbols, CMake default to use the
 # flag /Zi. However, it is not compatible with sccache. So we rewrite it off.
 # But some users don't use sccache; this override is for them.
@@ -539,8 +544,14 @@
 if(LINUX)
   set(CMAKE_SHARED_LINKER_FLAGS
       "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
-  set(CMAKE_SHARED_LINKER_FLAGS
-      "${CMAKE_SHARED_LINKER_FLAGS} $ENV{LDFLAGS}")
+
+  set(ENV_LDFLAGS "$ENV{LDFLAGS}")
+  string(STRIP "${ENV_LDFLAGS}" ENV_LDFLAGS)
+  # Do not append linker flags passed via env var if they already there
+  if(NOT ${CMAKE_SHARED_LINKER_FLAGS} MATCHES "${ENV_LDFLAGS}")
+     set(CMAKE_SHARED_LINKER_FLAGS
+         "${CMAKE_SHARED_LINKER_FLAGS} ${ENV_LDFLAGS}")
+  endif()
 endif()
 
 if(MSVC)
@@ -750,7 +761,6 @@
       CACHE STRING "Torch build version" FORCE)
 endif()
 caffe2_parse_version_str(TORCH ${TORCH_BUILD_VERSION})
-caffe2_parse_version_str(CAFFE2 ${TORCH_BUILD_VERSION})
 set(TORCH_SOVERSION "${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}")
 
 # ---[ CMake scripts + modules
@@ -873,6 +883,16 @@
   Will be disabled if not supported by the platform" ON
   "USE_CUDA OR USE_ROCM" OFF)
 
+#
+# Cannot be put into Dependencies.cmake due circular dependency:
+# USE_FLASH_ATTENTION -> USE_ROCM -> Dependencies.cmake -> aotriton.cmake
+#
+if(USE_ROCM)
+  if(USE_FLASH_ATTENTION OR USE_MEM_EFF_ATTENTION)
+    include(cmake/External/aotriton.cmake)
+  endif()
+endif()
+
 if(DEBUG_CUDA)
   string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -lineinfo")
   string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -lineinfo")
@@ -983,8 +1003,6 @@
   append_cxx_flag_if_supported("-Wno-array-bounds" CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wno-unknown-pragmas" CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
-  append_cxx_flag_if_supported("-Wno-unused-function" CMAKE_CXX_FLAGS)
-  append_cxx_flag_if_supported("-Wno-unused-result" CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wno-strict-overflow" CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wno-strict-aliasing" CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wno-stringop-overflow" CMAKE_CXX_FLAGS)
@@ -1032,15 +1050,8 @@
     endif()
   endif()
 
-  append_cxx_flag_if_supported("-Wno-error=pedantic" CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wno-error=old-style-cast" CMAKE_CXX_FLAGS)
-  append_cxx_flag_if_supported("-Wno-error=inconsistent-missing-override"
-                               CMAKE_CXX_FLAGS)
-  append_cxx_flag_if_supported(
-    "-Wno-error=inconsistent-missing-destructor-override" CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wconstant-conversion" CMAKE_CXX_FLAGS)
-  append_cxx_flag_if_supported("-Wno-invalid-partial-specialization"
-                               CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wno-aligned-allocation-unavailable"
                                CMAKE_CXX_FLAGS)
   append_cxx_flag_if_supported("-Wno-missing-braces" CMAKE_CXX_FLAGS)
@@ -1173,6 +1184,10 @@
   append_cxx_flag_if_supported("-Wno-missing-braces" CMAKE_CXX_FLAGS)
 endif()
 
+if(USE_XPU)
+  string(APPEND CMAKE_CXX_FLAGS " -DUSE_XPU")
+endif()
+
 if(EMSCRIPTEN)
   string(
     APPEND
@@ -1222,45 +1237,6 @@
 # ---[ Main build
 add_subdirectory(c10)
 add_subdirectory(caffe2)
-
-# --[ Documentation
-if(BUILD_DOCS)
-  # check if Doxygen is installed
-  find_package(Doxygen)
-  if(DOXYGEN_FOUND)
-    message("Generating documentation")
-
-    set(DOXYGEN_C_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/caffe2/.Doxyfile-c)
-    set(DOXYGEN_C_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/caffe2/Doxyfile-c)
-    set(DOXYGEN_P_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/caffe2/.Doxyfile-python)
-    set(DOXYGEN_P_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/caffe2/Doxyfile-python)
-
-    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/docs)
-      file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/docs)
-    endif()
-
-    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
-    configure_file(${DOXYGEN_C_IN} ${DOXYGEN_C_OUT} @ONLY)
-    configure_file(${DOXYGEN_P_IN} ${DOXYGEN_P_OUT} @ONLY)
-
-    add_custom_target(
-      doc_doxygen_c ALL
-      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_C_OUT}
-      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
-      COMMENT "Generating C++ API documentation with Doxygen"
-      VERBATIM)
-
-    add_custom_target(
-      doc_doxygen_python ALL
-      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_P_OUT}
-      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
-      COMMENT "Generating Python API documentation with Doxygen"
-      VERBATIM)
-  else()
-    message(
-      FATAL_ERROR "Doxygen needs to be installed to generate the documentation")
-  endif()
-endif()
 
 # ---[ CMake related files Uninistall option.
 if(NOT TARGET caffe2_uninstall)
@@ -1328,6 +1304,10 @@
     DESTINATION share/cmake/Caffe2/
     COMPONENT dev)
   install(
+    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindCUDSS.cmake
+    DESTINATION share/cmake/Caffe2/
+    COMPONENT dev)
+  install(
     FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindSYCLToolkit.cmake
     DESTINATION share/cmake/Caffe2/
     COMPONENT dev)
@@ -1380,6 +1360,7 @@
     # We have to specify the scope here. We do this by specifying the targets we
     # care about and caffe2/ for all test targets defined there
     if(BUILD_LIBTORCHLESS)
+      caffe2_update_option(USE_CUDA OFF)
       set(ALL_PT_TARGETS "torch_python;${C10_LIB};${TORCH_CPU_LIB};${TORCH_LIB}")
     else()
       # @todo test if we can remove this

```

