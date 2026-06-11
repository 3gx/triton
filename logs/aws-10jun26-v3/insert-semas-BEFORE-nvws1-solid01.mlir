#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#loc = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":425:0)
#loc1 = loc(unknown)
#loc2 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":497:8)
#loc26 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":175:12)
#loc27 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":378:12)
#loc36 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":58:42)
#loc44 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":67:25)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#loc71 = loc("sm_scale"(#loc))
#loc72 = loc("M"(#loc))
#loc73 = loc("Z"(#loc))
#loc74 = loc("H"(#loc))
#loc75 = loc("desc_q"(#loc))
#loc76 = loc("desc_k"(#loc))
#loc77 = loc("desc_v"(#loc))
#loc78 = loc("desc_o"(#loc))
#loc104 = loc(callsite(#loc27 at #loc2))
#loc116 = loc("m_ij"(#loc36))
#loc122 = loc("l_ij"(#loc44))
#loc160 = loc(callsite(#loc26 at #loc104))
#loc183 = loc(callsite(#loc116 at #loc160))
#loc189 = loc(callsite(#loc122 at #loc160))
#loc204 = loc(callsite(#loc1 at #loc183))
#loc206 = loc(callsite(#loc1 at #loc189))
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd(%sm_scale: f32 loc("sm_scale"(#loc)), %M: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("M"(#loc)), %Z: i32 loc("Z"(#loc)), %H: i32 {tt.divisibility = 16 : i32} loc("H"(#loc)), %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("desc_q"(#loc)), %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("desc_k"(#loc)), %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("desc_v"(#loc)), %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("desc_o"(#loc))) attributes {noinline = false} {
    %false = arith.constant false loc(#loc1)
    %true = arith.constant true loc(#loc1)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c128_i64 = arith.constant 128 : i64 loc(#loc1)
    %c1_i64 = arith.constant 1 : i64 loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc79)
    %cst = arith.constant 1.44269502 : f32 loc(#loc79)
    %c0_i32 = arith.constant 0 : i32 loc(#loc79)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear> loc(#loc1)
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc1)
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc1)
    %pid = tt.get_program_id x : i32 loc(#loc80)
    %off_hz = tt.get_program_id y : i32 loc(#loc81)
    %y_dim = arith.muli %Z, %H : i32 loc(#loc82)
    %y_dim_3 = arith.muli %y_dim, %c1024_i32 : i32 loc(#loc83)
    %desc_q_4 = tt.make_tensor_descriptor %desc_q, [%y_dim_3, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>> loc(#loc142)
    %desc_q_5 = tt.make_tensor_descriptor %desc_q, [%y_dim_3, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>> loc(#loc142)
    %desc_v_6 = tt.make_tensor_descriptor %desc_v, [%y_dim_3, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>> loc(#loc143)
    %desc_k_7 = tt.make_tensor_descriptor %desc_k, [%y_dim_3, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>> loc(#loc144)
    %desc_o_8 = tt.make_tensor_descriptor %desc_o, [%y_dim_3, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>> loc(#loc145)
    %desc_o_9 = tt.make_tensor_descriptor %desc_o, [%y_dim_3, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>> loc(#loc145)
    %off_z = arith.divsi %off_hz, %H : i32 loc(#loc146)
    %off_h = arith.remsi %off_hz, %H : i32 loc(#loc147)
    %offset_y = arith.muli %H, %c1024_i32 : i32 loc(#loc148)
    %offset_y_10 = arith.muli %off_z, %offset_y : i32 loc(#loc149)
    %offset_y_11 = arith.muli %off_h, %c1024_i32 : i32 loc(#loc150)
    %offset_y_12 = arith.addi %offset_y_10, %offset_y_11 : i32 loc(#loc151)
    %qo_offset_y = arith.muli %pid, %c256_i32 : i32 loc(#loc152)
    %qo_offset_y_13 = arith.addi %offset_y_12, %qo_offset_y : i32 loc(#loc153)
    %0 = arith.addi %qo_offset_y_13, %c128_i32 : i32 loc(#loc96)
    %q0 = arith.addi %qo_offset_y_13, %c128_i32 : i32 loc(#loc154)
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked> loc(#loc155)
    %offs_m0_14 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked> loc(#loc155)
    %offs_m0_15 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked> loc(#loc156)
    %offs_m0_16 = tt.splat %qo_offset_y : i32 -> tensor<128xi32, #blocked> loc(#loc156)
    %offs_m0_17 = arith.addi %offs_m0_15, %offs_m0 : tensor<128xi32, #blocked> loc(#loc156)
    %offs_m0_18 = arith.addi %offs_m0_16, %offs_m0_14 : tensor<128xi32, #blocked> loc(#loc156)
    %qk_scale = arith.mulf %sm_scale, %cst : f32 loc(#loc157)
    %q0_19 = tt.descriptor_load %desc_q_4[%qo_offset_y_13, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1> loc(#loc154)
    %q0_20 = tt.descriptor_load %desc_q_5[%q0, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1> loc(#loc154)
    %q0_0 = ttg.local_alloc %q0_19 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem> loc(#loc158)
    %q0_1 = ttg.local_alloc %q0_20 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem> loc(#loc159)
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc173)
    %m_ij_21 = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc173)
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear> loc(#loc174)
    %qk_22 = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear> loc(#loc174)
    %qk_0, %qk_0_23 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc175)
    %qk_1, %qk_1_24 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc176)
    %acc_0, %acc_0_25 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc177)
    %acc_1, %acc_1_26 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc178)
    %acc = ttng.tmem_store %cst_0, %acc_0[%acc_0_25], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc179)
    %acc_27 = ttng.tmem_store %cst_0, %acc_1[%acc_1_26], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc179)
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc161)
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc162)
    %alpha = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> loc(#loc180)
    %alpha_28 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> loc(#loc180)
    %offsetkv_y:9 = scf.for %offsetkv_y_48 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%offset_y_49 = %offset_y_12, %arg10 = %cst_2, %arg11 = %cst_1, %qk_0_50 = %qk_0_23, %acc_51 = %acc, %arg14 = %cst_2, %arg15 = %cst_1, %qk_1_52 = %qk_1_24, %acc_53 = %acc_27) -> (i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      nvws.descriptor_load %desc_k_7[%offset_y_49, %c0_i32] 32768 %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc161)
      %k_54 = ttg.memdesc_trans %k {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable> loc(#loc161)
      nvws.descriptor_load %desc_v_6[%offset_y_49, %c0_i32] 32768 %v {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable> loc(#loc162)
      %qk_55 = ttng.tc_gen5_mma %q0_0, %k_54, %qk_0[%qk_0_50], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc182)
      %qk_56 = ttng.tc_gen5_mma %q0_1, %k_54, %qk_1[%qk_1_52], %false, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc182)
      %qk_57, %qk_58 = ttng.tmem_load %qk_0[%qk_55] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1> loc(#loc182)
      %qk_59 = ttg.convert_layout %qk_57 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear> loc(#loc182)
      %qk_60, %qk_61 = ttng.tmem_load %qk_1[%qk_56] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1> loc(#loc182)
      %qk_62 = ttg.convert_layout %qk_60 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear> loc(#loc182)
      %m_ij_63 = "tt.reduce"(%qk_59) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%m_ij_128: f32 loc(callsite(#loc1 at #loc183)), %m_ij_129: f32 loc(callsite(#loc1 at #loc183))):
        %m_ij_130 = arith.maxnumf %m_ij_128, %m_ij_129 {ttg.partition = array<i32: 5>} : f32 loc(#loc210)
        tt.reduce.return %m_ij_130 {ttg.partition = array<i32: 5>} : f32 loc(#loc203)
      }) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc203)
      %m_ij_64 = "tt.reduce"(%qk_62) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%m_ij_128: f32 loc(callsite(#loc1 at #loc183)), %m_ij_129: f32 loc(callsite(#loc1 at #loc183))):
        %m_ij_130 = arith.maxnumf %m_ij_128, %m_ij_129 {ttg.partition = array<i32: 4>} : f32 loc(#loc210)
        tt.reduce.return %m_ij_130 {ttg.partition = array<i32: 4>} : f32 loc(#loc203)
      }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc203)
      %m_ij_65 = arith.mulf %m_ij_63, %m_ij {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc173)
      %m_ij_66 = arith.mulf %m_ij_64, %m_ij_21 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc173)
      %m_ij_67 = arith.maxnumf %arg11, %m_ij_65 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc184)
      %m_ij_68 = arith.maxnumf %arg15, %m_ij_66 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc184)
      %qk_69 = arith.mulf %qk_59, %qk {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> loc(#loc174)
      %qk_70 = arith.mulf %qk_62, %qk_22 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> loc(#loc174)
      %qk_71 = tt.expand_dims %m_ij_67 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear> loc(#loc185)
      %qk_72 = tt.expand_dims %m_ij_68 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear> loc(#loc185)
      %qk_73 = tt.broadcast %qk_71 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear> loc(#loc186)
      %qk_74 = tt.broadcast %qk_72 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear> loc(#loc186)
      %qk_75 = arith.subf %qk_69, %qk_73 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> loc(#loc186)
      %qk_76 = arith.subf %qk_70, %qk_74 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> loc(#loc186)
      %p = math.exp2 %qk_75 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> loc(#loc187)
      %p_77 = math.exp2 %qk_76 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> loc(#loc187)
      %alpha_78 = arith.subf %arg11, %m_ij_67 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc188)
      %alpha_79 = arith.subf %arg15, %m_ij_68 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc188)
      %alpha_80 = math.exp2 %alpha_78 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc180)
      %alpha_81 = tt.expand_dims %alpha_80 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear> loc(#loc180)
      %alpha_82 = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} true loc(#loc180)
      ttng.tmem_store %alpha_81, %alpha, %alpha_82 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> loc(#loc180)
      %alpha_83 = math.exp2 %alpha_79 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc180)
      %alpha_84 = tt.expand_dims %alpha_83 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear> loc(#loc180)
      %alpha_85 = arith.constant {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} true loc(#loc180)
      ttng.tmem_store %alpha_84, %alpha_28, %alpha_85 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> loc(#loc180)
      %l_ij = "tt.reduce"(%p) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%l_ij_128: f32 loc(callsite(#loc1 at #loc189)), %l_ij_129: f32 loc(callsite(#loc1 at #loc189))):
        %l_ij_130 = arith.addf %l_ij_128, %l_ij_129 {ttg.partition = array<i32: 5>} : f32 loc(#loc211)
        tt.reduce.return %l_ij_130 {ttg.partition = array<i32: 5>} : f32 loc(#loc205)
      }) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc205)
      %l_ij_86 = "tt.reduce"(%p_77) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%l_ij_128: f32 loc(callsite(#loc1 at #loc189)), %l_ij_129: f32 loc(callsite(#loc1 at #loc189))):
        %l_ij_130 = arith.addf %l_ij_128, %l_ij_129 {ttg.partition = array<i32: 4>} : f32 loc(#loc211)
        tt.reduce.return %l_ij_130 {ttg.partition = array<i32: 4>} : f32 loc(#loc205)
      }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc205)
      %acc_87, %acc_88 = ttng.tmem_load %acc_0[%acc_51] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1> loc(#loc179)
      %acc_89, %acc_90 = ttng.tmem_load %acc_1[%acc_53] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1> loc(#loc179)
      %7 = tt.reshape %acc_87 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2> loc(#loc190)
      %8 = tt.reshape %acc_89 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2> loc(#loc190)
      %9 = tt.trans %7 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3> loc(#loc191)
      %10 = tt.trans %8 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3> loc(#loc191)
      %outLHS, %outRHS = tt.split %9 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4> loc(#loc192)
      %outLHS_91, %outRHS_92 = tt.split %10 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4> loc(#loc192)
      %alpha_93, %alpha_94 = ttng.tmem_load %alpha[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5> loc(#loc180)
      %alpha_95 = tt.reshape %alpha_93 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6> loc(#loc180)
      %alpha_96 = ttg.convert_layout %alpha_95 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc180)
      %acc0_97 = tt.expand_dims %alpha_96 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear> loc(#loc193)
      %alpha_98, %alpha_99 = ttng.tmem_load %alpha_28[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5> loc(#loc180)
      %alpha_100 = tt.reshape %alpha_98 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6> loc(#loc180)
      %alpha_101 = ttg.convert_layout %alpha_100 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc180)
      %acc0_102 = tt.expand_dims %alpha_101 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear> loc(#loc193)
      %acc0_103 = ttg.convert_layout %acc0_97 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4> loc(#loc207)
      %acc0_104 = ttg.convert_layout %acc0_102 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4> loc(#loc207)
      %acc0_105 = tt.broadcast %acc0_103 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4> loc(#loc207)
      %acc0_106 = tt.broadcast %acc0_104 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4> loc(#loc207)
      %acc0_107 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS, %acc0_105 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4> loc(#loc207)
      %acc0_108 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS_91, %acc0_106 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4> loc(#loc207)
      %acc1 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS, %acc0_105 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4> loc(#loc208)
      %acc1_109 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS_92, %acc0_106 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4> loc(#loc208)
      %acc_110 = tt.join %acc0_107, %acc1 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3> loc(#loc196)
      %acc_111 = tt.join %acc0_108, %acc1_109 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3> loc(#loc196)
      %acc_112 = tt.trans %acc_110 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2> loc(#loc197)
      %acc_113 = tt.trans %acc_111 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2> loc(#loc197)
      %acc_114 = tt.reshape %acc_112 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear> loc(#loc198)
      %acc_115 = tt.reshape %acc_113 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear> loc(#loc198)
      %p_116 = arith.truncf %p {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear> loc(#loc199)
      %p_117 = arith.truncf %p_77 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear> loc(#loc199)
      %acc_118 = ttng.tmem_alloc %p_116 {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory> loc(#loc179)
      %acc_119 = ttng.tmem_alloc %p_117 {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory> loc(#loc179)
      %acc_120 = ttng.tmem_store %acc_114, %acc_0[%acc_88], %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 2>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc179)
      %acc_121 = ttng.tmem_store %acc_115, %acc_1[%acc_90], %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 3>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc179)
      %acc_122 = ttng.tc_gen5_mma %acc_118, %v, %acc_0[%acc_120], %true, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 2>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc179)
      %acc_123 = ttng.tc_gen5_mma %acc_119, %v, %acc_1[%acc_121], %true, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 3>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc179)
      %l_i0 = arith.mulf %arg10, %alpha_80 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc200)
      %l_i0_124 = arith.mulf %arg14, %alpha_83 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc200)
      %l_i0_125 = arith.addf %l_i0, %l_ij {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc201)
      %l_i0_126 = arith.addf %l_i0_124, %l_ij_86 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc201)
      %offsetkv_y_127 = arith.addi %offset_y_49, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : i32 loc(#loc164)
      scf.yield {ttg.partition = array<i32: 0, 1, 3, 4, 5>} %offsetkv_y_127, %l_i0_125, %m_ij_67, %qk_58, %acc_122, %l_i0_126, %m_ij_68, %qk_61, %acc_123 : i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token loc(#loc165)
    } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 3, 4, 5>, ttg.partition.outputs = [array<i32: 3>, array<i32: 5>, array<i32: 5>, array<i32: 1>, array<i32: 0>, array<i32: 4>, array<i32: 4>, array<i32: 1>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32} loc(#loc212)
    %acc_29, %acc_30 = ttng.tmem_load %acc_0[%offsetkv_y#4] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1> loc(#loc179)
    %acc_31 = ttg.convert_layout %acc_29 : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear> loc(#loc179)
    %acc_32, %acc_33 = ttng.tmem_load %acc_1[%offsetkv_y#8] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1> loc(#loc179)
    %acc_34 = ttg.convert_layout %acc_32 : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear> loc(#loc179)
    %m_i0 = math.log2 %offsetkv_y#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc166)
    %m_i0_35 = math.log2 %offsetkv_y#5 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc166)
    %m_i0_36 = arith.addf %offsetkv_y#2, %m_i0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc167)
    %m_i0_37 = arith.addf %offsetkv_y#6, %m_i0_35 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> loc(#loc167)
    %acc0 = tt.expand_dims %offsetkv_y#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear> loc(#loc168)
    %acc0_38 = tt.expand_dims %offsetkv_y#5 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear> loc(#loc168)
    %acc0_39 = tt.broadcast %acc0 : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear> loc(#loc169)
    %acc0_40 = tt.broadcast %acc0_38 : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear> loc(#loc169)
    %acc0_41 = arith.divf %acc_31, %acc0_39 : tensor<128x128xf32, #linear> loc(#loc169)
    %acc0_42 = arith.divf %acc_34, %acc0_40 : tensor<128x128xf32, #linear> loc(#loc169)
    %m_ptrs0 = arith.muli %off_hz, %c1024_i32 : i32 loc(#loc170)
    %m_ptrs0_43 = tt.addptr %M, %m_ptrs0 : !tt.ptr<f32>, i32 loc(#loc171)
    %m_ptrs0_44 = tt.splat %m_ptrs0_43 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked> loc(#loc172)
    %m_ptrs0_45 = tt.splat %m_ptrs0_43 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked> loc(#loc172)
    %m_ptrs0_46 = tt.addptr %m_ptrs0_44, %offs_m0_17 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked> loc(#loc172)
    %m_ptrs0_47 = tt.addptr %m_ptrs0_45, %offs_m0_18 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked> loc(#loc172)
    %1 = ttg.convert_layout %m_i0_36 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked> loc(#loc140)
    %2 = ttg.convert_layout %m_i0_37 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked> loc(#loc140)
    tt.store %m_ptrs0_46, %1 : tensor<128x!tt.ptr<f32>, #blocked> loc(#loc140)
    tt.store %m_ptrs0_47, %2 : tensor<128x!tt.ptr<f32>, #blocked> loc(#loc140)
    %3 = arith.truncf %acc0_41 : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear> loc(#loc141)
    %4 = arith.truncf %acc0_42 : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear> loc(#loc141)
    %5 = ttg.convert_layout %3 : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1> loc(#loc96)
    %6 = ttg.convert_layout %4 : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1> loc(#loc96)
    tt.descriptor_store %desc_o_8[%qo_offset_y_13, %c0_i32], %5 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1> loc(#loc96)
    tt.descriptor_store %desc_o_9[%0, %c0_i32], %6 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1> loc(#loc96)
    tt.return loc(#loc70)
  } loc(#loc)
} loc(#loc)
#loc3 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":447:24)
#loc4 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":448:27)
#loc5 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":449:16)
#loc6 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":449:20)
#loc7 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":240:70)
#loc8 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":451:8)
#loc9 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":457:8)
#loc10 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":463:8)
#loc11 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":469:8)
#loc12 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":331:22)
#loc13 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":332:21)
#loc14 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":334:32)
#loc15 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":334:24)
#loc16 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":334:45)
#loc17 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":334:37)
#loc18 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":335:39)
#loc19 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":335:29)
#loc20 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":416:35)
#loc21 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":347:21)
#loc22 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":337:47)
#loc23 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":337:34)
#loc24 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":345:16)
#loc25 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":58:47)
#loc28 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":62:22)
#loc29 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":51:19)
#loc30 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":96:23)
#loc31 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":156:12)
#loc32 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":157:24)
#loc33 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":65:25)
#loc34 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":152:12)
#loc35 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/triton/language/standard.py":195:40)
#loc37 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/triton/language/standard.py":170:27)
#loc38 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":58:31)
#loc39 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":62:38)
#loc40 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":62:33)
#loc41 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":63:21)
#loc42 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":65:31)
#loc43 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/triton/language/standard.py":315:36)
#loc45 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/triton/language/standard.py":275:15)
#loc46 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":74:33)
#loc47 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":74:65)
#loc48 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":74:21)
#loc49 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":76:42)
#loc50 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":256:8)
#loc51 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":76:36)
#loc52 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":77:36)
#loc53 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":81:28)
#loc54 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":81:48)
#loc55 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":81:59)
#loc56 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":94:13)
#loc57 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":100:22)
#loc58 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":100:30)
#loc59 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":178:22)
#loc60 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":178:8)
#loc61 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":412:25)
#loc62 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":412:12)
#loc63 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":413:23)
#loc64 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":413:18)
#loc65 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":414:27)
#loc66 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":414:18)
#loc67 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":414:35)
#loc68 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":415:22)
#loc69 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":416:43)
#loc70 = loc("/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/tutorials/fused-attention-ws-device-tma-1.py":475:4)
#loc79 = loc(callsite(#loc1 at #loc2))
#loc80 = loc("pid"(#loc3))
#loc81 = loc("off_hz"(#loc4))
#loc82 = loc("y_dim"(#loc5))
#loc83 = loc("y_dim"(#loc6))
#loc84 = loc("desc_q"(#loc8))
#loc85 = loc("desc_v"(#loc9))
#loc86 = loc("desc_k"(#loc10))
#loc87 = loc("desc_o"(#loc11))
#loc88 = loc("off_z"(#loc12))
#loc89 = loc("off_h"(#loc13))
#loc90 = loc("offset_y"(#loc14))
#loc91 = loc("offset_y"(#loc15))
#loc92 = loc("offset_y"(#loc16))
#loc93 = loc("offset_y"(#loc17))
#loc94 = loc("qo_offset_y"(#loc18))
#loc95 = loc("qo_offset_y"(#loc19))
#loc96 = loc(callsite(#loc20 at #loc2))
#loc97 = loc("q0"(#loc21))
#loc98 = loc("offs_m0"(#loc22))
#loc99 = loc("offs_m0"(#loc23))
#loc100 = loc("qk_scale"(#loc24))
#loc101 = loc("q0_0"(#loc21))
#loc102 = loc("q0_1"(#loc21))
#loc103 = loc("m_ij"(#loc25))
#loc105 = loc("qk"(#loc28))
#loc106 = loc("qk_0"(#loc29))
#loc107 = loc("qk_1"(#loc29))
#loc108 = loc("acc_0"(#loc30))
#loc109 = loc("acc_1"(#loc30))
#loc110 = loc("acc"(#loc30))
#loc111 = loc("k"(#loc31))
#loc112 = loc("v"(#loc32))
#loc113 = loc("alpha"(#loc33))
#loc114 = loc("acc0"(#loc34))
#loc115 = loc("qk"(#loc29))
#loc117 = loc("m_ij"(#loc38))
#loc118 = loc("qk"(#loc39))
#loc119 = loc("qk"(#loc40))
#loc120 = loc("p"(#loc41))
#loc121 = loc("alpha"(#loc42))
#loc123 = loc("acc0"(#loc49))
#loc124 = loc("acc0"(#loc51))
#loc125 = loc("acc1"(#loc52))
#loc126 = loc("acc"(#loc53))
#loc127 = loc("acc"(#loc54))
#loc128 = loc("acc"(#loc55))
#loc129 = loc("p"(#loc56))
#loc130 = loc("l_i0"(#loc57))
#loc131 = loc("l_i0"(#loc58))
#loc132 = loc("offsetkv_y"(#loc59))
#loc133 = loc("m_i0"(#loc61))
#loc134 = loc("m_i0"(#loc62))
#loc135 = loc("acc0"(#loc63))
#loc136 = loc("acc0"(#loc64))
#loc137 = loc("m_ptrs0"(#loc65))
#loc138 = loc("m_ptrs0"(#loc66))
#loc139 = loc("m_ptrs0"(#loc67))
#loc140 = loc(callsite(#loc68 at #loc2))
#loc141 = loc(callsite(#loc69 at #loc2))
#loc142 = loc(callsite(#loc7 at #loc84))
#loc143 = loc(callsite(#loc7 at #loc85))
#loc144 = loc(callsite(#loc7 at #loc86))
#loc145 = loc(callsite(#loc7 at #loc87))
#loc146 = loc(callsite(#loc88 at #loc2))
#loc147 = loc(callsite(#loc89 at #loc2))
#loc148 = loc(callsite(#loc90 at #loc2))
#loc149 = loc(callsite(#loc91 at #loc2))
#loc150 = loc(callsite(#loc92 at #loc2))
#loc151 = loc(callsite(#loc93 at #loc2))
#loc152 = loc(callsite(#loc94 at #loc2))
#loc153 = loc(callsite(#loc95 at #loc2))
#loc154 = loc(callsite(#loc97 at #loc2))
#loc155 = loc(callsite(#loc98 at #loc2))
#loc156 = loc(callsite(#loc99 at #loc2))
#loc157 = loc(callsite(#loc100 at #loc2))
#loc158 = loc(callsite(#loc101 at #loc2))
#loc159 = loc(callsite(#loc102 at #loc2))
#loc161 = loc(callsite(#loc111 at #loc104))
#loc162 = loc(callsite(#loc112 at #loc104))
#loc163 = loc("l_i0"(#loc114))
#loc164 = loc(callsite(#loc132 at #loc104))
#loc165 = loc(callsite(#loc60 at #loc104))
#loc166 = loc(callsite(#loc133 at #loc2))
#loc167 = loc(callsite(#loc134 at #loc2))
#loc168 = loc(callsite(#loc135 at #loc2))
#loc169 = loc(callsite(#loc136 at #loc2))
#loc170 = loc(callsite(#loc137 at #loc2))
#loc171 = loc(callsite(#loc138 at #loc2))
#loc172 = loc(callsite(#loc139 at #loc2))
#loc173 = loc(callsite(#loc103 at #loc160))
#loc174 = loc(callsite(#loc105 at #loc160))
#loc175 = loc(callsite(#loc106 at #loc160))
#loc176 = loc(callsite(#loc107 at #loc160))
#loc177 = loc(callsite(#loc108 at #loc160))
#loc178 = loc(callsite(#loc109 at #loc160))
#loc179 = loc(callsite(#loc110 at #loc160))
#loc180 = loc(callsite(#loc113 at #loc160))
#loc181 = loc("l_i0_1"(#loc163))
#loc182 = loc(callsite(#loc115 at #loc160))
#loc184 = loc(callsite(#loc117 at #loc160))
#loc185 = loc(callsite(#loc118 at #loc160))
#loc186 = loc(callsite(#loc119 at #loc160))
#loc187 = loc(callsite(#loc120 at #loc160))
#loc188 = loc(callsite(#loc121 at #loc160))
#loc190 = loc(callsite(#loc46 at #loc160))
#loc191 = loc(callsite(#loc47 at #loc160))
#loc192 = loc(callsite(#loc48 at #loc160))
#loc193 = loc(callsite(#loc123 at #loc160))
#loc194 = loc(callsite(#loc124 at #loc160))
#loc195 = loc(callsite(#loc125 at #loc160))
#loc196 = loc(callsite(#loc126 at #loc160))
#loc197 = loc(callsite(#loc127 at #loc160))
#loc198 = loc(callsite(#loc128 at #loc160))
#loc199 = loc(callsite(#loc129 at #loc160))
#loc200 = loc(callsite(#loc130 at #loc160))
#loc201 = loc(callsite(#loc131 at #loc160))
#loc202 = loc("m_i0"(#loc181))
#loc203 = loc(callsite(#loc35 at #loc183))
#loc205 = loc(callsite(#loc43 at #loc189))
#loc207 = loc(callsite(#loc50 at #loc194))
#loc208 = loc(callsite(#loc50 at #loc195))
#loc209 = loc("offsetkv_y"(#loc202))
#loc210 = loc(callsite(#loc37 at #loc203))
#loc211 = loc(callsite(#loc45 at #loc205))
#loc212 = loc(callsite(#loc209 at #loc104))


==== NVWS InsertSemas (commit 4: ACCESS-DAG + OWNER-DAG + SYNC-DAG + EMIT) ====
function: @_attn_fwd
groups: 6
GROUP buffer.id=4 memory=tmem members=3
  members: m0[0,128) m1[64,65) m2[0,64)
  pieces: P0=[0,64){m0,m2}c0 P1=[64,65){m0,m1}c0 P2=[65,128){m0}c0
  footprints: m0={P0,P1,P2} m1={P1} m2={P0}
ACCESS-DAG
|- func @_attn_fwd
|  |- scf.for (WS, tag=0) effects{P0:W,P1:W,P2:W}
|  |  |- W  m0  ttng.tc_gen5_mma {1}
|  |  |- R  m0  ttng.tmem_load {5}
|  |  |- W  m1  ttng.tmem_store {5}
|  |  |- R  m1  ttng.tmem_load {0}
|  |  |- W  m2  ttng.tmem_alloc {5}
|  |  |- R  m2  ttng.tc_gen5_mma {1}
OWNER-DAG
|- func @_attn_fwd
|  |- scf.for (WS, tag=0) pieces{P0:W:{1},P1:W:{1},P2:W:{1}}
|  |  |- ENTER pieces{P0:W:{1},P1:W:{1},P2:W:{1}}
|  |  |- W  m0  ttng.tc_gen5_mma {1}
|  |  |- R  m0  ttng.tmem_load {5}
|  |  |- W  m1  ttng.tmem_store {5}
|  |  |- R  m1  ttng.tmem_load {0}
|  |  |- W  m2  ttng.tmem_alloc {5}
|  |  |- R  m2  ttng.tc_gen5_mma {1}
|  |  |- EXIT pieces{P0:W:{1},P1:W:{1},P2:W:{1}}
SYNC-DAG
|- func @_attn_fwd
|  |- a  S3(2)  root  ; entry
|  |- scf.for (WS, tag=0) pieces{P0:W:{1},P1:W:{1},P2:W:{1}} parts{0,1,5} thread{c0:{1}}
|  |  |- ENTER pieces{P0:W:{1},P1:W:{1},P2:W:{1}}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r  S0  {1} [tc5mma]
|  |  |- a  S0  {5}
|  |  |- R m0  ttng.tmem_load {5}
|  |  |- W m1  ttng.tmem_store {5}
|  |  |- r  S1  {5} [none]
|  |  |- r  S3  {5} [none]
|  |  |- a  S1  {0}
|  |  |- R m1  ttng.tmem_load {0}
|  |  |- r  S3  {0} [none]
|  |  |- W m2  ttng.tmem_alloc {5}
|  |  |- r  S2  {5} [none]
|  |  |- a  S2  {1}
|  |  |- R m2  ttng.tc_gen5_mma {1}
|  |  |- a  S3(2)  {1}
|  |  |- EXIT pieces{P0:W:{1},P1:W:{1},P2:W:{1}} yield{c0: a S3}
  SEMAS c0: S0{count=1} S1{count=1} S2{count=1} S3{count=2 entry inherit={@0.1}}
  BACKING: numStages=1 anchor=before scf.for(tag=0)
GROUP buffer.id=5 memory=tmem members=3
  members: m0[0,128) m1[64,65) m2[0,64)
  pieces: P0=[0,64){m0,m2}c0 P1=[64,65){m0,m1}c0 P2=[65,128){m0}c0
  footprints: m0={P0,P1,P2} m1={P1} m2={P0}
ACCESS-DAG
|- func @_attn_fwd
|  |- scf.for (WS, tag=0) effects{P0:W,P1:W,P2:W}
|  |  |- W  m0  ttng.tc_gen5_mma {1}
|  |  |- R  m0  ttng.tmem_load {4}
|  |  |- W  m1  ttng.tmem_store {4}
|  |  |- R  m1  ttng.tmem_load {0}
|  |  |- W  m2  ttng.tmem_alloc {4}
|  |  |- R  m2  ttng.tc_gen5_mma {1}
OWNER-DAG
|- func @_attn_fwd
|  |- scf.for (WS, tag=0) pieces{P0:W:{1},P1:W:{1},P2:W:{1}}
|  |  |- ENTER pieces{P0:W:{1},P1:W:{1},P2:W:{1}}
|  |  |- W  m0  ttng.tc_gen5_mma {1}
|  |  |- R  m0  ttng.tmem_load {4}
|  |  |- W  m1  ttng.tmem_store {4}
|  |  |- R  m1  ttng.tmem_load {0}
|  |  |- W  m2  ttng.tmem_alloc {4}
|  |  |- R  m2  ttng.tc_gen5_mma {1}
|  |  |- EXIT pieces{P0:W:{1},P1:W:{1},P2:W:{1}}
SYNC-DAG
|- func @_attn_fwd
|  |- a  S3(2)  root  ; entry
|  |- scf.for (WS, tag=0) pieces{P0:W:{1},P1:W:{1},P2:W:{1}} parts{0,1,4} thread{c0:{1}}
|  |  |- ENTER pieces{P0:W:{1},P1:W:{1},P2:W:{1}}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r  S0  {1} [tc5mma]
|  |  |- a  S0  {4}
|  |  |- R m0  ttng.tmem_load {4}
|  |  |- W m1  ttng.tmem_store {4}
|  |  |- r  S1  {4} [none]
|  |  |- r  S3  {4} [none]
|  |  |- a  S1  {0}
|  |  |- R m1  ttng.tmem_load {0}
|  |  |- r  S3  {0} [none]
|  |  |- W m2  ttng.tmem_alloc {4}
|  |  |- r  S2  {4} [none]
|  |  |- a  S2  {1}
|  |  |- R m2  ttng.tc_gen5_mma {1}
|  |  |- a  S3(2)  {1}
|  |  |- EXIT pieces{P0:W:{1},P1:W:{1},P2:W:{1}} yield{c0: a S3}
  SEMAS c0: S0{count=1} S1{count=1} S2{count=1} S3{count=2 entry inherit={@0.1}}
  BACKING: numStages=1 anchor=before scf.for(tag=0)
GROUP buffer.id=2 memory=tmem members=1
  members: m0[0,128)
  pieces: P0=[0,128){m0}c0
  footprints: m0={P0}
ACCESS-DAG
|- func @_attn_fwd
|  |- W  m0  ttng.tmem_store root
|  |- scf.for (WS, tag=0) effects{P0:W}
|  |  |- R  m0  ttng.tmem_load {0}
|  |  |- W  m0  ttng.tmem_store {0}
|  |  |- W  m0  ttng.tc_gen5_mma {1}
|  |- R  m0  ttng.tmem_load root
OWNER-DAG
|- func @_attn_fwd
|  |- W  m0  ttng.tmem_store root
|  |- scf.for (WS, tag=0) pieces{P0:W:{0}}
|  |  |- ENTER pieces{P0:W:{0}}
|  |  |- R  m0  ttng.tmem_load {0}
|  |  |- W  m0  ttng.tmem_store {0}
|  |  |- W  m0  ttng.tc_gen5_mma {1}
|  |  |- EXIT pieces{P0:W:{0}}
|  |- R  m0  ttng.tmem_load root
SYNC-DAG
|- func @_attn_fwd
|  |- a  S1  root  ; entry
|  |- W m0  ttng.tmem_store root
|  |- scf.for (WS, tag=0) pieces{P0:W:{0}} parts{0,1} thread{c0:{0}}
|  |  |- ENTER pieces{P0:W:{0}}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- W m0  ttng.tmem_store {0}
|  |  |- r  S0  {0} [none]
|  |  |- a  S0  {1}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r  S1  {1} [tc5mma]
|  |  |- a  S1  {0}
|  |  |- EXIT pieces{P0:W:{0}} yield{c0: a S1}
|  |- r  S2  {@0.0} [none]
|  |- a  S2  root
|  |- R m0  ttng.tmem_load root
  SEMAS c0: S0{count=1} S1{count=1 entry inherit=root} S2{count=1}
  BACKING: numStages=1 anchor=before scf.for(tag=0)
GROUP buffer.id=3 memory=tmem members=1
  members: m0[0,128)
  pieces: P0=[0,128){m0}c0
  footprints: m0={P0}
ACCESS-DAG
|- func @_attn_fwd
|  |- W  m0  ttng.tmem_store root
|  |- scf.for (WS, tag=0) effects{P0:W}
|  |  |- R  m0  ttng.tmem_load {0}
|  |  |- W  m0  ttng.tmem_store {0}
|  |  |- W  m0  ttng.tc_gen5_mma {1}
|  |- R  m0  ttng.tmem_load root
OWNER-DAG
|- func @_attn_fwd
|  |- W  m0  ttng.tmem_store root
|  |- scf.for (WS, tag=0) pieces{P0:W:{0}}
|  |  |- ENTER pieces{P0:W:{0}}
|  |  |- R  m0  ttng.tmem_load {0}
|  |  |- W  m0  ttng.tmem_store {0}
|  |  |- W  m0  ttng.tc_gen5_mma {1}
|  |  |- EXIT pieces{P0:W:{0}}
|  |- R  m0  ttng.tmem_load root
SYNC-DAG
|- func @_attn_fwd
|  |- a  S1  root  ; entry
|  |- W m0  ttng.tmem_store root
|  |- scf.for (WS, tag=0) pieces{P0:W:{0}} parts{0,1} thread{c0:{0}}
|  |  |- ENTER pieces{P0:W:{0}}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- W m0  ttng.tmem_store {0}
|  |  |- r  S0  {0} [none]
|  |  |- a  S0  {1}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r  S1  {1} [tc5mma]
|  |  |- a  S1  {0}
|  |  |- EXIT pieces{P0:W:{0}} yield{c0: a S1}
|  |- r  S2  {@0.0} [none]
|  |- a  S2  root
|  |- R m0  ttng.tmem_load root
  SEMAS c0: S0{count=1} S1{count=1 entry inherit=root} S2{count=1}
  BACKING: numStages=1 anchor=before scf.for(tag=0)
GROUP buffer.id=none#1 memory=local members=1
  members: m0[0,128)
  pieces: P0=[0,128){m0}c0
  footprints: m0={P0}
ACCESS-DAG
|- func @_attn_fwd
|  |- scf.for (WS, tag=0) effects{P0:W}
|  |  |- W  m0  nvws.descriptor_load {3}
|  |  |- R  m0  ttng.tc_gen5_mma {1}
|  |  |- R  m0  ttng.tc_gen5_mma {1}
OWNER-DAG
|- func @_attn_fwd
|  |- scf.for (WS, tag=0) pieces{P0:W:{3}}
|  |  |- ENTER pieces{P0:W:{3}}
|  |  |- W  m0  nvws.descriptor_load {3}
|  |  |- R  m0  ttng.tc_gen5_mma {1}
|  |  |- R  m0  ttng.tc_gen5_mma {1}
|  |  |- EXIT pieces{P0:W:{3}}
SYNC-DAG
|- func @_attn_fwd
|  |- a  S1  root  ; entry
|  |- scf.for (WS, tag=0) pieces{P0:W:{3}} parts{1,3} thread{c0:{3}}
|  |  |- ENTER pieces{P0:W:{3}}
|  |  |- W m0  nvws.descriptor_load {3}
|  |  |- r  S0  {3} [tma_load]
|  |  |- a  S0  {1}
|  |  |- R m0  ttng.tc_gen5_mma {1}
|  |  |- R m0  ttng.tc_gen5_mma {1}
|  |  |- r  S1  {1} [tc5mma]
|  |  |- a  S1  {3}
|  |  |- EXIT pieces{P0:W:{3}} yield{c0: a S1}
  SEMAS c0: S0{count=1} S1{count=1 entry inherit={@0.3}}
  BACKING: numStages=1 anchor=before scf.for(tag=0)
GROUP buffer.id=none#2 memory=local members=1
  members: m0[0,128)
  pieces: P0=[0,128){m0}c0
  footprints: m0={P0}
ACCESS-DAG
|- func @_attn_fwd
|  |- scf.for (WS, tag=0) effects{P0:W}
|  |  |- W  m0  nvws.descriptor_load {3}
|  |  |- R  m0  ttng.tc_gen5_mma {1}
|  |  |- R  m0  ttng.tc_gen5_mma {1}
OWNER-DAG
|- func @_attn_fwd
|  |- scf.for (WS, tag=0) pieces{P0:W:{3}}
|  |  |- ENTER pieces{P0:W:{3}}
|  |  |- W  m0  nvws.descriptor_load {3}
|  |  |- R  m0  ttng.tc_gen5_mma {1}
|  |  |- R  m0  ttng.tc_gen5_mma {1}
|  |  |- EXIT pieces{P0:W:{3}}
SYNC-DAG
|- func @_attn_fwd
|  |- a  S1  root  ; entry
|  |- scf.for (WS, tag=0) pieces{P0:W:{3}} parts{1,3} thread{c0:{3}}
|  |  |- ENTER pieces{P0:W:{3}}
|  |  |- W m0  nvws.descriptor_load {3}
|  |  |- r  S0  {3} [tma_load]
|  |  |- a  S0  {1}
|  |  |- R m0  ttng.tc_gen5_mma {1}
|  |  |- R m0  ttng.tc_gen5_mma {1}
|  |  |- r  S1  {1} [tc5mma]
|  |  |- a  S1  {3}
|  |  |- EXIT pieces{P0:W:{3}} yield{c0: a S1}
  SEMAS c0: S0{count=1} S1{count=1 entry inherit={@0.3}}
  BACKING: numStages=1 anchor=before scf.for(tag=0)
