// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-assign-stage-phase -cse | FileCheck %s --check-prefix=ASP

// Two exact-alias epilogue members share one depth-2 physical allocation.
// The first member uses slot 0 and the second uses slot 1, so each
// read-to-next-write release must target the successor slot rather than the
// source read's slot.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // SEMA-LABEL: @fused_alias_depth_two
  // SEMA: [[BASE:%.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32}
  // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] true
  // SEMA: [[FULL0:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] false
  // SEMA: [[EMPTY1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] false
  // SEMA: [[FULL1:%.*]] = nvws.semaphore.create [[BASE]], [[BASE]] false
  // SEMA: scf.for
  // SEMA: [[W0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[W0_ZERO]]]
  // SEMA: [[W0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: nvws.semaphore.release [[FULL0]][[[W0_REL_ZERO]]], [[W0_TOK]]
  // SEMA: [[R0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
  // SEMA: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[R0_ZERO]]]
  // SEMA: [[TO_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
  // SEMA: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]]
  // SEMA: [[W1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[W1_ZERO]]]
  // SEMA: [[W1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: nvws.semaphore.release [[FULL1]][[[W1_REL_ZERO]]], [[W1_TOK]]
  // SEMA: [[R1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
  // SEMA: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[R1_ZERO]]]
  // SEMA: [[TO_NEXT_M0:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
  // SEMA: nvws.semaphore.release [[ENTRY]][[[TO_NEXT_M0]]], [[R1_TOK]]

  // ASP-LABEL: @fused_alias_depth_two
  // ASP: [[ENTRY:%.*]] = nvws.semaphore.create
  // ASP: [[FULL0:%.*]] = nvws.semaphore.create
  // ASP: [[EMPTY1:%.*]] = nvws.semaphore.create
  // ASP: [[FULL1:%.*]] = nvws.semaphore.create
  // ASP: scf.for {{.*}} iter_args([[CURSOR:%.*]] = {{%.*}}
  // ASP: [[SLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: arith.shli {{%.*}}, [[SLOT0]] {ttg.partition = array<i32: 4>} : i32
  // ASP: [[W0_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[SLOT0]], {{%.*}}]
  // ASP: nvws.semaphore.release [[FULL0]][[[SLOT0]]], [[W0_TOK]]
  // ASP: [[R0_TOK:%.*]] = nvws.semaphore.acquire [[FULL0]][[[SLOT0]], {{%.*}}]
  // ASP: [[TO_M1_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_M1_REM:%.*]] = arith.remsi [[TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
  // ASP: nvws.semaphore.release [[EMPTY1]][[[TO_M1]]], [[R0_TOK]]
  // ASP: [[NEXT_RAW:%.*]] = arith.addi [[SLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: [[SLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[NEXT_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: arith.shli {{%.*}}, [[SLOT1]] {ttg.partition = array<i32: 4>} : i32
  // ASP: [[W1_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY1]][[[SLOT1]], {{%.*}}]
  // ASP: nvws.semaphore.release [[FULL1]][[[SLOT1]]], [[W1_TOK]]
  // ASP: [[R1_TOK:%.*]] = nvws.semaphore.acquire [[FULL1]][[[SLOT1]], {{%.*}}]
  // ASP: [[TO_M0_RAW:%.*]] = arith.addi [[SLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_M0_REM:%.*]] = arith.remsi [[TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
  // ASP: nvws.semaphore.release [[ENTRY]][[[TO_M0]]], [[R1_TOK]]
  // ASP: scf.yield {{.*}} [[SLOT1]],
  tt.func @fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    %m0 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %m1 = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 500 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>

    scf.for %iv = %lb to %ub step %step : i32 {
      ttg.local_store %v0, %m0 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %r0 = ttg.local_load %m0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      ttg.local_store %v1, %m1 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %r1 = ttg.local_load %m1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Planner-authored aliases may be different views of one staged backing.
  // Here the smaller member covers the prefix of the larger member.  The
  // read-to-next-write handoff must still target the following physical slot.
  // SEMA-LABEL: @fused_partial_alias_depth_three
  // SEMA: [[PLARGE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32}
  // SEMA: [[PSMALL:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32}
  // SEMA: [[PENTRY:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] true
  // SEMA: [[PFULL0:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] false
  // SEMA: [[PHANDOFF:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] false
  // SEMA: [[PFULL1:%.*]] = nvws.semaphore.create [[PLARGE]], [[PSMALL]] false
  // SEMA: scf.for
  // SEMA: [[PW0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: [[PW0_TOK:%.*]] = nvws.semaphore.acquire [[PENTRY]][[[PW0_ZERO]]]
  // SEMA: [[PR0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
  // SEMA: [[PR0_TOK:%.*]] = nvws.semaphore.acquire [[PFULL0]][[[PR0_ZERO]]]
  // SEMA: [[TO_LARGE:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
  // SEMA: nvws.semaphore.release [[PHANDOFF]][[[TO_LARGE]]], [[PR0_TOK]]
  // SEMA: [[PW1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: [[PW1_TOK:%.*]] = nvws.semaphore.acquire [[PHANDOFF]][[[PW1_ZERO]]]

  // ASP-LABEL: @fused_partial_alias_depth_three
  // ASP: [[PENTRY:%.*]] = nvws.semaphore.create
  // ASP: [[PFULL0:%.*]] = nvws.semaphore.create
  // ASP: [[PHANDOFF:%.*]] = nvws.semaphore.create
  // ASP: [[PFULL1:%.*]] = nvws.semaphore.create
  // ASP: scf.for {{.*}} iter_args([[PCURSOR:%.*]] = {{%.*}}
  // ASP: [[PSLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: [[PW0_TOK:%.*]] = nvws.semaphore.acquire [[PENTRY]][[[PSLOT0]], {{%.*}}]
  // ASP: [[PR0_TOK:%.*]] = nvws.semaphore.acquire [[PFULL0]][[[PSLOT0]], {{%.*}}]
  // ASP: [[TO_LARGE_RAW:%.*]] = arith.addi [[PSLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_LARGE_REM:%.*]] = arith.remsi [[TO_LARGE_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[TO_LARGE_SLOT:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TO_LARGE_REM]] {ttg.partition = array<i32: 2>} : i32
  // ASP: nvws.semaphore.release [[PHANDOFF]][[[TO_LARGE_SLOT]]], [[PR0_TOK]]
  // ASP: [[PSLOT1_RAW:%.*]] = arith.addi [[PSLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: [[PSLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[PSLOT1_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: [[PW1_TOK:%.*]] = nvws.semaphore.acquire [[PHANDOFF]][[[PSLOT1]], {{%.*}}]
  tt.func @fused_partial_alias_depth_three(%lb: i32, %ub: i32, %step: i32) {
    %large = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %small = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 502 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %small_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %large_value = arith.constant dense<1.000000e+00> : tensor<256x64xf16, #blocked64>

    scf.for %iv = %lb to %ub step %step : i32 {
      ttg.local_store %small_value, %small {ttg.partition = array<i32: 4>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %small_read = ttg.local_load %small {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_small"(%small_read) {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked64>) -> ()
      ttg.local_store %large_value, %large {ttg.partition = array<i32: 4>} : tensor<256x64xf16, #blocked64> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %large_read = ttg.local_load %large {ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #blocked64>
      "consume_large"(%large_read) {ttg.partition = array<i32: 2>} : (tensor<256x64xf16, #blocked64>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }

  // SEMA-LABEL: @tmem_fused_alias_depth_two
  // SEMA: [[TBASE:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32}
  // SEMA: [[TENTRY:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] true
  // SEMA: [[TFULL0:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] false
  // SEMA: [[TEMPTY1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] false
  // SEMA: [[TFULL1:%.*]] = nvws.semaphore.create [[TBASE]], [[TBASE]] false
  // SEMA: scf.for
  // SEMA: [[TW0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: [[TW0_TOK:%.*]] = nvws.semaphore.acquire [[TENTRY]][[[TW0_ZERO]]]
  // SEMA: [[TW0_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: nvws.semaphore.release [[TFULL0]][[[TW0_REL_ZERO]]], [[TW0_TOK]]
  // SEMA: [[TR0_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
  // SEMA: [[TR0_TOK:%.*]] = nvws.semaphore.acquire [[TFULL0]][[[TR0_ZERO]]]
  // SEMA: [[T_TO_M1:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
  // SEMA: nvws.semaphore.release [[TEMPTY1]][[[T_TO_M1]]], [[TR0_TOK]]
  // SEMA: [[TW1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: [[TW1_TOK:%.*]] = nvws.semaphore.acquire [[TEMPTY1]][[[TW1_ZERO]]]
  // SEMA: [[TW1_REL_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 4>} 0 : i32
  // SEMA: nvws.semaphore.release [[TFULL1]][[[TW1_REL_ZERO]]], [[TW1_TOK]]
  // SEMA: [[TR1_ZERO:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
  // SEMA: [[TR1_TOK:%.*]] = nvws.semaphore.acquire [[TFULL1]][[[TR1_ZERO]]]
  // SEMA: [[T_TO_M0:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
  // SEMA: nvws.semaphore.release [[TENTRY]][[[T_TO_M0]]], [[TR1_TOK]]

  // ASP-LABEL: @tmem_fused_alias_depth_two
  // ASP: [[TENTRY:%.*]] = nvws.semaphore.create
  // ASP: [[TFULL0:%.*]] = nvws.semaphore.create
  // ASP: [[TEMPTY1:%.*]] = nvws.semaphore.create
  // ASP: [[TFULL1:%.*]] = nvws.semaphore.create
  // ASP: scf.for {{.*}} iter_args([[TCURSOR:%.*]] = {{%.*}}
  // ASP: [[TSLOT0:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: [[TW0_TOK:%.*]] = nvws.semaphore.acquire [[TENTRY]][[[TSLOT0]], {{%.*}}]
  // ASP: nvws.semaphore.release [[TFULL0]][[[TSLOT0]]], [[TW0_TOK]]
  // ASP: [[TR0_TOK:%.*]] = nvws.semaphore.acquire [[TFULL0]][[[TSLOT0]], {{%.*}}]
  // ASP: [[T_TO_M1_RAW:%.*]] = arith.addi [[TSLOT0]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[T_TO_M1_REM:%.*]] = arith.remsi [[T_TO_M1_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[T_TO_M1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[T_TO_M1_REM]] {ttg.partition = array<i32: 2>} : i32
  // ASP: nvws.semaphore.release [[TEMPTY1]][[[T_TO_M1]]], [[TR0_TOK]]
  // ASP: [[TSLOT1_RAW:%.*]] = arith.addi [[TSLOT0]], {{%.*}} {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: [[TSLOT1:%.*]] = arith.select {{%.*}}, {{%.*}}, [[TSLOT1_RAW]] {ttg.partition = array<i32: 2, 4>} : i32
  // ASP: [[TW1_TOK:%.*]] = nvws.semaphore.acquire [[TEMPTY1]][[[TSLOT1]], {{%.*}}]
  // ASP: nvws.semaphore.release [[TFULL1]][[[TSLOT1]]], [[TW1_TOK]]
  // ASP: [[TR1_TOK:%.*]] = nvws.semaphore.acquire [[TFULL1]][[[TSLOT1]], {{%.*}}]
  // ASP: [[T_TO_M0_RAW:%.*]] = arith.addi [[TSLOT1]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[T_TO_M0_REM:%.*]] = arith.remsi [[T_TO_M0_RAW]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
  // ASP: [[T_TO_M0:%.*]] = arith.select {{%.*}}, {{%.*}}, [[T_TO_M0_REM]] {ttg.partition = array<i32: 2>} : i32
  // ASP: nvws.semaphore.release [[TENTRY]][[[T_TO_M0]]], [[TR1_TOK]]

  tt.func @tmem_fused_alias_depth_two(%lb: i32, %ub: i32, %step: i32) {
    %v0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %v1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>

    scf.for %iv = %lb to %ub step %step : i32 {
      %m0 = ttng.tmem_alloc %v0 {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      %r0, %t0 = ttng.tmem_load %m0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume0"(%r0) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
      %m1 = ttng.tmem_alloc %v1 {buffer.copy = 2 : i32, buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      %r1, %t1 = ttng.tmem_load %m1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume1"(%r1) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2, 4>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
