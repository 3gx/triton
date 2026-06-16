// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse 2>&1 >/dev/null | FileCheck %s --check-prefix=DAG

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // S1: prefix For, p1-anchored transparent region.
  // DAG-LABEL: function: @uniform_hold_s1_prefix_for_p1
  // DAG: holdrule{c0:passthrough-drop}
  // DAG: holdrule{c0:pointofuse->ttg.local_store}
  // DAG: holdrule{c0:gated(entry-consumed)}
  tt.func @uniform_hold_s1_prefix_for_p1(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 981 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        scf.for %i2 = %lb to %ub step %step : i32 {
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 2>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S2: owner-change at op1 -> inner; negative expected.
  // The p1 hold cuts at op1; the p2-entering inner region is not transparent.
  // DAG-LABEL: function: @uniform_hold_s2_owner_change_cut
  // DAG: holdrule{c0:passthrough-drop}
  // DAG: holdrule{c0:pointofuse->ttg.local_store}
  // DAG: holdrule{c0:gated(result-consumed)}
  tt.func @uniform_hold_s2_owner_change_cut(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 982 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        scf.for %i2 = %lb to %ub step %step : i32 {
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S3: trailing read after regain; negative expected.
  // DAG-LABEL: function: @uniform_hold_s3_trailing_read_after_regain
  // DAG: holdrule{c0:gated(region-not-transparent)}
  // DAG: holdrule{c0:gated(trailing-use)}
  // DAG: holdrule{c0:gated(result-consumed)}
  tt.func @uniform_hold_s3_trailing_read_after_regain(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 983 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        scf.for %i2 = %lb to %ub step %step : i32 {
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S4: prefix For, p2-anchored owner mirror of S1.
  // DAG-LABEL: function: @uniform_hold_s4_prefix_for_p2
  // DAG: holdrule{c0:passthrough-drop}
  // DAG: holdrule{c0:pointofuse->ttg.local_store}
  // DAG: holdrule{c0:gated(entry-consumed)}
  tt.func @uniform_hold_s4_prefix_for_p2(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 984 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        scf.for %i2 = %lb to %ub step %step : i32 {
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S5: region-spanning at WS-body depth 1, no middle, no op4.
  // DAG-LABEL: function: @uniform_hold_s5_ws_body_depth1
  // DAG: holdrule{c0:pointofuse->ttg.local_store:regionTail}
  // DAG: holdrule{c0:gated(entry-consumed)}
  tt.func @uniform_hold_s5_ws_body_depth1(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 985 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      scf.for %i1 = %lb to %ub step %step : i32 {
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S6: same-owner trailing read after the inner region.
  // DAG-LABEL: function: @uniform_hold_s6_same_owner_trailing_read
  // DAG: holdrule{c0:gated(trailing-use)}
  // DAG: holdrule{c0:gated(entry-consumed)}
  tt.func @uniform_hold_s6_same_owner_trailing_read(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 986 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      scf.for %i1 = %lb to %ub step %step : i32 {
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 1, 2>}
      %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S7: cross-owner store out / load in.
  // DAG-LABEL: function: @uniform_hold_s7_cross_owner_store_load
  // DAG: holdrule{c0:pointofuse->ttg.local_store}
  tt.func @uniform_hold_s7_cross_owner_store_load(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 987 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      scf.for %i1 = %lb to %ub step %step : i32 {
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S8: If-as-prefix-region inside the WS body.
  // DAG-LABEL: function: @uniform_hold_s8_if_prefix_region
  // DAG: holdrule{c0:pointofuse->ttg.local_store:regionTail}
  // DAG: scf.if pieces{P0:W:{1}} parts{1,2} thread{c0:{1}}
  // DAG: EXIT pieces{P0:W:{1}} yield{c0: a S1}
  // DAG: EXIT yield{c0: pass}
  tt.func @uniform_hold_s8_if_prefix_region(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 988 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      scf.if %cond {
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S9: If inside the inner loop.
  // DAG-LABEL: function: @uniform_hold_s9_if_inside_inner_loop
  // DAG: holdrule{c0:gated(no-buf)}
  // DAG: holdrule{c0:gated(no-buf)}
  // DAG: scf.if pieces{P0:W:{1}} parts{1,2} thread{c0:{1}}
  // DAG: EXIT pieces{P0:W:{1}} yield{c0: a S1}
  // DAG: EXIT yield{c0: pass}
  tt.func @uniform_hold_s9_if_inside_inner_loop(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %alloc = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        scf.if %cond {
          %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S10: fan-out / multi-consumer.
  // Multiplicity violates the one-carrier-slot rule, so the producer-side
  // hold stays carrier-bearing.
  // DAG-LABEL: function: @uniform_hold_s10_fanout_multi_consumer
  // DAG: holdrule{c0:gated(rel-count)}
  // DAG: holdrule{c0:gated(result-consumed)}
  // DAG: SEMAS c0: S0{count=2}
  tt.func @uniform_hold_s10_fanout_multi_consumer(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 990 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      %v0 = "producer0"(%i0) {ttg.partition = array<i32: 3>} : (i32) -> !ty
      ttg.local_store %v0, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      scf.for %i1 = %lb to %ub step %step : i32 {
        %v1 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer1"(%v1) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1, 2, 3>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
