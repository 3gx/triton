# Plan: Mechanical InsertSemas Release/Acquire Placement

## Goal

Make InsertSemas emission follow the DAG directly.

The DAG is authoritative. The emitter must not infer different placement from
semantic edge endpoints, SSA users, post-dominance, or repair heuristics.

## Rule

For each transition produced by the DAG walk:

```mlir
SRC_DAG_ROW
semaphore.release ...
semaphore.acquire ...
DST_DAG_ROW
```

Release is placed from the source DAG row. Acquire is placed from the
destination DAG row.

If the source DAG row is an op row, emit release immediately after that op.

If the destination DAG row is an op row, emit acquire immediately before that
op.

If ENTER is the source or destination row, its insertion point is the start of
that region, before the first real body op.

If YIELD is the destination row, its insertion point is before the region
terminator.

If the DAG explicitly requires a release after an op, emit it immediately after
that op.

Do not move the release later. Do not move the acquire earlier. Do not rewrite
the DAG rule into a different source-op/destination-op rule.

## Invariants

1. Every emitted sync action must come from DAG rows.
2. Every source/destination DAG row must map to one legal MLIR insertion point.
3. Release and acquire for the same transition stay in DAG order: release first,
   acquire second.
4. ENTER maps to region start.
5. YIELD maps to before `scf.yield`.
6. Release-after-`scf.yield` is invalid. If it appears, stop and report the DAG
   row.
7. If the DAG is correct and the emitter cannot place the sync row, the emitter
   is wrong.
8. If the DAG row is invalid, report the exact DAG row instead of inventing a
   workaround.

## Implementation

1. Keep the existing RAW/OPT DAG structural verifier.
2. Add a small emission-site collector that walks the same DAG rows used for the
   DAG dump and verifier.
3. Build `EmitterTransitionPlan` from those collected DAG row sites.
4. Remove the endpoint-based mechanical verifier.
5. Remove placement repair heuristics from transition emission.
6. Keep semaphore class selection, token threading, backing allocation, and
   `EmitState` materialization unchanged unless a real verifier failure proves
   they must change.

## Testing

Run from:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12
```

Build first:

```bash
ninja triton triton-opt
```

Focused lit signal:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test/NVWS/insert_semas*.mlir test/NVWS/tmem-buffer-reuse-semas.mlir
```

Runtime gates:

```bash
PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python timeout 60s sh run_nvws.sh
PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python timeout 60s sh run_nvws_1.sh
PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python timeout 60s pytest -n16 python/test/unit/language/test_warp_specialization.py
```
