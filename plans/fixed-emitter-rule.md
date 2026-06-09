# Fixed InsertSemas Emitter Rule

The DAG dump is the source of truth.

For every rendered DAG transition:

```text
SRC_ROW
r  Sx  release
a  Sx  acquire
DST_ROW
```

emit exactly:

1. `release` immediately after `SRC_ROW`.
2. `acquire` immediately before `DST_ROW`.

Row mapping:

- op row: the op itself.
- ENTER row: first insertion point in that region.
- YIELD row as destination: immediately before the region terminator.
- parent-row after `scf.for`: immediately after the `scf.for` op.

Forbidden:

- Never use `srcYieldRegion` as a physical insertion point for a release.
- Never emit a release before `scf.yield`.
- Never reinterpret edge endpoints to choose another placement.
- Never use repair/post-processing to move sync ops after emission.
- If a rendered row cannot be mapped legally, fail and report the exact row.

Required proof for the AWS failure:

For DAG:

```text
|  |- a  S1  acquire  {1}
|  |- YIELD {1}
|- r  S2  release  {@0.1} -> root
|- a  S2  acquire  root
|- R  m0  ttng.tmem_load  root
```

the emitted IR must be:

```mlir
scf.for ... {
  ...
  scf.yield ...
}
nvws.semaphore.release ...
nvws.semaphore.acquire ...
ttng.tmem_load ...
```

Compliance gate:

- Build first.
- Run `--nvws-insert-semas` on `aws-9jun26-v1/before-insert-semas-attention-forward.mlir`.
- Verify no `nvws.semaphore.release` is emitted before `scf.yield`.
- Verify the `S2` release is after the `scf.for` and before the root load.
- Run `test/TritonGPU/automatic-warp-specialization.mlir`; it must pass unmodified.
- Do not run any pytest until `test/TritonGPU/automatic-warp-specialization.mlir` passes unmodified.
- After `test/TritonGPU/automatic-warp-specialization.mlir` passes, run only:
  `python/test/unit/language/test_warp_specialization.py::test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]`
- Run that pytest with:
  `PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/`
  and a hard `60s` timeout.
- If that single pytest does not finish within `60s`, treat it as a hang caused by the current change. Stop and root-cause it before running any other pytest.
- Do not run the full lit suite. Do not run any broader pytest selection for emitter work on this plan.
