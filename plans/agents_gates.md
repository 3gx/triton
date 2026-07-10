# Agent validation gates

These gates are ordered and fail-fast. Runtime tests and benchmarks must not be
started until the build and the mandatory LIT stage have completed as described
below.

## 1. Build before running any LIT test

Follow `AGENTS.md` exactly:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-02.git/build/cmake.linux-x86_64-cpython-3.12
ninja triton triton-opt
```

The build must succeed. Do not run LIT or pytest against a failed or stale
build.

## 2. Mandatory pre-runtime LIT gates

Run LIT from the build directory with:

```bash
LIT=/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit
```

### Automatic warp specialization

```bash
"$LIT" -v --filter='TritonGPU/automatic-warp-specialization\.mlir$' test
```

`test/TritonGPU/automatic-warp-specialization.mlir` is non-negotiable:

- do not modify the test or its `CHECK` lines;
- the pass pipeline must succeed;
- every existing `CHECK` must match, so the LIT test must be fully green.

### InsertSemas tests

```bash
"$LIT" -v --filter='NVWS/insert_semas.*\.mlir$' test
```

For every `insert_semas*` test, the compiler/pass execution must finish
successfully: no crash, assertion, verifier error, diagnostic failure, or
non-FileCheck failure is allowed. During initial development, an expected-IR
`CHECK` mismatch may be tolerated. Confirm such a failure by running the test's
`RUN` command without its FileCheck pipe; that command must exit successfully.

Runtime gates may begin only when:

1. the automatic-warp-specialization test is fully green without modification;
2. all InsertSemas pass executions succeed, with any remaining failures known
   to be FileCheck expectation mismatches only.

## 3. Establish the pre-change performance baseline

On the unchanged branch, complete the build and LIT gates above first. Only
then record the current branch's output for:

```bash
timeout 60s python 06-fa.py
timeout 180s sh run_nvwsX.sh
timeout 180s sh run_nvws.sh
```

Use the same GPU, environment, and benchmark arguments for baseline and patched
runs. A difference of roughly 2-3% can be measurement noise; a clear or
consistent regression is a gate failure.

After changing source code, repeat the build and mandatory LIT gates before
running any of the patched runtime gates below.

## 4. Runtime correctness gates

Run from the repository root. Each command must exit successfully before its
timeout:

```bash
timeout 120s pytest -n16 --tb=short \
  python/test/unit/language/test_warp_specialization.py

timeout 120s env \
  TRITON_FP8_PROMOTE_TO_TMEM=0 \
  NVWS_USE_SSA_TMEM=1 \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_NVWS_USE_META=1 \
  pytest -n16 python/tutorials/fused-attention-ws-device-tma.py
```

A timeout, failed test, compiler crash, or runtime error fails the gate.

## 5. Runtime performance gates

Run the patched branch with the same setup used for the recorded baseline:

```bash
timeout 60s python 06-fa.py
timeout 180s sh run_nvwsX.sh
timeout 180s sh run_nvws.sh
```

All three commands must complete successfully. Compare their results with the
pre-change baseline. Treat approximately 2-3% variation as noise, but do not
accept a clear or consistent performance regression.

## 6. Finish the LIT expectations

After the runtime gates pass, update only stale `insert_semas*` FileCheck
expectations to describe the intentional new IR. Do not weaken checks merely to
hide incorrect output, and do not modify
`TritonGPU/automatic-warp-specialization.mlir`.

Rerun the relevant InsertSemas LIT tests and finish with them fully green. If a
source change is needed while repairing them, restart at the build step before
running LIT again.
