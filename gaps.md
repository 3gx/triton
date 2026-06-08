# InsertSemas Emitter Plan Gaps

Plan checked: `plans/emiiter-tmem-semaphore-state-machine-plan.md`.

## Closed

1. Old post-emission release movement helpers are gone.
   - Evidence to re-check: no `moveAfterExistingReleasesBeforeAcquire`,
     `moveAfterLoopBeforeFollowingSemaphores`,
     `releaseShouldPrecedeFollowingSemaphores`, `latestTransitiveConsumer`,
     `findLastSameBlockNonTokenResultUser`, or `moveAfter(` in
     `InsertSemas*.{h,cpp}`.

2. Transition plan is no longer copied from DAG anchor maps.
   - Evidence to re-check: no `releaseBeforeOp`, `releaseAfterOp`,
     `acquireBeforeOp`, `releaseBeforeYield`, `releaseAfterYield`, or
     `acquireBeforeYield` in `InsertSemas*.{h,cpp}`.
   - Current status: closed. `OptSyncDag` no longer owns those maps and
     `buildEmitterTransitionPlan` derives transitions from `dag.groups` and
     `sp.edges`.

3. Carrier threading no longer depends on post-emission token-slot repair.
   - Evidence to re-check: no `reusedForCarrierSlots`,
     `reusedForTokenSlots`, `reusedForPoisonTokens`,
     `poisonUnbackedCarrierTokenSlots`, or
     `poisonDuplicateUnbackedTokenSlots` in `InsertSemas*.{h,cpp}`.
   - Current status: closed. Threaded loops carry one live carrier slot for
     the active semaphore class. Existing matching loop slots are selected
     when they already represent that live carrier; otherwise a carrier iter_arg
     is appended. Duplicate unbacked carrier slots are normalized locally after
     the loop's exit transitions, not by a final whole-function repair pass.

4. Main block walk no longer dispatches transitions around every operation.
   - Evidence to re-check: `emitResourceBlock` skips operations that are not
     structural operations, access events, or concrete transition anchors, and
     calls entry/exit transition emitters only for transition anchors.
   - Current status: closed. The traversal is still structural so regions can
     be entered in IR order, but non-event operations are only traversed for
     structure.

5. Release owner/payload is no longer re-derived in `EmitState::release`.
   - Evidence to re-check: no `computeReleaseOwner` or
     `computeReleasePayload` in `InsertSemasEmitter.h`. `PlannedRelease`
     carries `owner`, `payload`, `useCarriedOwner`, and `useCarriedPayload`;
     `resolvePlannedReleaseState` derives those fields in
     `InsertSemasOptSyncDag.h`; `EmitState::release` consumes the fields
     mechanically.
   - Current status: closed. Verified with build, focused
     `test/NVWS/insert_semas.mlir` plus
     `test/NVWS/insert_semas_nested_carrier.mlir`, and full lit baseline.

6. TMEM linear loop-exit drain is no longer semantic emitter logic.
   - Evidence to re-check: no `emitTmemLinearLoopExitDrain`,
     `PlannedLoopExitDrain`, `PlannedDrainRelease`, `loopExitDrains`,
     `buildLoopExitDrain*`, `emitPlannedLoopExitDrains`, or
     `emitPlannedDrainRelease` in `InsertSemas*.{h,cpp}`.
     `closeLoopSemaphoreStateAfter` / `closeLinearChainLoopStateAfter` close
     the loop from `EmitState::current` after the loop body has been walked.
   - Current status: closed. Verified with build, the 20-test
     `insert_semas` lit filter, `test/NVWS/tmem-buffer-reuse-semas.mlir`, and
     full NVWS lit.

7. LoC target is met.
   - Evidence to re-check:
     `wc -l third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas*.{h,cpp}`
     reports 7086 total, below the plan target of 7090.
   - Current status: closed. The obsolete `InsertSemasEmitSchedule.h`
     action-list diagnostic is deleted; the `RELEASED-SEMAPHORES` dump is
     retained without the old M1 violation text.

8. Source-yield producer payload promotion is no longer a transition-plan
   payload derivation rule.
   - Evidence to re-check: no `useCarriedProducerPayload`,
     `findLastProducerInRegion`, `canPromoteYieldProducerPayload`, or
     `destinationAccessIsInWarpSpecializedLoop` in `InsertSemas*.{h,cpp}`.
     `ActiveSemaphoreState` carries `producerPayload` plus `producerOwner`;
     `EmitState::release` uses `releaseShouldUseCarriedProducerState` to
     decide whether the current carried producer state is valid for the release.
   - Current status: closed. Verified with build, the 20-test
     `insert_semas` lit filter, and full NVWS lit.

## Caveats To Revisit Later

1. Placement predicates still exist in transition-plan derivation.
   - Current status: accepted caveat, not an open blocker for this iteration.
   - Evidence: `transitionLastValueUserInBlock`,
     `transitionHasLaterResourceAccess`, `transitionAccessCompletion`,
     `transitionReleaseBeforeAcquire`, and
     `transitionReleaseWaitsForReadCompletion` in
     `InsertSemasOptSyncDag.h`.

2. Carrier slot selection still uses existing loop token slots when they are
   already the byte-identical live carrier.
   - Current status: accepted caveat. Blindly appending new carriers changed
     golden loop arity and broke the plan's byte-identical constraint, so the
     implementation keeps local slot selection plus local normalization.

## Remaining Gaps

None for this iteration.
