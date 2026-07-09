---
name: draw-compiler-dags
description: Present compiler synchronization examples as a progressive sequence from input IR to synchronization-edge DAGs and then semaphore DAGs. Use when documenting access ordering, edge removal or merging, pending counts, tokens, ENTER/EXIT recurrence, nested regions, semaphore placement, schedules, or stage offsets, especially when a diagram is inconsistent or hard to follow.
---

# Draw Compiler DAGs

Make each example teach one continuous derivation. Start from the operations
the reader recognizes, preserve their names through every diagram, and add
detail only when the preceding step has established it.

## Use the pedagogical sequence

Present a complete synchronization example in this order:

```text
1. input IR or concise IR-shaped operations
2. buffer pieces, owners, and region summaries needed for the example
3. exact synchronization-edge table
4. synchronization-edge DAG
5. removed edges, followed by the resulting edge DAG
6. merge edges sharing a destination and source owner, then group by
   destination and destination owner
7. edge-to-semaphore table with pending counts and initial state
8. semaphore DAG with acquires, releases, tokens, and buffer uses
9. first entry, re-entry, final result, and zero-trip behavior when applicable
```

Do not start with an abstract graph when the reader has not seen the input.
Do not stop at the edge DAG when the point of the example is the emitted
semaphore flow.

Omit a stage only when it is genuinely irrelevant. State why it is omitted.
For example, an example about schedule legalization may begin from an already
established semaphore DAG, but it must identify the synchronization edge or
semaphore relationship being scheduled.

## Start from input IR

Show the smallest input that contains every operation used later:

```text
outer for
  W m0 {3}

  inner for
    R m0 {3}
    R m0 {2}
    W m0 {1}
    R m0 {0}
```

Use the document's established notation for reads, writes, owners, buffer
members, and iterations. Explain only facts needed to derive the graph:

- which physical pieces each member touches;
- which owner appears at a region boundary;
- whether a region reads or writes each piece; and
- which operation is the first or final use relevant to the example.

Do not replace concrete operations with `SOURCE`, `MIDDLE`, or `DESTINATION`
unless the example is intentionally generic and no concrete IR exists.

## Inventory exact synchronization edges

Read the nearby IR, test output, implementation, and existing tables. Do not
infer endpoints from prose when concrete endpoints are available.

Write the edge inventory before drawing:

```text
DAG node                 synchronization edge ending here
ENTER inner(i) {3}       none
R m0 {3}                 none
R m0 {2}                 c1: ENTER inner(i) {3} -> R m0 {2}
W m0 {1}                 c2: R m0 {3} -> W m0 {1}
                         c3: R m0 {2} -> W m0 {1}
R m0 {0}                 c4: W m0 {1} -> R m0 {0}
EXIT inner(i) {3}        c5: W m0 {1} -> EXIT inner(i) {3}
                         c6: R m0 {0} -> EXIT inner(i) {3}
```

Require every synchronization edge to have:

- one stable label;
- one exact source node;
- one exact destination node;
- the source and destination owners;
- the affected piece when it matters; and
- an iteration index when the edge reaches a loop boundary.

Keep edge labels stable through tables, prose, reduction diagrams, and
semaphore mapping.

## Draw the synchronization-edge DAG

Use only access, region-summary, `ENTER`, and `EXIT` nodes. Use labeled
synchronization edges and label program order as `walk` when it appears in the
same picture.

```text
                         ENTER inner(i) {3}
                         +---------+---------+
                    walk |                   | c1
                         v                   v
                     R m0 {3}            R m0 {2}
                      c2 |                   | c3
                         +---------+---------+
                                   v
                              W m0 {1}
                         +---------+---------+
                      c4 |                   | c5
                         v                   |
                     R m0 {0}                |
                      c6 |                   |
                         +---------+---------+
                                   v
                          EXIT inner(i) {3}
```

Do not put acquires, releases, semaphore names, tokens, pending counts, or
buffer operands in this DAG.

Draw parallel branches only when the synchronization relation is parallel.
Do not linearize independent reads merely because their operations have a
textual order. Conversely, do not draw same-owner operations as parallel when
their program order is essential to release placement.

## Show edge removal and merging explicitly

Show the complete edge set before removing anything. Explain a removal using
the stable edge labels, then draw the resulting DAG.

```text
initial

                         W m0 {1}
                         +------+------+
                      c4 |             | c5
                         v             |
                     R m0 {0}          |
                      c6 |             |
                         +------+------+
                                v
                        EXIT inner {3}

after removing c5

                         W m0 {1}
                             | c4
                             v
                         R m0 {0}
                             | c6
                             v
                     EXIT inner {3}
```

State the exact reason: `c4` followed by `c6` already makes `EXIT` wait for
the write. Do not replace this explanation with terms such as “covered,”
“wave,” “raw,” or “protocol.”

Perform and explain two distinct operations in this order:

1. Merge edges that have the same destination and source owner. They become
   one release by that source owner; preserve all represented edge labels and
   the latest required release position.
2. Group the remaining edges that have the same destination and destination
   owner. They share one semaphore and acquire, but different source owners
   still produce separate releases.

Do not silently replace several edge labels with one.

## Map remaining edges to semaphores

After all removals, merges, and destination grouping, provide a small mapping
table:

```text
edge        semaphore    release owner    pending_count    initial state
c1          S1           {3}              1                false
c2, c3      S2           {3}, {2}         2                false
c4          S3           {1}              1                false
c6          READY        {0}              1                initially released
```

For each source owner, one release contributes
`max(1, number of distinct completion kinds)`. Sum those contributions to
derive the acquire's pending count. One asynchronous release can therefore
contribute more than one signal. Never use a pending count to justify
removing an edge.

State when one semaphore serves more than one site, such as first entry and
next-iteration re-entry. Distinguish synchronization-edge removal from an
additional release used only to return a token for later use.
State which semaphore starts released and where iteration zero obtains its
first token.

## Draw the semaphore DAG

Replace each remaining synchronization edge with its actual release,
semaphore, acquire, token, and destination buffer use. Preserve the same
access names, owners, and orientation as the edge DAG.

The following is a focused `S2` fan-in excerpt from the mapping above. A
complete worked example must also draw `S1`, `S3`, and `READY`, unless it
links to an earlier diagram that already established those paths.

```text
       R m0(i) [r3tok] {3}              R m0(i) [r2tok] {2}
                  | walk                           | walk
                  v                                v
      release S2, r3tok {3} c2         release S2, r2tok {2} c3
               S2 |                                | S2
                  +---------------+----------------+
                                  v
               wtok = acquire S2 pending_count=2 {1}
                             wtok |
                                  v
                       W m0(i) [wtok] {1}
```

In these explanatory DAGs, `W m0 [wtok]` and `R m0 [rtok]` are shorthand for
creating `semaphore.buffer` with that token and using the resulting buffer at
the access. Define this shorthand in the document's notation. When exact IR
is the point, show the `semaphore.buffer` operation explicitly.

Show:

- the token result of every acquire;
- the token operand of every release;
- the token used by each read or write;
- the actual release position relative to the source access;
- the acquire position relative to the destination access; and
- the pending count when more than one signal is required.

If a release is anchored at `ENTER`, place it before later same-owner
operations. If an access reuses an existing token, show that same token on
the access instead of inventing another acquire.

## Show loops and regions without shortcuts

Keep the parent DAG and each child DAG distinct. A region summary is one node
in the parent; child accesses remain between the child's `ENTER` and `EXIT`.
When showing both levels together, use a box and state that the box overlays
the two views without joining their synchronization edges.

For the first inner iteration, never draw a fictitious `EXIT inner(-1)`.
Show where the first token actually comes from:

```text
outer token
    |
    v
ENTER inner(i,0)
```

When the region carries the token, show the acquire before the current
`EXIT`, the token through `EXIT`, and the next `ENTER`:

```text
next = acquire READY {3}
             next |
                  v
       EXIT inner(i,j) {3}
                  | next inner iteration
                  v
     ENTER inner(i,j+1) {3}
```

When the acquire moves to the first buffer use, show the alternate topology
instead. The synchronization edge still ends at `EXIT(i)`, but the acquire
appears only after the next `ENTER`:

```text
release READY, rtok {0} e2 ----- READY --------+
                                                |
EXIT(i) {3}                                     |
     | next iteration                           |
     v                                          |
ENTER(i+1) {3}                                  |
     | walk                                     |
     v                                          |
next = acquire READY {3} <----------------------+
          next |
               v
     W m0(i+1) [next] {3}
```

The semaphore arrow bypasses `EXIT` and `ENTER`; those nodes show control
flow only. Never label a release-to-boundary arrow `walk` when their owners
differ.

Explain separately:

- first entry;
- re-entry;
- the final token returned by the region;
- zero-trip behavior; and
- any acquire moved to its first buffer use.

Do not draw parent edges as child edges into `ENTER` or out of `EXIT`. Show
the concrete token crossing only in the semaphore DAG.

## Extend the sequence for schedules and offsets

For schedule examples, first show or reference the established semaphore
DAG. Then add `(loop.stage, loop.cluster)` to the same release, acquire, and
access names. Show the incorrect schedule and corrected schedule with
identical scaffolding.

For stage-offset examples, first show the input IR, synchronization edges,
and semaphore DAG. Add stage offsets to that same semaphore DAG afterward.
Do not present an offset table without showing which acquire, buffer, access,
and release use each offset.

Keep these concepts separate:

- synchronization edges decide which operations wait;
- semaphore conversion decides releases, acquires, tokens, and pending count;
- schedule decides when operations execute; and
- stage offset decides which semaphore stage or buffer copy is selected.

## Use plain, established terminology

Prefer: buffer, piece, read, write, owner, partition, synchronization edge,
semaphore, acquire, release, token, region, summary, `ENTER`, and `EXIT`.

Do not introduce a synonym when an established term works. In particular,
avoid “raw edge,” “protocol DAG,” “wave,” “covered sender,” “arrival
contribution,” and unexplained “chain.” Use implementation identifiers only
when the section explicitly documents that implementation detail.

## Validate every example

Before writing the final text, check:

1. Does the input contain every node used later?
2. Does every synchronization edge have exact endpoints and a stable label?
3. Does the synchronization-edge DAG contain no semaphore operations?
4. Do removed-edge diagrams differ only by the edges actually removed?
5. Does every merged edge identify the edges it represents?
6. Does the table distinguish source-owner merging from destination grouping?
7. Does the semaphore table account for every remaining edge and initial token?
8. Does every semaphore DAG preserve edge labels and use the correct token?
9. Does each pending count include every source-owner and completion-kind signal?
10. Are same-owner program order and real parallelism drawn truthfully?
11. Are parent and child region views kept distinct?
12. Is the carried-token or moved-acquire loop topology drawn accurately?
13. Are first entry, re-entry, final result, and zero-trip behavior accurate?
14. Do schedule and stage-offset overlays preserve the established DAG?

Inspect the implementation, lit test, or generated IR whenever an answer is
uncertain. Do not repair uncertainty with more prose.

## Respond to confusion by fixing the derivation

When a reader finds a diagram inconsistent:

1. Stop adding terminology.
2. Restate the input operations.
3. List the exact synchronization edges.
4. Retract the wrong endpoint, abstraction, or token path.
5. Redraw the synchronization-edge DAG.
6. Redraw the semaphore DAG from the corrected remaining edges.
7. Explain the result using only the stable labels and established terms.

## Edit existing documentation carefully

- Follow the document's established ASCII alignment and naming.
- Keep checks, test names, and implementation references accurate.
- Preserve useful examples, but remove obsolete diagrams and prose that the
  new progressive example replaces.
- Do not keep a stale diagram merely to avoid changing it.
- Do not expand every section mechanically; use the smallest complete
  derivation that teaches the section's point.
- Validate Markdown fences, anchors, and whitespace after editing.
- Do not modify lit tests or source unless the user explicitly includes them.
- Do not commit unless the user explicitly requests a commit.
