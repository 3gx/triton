---
name: draw-compiler-dags
description: Draw and revise consistent ASCII DAGs for compiler dependencies, synchronization, edge reduction, loop recurrence, and emitted semaphore protocols. Use when explaining or documenting graph edges, comparing before/after reduction, showing EXIT/ENTER recurrence, translating raw edges into semaphores, or fixing a diagram that a reader finds inconsistent or unreadable.
---

# Draw Compiler DAGs

Draw the graph so the reader can verify it mechanically. Prefer one accurate
diagram over paragraphs of explanation.

## Establish the abstraction first

Choose exactly one level for each diagram:

```text
raw DAG       access/ENTER/EXIT/region nodes connected by e1, e2, ...
reduced DAG   the same raw nodes after named edges have been removed
protocol DAG  acquire/access/release nodes connected by S0, S1, ...
```

Never mix levels in one comparison. In particular:

- Never put `a S`, `r S`, or pending counts in a raw-edge diagram.
- Never compare a raw DAG on one side with an emitted protocol DAG on the
  other.
- Never call a protocol acquire the destination of a raw edge.
- State explicitly when a loop-close picture is logically unrolled across
  iterations rather than showing the stored `source -> EXIT` edge.

When comparing two cases, use the same abstraction, orientation, node detail,
edge-label style, and iteration notation in both diagrams.

## Inventory nodes and edges before drawing

Read the nearby source, table, dump, and existing diagrams. Do not infer a new
graph from prose when concrete endpoints are available.

Write a private inventory first:

```text
nodes in walk order:
  N0 = ...
  N1 = ...

generated edges:
  e1: exact source node -> exact destination node
  e2: exact source node -> exact destination node

non-generated order needed in the picture:
  N1 -> N2: walk order
```

Require every edge to have:

- one stable, unique label;
- one exact source node;
- one exact destination node;
- one declared abstraction level;
- an iteration index when it crosses a loop boundary.

Do not draw until this inventory is internally consistent.

## Draw full nodes and labeled edges

Use the terminology and node labels already present in the document. If the
document says `node`, do not introduce `row`. Do not rename accesses between
related diagrams.

Use full paths. Never compress a path into prose such as:

```text
walk -> W -> R
[kept path]
source -- several operations --> destination
```

Place edge labels between their endpoints. Show every node that establishes
the claimed ordering.

Use this split/join form for a direct candidate and an alternate path:

```text
                         SOURCE
                         +------+------+
         e2 direct edge |             | e1
                         |             v
                         |          MIDDLE
                         |             | e3
                         +------+------+
                                v
                           DESTINATION
```

The reduced diagram must preserve the same names and direction:

```text
                           SOURCE
                              | e1
                              v
                           MIDDLE
                              | e3
                              v
                         DESTINATION
```

Explain the result only through edge labels: `e2` is removed because
`e1 -> e3` connects the same endpoints and the applicable reduction rule
allows deletion.

## Show parallelism and sequencing truthfully

- Draw parallel paths only when the DAG contains parallel dependency paths.
- Draw one chain when there is only one dependency path.
- Use a split and join when several edges leave or enter a node.
- Keep walk order visually distinct from generated edges. Label it `walk`
  consistently when the distinction matters.
- Do not force source walk order into a linear dependency DAG if two accesses
  are parallel in the dependency relation.
- Do not leave a node hanging merely because its connection is walk order;
  connect it and label the connection accurately.

## Draw loop recurrence at one consistent level

For the concrete raw representation, terminate a close at `EXIT`:

```text
SOURCE
   | e4
   v
EXIT(i) {destination owner}
   | walk
   v
ENTER(i+1)
```

For a wrap-around reduction proof, an equivalent unrolled picture may carry
the destination owner to its next access. Apply that convention to every
close being compared and say that the picture is unrolled:

```text
                         SOURCE(i)
                         +------+------+
       e2 direct close  |             | e1
                         |             v
                         |          MIDDLE(i)
                         |             | e3 close
                         +------+------+
                                v
                       DESTINATION(i+1)
```

Never put an emitted acquire at the join of this raw diagram. Semaphore
conversion comes later.

## Compare cases with identical scaffolding

When explaining why two isomorphic DAGs receive different policy decisions,
draw both with identical scaffolding and unique labels. Stack them vertically
instead of compressing their paths to fit side by side:

```text
case A

                         SOURCE-A
                         +------+------+
              e2 direct |             | e1
                         |             v
                         |         MIDDLE-A
                         |             | e3
                         +------+------+
                                v
                             DEST-A

case B

                         SOURCE-B
                         +------+------+
              f2 direct |             | f1
                         |             v
                         |         MIDDLE-B
                         |             | f3
                         +------+------+
                                v
                             DEST-B
```

Then state separately:

```text
graph fact:   e2 and f2 are both transitively redundant
policy fact:  the applicable rule removes e2 but protects f2
```

Do not distort either graph to make a policy difference look like a topology
difference.

## Convert edges to semaphores only after reduction

Maintain this order:

```text
raw edges
  -> remove redundant edges
  -> merge repeated edges
  -> group remaining edges by destination
  -> assign semaphore and pending count
```

Never use a pending count to justify whether a raw edge is redundant. The
pending count is a consequence of the remaining edges.

When drawing the protocol DAG:

- replace raw edge labels with the exact semaphore labels;
- show acquires and releases only at their actual placement nodes;
- show fan-in count only after confirming the number of remaining source
  edges;
- preserve the raw-to-protocol mapping in adjacent text or a small table.

## Validate every diagram

Before presenting or writing the diagram, check:

1. Are all compared diagrams at the same abstraction level?
2. Does every named edge have the inventoried source and destination?
3. Is every edge labeled so the reader can refer to it?
4. Are all necessary intermediate nodes shown explicitly?
5. Are parallel branches real dependency paths rather than visual invention?
6. Are walk-order connections distinguished from generated edges?
7. Are `EXIT(i)`, `ENTER(i+1)`, and iteration indices consistent?
8. Does the reduced DAG differ only by the edges actually removed?
9. Does a raw diagram contain no acquire, release, semaphore, or pending-count
   notation?
10. Does the protocol diagram contain only semaphore facts derived after
    reduction and merging?

If any answer is uncertain, inspect the source or dump again before drawing.

## Respond to diagram confusion

When the reader says a diagram is inconsistent or unreadable:

1. Stop adding conceptual explanation.
2. Identify and explicitly retract the mismatched abstraction or endpoint.
3. Restate the exact node and edge inventory.
4. Redraw all compared cases at one abstraction level with the same layout.
5. Keep the original edge labels stable.
6. Use full nodes; do not introduce shorthand to save space.
7. Explain the outcome in one or two sentences that refer only to labeled
   edges.

Do not answer repeated confusion with more terminology. Fix the picture first.

## Editing discipline

When adding diagrams to an existing document:

- copy its established ASCII style;
- place raw DAGs immediately after raw-edge tables;
- place final protocol DAGs immediately after emitted protocol listings;
- reduce prose that merely repeats paths now visible in the diagram;
- preserve precise implementation caveats that the diagram cannot express;
- do not commit unless the user explicitly requests a commit.
