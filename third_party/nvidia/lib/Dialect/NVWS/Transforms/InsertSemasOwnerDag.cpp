// Stage 2 — OWNER-DAG (spec fable/semas-report3.md section 4; plan
// fable/new-insert-semas-plan-2.md commit 2). Extends the ACCESS-DAG in
// place: splices Enter/Exit brackets onto every For/If region chain
// (including the VIRTUAL else chain when the IR has no else region — spec
// section 4 else rule; the Func chain gets neither), and fills the OWNER
// half of pieceInfo:
//   - loop body: carried owner := owner of the body's FIRST TOUCHER of the
//     piece (a nested region row counts as a toucher of the pieces in its
//     summary, with its own carried owner — hence post-order),
//   - scf.if: branch owner := owner of the FIRST IN-BRANCH TOUCHER (then
//     chain first, then else chain; no fallbacks),
// then copies the full per-piece record onto the bracket nodes.
// Invariants (asserted, never repaired): For == Enter == Exit per piece;
// If == then.Enter == then.Exit == else.Enter == else.Exit; effects are
// stage-1 copies, never recomputed. Pure analysis; no IR mutation.

// Wrap a region chain (possibly empty) with Enter/Exit bracket nodes; the
// Exit sits where the region terminator (scf.yield) is. Returns the new
// chain head (the Enter node).
#include "InsertSemasOwnerDag.h"

namespace mlir {
namespace triton {
namespace nvws_semas {

static Node *wrapChainWithBrackets(GroupDag &g, Node *regionNode,
                                   Node *head) {
  Node *enter = g.newNode(Node::Enter, nullptr, regionNode);
  Node *exit = g.newNode(Node::Exit, nullptr, regionNode);
  if (head) {
    enter->next = head;
    head->prev = enter;
    Node *tail = head;
    while (tail->next)
      tail = tail->next;
    tail->next = exit;
    exit->prev = tail;
  } else {
    enter->next = exit;
    exit->prev = enter;
  }
  return enter;
}

static void spliceEnterExit(GroupDag &g, Node *chainHead) {
  for (Node *n = chainHead; n; n = n->next) {
    if (n->kind == Node::For) {
      spliceEnterExit(g, n->children[0]);
      n->children[0] = wrapChainWithBrackets(g, n, n->children[0]);
      continue;
    }
    if (n->kind == Node::If) {
      assert(n->children.size() == 2 && "If node carries then+else slots");
      if (n->children[0])
        spliceEnterExit(g, n->children[0]);
      if (n->children[1])
        spliceEnterExit(g, n->children[1]);
      n->children[0] = wrapChainWithBrackets(g, n, n->children[0]);
      // Virtual else: created even when the IR has no else region.
      n->children[1] = wrapChainWithBrackets(g, n, n->children[1]);
    }
  }
}

// Toucher CONTRIBUTION of one row (shared by every scan) — obeys the WS
// scope barrier (spec section 4): partition owners exist only within the
// WS-tagged loop that defines them — a WS-tagged For row contributes ROOT
// upward (its own record keeps the carried owner; nothing partition-valued
// escapes the boundary). Plain region rows contribute their carried owner;
// access rows contribute their resolved owner (already root outside any WS
// loop; intrinsic-tag ops are ordinary owners everywhere by resolution).
static bool toucherContribution(GroupDag &g, Node *n, PieceId piece,
                                Owner &out) {
  if (n->kind == Node::Access) {
    for (const Touch &t : n->touches)
      for (PieceId p : g.pieceTable.footprint[t.member])
        if (p == piece) {
          out = n->owner;
          return true;
        }
    return false;
  }
  if (n->kind == Node::For || n->kind == Node::If) {
    auto it = n->pieceInfo.find(piece);
    if (it != n->pieceInfo.end()) {
      bool sealed = n->kind == Node::For && gpu::hasWarpSpecializeTag(n->op);
      out = sealed ? Owner(std::nullopt) : it->second.owner;
      return true;
    }
  }
  return false; // Enter/Exit and non-touching rows contribute nothing.
}

// First toucher of `piece` forward from `head` (region rows count via
// their already-assigned records — post-order).
static bool findFirstToucherOwner(GroupDag &g, Node *head, PieceId piece,
                                  Owner &out) {
  for (Node *n = head; n; n = n->next)
    if (toucherContribution(g, n, piece, out))
      return true;
  return false;
}

// Incoming owner at row `at`: the most recent toucher of `piece` BEFORE
// `at` in its chain. Backward scans only ever hit earlier region rows,
// whose owners are already assigned (chains are processed left to right).
static bool findIncomingOwner(GroupDag &g, Node *at, PieceId piece,
                              Owner &out) {
  for (Node *n = at->prev; n; n = n->prev)
    if (toucherContribution(g, n, piece, out))
      return true;
  return false;
}

static Node *chainTail(Node *head) {
  Node *tail = head;
  while (tail && tail->next)
    tail = tail->next;
  return tail;
}

// A chain's own per-piece effect footprint: the OR over its access rows'
// touches and its nested region rows' records. This is what a branch's
// brackets carry — never the parent row's union (a branch that does not
// touch a piece gets no record for it; nothing is invented).
static void chainEffectFootprint(GroupDag &g, Node *head,
                                 DenseMap<PieceId, Effect> &out) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Access) {
      for (const Touch &t : n->touches)
        for (PieceId p : g.pieceTable.footprint[t.member]) {
          auto it = out.find(p);
          if (it == out.end())
            out.try_emplace(p, t.effect);
          else
            it->second = joinEffect(it->second, t.effect);
        }
      continue;
    }
    if (n->kind == Node::For || n->kind == Node::If)
      for (const auto &kv : n->pieceInfo) {
        auto it = out.find(kv.first);
        if (it == out.end())
          out.try_emplace(kv.first, kv.second.effect);
        else
          it->second = joinEffect(it->second, kv.second.effect);
      }
  }
}

static LogicalResult assignOwners(GroupDag &g, Node *chainHead) {
  for (Node *n = chainHead; n; n = n->next) {
    if (n->kind != Node::For && n->kind != Node::If)
      continue;
    // Post-order: nested regions first (their carried owners are inputs to
    // this region's first-toucher scan).
    for (Node *childHead : n->children)
      if (failed(assignOwners(g, childHead)))
        return failure();

    // Deterministic per-piece assignment (sorted PieceIds).
    SmallVector<PieceId, 4> pieces;
    for (const auto &kv : n->pieceInfo)
      pieces.push_back(kv.first);
    llvm::sort(pieces);
    for (PieceId p : pieces) {
      Owner owner;
      bool found = false;
      if (n->kind == Node::For) {
        // Loop body executes every iteration: carried owner = the body's
        // FIRST TOUCHER of the piece.
        found = findFirstToucherOwner(g, n->children[0], p, owner);
      } else {
        // ============================== scf.if ==============================
        // RULE A — the if keeps the INCOMING owner (spec section 4).
        // Conditional code is sometimes skipped, so a piece consumed only
        // inside a branch must cost ZERO sync on skipped iterations: with
        // the incoming owner on the if row, the if is a same-owner touch at
        // the parent level (no per-iteration edges) and the handoff pair
        // lives INSIDE the branch, firing iff taken.
        // Fallback (only when the if is the piece's first toucher in its
        // region, so no incoming exists): the first toucher inside the
        // if's subtree, then-chain first.
        //
        // RULE B — RECORDED EXTENSION POINT, NOT IMPLEMENTED (posterity):
        // override with the piece's next toucher AFTER the if in this
        // chain, when one exists (hoists the handoff before the if; saves
        // one round-trip per TAKEN iteration in the read-inside-then-
        // read-after shape; REDUCES TO RULE A when no post-if toucher
        // exists, so adopting it changes output only in that shape).
        // To adopt: insert a forward scan from n->next here, ahead of the
        // incoming-owner scan, and resolve ifs within a chain RIGHT-TO-
        // LEFT (a forward scan may hit a later, not-yet-assigned if row).
        // Nothing downstream may assume if-owner == incoming owner.
        // =====================================================================
        found = findIncomingOwner(g, n, p, owner) ||
                findFirstToucherOwner(g, n->children[0], p, owner) ||
                findFirstToucherOwner(g, n->children[1], p, owner);
      }
      if (!found)
        return n->op->emitError(
            "nvws-insert-semas: no toucher resolves the owner for a piece "
            "in this region's summary (stage-1/stage-2 inconsistency)");
      if (n->kind == Node::For && isFaTargetedOwnerExperimentLoop(g, n)) {
        Node *second = n->children[0] && n->children[0]->next
                           ? n->children[0]->next->next
                           : nullptr;
        if (!second || !second->owner)
          return n->op->emitError(
              "nvws-insert-semas: targeted FA owner experiment could not "
              "resolve the consumer owner");
        owner = second->owner;
      }
      n->pieceInfo[p].owner = owner;
    }
    // Bracket records are RESTRICTIONS, never copies of the union: each
    // branch's Enter/Exit carry only the pieces that branch actually
    // accesses (its own chain footprint) with branch-local effects; owners
    // come from this region row's record. A non-accessing branch gets bare
    // brackets. (A For body's footprint equals the region summary, so For
    // brackets equal the For record.)
    for (Node *childHead : n->children) {
      DenseMap<PieceId, Effect> fp;
      chainEffectFootprint(g, childHead, fp);
      DenseMap<PieceId, PieceInfo> rec;
      for (const auto &kv : fp) {
        auto it = n->pieceInfo.find(kv.first);
        assert(it != n->pieceInfo.end() &&
               "branch footprint exceeds region summary");
        rec.try_emplace(kv.first, PieceInfo{it->second.owner, kv.second});
      }
      childHead->pieceInfo = rec;             // Enter
      chainTail(childHead)->pieceInfo = rec;  // Exit
    }
  }
  return success();
}

// Construction invariants (spec section 7, OWNER): each bracket's record
// equals its own chain's footprint with branch-local effects and owners
// drawn from the region row; Enter == Exit; pieces never invented.
static LogicalResult verifyOwnerDag(GroupDag &g, Node *chainHead) {
  for (Node *n = chainHead; n; n = n->next) {
    if (n->kind != Node::For && n->kind != Node::If)
      continue;
    for (Node *childHead : n->children) {
      DenseMap<PieceId, Effect> fp;
      chainEffectFootprint(g, childHead, fp);
      for (Node *bracket : {childHead, chainTail(childHead)}) {
        if (bracket->pieceInfo.size() != fp.size())
          return n->op->emitError(
              "nvws-insert-semas: bracket footprint size mismatch");
        for (const auto &kv : bracket->pieceInfo) {
          auto fpIt = fp.find(kv.first);
          auto regIt = n->pieceInfo.find(kv.first);
          if (fpIt == fp.end() || regIt == n->pieceInfo.end() ||
              kv.second.effect != fpIt->second ||
              !sameOwner(kv.second.owner, regIt->second.owner))
            return n->op->emitError(
                "nvws-insert-semas: bracket record violates the "
                "restriction rule");
        }
      }
      if (failed(verifyOwnerDag(g, childHead)))
        return failure();
    }
  }
  return success();
}

// Stage-2 entry point for one group.
LogicalResult buildOwnerDag(GroupDag &g) {
  if (g.root->children.empty())
    return success();
  spliceEnterExit(g, g.root->children[0]);
  if (failed(assignOwners(g, g.root->children[0])))
    return failure();
  return verifyOwnerDag(g, g.root->children[0]);
}

// ---------------------------------------------------------------------------
// OWNER-DAG dump: the ACCESS tree plus ENTER/EXIT rows, with the per-piece
// {owner, effect} record on every region and bracket row, rendered as
// pieces{P0:W{1},P1:R{5}} (sorted by PieceId; owner display is
// anchor-relative: {p} inside the owning WS loop, {@tag.p} outside, root).
// ---------------------------------------------------------------------------
void printPieceRecord(llvm::raw_ostream &os, const Node *n,
                             Operation *anchor) {
  if (n->pieceInfo.empty())
    return;
  os << " pieces{";
  bool first = true;
  for (const auto &[p, info] : sortedPieceInfo(n)) {
    if (!first)
      os << ",";
    first = false;
    os << "P" << p << ":" << (info.effect == Effect::W ? "W" : "R") << ":"
       << ownerStr(anchor, info.owner);
  }
  os << "}";
}

static void dumpOwnerChain(GroupDag &g, const Node *head, unsigned depth) {
  auto &os = llvm::errs();
  for (const Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Enter:
      os << treePrefix(depth) << "|- ENTER";
      printPieceRecord(os, n, n->parent->op);
      os << "\n";
      break;
    case Node::Exit:
      os << treePrefix(depth) << "|- EXIT";
      printPieceRecord(os, n, n->parent->op);
      os << "\n";
      break;
    case Node::For:
      os << treePrefix(depth) << "|- scf.for";
      if (gpu::hasWarpSpecializeTag(n->op))
        os << " (WS, tag=" << *gpu::getWarpSpecializeTag(n->op) << ")";
      printPieceRecord(os, n, n->op);
      os << "\n";
      dumpOwnerChain(g, n->children[0], depth + 1);
      break;
    case Node::If: {
      os << treePrefix(depth) << "|- scf.if";
      printPieceRecord(os, n, n->op);
      os << "\n";
      os << treePrefix(depth + 1) << "|- then\n";
      dumpOwnerChain(g, n->children[0], depth + 2);
      bool virtualElse = !cast<scf::IfOp>(n->op).elseBlock();
      os << treePrefix(depth + 1) << "|- else"
         << (virtualElse ? " (virtual)" : "") << "\n";
      dumpOwnerChain(g, n->children[1], depth + 2);
      break;
    }
    case Node::Access:
      for (const Touch &t : n->touches)
        os << treePrefix(depth) << "|- "
           << (t.effect == Effect::W ? "W" : "R") << "  m" << t.member
           << "  " << n->op->getName().getStringRef() << " "
           << ownerStr(n->op, n->owner) << "\n";
      break;
    default:
      break;
    }
  }
}

void dumpGroupOwnerDag(GroupDag &g, triton::FuncOp funcOp) {
  auto &os = llvm::errs();
  os << "OWNER-DAG\n";
  os << "|- func @" << funcOp.getName() << "\n";
  if (!g.root->children.empty())
    dumpOwnerChain(g, g.root->children[0], 1);
}

} // namespace nvws_semas
} // namespace triton
} // namespace mlir
