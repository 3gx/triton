#!/usr/bin/env python3
"""Regenerate FileCheck lines for an insert_semas lit test, in the house
style: explicit [[Vn]] captures for every semaphore/token/backing, full
types, CHECKs interleaved immediately before the source ops that generate
them. No wildcards on tokens: acquire->buffer/release weaves are pinned.

Usage: gen_insert_semas_checks.py <test.mlir> <triton-opt> [extra args...]
Rewrites the file in place (CHECK lines only; source untouched).
Conventions (user ruling 10jun26): same structure as existing tests;
buffer/acquire/release/create must be present; tok explicit in release
and buffer; checks adjacent to generating ops.
"""
import re, subprocess, sys

CHECKED = re.compile(
    # tmem_alloc only when operand-LESS: a sourceful unmanaged alloc
    # survives in the emitted IR and must bump the access anchor instead
    # (its managed sibling becomes buffer + synthesized store).
    r'= (nvws\.semaphore\.(create|acquire|buffer)|ttg\.local_alloc \{buffer'
    r'|ttng\.tmem_alloc(?! %)|ttng\.tmem_subslice|ttg\.memdesc_reinterpret)')
SYNC_NORESULT = re.compile(r'^\s*nvws\.semaphore\.release ')
SSA = re.compile(r'%[A-Za-z0-9_#]+(?::\d+)?')
WILD = '%{{[-A-Za-z0-9_.$#]+}}'


def run_pass(test, opt, extra):
    # use the test's own RUN line (the default-prefix FileCheck one)
    args = ['-split-input-file', '-allow-unregistered-dialect',
            '--nvws-insert-semas', '-cse']
    for ln in open(test).read().splitlines():
        m = re.match(r'// RUN: triton-opt %s (.*?)\s*\| FileCheck %s\s*$', ln)
        if m:
            args = m.group(1).split()
            break
    out = subprocess.run([opt, test] + args + extra,
                         capture_output=True, text=True)
    if out.returncode:
        sys.exit(f'pass failed on {test}:\n{out.stderr[:2000]}')
    return out.stdout


def funcs_of(text):
    """name -> list of body lines (between tt.func and its closing)."""
    res = {}
    for m in re.finditer(r'tt\.func [^\n]*@([A-Za-z0-9_]+)', text):
        depth, body, i = 0, [], text.index('\n', m.start()) + 1
        line_start = i
        for j in range(m.start(), len(text)):
            pass
        lines = text[i:].splitlines()
        for ln in lines:
            if ln.strip() == '}' and depth == 0:
                break
            depth += ln.count('{') - ln.count('}')
            body.append(ln)
        res[m.group(1)] = body
    return res


class Caps:
    def __init__(self):
        self.map, self.n = {}, 0

    def define(self, ssa):
        self.n += 1
        self.map[ssa] = f'[[V{self.n}:%.*]]'
        return self.map[ssa]

    def ref(self, ssa):
        base = ssa.split('#')[0]
        if base in self.map:
            r = self.map[base].split(':')[0] + ']]'
            return r + ('#' + ssa.split('#')[1] if '#' in ssa else '')
        return None


def render(line, caps):
    """One emitted op line -> CHECK text with captures/wildcards."""
    s = line.strip()
    # scf.for: capture iter_arg block args (tokens must be explicit) and
    # the result; wildcard bounds and non-tracked inits.
    if re.search(r'= scf\.for |^scf\.for ', s):
        m = re.search(r'iter_args\(([^)]*)\)', s)
        if m:
            inner = []
            for pair in m.group(1).split(', '):
                lhs, rhs = pair.split(' = ')
                ref = caps.ref(rhs) or WILD
                inner.append(f'{caps.define(lhs)} = {ref}')
            s = s[:m.start()] + 'iter_args(' + ', '.join(inner) + ')' \
                + s[m.end():]
        mdef = re.match(r'(%[A-Za-z0-9_]+)(:\d+)? = ', s)
        if mdef:
            cap = caps.define(mdef.group(1)) + (mdef.group(2) or '')
            s = cap + ' = ' + s.split(' = ', 1)[1]
        s = re.sub(r'%[A-Za-z0-9_]+ = (%[A-Za-z0-9_]+) to', WILD + ' = '
                   + WILD + ' to', s, count=1)
        s = s.replace(' to %', ' to ' + WILD.replace('%', '%', 1), 1) if False else s
        # wildcard remaining bare SSA (bounds/steps)
        s = SSA.sub(lambda mm: mm.group(0) if mm.group(0).startswith('[[')
                    else mm.group(0), s)
        s = re.sub(r'(?<!\[)%(?!{)[A-Za-z0-9_]+', WILD, s)
        s = re.sub(r'\[\[V(\d+):' + re.escape(WILD[1:]) , r'[[V\1:%.*', s)
        return '// CHECK: ' + s
    mdef = re.match(r'(%[A-Za-z0-9_]+)(:\d+)? = ', s)
    out = s
    # replace operand SSA uses right-to-left to keep spans valid
    spans = [(m.start(), m.end(), m.group(0)) for m in SSA.finditer(s)]
    repl = []
    for st, en, tok in spans:
        if mdef and st == 0:
            continue
        r = caps.ref(tok)
        repl.append((st, en, r if r else WILD))
    for st, en, r in reversed(repl):
        out = out[:st] + r + out[en:]
    if mdef:
        cap = caps.define(mdef.group(1))
        suffix = mdef.group(2) or ''
        if suffix:
            cap = cap.replace(':%.*]]', ':%.*]]')
            out = cap + suffix + out[len(mdef.group(0)) - 3:]
            out = cap + suffix + ' = ' + out.split(' = ', 1)[1]
        else:
            out = cap + ' = ' + out.split(' = ', 1)[1]
    return '// CHECK: ' + out


def gen_for_func(body):
    """emitted body lines -> list of (anchor_kind, anchor_key, check)."""
    caps = Caps()
    checks = []   # (anchor, text); anchor = ('access', k) | ('for', k) | ('yield', k) | ('pre', 0)
    acc_k = for_k = yield_k = 0
    pending = []
    stack = []   # region kinds: 'for' | 'if'; yields anchor only in 'for'
    for ln in body:
        s = ln.strip()
        if s.startswith('}'):
            if stack:
                stack.pop()
            if re.search(r'\belse\b.*{\s*$', s):
                stack.append(('if', 0))
            continue
        if CHECKED.search(ln) or SYNC_NORESULT.match(ln):
            pending.append(render(ln, caps))
            continue
        if re.match(r'(%[^=]+= )?scf\.for ', s):
            for_k += 1
            pending.append(render(ln, caps).rstrip('{').rstrip() + ' {')
            for p in pending:
                checks.append((('for', for_k), p))
            pending = []
            stack.append(('for', for_k))
            continue
        if re.match(r'(%[^=]+= )?scf\.if ', s):
            stack.append(('if', 0))
            continue
        if s.startswith('scf.yield'):
            # if-region yields are NOT anchors: the emitter's if-split
            # changes their count; pendings flow to the next access row.
            # Loop yields key on the ENCLOSING for index ('yieldof', k):
            # for counts are emission-stable, global yield counts are not
            # (the pass hoists carriers out of inner loops / adds them).
            if not stack or stack[-1][0] != 'for':
                continue
            tokops = [caps.ref(t) for t in SSA.findall(s) if caps.ref(t)]
            if tokops:
                # no space between {{.*}} and the ref: an attr-less
                # `scf.yield %16` has only one whitespace run to consume.
                pending.append('// CHECK: scf.yield {{.*}}' +
                               ', '.join(tokops))
            for p in pending:
                checks.append((('yieldof', stack[-1][1]), p))
            pending = []
            continue
        # sourceful UNMANAGED allocs survive in the emitted IR; they must
        # bump the anchor counter exactly like their source rows do (the
        # MANAGED ones become a synthesized store at the same position).
        if re.search(r'ttg\.local_(load|store)|ttng\.tmem_(load|store)'
                     r'|tc_gen5_mma|descriptor_(load|gather)|"[A-Za-z_]+"\('
                     r'|= ttg\.local_alloc %|= ttng\.tmem_alloc %', s):
            acc_k += 1
            if CHECKED.search(ln):
                pending.append(render(ln, caps))
            elif re.search(r'local_(load|store)|tmem_(load|store)', s):
                pending.append(render(ln, caps))
            for p in pending:
                checks.append((('access', acc_k), p))
            pending = []
            continue
    for p in pending:
        checks.append((('end', 0), p))
    return checks


SRC_ACCESS = re.compile(
    r'ttg\.local_(load|store)|ttng\.tmem_(load|store)|tc_gen5_mma'
    r'|descriptor_(load|gather)|"[A-Za-z_]+"\('
    # sourceful allocs: the emitter splits them into view + synthesized
    # store, which lands at the same position -> they ARE anchor rows.
    r'|= ttg\.local_alloc %|= ttng\.tmem_alloc %')


def apply_to_test(test, gen):
    """Interleave generated checks into the test file at source anchors."""
    lines = open(test).read().splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        m = re.search(r'tt\.func [^\n]*@([A-Za-z0-9_]+)', ln)
        if not m or m.group(1) not in gen or '// ' in ln:
            out.append(ln); i += 1; continue
        by = {}
        for anchor, text in gen[m.group(1)]:
            by.setdefault(anchor, []).append(text)
        out.append(ln); i += 1
        # multi-line signature: pass continuation lines through verbatim
        # so the body walk starts after the '{' opener (else this walk
        # runs long and swallows the NEXT function, stripping its checks).
        while i < n and not out[-1].rstrip().endswith('{'):
            out.append(lines[i]); i += 1
        depth = 0
        acc_k = for_k = yield_k = 0
        stack = []
        while i < n:
            bl = lines[i]; s = bl.strip()
            if s.startswith('// CHECK') and 'CHECK-LABEL' not in s:
                i += 1; continue          # strip stale default-prefix checks
            if s == '}' and depth == 0:
                for t in by.get(('end', 0), []):
                    out.append('    ' + t)
                by.pop(('end', 0), None)
                out.append(bl); i += 1; break
            here = []
            if s.startswith('}'):
                if stack:
                    kind, kk = stack.pop()
                    # void source loop: no written scf.yield, but the
                    # emitted loop gained token results -> its yield
                    # checks anchor at the loop's closing brace.
                    if kind == 'for':
                        here = by.pop(('yieldof', kk), [])
                if re.search(r'\belse\b.*{\s*$', s):
                    stack.append(('if', 0))
            elif re.match(r'(%[^=]+= )?scf\.for ', s):
                for_k += 1; here = by.pop(('for', for_k), [])
                stack.append(('for', for_k))
            elif re.match(r'(%[^=]+= )?scf\.if ', s):
                stack.append(('if', 0))
            elif s.startswith('scf.yield'):
                # mirror the generator: only loop-body yields anchor (and
                # the printer omits bare void yields in emitted output).
                if s != 'scf.yield' and stack and stack[-1][0] == 'for':
                    here = by.pop(('yieldof', stack[-1][1]), [])
            elif SRC_ACCESS.search(s):
                acc_k += 1; here = by.pop(('access', acc_k), [])
            elif s.startswith('tt.return'):
                here = by.pop(('end', 0), [])
            ind = bl[:len(bl) - len(bl.lstrip())] or '    '
            for t in here:
                out.append(ind + t)
            depth += bl.count('{') - bl.count('}')
            out.append(bl); i += 1
        for anchor in by:
            print(f'  WARNING {test} @{m.group(1)}: unplaced {anchor} '
                  f'({len(by[anchor])} checks)')
    open(test, 'w').write('\n'.join(out) + '\n')


def main():
    argv = [a for a in sys.argv[1:] if a != '--apply']
    do_apply = '--apply' in sys.argv
    test, opt, *extra = argv
    emitted = run_pass(test, opt, extra)
    gen = {name: gen_for_func(body) for name, body in funcs_of(emitted).items()}
    if do_apply:
        apply_to_test(test, gen)
        print(f'applied: {test}')
        return
    for name, body in gen.items():
        print(f'\n===== @{name}')
        for anchor, text in body:
            print(f'  {anchor}: {text}')


if __name__ == '__main__':
    main()
