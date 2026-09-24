# Race detection corpus

A mutation-testing corpus for GPU race and hazard detectors.

The baseline (non-mutated) version of each kernel here is correct. 
The corpus takes each kernel, compiles it to GCN assembly, 
and `"mutates"` the kernel to introduces a known defect —
removing one `s_wait_*`, or rewriting one sub-dword load as a store. 

The specified detector is then invoked, which ideally identifies
no hazards for each of the baseline kernels, and one or more for 
each of the mutants.



---

## Support

### Detectors

| `--detector` | What it is | Output it emits |
|---|---|---|
| `data_hazard` | rocJITsu plugin | JSON list at `report_path` |
| `race_detector` | rocJITsu plugin | free-form text to a sink |
| `memory_wait` | core simulator feature | `memory-wait:` warnings on the log |

> [!NOTE]
> `data_hazard` and `memory_wait` are currently in flight, but still supported in the current corpus.

### Targets

`gfx950` and `gfx1250`. 16 of the 21 kernels are portable; 5 need gfx1250 for
tensor-DMA builtins, gfx12 DS mnemonics, or wave32. `mutants.toml` currently
holds recipes for both.

See [Regenerating](#regenerating-the-manifest) for regenerating `mutants.toml` after changes or additions.

### Mutation kinds

| kind | Edit | Makes |
|---|---|---|
| `wait` | one `s_wait_*` replaced by `s_nop` | whatever that wait ordered is now unguarded |
| `memory` | one sub-dword load rewritten as a store to the same address | an all-reads kernel becomes racy |

`s_wait_xcnt` is excluded by default: it tracks address translation rather than
data completion, so removing it produces no data hazard to find.

---

## How it works

### One case per mutant

```
race.gfx950.lds_war_pattern.baseline    the unmutated kernel — must report nothing
race.gfx950.lds_war_pattern.wait000     first s_wait_* removed
race.gfx950.lds_war_pattern.wait001     second removed
...
```

Each case is independently selectable with `--case`.

### Mutation ordinals

A recipe specifies to remove the `nth s_wait_*` in this kernel's assembly, as opposed to
specifying a line number for prosperity across toolchains.
This allows `mutants.toml` be committed while the assembly it describes is
generated fresh by whatever compiler is installed.


### Build pipeline

A mutant exists only as edited assembly, so it cannot be recompiled from source:

```
kernel.hip ── hipcc -S --cuda-device-only ──> baseline.s ─┐  cached per kernel
kernel.hip ── hipcc -c --cuda-host-only ────> host.o ─────┘

baseline.s ── apply recipe ──> mutant.s
           ── clang -target amdgcn ──> .o
           ── clang-offload-bundler ──> .hipfb
           ── llvm-mc (.incbin) ──> fatbin.o
           ── hipcc link (+ host.o) ──> executable
```

### Capability scoping

Detectors cover different hazards, and a detector should not be scored as a miss
for something it never claimed. Each mutant is tagged with the resource its
hazard lands on and the kind of hazard it is; each detector declares the
`(resource, kind)` **pairs** it supports.

```
resources:  global  lds  scratch  vgpr  agpr  sgpr
kinds:      RAW  WAR  WAW  LocalMemoryRace  GlobalMemoryRace
```

Mutants outside a detector's declared pairs are **skipped and removed from its
denominator**, so scores are computed per-detector over its own scope.

---

## Running it

```bash
export RJ=/path/to/rocm-systems/emulation/rocjitsu
export W="$RJ/build/tools/rocjitsu/rocjitsu --config $RJ/configs/gfx950_mi355x.json --"
cd <this repo>
```

### `data_hazard` — plugin

```bash
python -m pytest tests/test_corpus.py --suite race --target gfx950 \
  --detector data_hazard --run-wrapper "$W"
```

The adapter derives a per-mutant config from the one your wrapper names,
enabling `plugins.data_hazard` and pointing `report_path` at the mutant's
workdir. Needs the branch where that plugin exists.

### `race_detector` — plugin

```bash
python -m pytest tests/test_corpus.py --suite race --target gfx950 \
  --detector race_detector --run-wrapper "$W" -rs
```

Enables `plugins.race` with a file sink and counts `RACE kernel=…` records in
`race.log`. `-rs` shows what it skipped — expect global-memory mutants
(no cross-workgroup shadow) and LDS WAW (not implemented).

The count deliberately does **not** come from the plugin's summary banner: that
is written from `~RaceDetectorPlugin`, by which point the sink is gone, so it
never reaches the log.

### `memory_wait` — core simulator feature

```bash
python -m pytest tests/test_corpus.py --suite race --target gfx950 \
  --detector memory_wait --run-wrapper "$W"
```

### Narrowing down

```bash
... --detector data_hazard --run-wrapper "$W" -k lds_war_pattern     # one kernel
... --detector data_hazard --run-wrapper "$W" --case race.gfx950.lds_war_pattern.wait002
python -m pytest tests/test_corpus.py --suite race --target gfx950 --collect-only -q
```

---

## Reading the results

### Per case

| pytest | Means |
|---|---|
| **pass** on a mutant | in scope, detector reported a hazard — found it |
| **pass** on a baseline | the unmutated kernel reported nothing, as it must |
| **fail** on a mutant | in scope, reported nothing — **a real gap** |
| **fail** on a baseline | the kernel reports a hazard before anything was injected — a corpus defect or a false positive |
| **skip** | out of scope, tool unavailable, or a stale ordinal — never a gap |

A skip is never a miss. If a baseline fails, every mutant of that kernel also
skips, because a dirty baseline says nothing about the mutant.

## Artifacts

Everything for a case lands in one directory:

```
.pytest-artifacts/race/<target>/<kernel>/<mutant>/
    <kernel>.s                        the MUTATED assembly; grep MUTANT
    <kernel>                          standalone executable
    data_hazard.json  .log            data_hazard plugin output
    race.log                          race_detector plugin output
    memory_wait.log                   captured stdout+stderr
    rocjitsu-config.<detector>.json   the config each detector ran under
```

Configs are namespaced per detector because several share a workdir; an
unqualified name means whichever ran last decides which plugin the next one
enables. Logs are overwritten per run — copy anything you want to keep.

---

## Regenerating the manifest

Only needed when a kernel changes, for a new target, or to re-tag against newer
codegen.

```bash
python corpus/race/scripts/generate.py --target gfx950
python corpus/race/scripts/generate.py --target gfx950 --target gfx1250
git diff corpus/race/mutants.toml
```

Review the diff. Ordinal churn means codegen moved; changed `resource` or
`hazard` tags need a real read, not a skim.

---

## Adding a detector

One file in `scripts/adapters/`, then add its name to `KNOWN_DETECTORS`.

```python
name = "my_detector"
supports = capability({"vgpr", "sgpr"}, {"RAW", "WAR"})   # (resource, kind) pairs
needs_execution = True                                     # False for static

def detect(artifacts, run_wrapper) -> Detection:
    # artifacts gives .asm, .code_object, .executable, .workdir, .target
    # take what you need; the corpus never assumes a run happens
    return Detection(count=..., raw=..., command=...)
```

Two rules that matter more than they look:

- **Never report zero when you do not know.** A missing binary, an unreadable
  report, or a failed run must raise `DetectorUnavailable` or `DetectorError`.
  Zero means "ran correctly and found nothing", and anything else masquerading
  as zero turns the whole corpus into false negatives.
- **Declare only what your tool implements.** An over-broad `supports` produces
  misses that are really scope gaps; an under-broad one silently stops testing
  things that work.

---

## Layout

```
corpus/race/
  cases.toml          kernel metadata: arch, mutation kinds, intent
  mutants.toml        GENERATED, COMMITTED: per-mutant recipes + tags
  configs/            target configs
  kernels/            21 HIP kernels + utils.hpp
  scripts/
    mutate.py             mutation engine — no ROCm, no simulator, text in/out
    build.py              HIP source to executable
    generate.py           writes mutants.toml
    detector_protocol.py  the detector contract, scoping, scoring
    adapters/             one module per detector

tests/test_suites/race.py   suite adapter (discover / build / run)
```
