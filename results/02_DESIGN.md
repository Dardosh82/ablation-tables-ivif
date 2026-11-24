# Chapter 6 — LRRFuse-VA: design of record

**Part III** of *Variance-Aware Design and Evaluation of Deep Learning-Based Infrared and Visible
Image Fusion*. This file is the **design brief and plan of record** for the proposed method. Numbers
go to `studied_papers/LRRNet/LRRFUSE_RESULTS.md` once runs exist; plain-language narrative to
[03_EXPLAINED.md](03_EXPLAINED.md) and [EXPLAIN.md](../../../thesis/EXPLAIN.md); operational state to
[CONTEXT.md](../../../CONTEXT.md).

**Created:** 2026-09-03 · **Status:** ✅ **all runs complete 2026-09-07** — results in
[`05_RESULTS.md`](05_RESULTS.md). This file is the *design of record* and is deliberately left as
written, so the pre-registered readings can be compared against what actually happened; status
columns below are updated, the reasoning is not rewritten.

---

## 0. The decision this chapter answers

Part I measured σ. Part II showed two published ablation tables cannot support the choices they
justify. Part III must now show the protocol works **for** a method, not only against other people's.

**Two decisions taken 2026-09-03, before any design detail:**

| decision | choice | consequence |
|---|---|---|
| where to start | **M1 (weight tying)** | cheapest, funds M2, and is a categorical result even under a null |
| the headline claim | **stability + efficiency** | "lower σ and fewer parameters at indistinguishable quality" — as planned; measured (Run 12): fewer parameters with **no metric differing after correction**, equivalence on 2 of 9 (claim #15) |

The claim is deliberately **not** "higher means". At 4 epochs on LLVIP, a mean improvement is a bet;
σ and parameter count are measurable to arbitrary precision. Chasing means would repeat the exact
error this thesis documents.

---

## 1. Why LRRNet is the right host, stated as evidence

`studied_papers/LRRNet/scripts/_lrrnet_src/net_lista.py`, read line by line. Three design properties of the
released network are in direct tension with things **this project measured**.

### D1 — the network is not low-rank anywhere in its forward pass

```python
L = tensor_z[:, :self.n, :, :]          # net_lista.py:112
S = tensor_z[:, self.n: 2 * self.n, :, :]
```

`L` and `S` are two **channel slices of one tensor**. No rank constraint, no nuclear-norm
projection, no structural separation. The low-rank / sparse split is carried entirely by the loss
(`lam2`, `w_ir`, `lam3_gram`) and by the name. The only structural asymmetry in the whole model is
which half of the channel axis a slice lands in.

This is not a criticism that requires a run to make — it is readable from the source — but it makes
the paper's central claim (a *representation-learning guided* fusion network) an empirical question
rather than an architectural fact.

### D2 — L and S are then processed identically

```python
self.conv_ReZx = ConvLayer(n, channel, s, stride)   # the L path   net_lista.py:173
self.conv_ReUx = ConvLayer(n, channel, s, stride)   # the S path   net_lista.py:174
```

Same type, same shape, same kernel. **Our own A4 + A5 result says this is wrong**: at *matched
capacity* a channel-attention block and a CNN block are not interchangeable — the attention block
buys perceptual fidelity (VIF, Qabf, MI), the deep CNN buys contrast and sharpness (EN, SD, SF, AG).
CrossFuse assigns them asymmetrically and our capacity-matched control vindicated that assignment.
LRRNet has the same low/high split available and does not use it.

Worse, both paths collapse **128 channels → 1 channel** in a single 3×3 convolution before anything
is done with them. Whatever the decomposition found is discarded at the decoder's front door.

### D3 — untied unrolling is the source of confound L1

Each `LRR_Block_lista` owns its own `conv_Wdz` / `conv_Wdtz` (4,865 parameters). So `num_block`
moves depth and capacity together — the 2.96× spread recorded as **L1**. Classical LISTA/ISTA
unrolling **ties** the weights across iterations: the block *is* one iteration of one solver.
LRRNet's untied variant is a departure from the algorithm it is named after, and it is precisely
what makes its own Table IV unreadable.

---

## 2. The method: four modifications, each traceable to one of our measurements

| ID | modification | justified by | predicted effect | status |
|---|---|---|---|---|
| **M1** | weight-tied unrolling — one `LRR_Block_lista` applied `num_block` times | **L1** | 49,204 → **20,014** params; `num_block` becomes a free depth knob | ✅ **PASS — 0 of 9, [05_RESULTS.md](05_RESULTS.md)** |
| **M2** | asymmetric refiners before the channel collapse: channel-attention (MDTA) on `L`, small CNN on `S` | **A4 + A5**, **D2** | VIF / Qabf / MI ↑ | ❌ failed — 0 of 9 paired, and MI fell |
| **M3** | Top-K selective, distribution-aligned corpus augmentation — a **data** mechanism, not a module (see §2.5) | **Tier 3 §0.2** | **σ ↓** | ⚪ null — σ ratio 0.89 at the paper's own k |
| **M4** | *(optional)* consistency penalty between two stochastic forward passes | new hypothesis | σ ↓ at training time | ⬜ held back |

### ⭐ 2.1 M1 funds M2 exactly — capacity matching by construction

Parameter accounting, derived from the source and cross-checked against the measured 49,204:

| component | untied (published, nb=4) | tied (M1) |
|---|---|---|
| `conv_W00` per tower | 2,560 | 2,560 |
| `lamj` + `lamz` per tower | 257 | 257 |
| LRR blocks per tower | 4 × 4,865 = **19,460** | 1 × 4,865 = **4,865** |
| two towers | 44,554 | 15,364 |
| `Decoder_simple` (`cat`) | 4,650 | 4,650 |
| **total** | **49,204** | **20,014** |

> ✅ **Both columns verified on CPU, 2026-09-03** — `scripts/check_tied_capacity.py`, built from the
> patch itself rather than a re-implementation. 49,204 → 20,014, freeing **exactly 29,190**; the tied
> column is **constant across `num_block` (1.00× spread)** against the untied range's 2.96×; and the
> default (`LRR_TIED` unset) is **tensor-for-tensor identical to upstream** at every block count, so
> stages 1–4 remain reproducible from the same file. Details: [`04_PATCH_NOTES.md`](04_PATCH_NOTES.md).

> **2.46× fewer parameters at nb=4, and 4.40× fewer at nb=8** — where the published model needs
> 88,124 and, as **L7** records, will not train on a 16 GB card at all.

M1 frees **29,190 parameters**. M2's refiner widths are then chosen so that

```
params(M1 + M2)  ==  49,204   exactly
```

**This is the design principle of the chapter.** The comparison against published LRRNet is
capacity-matched *by construction*, not by a control run bolted on afterwards. We cannot be accused
of the confound we spent two case studies documenting, because the architecture makes it
unconstructible.

The decoder collapse (D2) is why M2's budget goes **before** `conv_ReZx` / `conv_ReUx`, not after.

### 2.1b Two implementation decisions, taken 2026-09-03

**Decision A — the tied block is per tower, not shared across modalities.**

`LRR_NET` holds two `GetLS_Net` towers, one per modality. Tying can be done along two independent
axes, and only one of them is M1:

| axis | what it ties | is it M1? |
|---|---|---|
| across **iterations** (blocks 0…3 within a tower) | the unrolled solver's steps | ✅ **yes — this is M1** |
| across **modalities** (the IR tower and the VIS tower) | the operator applied to two different sensors | ❌ no — a second, separate hypothesis |

Doing both at once would change two things in one run — **the exact confound this thesis is built on
criticising**, committed in the first modification we make ourselves. It is also a strong physical
assumption: IR is smooth and blob-like, visible is textured and edge-rich; forcing one solver on both
is a claim, not housekeeping.

| variant | parameters | vs published |
|---|---|---|
| published, untied | 49,204 | 100% |
| **M1 — tied per tower (chosen)** | **20,014** | **40.7% (2.46× smaller)** |
| M1b — additionally tied across modalities | 15,149 | 30.8% (3.25× smaller) |

**M1b is deferred, not discarded** — logged as an optional row (3.3 h) answering a question nobody
has asked: *do the two modalities need different decomposition operators at all?* It is a cleaner
result standing alone than buried inside M1 as an uncontrolled second variable.

**Decision B — Stage 4 and M1 run in the same week, in separate notebooks, Stage 4 first.**

Quota is not the constraint (4.4 + 3.3 = 7.7 h of 30). **Independent failure** is. Stage 3's first
attempt died of OOM; had stage 4 shared that notebook, both rows would have been lost to a fault
unrelated to either. Two notebooks cost nothing and fail independently.

Stage 4 goes first because it is **runnable now with no new code**, while M1 needs patch #12 written
and dry-run against upstream first — the discipline that has already caught two would-be-silent
errors. Writing that patch while stage 4 occupies the GPU costs no wall-clock time. There is no
dependency between them: M1's baseline is the completed `nb=4, n=5` set, not stage 4.

### 2.2 M1 is a result under either outcome

Stages 1–3 already established that **4, 6 and 8 blocks are mutually indistinguishable** (0 of 9
metrics, twice). The tied model has one block's worth of parameters at any depth. So:

| outcome | reading |
|---|---|
| tied ≈ untied (expected) | the extra 29,190 parameters buy nothing — **L6 confirmed from the design side**, and the efficiency claim is categorical (σ ≈ 0 on parameter counts) |
| tied worse | untied capacity does matter after all — which would mean Table IV's rows differ for a reason the table cannot see. Also a finding |
| tied better | regularisation effect; report and investigate |

There is no null outcome. That is why it goes first.

### 2.3 M1 also makes the depth question answerable for the first time

With tied weights, `num_block` costs **zero** parameters. `tied@nb=8` vs `tied@nb=4` is therefore a
**pure depth ablation** — the experiment LRRNet's Table IV was presented as, and was not. To our
knowledge no IVIF paper has run one.

### 2.4 M2 — what goes where

Following the assignment A5 vindicated:

- **`L` (low-rank / long-range):** MDTA-style channel attention, ported from CrossFuse
  `network.py:229` (`Attention`). Channel attention is O(C²) not O(HW²), so it is affordable at
  128×128. Applied through a bottleneck (1×1 down → MDTA(d) → 1×1 up) with `d` set by the budget
  identity in §2.1.
- **`S` (sparse / high-frequency):** a small depthwise-separable CNN, in the spirit of the
  `DetailNode` block that A4 showed the Transformer cannot replace.

**If M2 fails, that is a result too:** it would say `L` and `S` are not in fact a low/high-frequency
split, i.e. D1 is not merely a structural observation but an empirical one — the names do not
describe the tensors.

### 2.5 ⚠️ M3 — CORRECTED 2026-09-06, after reading the released source

**The description below §2.5 was wrong, and the correction is left visible rather than silently
rewritten.** It described Top-K as an architectural module inserted between the two encoder towers.
It is not. Reading `top_k_alignment.py` in the released CrossFuse code shows the mechanism is
entirely **data-level**:

```
1. a PRIMARY corpus and an AUXILIARY corpus
2. compute the primary's per-channel pixel means            -> the target
3. for every auxiliary patch: sum_c |mean_c(patch) - target_c|  -> its difference
4. keep the k patches with the SMALLEST difference          <- "Top-K selective"
5. optimise a per-channel gamma so the kept patches' means match the target  <- "alignment"
6. merge: primary patches + gamma-corrected top-k auxiliary patches
```

So Top-K Selective Channel Alignment is **selective, distribution-aligned corpus augmentation**.
The earlier description was inferred from the name rather than the code — the exact error this
thesis documents in others, committed here and corrected on inspection.

**What this changes for the port.** LRRNet trains on one grayscale corpus, so the mechanism does not
apply as written: there is no auxiliary corpus and no RGB channels. The port therefore has to
construct the setting the mechanism needs:

| | CrossFuse | M3 on LRRNet |
|---|---|---|
| primary | RoadScene | **LLVIP**, 8,415 images |
| auxiliary | M3FD | **MSRS**, cropped to 128×128 patches |
| channels | 3 (B, G, R) | **1 (grayscale)** — the same arithmetic at C=1 |
| merged size | fixed | **12,025 — identical to every previous run**, so iterations per epoch are unchanged |

**The control is what makes it a measurement.** Two runs of identical corpus size and identical
iteration count, differing only in how the 3,610 auxiliary patches were chosen and treated:

| run | auxiliary patches |
|---|---|
| **4a (k=0)** | 3,610 chosen **at random**, no gamma alignment |
| **4b (Top-K)** | 3,610 **top-K selected** and **gamma-aligned** |

`4b − 4a` is the Top-K mechanism alone. This mirrors CrossFuse's own k=0 vs k=1000 contrast, which
is where chapter 4 read a stability effect (σ ratio median 3.35×) rather than a mean-quality one —
the finding M3 exists to test for transfer. ⚠️ **That reading did not survive this very design:**
because `4a` and `4b` hold corpus size fixed and chapter 4's contrast did not, Run 4b showed the
effect belongs to the added data, not the mechanism. See the banner in §2.5b.

Evaluation stays TNO-only, so using MSRS as auxiliary *training* data contaminates nothing: TNO is
never trained on, in any run in this thesis.

### 2.5b The original motivation, unchanged

> ### ⚠️ Superseded 2026-09-06 — read this first
>
> **The premise below was refuted by the experiment it motivated.** Run 4b held corpus size fixed
> and found the mechanism does nothing; the stability effect belongs to the auxiliary *data* Top-K
> adds, not to the selection procedure. Chapter 4's `k=0` arm removed the mechanism **and 47 % of
> the training corpus**, so "Top-K buys stability" was confounded with training-set size all along.
>
> The text is kept as written because this document's job is to record what was believed **before**
> the run, which is what makes the pre-registration real. It is not a current claim. See
> [`05_RESULTS.md`](05_RESULTS.md) Run 4b and withdrawn claim #6 in
> [`../../../CONTEXT.md`](../../../CONTEXT.md).

Tier 3 §0.2 measured that removing Top-K raises seed instability in 22 of 27 cells, median 3.35×.
Top-K's real product is **stability**, and Table VI cannot see it because it reports single-run
means. LRRNet is the noisiest thing we have measured (MI CV 5.43%, VIF 5.24%) and has two separate
encoder towers — a natural insertion point, at ≈ 0 parameters.

**H3: Top-K alignment reduces σ_LRRNet by ≥ 2× at statistically unchanged means.**

A negative result is publishable: *the variance-reducing mechanism does not transfer across
architecture families*, which is itself a constraint on how such mechanisms should be reported.

---

## 3. Protocol — fixed before any run

Inherited unchanged from stages 1–3, which is the point of having run them first.

- **Corpus:** LLVIP, the same 12,025 pairs. Deviation L4 stands and is restated wherever used.
- **Evaluation:** 25 TNO pairs, CrossFuse's `Evaluator`, nine metrics. Same evaluator as every other
  number in this thesis.
- **Baseline:** our own `nb=4, n=5` runs (σ measured). **Never** the published table — L2 forbids it.
- **Seeds:** 3 per configuration (42–44), matching stages 2–3, with the n=5 baseline retained.
- **Statistics:** Welch *t* on pooled σ, plus the effect stated in σ units. Nine correlated metrics
  are tested at once, so a claim requires several agreeing — the rule applied to LRRNet is applied
  to us.
- **Nondeterminism:** cudnn left free, as in stages 1–3. L5's floor (up to 44% of σ at a fixed seed)
  is part of what σ must contain.
- **Patches:** extend `scripts/patch_lrrnet.py`, keep every new flag **defaulting to released
  behaviour**, dry-run against upstream before upload. Eleven patches exist; the same discipline
  applies to the twelfth.

### Reporting obligation

Every result feeds the **four-paper comparison table** at [`03_EXPLAINED.md`](03_EXPLAINED.md)
**§6.2b** — CDDFuse · CrossFuse (Shi et al., T-CSVT 2025) · LRRNet · LRRFuse-VA, across the nine
metrics, parameter count, inference time, training cost, and **mean σ across the nine metrics**, which
is the headline row. Updated the same day a run lands, in **both** the English and Arabic copies of
that table, alongside `../../../CONTEXT.md`, `../EXPLAIN.md` and the results file.

No cell is filled from a published table (**L2**), and CDDFuse's column stays `—`: it was never
trained here at 37.5 h/run, and that emptiness is a finding about the field's compute economics
rather than a gap in the work.

### Pre-registered kill criteria

| # | if | then |
|---|---|---|
| 1 | M1 loses on ≥ 3 of 9 metrics beyond 3σ | report it, and M2/M3 are built on the untied backbone instead |
| 2 | M3 fails to move σ by ≥ 1.5× | report the negative result; do not rescue it by tuning K post hoc |
| 3 | any configuration cannot be capacity-matched to 49,204 within ±1% | report the mismatch in the table rather than reporting the delta as clean |

Written down now so that no result can be re-framed after it is seen. This is the fourth time this
project pre-registers a reading before a run (A4 was the first).

---

## 4. Run plan — ≈ 17 GPU-h, one weekly quota

At the measured 1.10 h per run (nb=4) and 1.06 h (nb=8, 2× T4 via the restored `DataParallel`).

| # | configuration | seeds | GPU-h | question |
|---|---|---|---|---|
| — | **B0** baseline nb=4 | 5 | **0 (done)** | σ_LRRNet, already measured |
| 1 | **M1** tied, nb=4 | 3 | 3.3 | do 29,190 parameters buy anything? |
| 2 | **M1** tied, nb=8 | 3 | 3.3 | ⭐ a clean depth ablation (none found in the tables we examined) |
| 3 | **M1 + M2** capacity-matched to 49,204 | 3 | 3.5 | does the A4/A5 assignment transfer to an LRR split? |
| 4 | **M3** alone, on the published backbone | 3 | 3.3 | does Top-K's variance effect transfer? — isolated |
| 5 | **M1 + M2 + M3** full method | 3 | 3.5 | the proposed method |

Additive rather than 2³ factorial: 5 rows instead of 8, and every row answers one question. Runs 1
and 4 are independent of each other and of run 3, so the order can absorb a failure without
stranding the chapter.

**Completed since this plan was written:** LRRNet stage 4 (`nb=2`, ×3) finished on 2026-09-03 and
closed Table IV at 0 of 27. The paragraph below is the reasoning as it stood beforehand. It was the
one row the published numbers separate from the rest, so it is the table's remaining chance to carry
a real effect. It belongs to Chapter 5 and is scheduled independently of the above.

---

## 5. Risks, stated now

| risk | response |
|---|---|
| Will not beat SOTA on means | Not claimed. The claim is efficiency (categorical, σ ≈ 0) + stability (measured) + a confound-free ablation |
| Our absolute numbers are not comparable to any published table | **L2** establishes this holds for every paper in the field. We compare only against our own n=5 baseline |
| 4 epochs is a small budget for an architecture claim | True, and stated. It is the published budget; changing it would break comparability with stages 1–3 |
| M2 could exceed the parameter budget at any usable width | Then `d` shrinks until the identity holds, or the refiner is shared across towers. The budget is a hard constraint, not a target |
| "This is LRRNet with three patches, not a new architecture" | The strongest objection. Answer: weight tying changes what the model *is* (an unrolled solver rather than a deep CNN wearing its name), and the asymmetric refiners change what the decomposition is *for*. The chapter should defend the design principle — architecture chosen so the confound cannot exist — as the contribution, with the network as its instance |

---

## 6. The sentence the chapter has to earn

> **The first infrared–visible fusion method to state its advantage in units of the run-to-run
> standard deviation measured for its own architecture — and whose advantage is a reduction in that
> standard deviation, the quantity the field does not measure.**

Competitors claim +0.02. This claims: σ down, parameters down 2.46×, and an ablation table in which
the capacity confound is unconstructible.

---

## 7. Open questions for the supervisor

1. Is "stability + efficiency" a strong enough headline for a Chapter 6 contribution, or does the
   examiner expect a mean improvement? (Our position: means at n=1 are what Chapters 4–5 refute.)
2. Should M4 (variance-regularised training) be attempted, or does it over-claim against the
   registered title? The title's `Design` covers it; the compute does too (3.3 h).
3. Does the LLVIP substitution (L4) weaken a *method* claim more than it weakened a *replication*
   claim? Our reading: no, because every row including the baseline uses it.
