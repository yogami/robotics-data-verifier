## VERDICT
**BLOCK**

This pipeline has multiple independent, complete-forgery-capable security holes, a statistically degenerate primary result that is being mis-reported as a valid null finding, and strong circumstantial evidence that the uniform 0% success rate is an artifact of a broken evaluation harness rather than a genuine experimental outcome. None of the "trust boundary" security claims in the architecture description are actually enforced by the code as written.

---

## BLOCKING ISSUES

### 1. Signature verification is a no-op stub — the entire crypto chain is theater
**File:** `scratch/mixed_effects_model.py`, `verify_ed25519_signature()`
```python
def verify_ed25519_signature(public_key_openssh: str, entry: dict) -> bool:
    if "signature" in entry:
        entry.pop("signature")
    return True
```
This never loads the public key, never calls `.verify()`, and **always returns `True`**. Combined with the fact that `verification_ledger.jsonl` is assembled locally by the untrusted Executor (`autonomous_researcher.py::trigger_github_workflow` appends downloaded artifact bytes to a local file it fully controls), the Executor can fabricate the entire ledger — including fake `eval_info.json` files with matching SHA-256 hashes and arbitrary `"signature"` strings — and `mixed_effects_model.py` will accept it without complaint. **This single bug defeats 100% of the claimed cryptographic guarantees.**

**Fix:**
```python
from cryptography.exceptions import InvalidSignature

def verify_ed25519_signature(public_key_openssh: str, entry: dict) -> bool:
    entry = dict(entry)
    sig_hex = entry.pop("signature", None)
    if not sig_hex:
        return False
    payload_str = json.dumps(entry, sort_keys=True)
    public_key = serialization.load_ssh_public_key(public_key_openssh.encode("utf-8"))
    try:
        public_key.verify(bytes.fromhex(sig_hex), payload_str.encode("utf-8"))
        return True
    except InvalidSignature:
        return False
```
Also fix the call site — it currently discards the return value context (it does check it, but only because the stub always returns True; add a unit test that asserts a tampered entry is rejected).

### 2. `torch.load(weights_only=False)` inside the trusted signing boundary = RCE + signing-key theft
**Files:** `eval_policy.py` (`eval_policy()`), `train_bc_policy.py` (`evaluate()`)
```python
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
```
GHA (the "Trusted Evaluation Boundary") downloads a `.pt` file from an HF repo that the *untrusted Executor controls the upload token for*, and deserializes it with full pickle execution enabled, **in the same runner process that holds `secrets.EVAL_PRIVATE_KEY`**. A malicious Executor can craft a `.pt` file with a `__reduce__` payload, upload it as `bc_model.pt`, dispatch the eval workflow pointed at that revision, and achieve arbitrary code execution with access to the Ed25519 signing key — enabling forging of *any* future ledger entry with a valid signature. This is the most severe single vulnerability in the system: it breaks even a *correctly implemented* signature check (issue #1).

**Fix:**
```python
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
```
Confirm the checkpoint only contains tensors/primitives (it does: `model_state`, `obs_dim`, `action_dim`, `dataset_hash`, `seed`, `infection_level` — all safe under `weights_only=True`). Additionally, the signing step should be isolated into a separate, minimal job/service that never executes untrusted code or model files, and receives only the final numeric result to sign.

### 3. No training provenance / attestation — Executor can fabricate checkpoints without training anything
**Files:** `train_bc_policy.py::AlohaDataset.__init__`, `autonomous_researcher.py::run_ssh_training`
The `dataset_hash` embedded in the checkpoint is **self-reported by code running on infrastructure fully controlled by the untrusted Executor** (root SSH on RunPod). Nothing prevents the Executor from:
- Not running `train_bc_policy.py` at all,
- Hand-crafting a `.pt` file with any `model_state`, and copying the correct pre-registered `dataset_hash`/`seed`/`infection_level` values verbatim from `experiment_manifest.yaml`,
- Uploading it via the HF API themselves, extracting a real commit SHA, and dispatching eval.

There is no TEE/enclave, no remote attestation, no compute-time proof, and no check anywhere that gradient descent genuinely occurred on the declared parquet file. The GHA "trust boundary" only validates the evaluation step, which is real — but it evaluates whatever bytes the Executor chose to upload, with **zero causal link to actual training on infected/clean data.** Given the stated research claim ("does training data infection degrade performance"), this makes the entire causal claim of the experiment unverifiable under the declared threat model.

**Fix:** This requires an architectural change, not a patch:
- Run training itself inside a similarly attested/trusted environment (e.g., a GHA self-hosted runner or a signed, reviewed job spec), or
- Require the training log itself to be included in the signed ledger with a hash of intermediate loss curves/gradient checkpoints that can be spot-checked, or
- At minimum, log and independently re-verify a deterministic re-hash of a random subset of raw parquet rows against the manifest hash from a copy staged by a *trusted* party (not the RunPod box), and have the GHA workflow independently re-derive part of the training result (e.g., re-run a few epochs) as a probabilistic proof-of-work spot check.

### 4. Executor holds credentials sufficient to modify the "trusted" evaluation code before dispatch
**Files:** `autonomous_researcher.py` (`GITHUB_PAT` used for both `git clone` push-capable auth and workflow dispatch), `.github/workflows/evaluator.yml`
The evaluator workflow checks out `ref: main` and merely asserts `git rev-parse HEAD == eval_commit_sha`. This prevents *train/eval commit drift* but does **nothing** to prevent the Executor — who possesses `GITHUB_PAT` with clone/push rights implied by the training clone command — from pushing a malicious commit to `main` (e.g., replacing `eval_policy.py`'s success criterion with `True`, or `verify_logs.py` with a stub) and then dispatching evaluation against that exact SHA. There is no branch protection, required review, or CODEOWNERS gate mentioned or enforced anywhere.

**Fix:** Enforce GitHub branch protection on `main` (required PR review from a second, independent identity; disallow force-push; disallow direct pushes). Issue the Executor a fine-grained PAT scoped only to `actions:write` (workflow dispatch) and read access — never `contents:write`. Move `eval_policy.py`/`verify_logs.py`/`mixed_effects_model.py` provenance-critical code into a protected path requiring signed commits from a separate reviewer identity.

### 5. `seed` never actually seeds anything — the "5 random seeds" design is not implemented
**Files:** `train_bc_policy.py::train()`, `eval_policy.py::eval_policy()`
`seed` is accepted as a CLI arg and stored in checkpoint metadata purely for **provenance labeling**, but `torch.manual_seed`, `np.random.seed`, `random.seed`, and `env.reset(seed=...)` are **never called anywhere in the codebase**. The DataLoader uses `shuffle=True` with no seeded generator; `env.reset()` in both training-side and GHA-side evaluation is unseeded. This means:
- The "seed" independent variable does not control any actual randomness,
- Runs are not reproducible,
- The premise of "5 independent replicate seeds per condition" for variance estimation is not actually satisfied — the observed run-to-run variability is uncontrolled ambient noise, not a designed randomization.

**Fix:**
```python
def train(..., seed=None, ...):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        g = torch.Generator(); g.manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, ...)
```
```python
observation, info = env.reset(seed=seed + ep)  # in eval_policy.py, per-episode reseed
```

### 6. Pre-registered `baseline_check` safety gate is dead code — it never runs, and the current 0% result would have failed it
**Files:** `.github/workflows/evaluator.yml`, `scratch/verify_logs.py`, `scratch/experiment_manifest.yaml`
`evaluator.yml` hardcodes `phase="sweep_logging"` for every dispatched job — there is no `phase` input on `workflow_dispatch`, and `phase == "baseline_check"` in `verify_logs.py` is unreachable in production. The manifest specifies `baseline_target_success_rate: 50.0` with `exploratory_seeds: [9001,9002,9003]` intended as a pre-flight sanity gate before running the full 25-job sweep. Had this check actually run against the observed 0/500 baseline, `proportion_confint` would yield `lo≈0 < 0.5` and the run would `sys.exit(1)`, **aborting the sweep**. Instead, the sweep proceeded to completion on data that should have triggered an automatic stop.

Compounding this, `autonomous_researcher.py`'s human sign-off gate ("Type PROCEED") **never displays the baseline's actual success rate or reward** to the human before asking for approval — it only prints "BASELINE COMPLETED SUCCESSFULLY," which merely means the (currently no-op) GHA job exited 0, not that the model actually worked. Both automated and human gates were vacuous in this run.

**Fix:** Add `phase` and `nonce` to `workflow_dispatch.inputs`, thread a genuine baseline job through `phase="baseline_check"` before dispatching the 25 sweep jobs, and have `autonomous_researcher.py` fetch and print the actual baseline success-rate/reward from the downloaded ledger/eval_info before prompting for PROCEED.

### 7. GEE fit is degenerate (NaN std error / complete separation) and is silently reported as a valid null result
**File:** `scratch/mixed_effects_model.py`
```
infection   -2.829e-16    nan    nan    nan    nan    nan
```
With all 25 runs having exactly 0/500 successes, the binomial response has zero variance — this is a textbook complete-separation degenerate MLE, not a valid hypothesis test. The code does:
```python
if not result.converged:
    sys.exit(1)
...
p_val_t = scipy.stats.t.sf(abs(t_stat), df=df_resid) * 2
if p_val_t < alpha: ...
else: print("NO STATISTICALLY SIGNIFICANT EFFECT FOUND.")
```
`result.converged` may report `True` even though the sandwich variance is singular (as shown by the NaN std err), and `t_stat` will be `nan`; `nan < alpha` is `False` in Python, so the code silently falls through to the "no effect" branch. **This conflates "model is undefined" with "no effect found,"** which is scientifically false and misleading.

**Fix:**
```python
import numpy as np
if df["successes"].sum() == 0:
    print("ABORTING: Zero successes across all runs. Binomial GEE is degenerate. "
          "Cannot assess treatment effect on success-rate outcome.")
    sys.exit(1)
...
if not np.isfinite(t_stat) or not np.isfinite(result.bse["infection"]):
    print("ABORTING: Degenerate GEE fit (non-finite standard error). "
          "Do not report a p-value from this model.")
    sys.exit(1)
```
Per Q2: the continuous `max_reward` metric should be the fallback/primary here — but note this must be declared as a **post-hoc deviation from the pre-registered binomial analysis plan**, clearly labeled as exploratory, not substituted silently. Given the reward distribution is bounded, zero-inflated, and non-normal, recommend a rank-based ordered-alternative test (Jonckheere–Terpstra across the ordinal infection levels, stratified/blocked by seed) or a permutation test on the linear trend of mean `max_reward`, rather than assuming Gaussian LMM.

### 8. Suspicious observation-alignment hack is the most likely root cause of the uniform 0% result and is unverified
**File:** `eval_policy.py`, `BCPolicyWrapper.select_action`
```python
if obs.shape[-1] == 16 and expected_dim == 14:
    obs = obs[:, 2:16]
```
There is no justification, logging, or test confirming that `agent_pos` returned by `gym_aloha` at index `[2:16]` semantically corresponds to the same physical joints, in the same order, as the `observation.*`/`state.*` columns used to build `obs` in `AlohaDataset` at training time. If this alignment is wrong (e.g., should be `[:14]`, or the parquet column order differs from the sim's `qpos` order), the policy receives systematically garbage/misaligned inputs at eval time regardless of training quality or infection level — which is fully consistent with the observed uniform near-zero, non-monotonic reward across **all** 25 runs. This must be resolved before the null result can be trusted.

**Fix:** Before running the real sweep again:
1. Log `obs.shape[-1]` and which branch is taken on every eval run.
2. Add a **positive control**: replay the raw teleoperation action sequences from the training parquet directly through `env.step()` and confirm they reliably achieve `max_reward >= 4.0` in this same harness. If the positive control also fails, the evaluation harness itself is broken and no conclusion whatsoever can be drawn from the current results.
3. Add an assertion that state/action column semantics (units, ranges) match between parquet and sim `agent_pos`/action space.

### 9. Pre-registered hyperparameter search space and analysis plan do not correspond to the code actually run
**File:** `scratch/experiment_manifest.yaml` vs `train_bc_policy.py`
The manifest declares an ACT-style hyperparameter search space (`chunk_size: [50,100]`, `training_steps: 200000`, `batch_size: [8,16,32]`, `learning_rate: [1e-5,5e-5,1e-4]`) — these are hyperparameters for a transformer-based action-chunking policy (ACT), not a plain per-timestep MLP. The actual implementation (`BCPolicy`, 105K params, hardcoded `lr=1e-3`, `batch_size=256`, `epochs=100`) ignores every value in this search space entirely. Nothing in the codebase enforces that the manifest's declared hyperparameters were used. This means:
- The "pre-registration" doesn't bind the actual model/training procedure at all,
- The manifest almost certainly documents a different (and considerably more capable) intended architecture than what was executed,
- This substitution plausibly explains the floor-effect null result (plain per-step MLP-BC is known to perform very poorly on ALOHA insertion vs. ACT, independent of data quality).

**Fix:** Either implement ACT per the manifest, or rewrite the manifest to match what is actually implemented and re-declare it as a new pre-registration before the sweep is run again. Add a manifest field enumerating the *actual* trainer hyperparameters used, hashed and checked by `verify_logs.py` the same way `dataset_hash` is checked.

### 10. Manifest's declared `success_metric: "recomputed_from_episodes"` is not implemented
**File:** `scratch/mixed_effects_model.py`
```python
successes = sum(1 for e in episodes if e.get("success", False))
```
This trusts the boolean `"success"` field written by `eval_policy.py` rather than recomputing it from the raw `max_reward` field per the manifest's stated intent (`"success_metric": "recomputed_from_episodes"`). Given issue #2 (RCE risk in the same process that writes this field), independent recomputation is a meaningful defense-in-depth control that is currently absent.

**Fix:**
```python
SUCCESS_THRESHOLD = 4.0
successes = sum(1 for e in episodes if e.get("max_reward", -1) >= SUCCESS_THRESHOLD)
```

---

## NON-BLOCKING IMPROVEMENTS

1. **Duplicate/dead code** — `autonomous_researcher.py::run_ssh_training` has the entire "extract HF commit SHA" block duplicated verbatim after the first `return` statement (fully unreachable). Delete the second copy.
2. **BCPolicy class defined 3 separate times** (`train_bc_policy.py`, `eval_policy.py`, and inline in `train_bc_policy.py::evaluate`) with hardcoded `hidden=256`. Refactor into a shared `policy.py` module imported by all three to prevent architecture drift causing silent state-dict load failures.
3. **`train_bc_policy.py::evaluate()` is effectively dead code in production** — RunPod's `pip install` in `autonomous_researcher.py::run_ssh_training` never installs `gym_aloha`/`gymnasium`, so this local eval always short-circuits to the `ImportError` fallback (`success_rate: -1.0`). Either install the dependency or remove this unused code path to stop wasting GPU time per run.
4. **Torch not explicitly installed on RunPod** — relies on an undocumented assumption that the base image has it pre-installed. Add an explicit `pip install torch==<pinned>` with hash pinning, or assert its presence and version at start of `run_ssh_training`.
5. **Unpinned dependency installs inside the trusted GHA runner** (`pip install gym-aloha==0.1.1 lerobot cryptography pyyaml statsmodels` with no hash pinning) — supply-chain risk inside the boundary holding `EVAL_PRIVATE_KEY`. Use `pip install --require-hashes` with a locked requirements file.
6. **Single shared credential (`GITHUB_PAT`, `HF_TOKEN`) reused across multiple privilege scopes** (clone, dispatch, upload). Violates least privilege; scope tokens separately per function.
7. **`HF_TOKEN` extracted via `/proc/1/environ` and appended to `/etc/environment` in plaintext** on the RunPod box (`run_ssh_training`). Fragile and insecure; pass the token explicitly via a scoped SSH env-forwarding mechanism or a short-lived secret file with restrictive permissions, then shred it.
8. **Shell command construction via raw f-strings with `shell=True`** throughout `autonomous_researcher.py`. Even though current inputs are operator-controlled, this is bad practice; use `shlex.quote()` or argument lists.
9. **Evaluator workflow requires `main` HEAD to remain frozen at exactly `eval_commit_sha`** during the entire polling window rather than checking out `ref: ${{ inputs.eval_commit_sha }}` directly — this creates unnecessary operational fragility (any concurrent push during a long training run breaks unrelated jobs). Checkout the specific SHA directly instead, and separately verify (e.g., via `git merge-base --is-ancestor`) that it is reachable from a protected `main`.
10. **Signed ledger payload omits `eval_commit_sha`/GHA `run_id`/`nonce`** — even after fixing issue #1, a legitimately signed entry from one context could be replayed into an unrelated experiment sharing the same manifest hash. Include these fields in the signed payload.
11. **GEE uses only 5 clusters (seeds)** with `cov_type='bias_reduced'` — statistically underpowered per general GEE small-sample guidance (commonly ≥30–40 clusters recommended even with bias correction). This is a design-time limitation to flag even independent of the degenerate fit.
12. **Infection level modeled as strictly linear/continuous** in the GEE formula — assumes monotonic dose-response; the raw reward data shown (50% > 100% > 25% > 75% > 0%) is non-monotonic, suggesting either noise or a genuinely non-linear relationship that a linear-on-logit model would not detect. Consider treating infection as categorical/ordinal with a trend contrast as a secondary check.
13. **No positive control / harness self-test** anywhere in the pipeline (see blocking #8) — should be a standing regression test, not a one-off debugging step.
14. **`AlohaDataset`'s column-identification fallback** (`mid = len(feature_cols)//2` 50/50 split when no naming convention matches) has zero validation that the resulting `obs_dim`/`action_dim` are physically plausible. Add assertions (e.g., expected dims for ALOHA ~14).
15. **No train/validation split** reported for BC training — cannot assess overfitting/generalization of the offline fit at all before sim rollout.
16. **No observation/action normalization** before MSE regression — plausible cause of poor convergence independent of infection.

---

## METHODOLOGY CRITIQUE

**Q1 — Is 0% success across all 25 conditions a valid outcome or a pipeline flaw?**
Almost certainly a pipeline flaw, not a valid negative result, for three independent reasons: (a) the observation-alignment hack in `BCPolicyWrapper` (blocking #8) is unverified and could easily be misaligning inputs; (b) the `seed` bug (blocking #5) means there is no genuine controlled randomization behind the reported variance; (c) the manifest/implementation mismatch (blocking #9) suggests a much weaker model class than intended was actually trained. A hard task producing a genuine floor effect is *possible*, but nothing in this pipeline lets you distinguish "task too hard" from "harness broken," because there is no positive control (blocking #8). **Do not publish this as a valid negative finding until a positive control passes.**

**Q2 — Is the GEE binomial analysis valid with all-zero outcomes?**
No. This is complete separation — the model is mathematically degenerate (as the NaN std. error demonstrates), and the current code path silently reports it as "no significant effect," which is a false characterization (blocking #7). If a genuine null on `max_reward` is desired, use a distribution-appropriate, pre-registered-as-exploratory secondary analysis (rank-based ordered-alternative test, e.g. Jonckheere–Terpstra, blocked by seed) — not a Gaussian LMM, given the boundedness/skew of reward.

**Q5 — Minimum changes for publishable results:**
1. Fix the observation-alignment bug and add a positive control (blocking #8).
2. Fix seeding (blocking #5).
3. Reconcile manifest vs. actual model/hyperparameters, and if the intent was ACT, implement ACT — a plain MLP-BC is known to be inadequate for bimanual insertion (blocking #9).
4. Re-run the pre-flight `baseline_check` for real and require it to pass before dispatching the sweep (blocking #6).
5. Fix the degenerate-GEE detection and pre-register (as a distinct, dated amendment) the fallback continuous-outcome analysis before looking at the sweep data (blocking #7, #10).
6. Only after a genuine non-floor baseline is achieved does the infection-level sweep become scientifically meaningful.

---

## SECURITY AUDIT

**Q3 — Can the Executor forge results?** Yes, in at least four independent, sufficient ways:
1. Fake ledger signatures (blocking #1) — trivial, no cryptography required at all currently.
2. RCE via `torch.load(weights_only=False)` inside the signing boundary → steal `EVAL_PRIVATE_KEY` → forge anything, forever (blocking #2).
3. No training attestation → fabricate a checkpoint wholesale, upload it, and have it "genuinely" evaluated by the real harness with no relationship to the declared training data (blocking #3).
4. Push malicious code to `main` (no branch protection) before dispatch, altering `eval_policy.py`/`verify_logs.py` themselves (blocking #4).

Any single one of these fully compromises the pipeline's integrity claims; all four are present simultaneously.

**Q4 — Is the dataset hash verification chain complete?**
Functionally complete for *detecting a data-file substitution after the fact* (parquet → checkpoint → HF → GHA → eval_info → ledger all correctly propagate and cross-check a SHA-256 of the parquet bytes). But the chain's root of trust — the hash computed inside `AlohaDataset.__init__` — is computed by code running entirely on Executor-controlled infrastructure with no attestation that it was genuinely computed over genuinely-used training data (see blocking #3). The chain is "complete" as a bookkeeping mechanism but does not constitute a security proof under the stated threat model.

**Trust boundary architecture assessment:** The design intent (Executor untrusted → GHA trusted → Verifier checks signatures) is reasonable in outline, but every implementation detail that would make it actually enforceable is either missing or stubbed out. As shipped, the "Trusted Evaluation Boundary" is trusted in name only.

---

## CODE QUALITY

- `scratch/mixed_effects_model.py`: unused import `ed25519` module is imported but the actual `Ed25519PublicKey` API is never used (since verification is stubbed) — dead import once fixed, will become live.
- `autonomous_researcher.py::run_ssh_training`: duplicated dead code block after `return` (noted above).
- Broad `except Exception: pass` swallowing in `eval_policy.py`'s dataset-hash fallback path — masks real errors; should at least `print`/log the exception even if using a default.
- Inconsistent success criteria between `train_bc_policy.py::evaluate()` (`reward > 0`) and `eval_policy.py` (`ep_max_reward >= 4.0`) — confusing given both nominally evaluate the same env/task; consolidate into one shared function.
- `verify_logs.py` and `mixed_effects_model.py` duplicate significant logic (manifest loading/hashing, dataset hash lookup with `str(key)`/`int(key)` fallback) — factor into a shared module to avoid divergence.
- Magic numbers throughout (`hidden=256`, `4.0` success threshold, `400` max steps) with no named constants or documentation of their provenance (e.g., why 4.0 is the ALOHA insertion success threshold should be cited/commented).
- `scratch/verify_logs.py::verify_log` argument parsing via raw `sys.argv` positional indices with no argparse/help text — fragile CLI, easy to misuse.