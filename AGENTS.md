# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.
- `uv run mkdocs serve`: View the documentation locally at http://127.0.0.1:8000/
- `uv run mkdocs build`: Build the documentation to the `site/` directory.

# Workflow
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- Launch tool use experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/tool_grpo_fast.sh`.
- Launch multi-node non-tool experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh`.
- Launch the GPU tests with `./scripts/train/build_image_and_launch.sh scripts/train/debug/run_gpu_tests.sh`.

# Documentation
To verify that documentation changes don't alter the generated output:
1. Build docs on your branch: `uv run mkdocs build && cp -r site site-branch`
2. Switch to main branch and build: `cd /path/to/main && uv run mkdocs build`
3. Compare the builds: `diff -rq site-branch /path/to/main/site`
4. If no output, the docs are identical. If differences exist, review with: `diff -r site-branch /path/to/main/site`

## AWS

Only use AWS account **723951822728** for all AWS operations in this project.

# Benchmarking and reproducibility

## Action / control tokens
The trained tokens are defined ONCE in `open_instruct/action_tokens.py`:
`<ACT:GEN>`, `<ACT:RET>`, `<ACT:THINK>`, `<ACT:STOP>`.
Both training (`open_instruct/unified_finetune.py`) and every benchmark/eval/demo
script imports from there. **Never hardcode token lists in scripts.** Earlier
versions referenced `<ACT:TOOL>` and `<ACT:CODE>` which were never trained;
those references have been removed. If a future training round adds new tokens,
update the registry first, then everything downstream picks them up.

## Smoke tests vs research benchmarks
`benchmarks/README.md` has the canonical table. TL;DR:
- `run_generation_eval.py`, `run_embedding_eval.py` → smoke tests, regression
  detection only, do not cite numbers from these in reports
- `run_lm_eval.py`, `run_mteb.py`, `run_mteb_comparison.py` → research-grade,
  reportable
- `run_ragas.py` → research with caveat (full RAGAS needs OpenAI judge key)

## Reproducing the report
`benchmarks/REPRODUCTION.md` has exact `lm_eval` invocations to reproduce any
cell in `benchmarks/BENCHMARK_REPORT.md`. Use `--seed 42` and pin
`lm-eval==0.4.11` to match the most recent independent reproduction
(`benchmarks/REPRODUCTION_RESULTS.md`).

## Decontamination
Before publishing benchmark numbers, dump the eval prompts and scan against
training data:
```
python benchmarks/dump_smoke_prompts.py
# then on cluster: see decontamination/EVAL_CONTAMINATION_CHECK.md
```
Known structural leak: Stage-1.5-v3 trains on
`sentence-transformers/natural-questions` train and `run_embedding_eval.py::eval_retrieval`
evaluates on the same source — flagged inline with `# TODO(decontam)`. Fix
before reporting Stage-1.5-v3 retrieval numbers.

# Nautilus job conventions (lessons learned the hard way)

## Always mount the data PVC for results
Eval Job specs MUST mount `unified-model-data-vol` (CephFS RWX, 200GB) at
`/workspace/results` with `subPathExpr: "$(JOB_NAME)"`. The `JOB_NAME` env
var comes from the Downward API
(`fieldPath: "metadata.labels['batch.kubernetes.io/job-name']"`). Reference
implementation: `scripts/nautilus/repro-stage1-with-base-baseline.yaml`.

Why this matters: if results land in pod-ephemeral storage instead of a PVC,
`kubectl cp` is the only extraction path, and `kubectl cp` is BLOCKED on
`Succeeded` pods (it uses `kubectl exec` under the hood). Once the pod
completes you cannot get the data out. We lost a full A100 reproduction
this way once; don't repeat.

## Always fetch BEFORE delete
```
bash scripts/nautilus/fetch_repro_results.sh   # MUST run first
kubectl delete job <name> -n svcl-self-improve
```
PVC mount makes this safer (data survives pod deletion) but the muscle
memory still matters because `ttlSecondsAfterFinished: 86400` cleans the
pod automatically after 24h.

## backoffLimit: 0 for eval jobs
Failed eval pods are debugging gold. `backoffLimit: 0` keeps them around;
`backoffLimit: 1+` retries can burn another GPU allocation and write into
the same PVC subdir. Arjun's preference per `d989fb16`. Training jobs may
differ.

## Validate without launching
```
kubectl apply --dry-run=server --validate=true -f <yaml> -n svcl-self-improve
```
Catches admission-policy violations (CPU/mem ratio, etc.) before scheduling.
Schema and PVC existence both verified. Mount/runtime can only be proven by
launching a tiny smoke pod.

## When the canonical YAML location is unsettled
Eval YAMLs live in BOTH `scripts/nautilus/` and `benchmarks/kubernetes-jobs/`
on the merge branch (the latter was Arjun's reorganization). Until that's
resolved, prefer editing `scripts/nautilus/` and propose moves explicitly.

# Personal log
This repo follows the personal-log convention from `~/.claude/CLAUDE.md`:
append a dated entry to `.agents/log.md` before stopping if the turn made
any decision, took action with non-obvious rationale, made an assumption,
opened a question, or made meaningful progress.
