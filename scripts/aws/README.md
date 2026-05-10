# AWS launch infrastructure

Parallel infra to `scripts/nautilus/`. Same model, different cloud — use AWS when:

- Nautilus contention/admission is blocking
- We need predictable wall-clock for a paper-deadline-adjacent experiment
- We want to run multiple parallel jobs that Nautilus's GPU contention can't accommodate

## What's set up

| Resource | Identifier | Purpose |
|---|---|---|
| S3 bucket | `unified-model-results-723951822728-us-west-2` | Durable result storage (encrypted, versioned, public-blocked) |
| IAM role | `unified-model-eval-instance-role` | EC2 → S3 write + CloudWatch logs + ECR pull |
| Instance profile | `unified-model-eval-instance-profile` | Wraps the role for EC2 attachment |
| Security group | `sg-01f0a2c1e71f022da` (`unified-model-eval-sg`) | Egress-only; no SSH from the world |
| Launch template | `unified-model-eval-lt` | DL Base AMI (Ubuntu 22.04 + NVIDIA drivers + CUDA), 200GB gp3 EBS, IMDSv2-required, terminate-on-shutdown |
| Default region | `us-west-2` | per project's `CLAUDE.md` mandate (account 723951822728) |

All identifiers are persisted in `.agents/aws_infra/ids.env` (source it to get them as env vars).

## Quotas — currently the bottleneck

| Quota | Limit | Status |
|---|---|---|
| Running On-Demand G+VT (g5/g6) vCPU | 4 (= 1× g5.xlarge) | active |
| Running On-Demand P (A100/H100) vCPU | 0 | requested 64 (PENDING) |
| Spot G+VT vCPU | 0 | n/a (was 0; can't request lower) |
| Spot P vCPU | 0 | requested 64 (PENDING) |

Until quotas land we can run **one** g5.xlarge (1× A10G GPU, 24GB) at a time on-demand — about $1.00/hr. This is fine for 3B inference; for training it's slower than the A100s on Nautilus.

## Usage

```bash
# 1. Launch a job
./scripts/aws/launch_job.sh <job-name> <job-script-path> [instance-type]

# Example: run the eval-matrix on AWS instead of Nautilus
./scripts/aws/launch_job.sh eval-matrix-aws scripts/aws/jobs/eval_matrix.sh

# Example with a bigger instance (once P-class quota lands):
./scripts/aws/launch_job.sh stage2-pilot scripts/aws/jobs/stage2_pilot.sh p4d.24xlarge

# 2. Watch instance state
aws ec2 describe-instances --filters "Name=tag:Job,Values=eval-matrix-aws" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]' --output table

# 3. Fetch results (any time after job completes; S3 is durable)
./scripts/aws/fetch_results.sh eval-matrix-aws
```

## Per-job script convention

Per-job scripts live in `scripts/aws/jobs/`. They:
- Run on the launched instance after the bootstrap
- Have `JOB_NAME`, `S3_BUCKET`, `GIT_SHA`, `GIT_BRANCH` exported
- Should write all output to `/workspace/results/$JOB_NAME/`
- Don't need to handle S3 upload — the bootstrap does that

The bootstrap auto-terminates the instance on completion, so per-job scripts don't need cleanup logic. Failed jobs still upload (the bootstrap captures exit code separately and force-syncs).

## Cost discipline

- `InstanceInitiatedShutdownBehavior: terminate` — when the bootstrap calls `shutdown -h`, the instance is **terminated** (not stopped), so EBS billing also ends.
- Default 200 GB gp3 EBS = ~$16/mo if a volume ever leaks. Periodic check: `aws ec2 describe-volumes --filters Name=status,Values=available` should return empty.
- All instances are tagged `Project=unified-model-open-instruct` for cost allocation.

## When NOT to use AWS

- Short eval jobs (<1 GPU-h) — Nautilus contention is annoying but not 1-hour annoying. Stay on Nautilus.
- Anything that needs a specific Nautilus-only resource (e.g. the existing CephFS PVC).
- When AWS quotas are still 0 for the GPU class you need.

## Verify after creating a new job

`aws s3 ls s3://unified-model-results-723951822728-us-west-2/` should show the new prefix appearing within a few minutes of launch (the bootstrap creates `start.txt` early).
