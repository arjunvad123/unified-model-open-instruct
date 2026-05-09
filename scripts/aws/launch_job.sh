#!/usr/bin/env bash
# Launch an AWS EC2 GPU instance that runs a job script and uploads results to S3.
#
# Usage:
#   ./scripts/aws/launch_job.sh <job-name> <job-script-path> [instance-type]
#
# Examples:
#   ./scripts/aws/launch_job.sh eval-matrix scripts/aws/jobs/eval_matrix.sh
#   ./scripts/aws/launch_job.sh stage2-smoke scripts/aws/jobs/stage2_smoke.sh g5.2xlarge
#
# The instance:
#   - boots Deep Learning AMI (Ubuntu 22.04, NVIDIA drivers + CUDA preinstalled)
#   - clones this repo at the current commit (passed via --git-ref)
#   - runs the supplied job script with JOB_NAME, S3_BUCKET env vars exported
#   - uploads /workspace/results/$JOB_NAME/ to s3://$S3_BUCKET/$JOB_NAME/
#   - terminates on shutdown (no idle billing)
#
# Default instance: g5.xlarge (1x A10G, 24GB GPU mem, ~$1.00/hr on-demand).
# Quotas in this account currently allow only 4 vCPU on-demand G/VT,
# i.e. one g5.xlarge at a time. Quota increase requests pending.

set -euo pipefail

JOB_NAME="${1:?usage: launch_job.sh <job-name> <job-script-path> [instance-type]}"
JOB_SCRIPT_PATH="${2:?usage: launch_job.sh <job-name> <job-script-path> [instance-type]}"
INSTANCE_TYPE="${3:-g5.xlarge}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ ! -f "$JOB_SCRIPT_PATH" ]]; then
  echo "ERROR: job script not found: $JOB_SCRIPT_PATH" >&2
  exit 1
fi

# Load AWS infra IDs (created by setup once)
# shellcheck disable=SC1091
source "$REPO_ROOT/.agents/aws_infra/ids.env"

GIT_SHA=$(git rev-parse HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
JOB_SCRIPT_B64=$(base64 -i "$JOB_SCRIPT_PATH" | tr -d '\n')

USER_DATA=$(cat <<EOF
#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/unified-job.log | logger -t unified-job) 2>&1

export JOB_NAME=${JOB_NAME}
export S3_BUCKET=${S3_BUCKET}
export GIT_SHA=${GIT_SHA}
export GIT_BRANCH=${GIT_BRANCH}

mkdir -p /workspace/results/\$JOB_NAME
cd /workspace

git clone https://github.com/arjunvad123/unified-model-open-instruct.git repo
cd repo
git checkout \$GIT_SHA

# Decode and run the per-job script
echo "${JOB_SCRIPT_B64}" | base64 -d > /tmp/job.sh
chmod +x /tmp/job.sh

# Mark start
date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ" > /workspace/results/\$JOB_NAME/start.txt

# Run job; capture exit code; never abort the upload step
set +e
/tmp/job.sh
JOB_EXIT=\$?
set -e

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" > /workspace/results/\$JOB_NAME/end.txt
echo "\$JOB_EXIT" > /workspace/results/\$JOB_NAME/exit_code.txt

# Always sync results, even on failure (logs are debugging gold)
cp /var/log/unified-job.log /workspace/results/\$JOB_NAME/job.log 2>/dev/null || true
aws s3 sync /workspace/results/\$JOB_NAME/ s3://\$S3_BUCKET/\$JOB_NAME/ --no-progress

# Auto-terminate (InstanceInitiatedShutdownBehavior=terminate stops billing)
shutdown -h +1 "job complete (exit \$JOB_EXIT)"
EOF
)

# AWS CLI base64-encodes the user-data automatically when you pass a string or
# `file://` reference. Pre-encoding ourselves caused double-encoding (cloud-init
# saw base64 text instead of an executable script). Fix per Codex review on
# PR #2: write to a tempfile and reference via `file://`.
USER_DATA_FILE=$(mktemp -t unified-userdata.XXXXXX)
trap 'rm -f "$USER_DATA_FILE"' EXIT
printf '%s' "$USER_DATA" > "$USER_DATA_FILE"

echo "=== launching $INSTANCE_TYPE for job=$JOB_NAME (git $GIT_SHA on $GIT_BRANCH) ==="
INSTANCE_ID=$(aws ec2 run-instances \
  --launch-template "LaunchTemplateName=${LAUNCH_TEMPLATE_NAME},Version=\$Latest" \
  --instance-type "$INSTANCE_TYPE" \
  --user-data "file://$USER_DATA_FILE" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=unified-${JOB_NAME}},{Key=Job,Value=${JOB_NAME}},{Key=GitSha,Value=${GIT_SHA}}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "InstanceId: $INSTANCE_ID"
echo "Job results will land at: s3://${S3_BUCKET}/${JOB_NAME}/"
echo "Tail logs (after ~2 min once instance is up):"
echo "  aws ssm start-session --target $INSTANCE_ID  # if SSM agent + session manager configured"
echo "  aws s3 cp s3://${S3_BUCKET}/${JOB_NAME}/job.log -  # after run completes"
echo "Watch instance state:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[].Instances[].[State.Name,InstanceType,LaunchTime]' --output table"
