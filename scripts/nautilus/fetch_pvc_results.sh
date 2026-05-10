#!/usr/bin/env bash
# Fetch results from a Nautilus job that wrote into the unified-model-data-vol PVC.
#
# Usage:
#   ./scripts/nautilus/fetch_pvc_results.sh <job-name>
#
# Why this exists:
#   `kubectl cp` is blocked on Succeeded pods (it uses kubectl exec under
#   the hood, and exec is not allowed on completed pods). Jobs that mount
#   the CephFS PVC at /workspace/results survive pod deletion -- the data
#   is durably on the PVC -- but the only way to read it back to a laptop
#   is to mount the same PVC into a fresh "reader" pod and kubectl-cp from
#   that. This script automates the round-trip: spin up reader, copy out,
#   tear down reader.
#
# Companion to scripts/nautilus/fetch_repro_results.sh, which is for the
# OLDER pattern where a job wrote to pod-ephemeral storage and we copied
# from the still-Running pod. Use THIS script for any job whose YAML mounts
# unified-model-data-vol with subPathExpr "$(JOB_NAME)" -- which is the
# convention going forward (eval-matrix-stage1_5.yaml,
# variance-stage1-stage1_5.yaml, repro-stage1-with-base-baseline.yaml,
# and the 9 eval YAMLs swept in Phase 1.6).

set -euo pipefail

JOB_NAME="${1:?usage: fetch_pvc_results.sh <job-name>}"
NS="${NAMESPACE:-svcl-self-improve}"
PVC_NAME="${PVC_NAME:-unified-model-data-vol}"
# subPathExpr "$(JOB_NAME)" expands to the literal job name on the writer
# pod, so the reader pod just mounts a static subPath of the same value.
PVC_SUBPATH="${PVC_SUBPATH:-$JOB_NAME}"
LOCAL_DIR="${LOCAL_DIR:-results/$JOB_NAME}"

# Reader pod name: short, unique, kubernetes-DNS-safe (no underscores).
# Truncate JOB_NAME at 40 chars so the full pod name stays under k8s's
# 63-char limit even with the timestamp suffix.
SHORT_JOB="$(printf '%s' "$JOB_NAME" | tr '_' '-' | cut -c1-40)"
READER_POD="pvc-reader-${SHORT_JOB}-$(date +%s)"

# Sanity-check the Job: if it isn't done yet, the data on the PVC may be
# partial. We don't refuse to fetch (sometimes you want a peek mid-run),
# but we warn loudly so the user notices.
if kubectl -n "$NS" get job "$JOB_NAME" >/dev/null 2>&1; then
  COMPLETIONS=$(kubectl -n "$NS" get job "$JOB_NAME" -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")
  ACTIVE=$(kubectl -n "$NS" get job "$JOB_NAME" -o jsonpath='{.status.active}' 2>/dev/null || echo "0")
  if [[ "${COMPLETIONS:-0}" != "1" ]]; then
    echo "WARNING: job $JOB_NAME is not Succeeded yet (succeeded=${COMPLETIONS:-0}, active=${ACTIVE:-0})." >&2
    echo "WARNING: results on the PVC may be partial. Continuing anyway." >&2
  fi
else
  echo "Note: no Job named '$JOB_NAME' found in namespace $NS. Continuing -- maybe the Job was already deleted but data survives on the PVC." >&2
fi

mkdir -p "$LOCAL_DIR"

# Always tear down the reader pod, even if kubectl cp fails partway.
cleanup() {
  if kubectl -n "$NS" get pod "$READER_POD" >/dev/null 2>&1; then
    echo "Cleaning up reader pod $READER_POD ..."
    kubectl -n "$NS" delete pod "$READER_POD" --wait=false >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Reader pod manifest. busybox is small (~5MB) and has the `sh` we need
# for kubectl cp's tar-pipe to work. 1h sleep is plenty -- we delete it
# manually right after the cp.
READER_MANIFEST=$(cat <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $READER_POD
  namespace: $NS
  labels:
    app: pvc-reader
    target-job: "$JOB_NAME"
spec:
  restartPolicy: Never
  containers:
    - name: reader
      image: busybox:1.36
      command: ["sh", "-c", "sleep 3600"]
      volumeMounts:
        - name: results
          mountPath: /data
          subPath: "$PVC_SUBPATH"
      resources:
        requests:
          cpu: "50m"
          memory: "32Mi"
        limits:
          cpu: "200m"
          memory: "128Mi"
  volumes:
    - name: results
      persistentVolumeClaim:
        claimName: $PVC_NAME
EOF
)

echo "=== creating reader pod $READER_POD (PVC=$PVC_NAME subPath=$PVC_SUBPATH) ==="
echo "$READER_MANIFEST" | kubectl apply -f - >/dev/null

echo "=== waiting for reader pod to be Ready (timeout 60s) ==="
if ! kubectl -n "$NS" wait --for=condition=Ready "pod/$READER_POD" --timeout=60s; then
  echo "ERROR: reader pod did not become Ready in 60s. Recent events:" >&2
  kubectl -n "$NS" describe pod "$READER_POD" | tail -30 >&2
  exit 2
fi

echo "=== copying /data from reader pod -> $LOCAL_DIR/ ==="
# Trailing /. on the source means "copy contents into" rather than "copy
# the dir as a child." That keeps the local path layout compatible with
# the older fetch_repro_results.sh script.
kubectl cp -n "$NS" "$READER_POD:/data/." "$LOCAL_DIR"

# cleanup() runs via trap on EXIT, so no explicit delete here.

echo
echo "=== contents of $LOCAL_DIR ==="
ls -lh "$LOCAL_DIR"

# Auto-detect known summary JSON shapes and pretty-print the head so the
# user can eyeball results without leaving the terminal.
echo
for cand in eval_matrix.json reproduction_summary.json variance_summary.json; do
  if [[ -f "$LOCAL_DIR/$cand" ]]; then
    echo "=== $cand (head) ==="
    python3 -m json.tool "$LOCAL_DIR/$cand" | head -80
    break
  fi
done
