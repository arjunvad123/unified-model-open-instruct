# Nautilus Deployment Guide for Unified Agentic Model

This guide follows Nautilus cluster policies exactly. Read the golden rules first.

## Golden Rules (Memorize These)

1. **NO protected data** - No HIPAA, FERPA, PID data ever
2. **Use Jobs, not Pods** - All training must use `kind: Job`
3. **NO sleep infinity** - Script must run to completion
4. **GPU utilization >40%** - Or you get violations and potential ban
5. **Clean up when done** - Delete jobs after completion
6. **Storage is temporary** - Move results to permanent storage within 6 months

## Prerequisites

### 1. Install kubectl
```bash
# macOS
brew install kubectl

# Verify
kubectl version --client
```

### 2. Configure kubectl
1. Download kubeconfig from: https://portal.nrp-nautilus.io
2. Copy to config:
```bash
mkdir -p ~/.kube
cp ~/Downloads/config ~/.kube/config
chmod 600 ~/.kube/config
```

### 3. Set Your Namespace (REQUIRED)
```bash
# Set the namespace for svcl-self-improve
kubectl config set-context nautilus --namespace=svcl-self-improve

# Verify
kubectl config get-contexts
```

## Deployment Steps

### Step 1: Create Storage (One-Time Setup)

Create the PVCs for your storage:

```bash
# Create conda environment storage (RBD - 50GB)
kubectl apply -f pvc-conda.yaml

# Create data storage for datasets/checkpoints (CephFS - 200GB)
kubectl apply -f pvc-data.yaml

# Verify PVCs are created
kubectl get pvc
```

Expected output:
```
NAME                      STATUS   VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS
unified-model-conda-vol   Bound    ...      50Gi       RWO            rook-ceph-block
unified-model-data-vol    Bound    ...      200Gi      RWX            rook-cephfs
```

### Step 2: Submit the Training Job

```bash
# Submit the job
kubectl apply -f unified-training-job.yaml

# Verify job is created
kubectl get jobs
```

### Step 3: Monitor Training

```bash
# Get pod name
kubectl get pods

# Watch logs (replace <pod-name> with actual name)
kubectl logs -f <pod-name>

# Describe job for details
kubectl describe job unified-agentic-training

# Check for violations (IMPORTANT - do this regularly)
# Visit: https://portal.nrp-nautilus.io -> Violations page
```

### Step 4: Clean Up When Done (REQUIRED)

```bash
# Delete the job when training completes
kubectl delete job unified-agentic-training

# Verify deletion
kubectl get jobs
kubectl get pods
```

## Monitoring Resources

Check these dashboards to ensure you're not getting violations:

- **GPU Usage**: https://grafana.nrp-nautilus.io/d/dRG9q0Ymz/k8s-compute-resources-namespace-gpus
- **CPU/Memory**: https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods
- **Violations**: Check the Nautilus portal regularly

## Retrieving Results

After training completes, copy results to your local machine:

```bash
# First, create a temporary pod to access the data
kubectl run data-access --image=alpine --restart=Never -- sleep 3600

# Wait for pod to start
kubectl wait --for=condition=Ready pod/data-access

# Copy files from the data volume
kubectl cp data-access:/data/unified_model_output ./local_output

# Clean up the temporary pod
kubectl delete pod data-access
```

## Troubleshooting

### Job Not Starting
```bash
# Check job status
kubectl describe job unified-agentic-training

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

### Out of Memory (OOM)
If your job gets OOM-killed, increase memory in the YAML:
```yaml
resources:
  requests:
    memory: "60Gi"  # Increase this
  limits:
    memory: "72Gi"  # Keep within 20% of request
```

### GPU Not Available
Check available GPUs:
```bash
kubectl get nodes -L nvidia.com/gpu.product
```

### Violation Warnings
If you get fair-share violations:
1. Check GPU utilization on Grafana
2. Ensure your training is actually using the GPU
3. Don't run interactive sessions with GPUs

## File Structure

```
scripts/nautilus/
├── README.md                    # This file
├── pvc-conda.yaml              # Conda environment storage (RBD)
├── pvc-data.yaml               # Data/checkpoint storage (CephFS)
└── unified-training-job.yaml   # Main training job
```

## Important Policies Summary

| Policy | Limit | Consequence |
|--------|-------|-------------|
| GPU Utilization | Must be >40% | Violation → potential ban |
| Resource Limits | Within 20% of requests | Job rejected or OOM-killed |
| Storage Access | 6 months | Data purged without notice |
| Interactive Pods | 6 hours max | Automatically destroyed |
| Jobs | Run to completion | Must finish (no sleep infinity) |
| Violations | Max 4 pods | Namespace banned |

## Quick Reference Commands

```bash
# Submit job
kubectl apply -f unified-training-job.yaml

# Check job status
kubectl get jobs

# Get pod name
kubectl get pods

# View logs
kubectl logs -f <pod-name>

# Delete job
kubectl delete job unified-agentic-training

# Check PVCs
kubectl get pvc

# Check storage usage
kubectl exec <pod-name> -- df -h
```
