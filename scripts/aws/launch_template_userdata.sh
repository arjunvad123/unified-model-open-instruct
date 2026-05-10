#!/bin/bash
# EC2 user-data: minimal bootstrap. Per-job user-data overrides this entirely.
set -euo pipefail
exec > >(tee -a /var/log/unified-model-bootstrap.log | logger -t unified-model) 2>&1
echo "unified-model bootstrap: no per-job user-data provided; idling for inspection."
sleep 3600
shutdown -h now
