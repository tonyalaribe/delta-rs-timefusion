#!/usr/bin/env bash
set -euo pipefail

hdfs_diagnostics="$RUNNER_TEMP/hdfs-diagnostics"
hdfs_bin="$RUNNER_TEMP/hdfs-diagnostic-bin"
mkdir -p "$hdfs_diagnostics" "$hdfs_bin"
export HDFS_REAL_MVN="$(command -v mvn)"
export HDFS_MAVEN_STDERR_LOG="$hdfs_diagnostics/maven.stderr"
export HDFS_MAVEN_WRAPPER="$GITHUB_WORKSPACE/.github/scripts/hdfs-maven.py"
# MiniDfs discards Maven stderr. Forward the real command unchanged and retain it.
cat > "$hdfs_bin/mvn" <<'MVN'
#!/bin/sh
exec python3 "$HDFS_MAVEN_WRAPPER" "$@"
MVN
chmod +x "$hdfs_bin/mvn"
export PATH="$hdfs_bin:$PATH"
cat > "$hdfs_diagnostics/nextest.toml" <<'NEXTEST'
[profile.default]
slow-timeout = { period = "60s", terminate-after = 5 }
NEXTEST

result=0
timeout 20m cargo llvm-cov nextest \
    --config-file "$hdfs_diagnostics/nextest.toml" \
    --no-capture \
    --cargo-profile=ci \
    --features integration_test \
    --package deltalake-hdfs \
    --codecov \
    --output-path codecov.json || result=$?
printf '%s\n' "$result" > "$hdfs_diagnostics/test-exit-code"
exit "$result"
