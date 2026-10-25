#!/usr/bin/env bash
set -euo pipefail

hdfs_diagnostics="$RUNNER_TEMP/hdfs-diagnostics"
hdfs_bin="$RUNNER_TEMP/hdfs-diagnostic-bin"
mkdir -p "$hdfs_diagnostics" "$hdfs_bin"
export HDFS_REAL_MVN="$(command -v mvn)"
export HDFS_MAVEN_STDERR_LOG="$hdfs_diagnostics/maven.stderr"
# The first stdout line is a readiness protocol, so Hadoop logging uses stderr.
export HDFS_MAVEN_LOGGING_OPTIONS="-Dlog4j.configuration=file:$GITHUB_WORKSPACE/.github/scripts/hdfs-log4j.properties -Dlog4j.configurationFile=file:$GITHUB_WORKSPACE/.github/scripts/hdfs-log4j2.xml"
# MiniDfs discards Maven stderr. Forward the real command unchanged and retain it.
cat > "$hdfs_bin/mvn" <<'MVN'
#!/bin/sh
export MAVEN_OPTS="${MAVEN_OPTS:-} $HDFS_MAVEN_LOGGING_OPTIONS"
exec "$HDFS_REAL_MVN" "$@" 2>>"$HDFS_MAVEN_STDERR_LOG"
MVN
chmod +x "$hdfs_bin/mvn"
export PATH="$hdfs_bin:$PATH"
cat > "$hdfs_diagnostics/nextest.toml" <<'NEXTEST'
[profile.default]
slow-timeout = { period = "60s", terminate-after = 5 }
NEXTEST

# Capture the actual Maven JVM while the unchanged integration test runs.
(
    captures=0
    while [ "$captures" -lt 3 ]; do
        while read -r java_pid java_main; do
            case "$java_main" in
                *plexus*|main.Main)
                    timeout 10s jstack -l "$java_pid" > "$hdfs_diagnostics/jvm-$java_pid-$captures.txt" 2>&1 || true
                    captures=$((captures + 1))
                    [ "$captures" -lt 3 ] || break
                    ;;
            esac
        done < <(jps -l || true)
        [ "$captures" -lt 3 ] && sleep 30
    done
) &
hdfs_watcher=$!
trap 'kill "$hdfs_watcher" 2>/dev/null || true; wait "$hdfs_watcher" 2>/dev/null || true' EXIT

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
