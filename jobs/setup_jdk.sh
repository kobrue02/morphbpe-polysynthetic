#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p tools/jdk
curl -fL -o /tmp/portable-jdk.tar.gz \
  "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
tar -xzf /tmp/portable-jdk.tar.gz -C tools/jdk --strip-components=1
rm -f /tmp/portable-jdk.tar.gz

tools/jdk/bin/java -version
echo "JDK installed at tools/jdk/ -- fst_boundaries_iku.py and the job scripts will find it automatically."
