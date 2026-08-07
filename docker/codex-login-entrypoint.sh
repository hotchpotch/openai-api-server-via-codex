#!/bin/sh
set -eu

# Codex's login server binds to loopback inside the container, which Docker's
# published ports cannot reach. Bridge 0.0.0.0:1456 to it so the host browser
# can complete the http://localhost:1455 callback via the port mapping
# 127.0.0.1:1455 -> 1456.
socat TCP-LISTEN:1456,fork,reuseaddr,bind=0.0.0.0 TCP:127.0.0.1:1455 &

exec codex "$@"
