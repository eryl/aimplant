#!/bin/bash

case "$1" in
  simulate)
    echo "Starting FLARE simulator..."
    shift
    nvflare simulator /app/apps/xlmroberta_mlm -w /app/workspace/xlmroberta-mlm -n 2 -t 1 -gpu 0
    ;;
  client)
    echo "Starting FLARE client..."
    shift
    python run_client.py "${@}"
    ;;
  bash)
    exec /bin/bash
    ;;
  *)
    echo "Unknown command: $1"
    exec "$@"
    ;;
esac
