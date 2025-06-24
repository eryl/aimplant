#!/bin/bash

case "$1" in
  simulate)
    echo "Starting FLARE simulator..."
    shift
    nvflare simulator "${@}"
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
