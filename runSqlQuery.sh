#!/usr/bin/env bash
#
# run_query.sh <your natural‑language query>
#
#   ./run_query.sh "Show all employees in the Engineering department"

if [ -z "$1" ]; then
  echo "Usage: $0 \"<query>\""
  exit 1
fi

QUERY="$1"

curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"${QUERY//\"/\\\"}\",\"db_type\":\"mysql\"}" | jq .