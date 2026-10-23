#!/usr/bin/env bash
# Verification script for the TensorFlow Serving lab.
#
# Checks, cheapest first:
#   1. docker-compose.yml syntax
#   2. backend unit tests (TensorFlow Serving mocked with respx)
#   3. backend running live against a fake TensorFlow Serving, over real HTTP
#   4. the frontend script, executed headlessly
#   5. the frontend/backend contract on the upload path
#
# Everything here runs WITHOUT Docker and without the real model server, so it
# isolates application bugs from packaging bugs. Once this is green, run the
# real thing with: docker compose up --build
#
# Usage: ./verify.sh

set -u

GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YELLOW=$'\033[0;33m'; NC=$'\033[0m'
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv-verify"
TMP="$PROJECT_DIR/.verify-tmp"
TFS_PORT=8599
BACKEND_PORT=8088
FAILURES=0

pass() { echo "${GREEN}PASS${NC}  $1"; }
fail() { echo "${RED}FAIL${NC}  $1"; FAILURES=$((FAILURES + 1)); }
skip() { echo "${YELLOW}SKIP${NC}  $1"; }
step() { echo ""; echo "=== $1 ==="; }

cleanup() {
  [ -n "${TFS_PID:-}" ] && kill "$TFS_PID" 2>/dev/null
  [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null
  rm -rf "$TMP"
}
trap cleanup EXIT

# ------------------------------------------------------------- 1. compose
step "1. docker-compose.yml syntax"
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  if (cd "$PROJECT_DIR" && docker compose config --quiet 2>/dev/null); then
    pass "compose file is valid"
  else
    fail "compose file is invalid"
    (cd "$PROJECT_DIR" && docker compose config 2>&1 | tail -5)
  fi
else
  skip "docker not available, cannot validate compose file"
fi

# ---------------------------------------------------------- 2. unit tests
step "2. Environment and backend unit tests"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV" || { fail "could not create venv"; exit 1; }
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install -q -r "$PROJECT_DIR/backend/requirements-dev.txt" \
               -r "$PROJECT_DIR/frontend/requirements.txt" || {
  fail "dependency installation failed"; exit 1; }

if (cd "$PROJECT_DIR/backend" && python -m pytest tests/ -q 2>&1 | tail -3 | grep -q "passed"); then
  pass "backend unit tests"
else
  fail "backend unit tests"
  (cd "$PROJECT_DIR/backend" && python -m pytest tests/ -q 2>&1 | tail -15)
fi

# ------------------------------------------------- 3. live HTTP round trip
step "3. Backend against a fake TensorFlow Serving (real HTTP)"
mkdir -p "$TMP"
python - "$TMP" <<'PY'
import json, sys
from PIL import Image
tmp = sys.argv[1]
Image.new("RGB", (640, 480), (90, 140, 60)).save(f"{tmp}/photo.jpg")
json.dump({"207": "golden_retriever"}, open(f"{tmp}/labels.json", "w"))
PY

python "$PROJECT_DIR/tools/fake_tf_serving.py" "$TFS_PORT" > "$TMP/tfs.log" 2>&1 &
TFS_PID=$!

# Wait for the fake model server before starting the backend, otherwise the
# first status call races against its startup.
for _ in $(seq 1 25); do
  curl -sf "http://127.0.0.1:$TFS_PORT/v1/models/mobilenet" >/dev/null 2>&1 && break
  sleep 1
done
if curl -sf "http://127.0.0.1:$TFS_PORT/v1/models/mobilenet" >/dev/null 2>&1; then
  pass "fake TensorFlow Serving is up"
else
  fail "fake TensorFlow Serving did not start"
  tail -15 "$TMP/tfs.log"
  exit 1
fi

(cd "$PROJECT_DIR/backend" && \
  TF_SERVING_URL="http://127.0.0.1:$TFS_PORT" \
  MODEL_NAME=mobilenet \
  LABELS_PATH="$TMP/labels.json" \
  python -m uvicorn app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" \
  > "$TMP/backend.log" 2>&1) &
BACKEND_PID=$!

for _ in $(seq 1 25); do
  curl -sf "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

if curl -sf "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then
  pass "backend is up"
else
  fail "backend did not start"
  tail -15 "$TMP/backend.log"
  exit 1
fi

STATUS=$(curl -sS "http://127.0.0.1:$BACKEND_PORT/model/status")
if echo "$STATUS" | grep -q '"available":true'; then
  pass "backend reached the model server and reports it available"
else
  fail "model status wrong: $STATUS"
fi

CODE=$(curl -sS -X POST "http://127.0.0.1:$BACKEND_PORT/predict" \
  -F "file=@$TMP/photo.jpg" -o "$TMP/response.json" -w "%{http_code}")

if [ "$CODE" = "200" ]; then
  pass "POST /predict returned 200"
else
  fail "POST /predict returned $CODE: $(cat "$TMP/response.json")"
fi

# Note: the response is read from a file, not piped, because `python - <<EOF`
# already uses stdin for the script itself.
RESPONSE_FILE="$TMP/response.json" python - <<'PY' && pass "response shape and ranking are correct" || fail "response shape is wrong"
import json, os
with open(os.environ["RESPONSE_FILE"], encoding="utf-8") as handle:
    payload = json.load(handle)
assert len(payload["predictions"]) == 5, "expected top-5"
probs = [p["probability"] for p in payload["predictions"]]
assert probs == sorted(probs, reverse=True), "predictions are not sorted"
assert all(0.0 <= p <= 1.0 for p in probs), "probability outside 0..1"
for key in ("label", "class_index", "probability"):
    assert key in payload["predictions"][0], f"missing {key}"
for key in ("filename", "model_name", "inference_time_ms"):
    assert key in payload, f"missing {key}"
PY

# Wrong payloads must be rejected, not silently accepted.
printf 'this is not an image at all' > "$TMP/notes.txt"
BAD=$(curl -sS -o /dev/null -w "%{http_code}" -X POST \
  "http://127.0.0.1:$BACKEND_PORT/predict" -F "file=@$TMP/notes.txt")
if [ "$BAD" = "422" ]; then
  pass "non-image upload rejected with 422"
else
  fail "non-image upload returned $BAD (expected 422)"
fi

# ------------------------------------------------------------ 4. frontend
step "4. Frontend script executed headlessly"
FRONT=$(cd "$PROJECT_DIR/frontend" && \
  APP_BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py', default_timeout=60)
at.run()
if at.exception:
    print('EXCEPTION:', [str(e.value) for e in at.exception]); raise SystemExit(1)
print('SIDEBAR_OK:', bool(at.sidebar.success))
" 2>&1)

if echo "$FRONT" | grep -q "EXCEPTION"; then
  fail "frontend raised an exception"
  echo "$FRONT" | grep EXCEPTION
elif echo "$FRONT" | grep -q "SIDEBAR_OK: True"; then
  pass "frontend runs and sees the model server as ready"
else
  fail "frontend did not detect the model server"
  echo "$FRONT"
fi

# --------------------------------------------- 5. upload contract
# Streamlit's AppTest cannot drive file_uploader, so the upload path is
# verified by issuing the exact request the frontend builds.
step "5. Frontend/backend contract on the upload path"
(cd "$PROJECT_DIR" && BACKEND="http://127.0.0.1:$BACKEND_PORT" IMG="$TMP/photo.jpg" python - <<'PY'
import os, requests
backend, img = os.environ["BACKEND"], os.environ["IMG"]
with open(img, "rb") as handle:
    data = handle.read()
r = requests.post(f"{backend}/predict",
                  files={"file": ("photo.jpg", data, "image/jpeg")}, timeout=60)
assert r.status_code == 200, r.status_code
payload = r.json()
# Every key frontend/app.py reads after a successful upload:
assert "predictions" in payload and "inference_time_ms" in payload
best = payload["predictions"][0]
for key in ("label", "probability", "class_index"):
    assert key in best, f"missing {key}"
assert 0.0 <= best["probability"] <= 1.0, "st.progress would reject this value"
PY
) && pass "frontend's upload request works against the backend" \
  || fail "frontend/backend contract is broken"

# ------------------------------------------------------------- summary
step "Summary"
if [ "$FAILURES" -eq 0 ]; then
  echo "${GREEN}All checks passed.${NC}"
  echo "Next step: docker compose up --build   (first run downloads TensorFlow, be patient)"
  exit 0
else
  echo "${RED}$FAILURES check(s) failed.${NC}"
  exit 1
fi
