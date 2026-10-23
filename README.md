# Serving a Model with TensorFlow Serving, Docker Compose and Streamlit

Module 5 homework for *Solution Deployment & Communication* (FHTW):
*Serving Models with Docker, TensorFlowServe, and TorchServe*.

Upload an image in the browser, get a classification back from a MobileNetV2
model served by **TensorFlow Serving**, with a FastAPI backend in between.
Everything is started by a single `docker compose up --build`.

## Architecture

```
                         app-network (bridge)

  ┌──────────────┐    ┌───────────────┐    ┌────────────────────┐
  │   frontend   │───▶│    backend    │───▶│    tf-serving      │
  │  Streamlit   │HTTP│   FastAPI     │REST│ TensorFlow Serving │
  │    :8501     │    │    :8000      │    │  :8501 / :8500     │
  └──────────────┘    └───────────────┘    └────────────────────┘
        │                    │                      │
   host :8501          host :8000             host :8601
                             │                      │
                             └──── model-store ─────┘
                                   (named volume)
                                        ▲
                                  ┌───────────┐
                                  │model-init │  runs once, then exits
                                  │TensorFlow │
                                  └───────────┘
```

Four services:

| Service | Role |
|---|---|
| `model-init` | Runs once: exports MobileNetV2 as a SavedModel, writes `labels.json` and `models.config` into the shared volume, then exits. |
| `tf-serving` | Serves the exported model over REST and gRPC. |
| `backend` | Receives the uploaded image, preprocesses it, calls TensorFlow Serving, maps class indices to names. |
| `frontend` | Streamlit UI with the file uploader. |

Startup order is enforced by Compose:
`model-init` (`service_completed_successfully`) → `tf-serving` (`service_healthy`)
→ `backend` (`service_healthy`) → `frontend`.

### Why a separate model-init service

TensorFlow is a large dependency and is only needed to *export* the model, not
to serve it or to call it. Keeping the export in its own container that runs
once and exits means the backend image stays small: it talks to TensorFlow
Serving over plain HTTP and does its preprocessing with Pillow and NumPy
instead of TensorFlow.

## How to run

```bash
docker compose up --build
```

Then open **http://localhost:8501**, upload an image and click *Classify*.

The first run takes a while: `model-init` installs TensorFlow and downloads
the MobileNetV2 weights. Later runs reuse the volume and skip the export.

Useful addresses while it runs:

| URL | What |
|---|---|
| http://localhost:8501 | Streamlit frontend |
| http://localhost:8000/docs | Backend API docs |
| http://localhost:8601/v1/models/mobilenet | TensorFlow Serving model status |

To stop, and to also discard the exported model:

```bash
docker compose down          # stop
docker compose down -v       # stop and delete the model volume
```

## Based on the course material

- **Request format.** The class notebook (Moodle, "code for next batches")
  calls the model with
  `requests.post('http://localhost:8501/v1/models/animal_classifier:predict', data=json.dumps({"instances": ...}))`
  and reads `json.loads(response.text)['predictions']`. `backend/app/services/tfserving.py`
  uses exactly that contract.
- **models.config.** The lecturer's gist configures TensorFlow Serving through
  a config file with an explicit `model_version_policy` and `version_labels`,
  rather than through `--model_name` / `--model_base_path`. This project does
  the same: `tf-serving` starts with `--model_config_file=/models/models.config`
  (see `model-init/models.config`, which also documents how to add more
  versions and labels).
- **Model choice.** The class model (`animal_classifier`) lives in the lab
  repository, which is no longer reachable since GitHub Classroom was retired
  on 28 August 2026. The assignment allows choosing a different model, so this
  project uses MobileNetV2 with ImageNet weights, which downloads reproducibly
  from Keras.

## API

### `POST /predict`

Multipart upload with a `file` field.

```bash
curl -X POST http://localhost:8000/predict -F "file=@cat.jpg"
```

```json
{
  "filename": "cat.jpg",
  "model_name": "mobilenet",
  "predictions": [
    {"label": "tabby", "class_index": 281, "probability": 0.71},
    {"label": "tiger_cat", "class_index": 282, "probability": 0.18}
  ],
  "inference_time_ms": 42.7
}
```

Status codes: `200` success, `422` not a readable image or empty file,
`413` larger than 10 MB, `503` TensorFlow Serving unreachable or not ready.

### `GET /model/status`

Reports whether TensorFlow Serving has the model loaded. The frontend uses it
for the status indicator in the sidebar.

### `GET /health`

Liveness of the backend itself, without touching the model server.

## Testing

```bash
./verify.sh
```

Runs, from cheapest to most expensive: compose file syntax, the backend unit
tests, the backend live against a **fake TensorFlow Serving** over real HTTP,
the frontend script executed headlessly, and the frontend/backend contract on
the upload path. None of it needs Docker or the real model server, so a
failure points at application code rather than packaging.

The unit tests alone:

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

TensorFlow Serving is mocked with `respx`, so the suite runs offline.

## Project layout

```
.
├── docker-compose.yml
├── verify.sh                     end-to-end checks without Docker
├── model-init/
│   ├── Dockerfile
│   ├── export_model.py           exports MobileNetV2 + labels.json
│   └── models.config             TensorFlow Serving model configuration
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt          runtime deps (no TensorFlow)
│   ├── requirements-dev.txt      + pytest, respx
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── routers/predict.py    POST /predict, GET /model/status
│   │   └── services/
│   │       ├── preprocessing.py  Pillow/NumPy, matches preprocess_input
│   │       └── tfserving.py      TensorFlow Serving REST client
│   └── tests/test_api.py
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                    Streamlit, file uploader + top-5
└── tools/
    └── fake_tf_serving.py        stand-in model server used by verify.sh
```

## Running in GitHub Codespaces

`.devcontainer/devcontainer.json` enables Docker-in-Docker, so `docker compose`
works inside a Codespace.

1. **Code > Codespaces > Create codespace on main**.
2. `docker compose up --build`
3. Open the forwarded port **8501** from the *Ports* tab.

If the Streamlit page keeps reconnecting behind the Codespaces proxy, add
`--server.enableCORS=false --server.enableXsrfProtection=false` to the `CMD`
in `frontend/Dockerfile` and rebuild.

## Notes and limitations

- **Preprocessing is duplicated conceptually.** The backend reimplements
  `mobilenet_v2.preprocess_input` (`x / 127.5 - 1`) with NumPy so it does not
  need TensorFlow. If the model is swapped for one with different
  preprocessing, `backend/app/services/preprocessing.py` must be updated to
  match, otherwise predictions silently degrade.
- **Single model version.** `models.config` pins version 1. The file documents
  how to serve several versions with `stable` / `canary` labels, the pattern
  shown in class.
- **No authentication** on any of the services; they are meant to run locally.
- **The `tf-serving` healthcheck** tries `curl` first and falls back to a plain
  TCP check through bash, because the serving image is not guaranteed to ship
  with curl.
