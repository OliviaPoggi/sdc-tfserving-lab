# Image Classification with TensorFlow Serving and Docker Compose

Module 5 homework for *Solution Deployment & Communication* (FHTW).

A Streamlit web app where you upload an image and get a prediction from a
MobileNetV2 model served by **TensorFlow Serving**. Everything starts with a
single `docker compose up --build`.

## Architecture

Three services on a shared Docker network:

```
  ┌────────────┐        ┌──────────────────┐
  │  frontend  │───────▶│    tf-serving    │
  │ Streamlit  │  HTTP  │ TensorFlow Serv. │
  │   :8501    │        │      :8501       │
  └────────────┘        └──────────────────┘
         │                       │
         └──── model-store ──────┘
              (shared volume)
                     ▲
              ┌─────────────┐
              │ model-init  │  runs once, then exits
              └─────────────┘
```

| Service | What it does |
|---|---|
| `model-init` | Downloads MobileNetV2, saves it into the shared volume, exits. |
| `tf-serving` | Loads that model and serves it over a REST API. |
| `frontend` | Streamlit page: upload an image, show the prediction. |

The frontend reaches the model server at `http://tf-serving:8501`.
`tf-serving` is the service name, which Docker Compose resolves on its
internal network, so no IP addresses are hardcoded.

`model-init` exists so that TensorFlow is only installed where it is actually
needed. The frontend talks to the model over HTTP, so it stays a small image
without TensorFlow.

## How to run

```bash
docker compose up --build
```

Then open **http://localhost:8501**, upload an image and click *Classify*.

The first run takes several minutes: `model-init` has to install TensorFlow
and download the model weights. It is finished when the log shows:

```
Done. TensorFlow Serving can now load the model.
```

Later runs reuse the volume and start immediately.

To stop:

```bash
docker compose down          # stop the containers
docker compose down -v       # also delete the downloaded model
```

## How a prediction works

1. The user uploads an image in the browser.
2. The frontend resizes it to 224x224 and scales the pixel values from
   `[0, 255]` to `[-1, 1]`, which is what MobileNetV2 expects.
3. It sends `{"instances": [image]}` to
   `http://tf-serving:8501/v1/models/mobilenet:predict`.
4. TensorFlow Serving answers `{"predictions": [[...1000 scores...]]}`,
   one score per ImageNet class.
5. The frontend sorts the scores, keeps the five highest, and looks up their
   names in `labels.json`, which `model-init` wrote into the shared volume.

The request and response format is the one used in the course notebook
("code for next batches" on Moodle).

## Testing

Tested manually through the browser. Uploading a photograph of a white wolf
returns `white wolf` with about 72% confidence, and the rest of the top 5 are
related animals (Arctic fox, dingo, ice bear, timber wolf).

Two observations from testing:

- The first prediction takes around 1300 ms, later ones around 120 ms.
  TensorFlow Serving is slower on the first call because it initialises the
  model, and fast afterwards.
- Images that are not photographs, such as a logo, give a low confidence
  spread over unrelated classes. That is expected: MobileNetV2 was trained on
  photographs of 1000 everyday categories and has no class for abstract
  graphics.

## Notes

- MobileNetV2 is trained on ImageNet, so it recognises animals, objects,
  food and vehicles. There is no class for people.
- `tf-serving` is published on port 8601 on the host instead of 8501,
  because the frontend already uses 8501 there. Inside the Docker network
  both still use 8501.
- The model is downloaded at build time, so the first start needs an internet
  connection.

## Running in GitHub Codespaces

`.devcontainer/devcontainer.json` enables Docker, so `docker compose` works
inside a Codespace. Create the codespace, run `docker compose up --build`,
then open port **8501** from the *Ports* tab.
