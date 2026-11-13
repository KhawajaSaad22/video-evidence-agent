# Video evidence-search agent — project 2

CLIP encoding can use CUDA. See [GPU.md](GPU.md) for the optional GPU container;
the default published image uses CPU PyTorch.

Reliability update: second-pass verifier frames remain chronologically ordered with
their timestamps and evidence references. In evaluation manifests, use an empty
`relevant_intervals` list for a negative query. Negative false-positive rate is
reported separately from positive-query recall. An empty predictions list counts
as abstention; this does not automatically calibrate raw retrieval scores.

Status: working visual retrieval baseline with a deterministic, budgeted inspection
controller. Uses CLIP image/text embeddings, cached frames, timestamped candidates,
fine-grained resampling, a CLI, and a Streamlit interface. An optional local Ollama
VLM verifier can assess candidate frames and request more inspection within a hard
budget. Free-form planning and reliable action-order reasoning are not implemented.

## Setup and run

Use a dedicated Python environment and `pip install -r requirements.txt`. This workspace
has the needed packages in the sibling `../hybrid-rag/.venv`; its PyTorch build is CPU-only.
On Windows substitute `..\hybrid-rag\.venv\Scripts\python.exe` for `python` below.

```powershell
python make_demo.py
python video_search.py index data/demo.avi --output artifacts/demo --step 2
python video_search.py search artifacts/demo "a blue circle" --output demo-result.json
python -m streamlit run app.py
python -m unittest -v
```

The first run downloads `openai/clip-vit-base-patch32` weights. No paid API is used.
The generated 12-second AVI contains colored shapes and only checks decoding/retrieval;
it cannot validate repair-video reasoning. To index my own video:

```powershell
python video_search.py index C:/videos/repair.mp4 --output artifacts/repair --step 2
python video_search.py search artifacts/repair "a hand holding a battery" --refine-budget 12
```

The default index cap is 600 frames. Increase sampling interval for long videos, or
explicitly raise the cap. The full video is hashed for index consistency. The index
contains an absolute source path; keep the original video in place. Index folders
must be new. Frames are decoded locally and are not sent to external services.

## Controller and limitations

Coarse frame embeddings are cached. Each question is encoded into the same image/text
space. Ranked coarse candidates receive non-overlapping intervals, then share a fixed
additional-frame inspection budget. The controller scores denser samples and reorders
candidates. The trace records each inspection action and its frame count.

This controller is deterministic, not an LLM agent. CLIP similarity cannot verify
whether an action occurred, whether a connector was disconnected first, or whether
the returned interval contains the entire event. Interval boundaries are sampling
windows, not learned event boundaries. CLIP uses center crops and can miss edge content.
Repeated scenes can produce multiple similar candidates. Comparisons against negatives
remain future work; optional VLM verification is described below.

Optional `--min-score` filters candidates, but there is no default calibrated abstention
threshold. Similarities are not probabilities. Tune any threshold only on held-out
validation queries, then measure false positives on unanswerable questions.

The UI shows coarse candidate thumbnails and timestamp intervals, not playable clips.
Open the original video at those timestamps for review. No generated factual answer is
presented. Models run on CPU in this workspace; CUDA is selected automatically if a
compatible PyTorch build is installed. Independent installation avoids coupling projects.

## Optional bounded VLM verification

Install and run a vision-capable model in local Ollama, then specify its installed name:

```powershell
python verifier.py artifacts/demo "a blue circle" --model YOUR_INSTALLED_VISION_MODEL --budget 12
```

Or enter the installed model name in the interface. The verifier receives time-ordered
JPEG frames and returns supported, unsupported, unknown, or inspect_more plus frame
references. An inspect_more decision triggers another bounded inspection of the same
interval, with at most two turns per interval. Exhausted requests become unknown.
References outside supplied frames are rejected. Only the local localhost:11434 API
is contacted. This is a restricted inspection loop, not an autonomous general-purpose
agent. Frame text is treated as untrusted data and no execution tools are exposed.

In verifier mode the coarse embedding cache is reused and additional CLIP refinement
is disabled; the specified budget caps newly decoded verification frames across all
candidates. The JSON result records assessments, frame times, budget use, and abstention.
Model wording and evidence references can still be wrong. The live Ollama vision path
has not been validated with an installed vision model here; bounded logic and response
validation were exercised with simulated verifier responses.

API references: https://docs.ollama.com/capabilities/vision and
https://docs.ollama.com/capabilities/structured-outputs .

Goal: find and verify visual evidence in repair/training footage, returning timestamped
clips instead of unsupported answers. Research: https://github.com/yeliudev/VideoMind

## Contract

Each evidence record has `video_id`, `start`, `end`, `description`, and `frame_paths`.
Descriptions in this older lexical module are manually authored, not model predictions.
Use `evidence.py` now to validate and search a manually authored evidence manifest.
This lexical module is separate from the implemented `video_search.py` visual pipeline.

```powershell
python evidence.py sample_evidence.json "battery connector"
```

## Build order

1. Record five short repair clips; manually annotate 30 questions and evidence intervals.
2. Done: frame extraction and cached CLIP visual embeddings.
3. Done: deterministic coarse-to-fine candidate retrieval and denser similarity inspection.
4. Give a small VLM tools to inspect bounded frame sets, refine intervals, and abstain.
5. Compare fixed sampling against agent-selected inspection under identical frame budgets.

Evaluate temporal IoU, recall at IoU thresholds, answer/evidence correctness, abstention,
frames inspected, latency, and GPU memory. Split by video, not extracted frame. Restrict
tools to read-only local media. On the 8 GB GPU, load models sequentially and cache features.

## Docker image

This project has its own CPU image: `ghcr.io/khawajasaad22/video-evidence-agent:latest`.
Images are stored in GitHub Container Registry; the repository contains the Dockerfile,
Compose configuration, and build/publish workflow. The workflow tests the image and
checks the running app before publishing `latest` and a full commit-SHA tag. A failed
build does not publish. Use the Actions tab to confirm the first successful publication.

With Docker Engine/Desktop and Compose installed, from this repository:

```sh
docker compose up --build -d
```

Open http://localhost:8503. To use the registry image after it is published:

```sh
docker compose pull app
docker compose up -d --no-build app
```

If the registry package is private, authenticate with `docker login ghcr.io` using an
account/token allowed to read the package. Repository visibility alone does not ensure
anonymous image access. No custom registry secret is needed by the publishing workflow:
it uses the repository's built-in GitHub token with package-write permission.

The container runs as UID 10001. Named volumes preserve inputs, generated artifacts,
and downloaded model weights. Weights and private datasets are not baked into the image.
First model use requires internet access. These are Linux/amd64 CPU images; GPU/CUDA
acceleration is not included. Initial builds can take several minutes.

Use `docker compose exec app python ...` for the documented CLI commands. Inputs are
container paths, not Windows host paths. Copy real files with `docker compose cp`:

```sh
docker compose cp ./my-input app:/app/data/my-input
docker compose logs -f app
docker compose down
```

`down` keeps named volumes; `down -v` deletes them, including my generated work.
The app port is published on localhost only. Container health means Streamlit is running;
you still need to prepare data and fit/index models before using the app workflow.

```sh
docker compose exec app python make_demo.py
docker compose exec app python video_search.py index data/demo.avi --output artifacts/demo --step 2
docker compose exec app python video_search.py search artifacts/demo "a blue circle"
```

### Optional Ollama service

```sh
docker compose --profile generation up -d
docker compose exec ollama ollama pull YOUR_MODEL_NAME
```

Choose a text model for RAG or a vision-capable model for video verification, then enter
its installed name in the UI. Ollama is accessible only inside the Compose network;
no host port is published. `OLLAMA_BASE_URL` selects the service endpoint. This optional
service also runs on CPU. Its model quality and memory needs depend on my selection.
