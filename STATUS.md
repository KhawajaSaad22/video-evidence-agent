# Verified first implementation

- Real CLIP vision/text model loaded successfully.
- Generated and decoded a 12-second 320 x 240 MJPG video at 10 FPS.
- Cached six frame embeddings at a two-second interval.
- Queried “a blue circle”; the highest-ranked interval was 5–7 seconds, inside the
  blue-circle section (4–8 seconds).
- The controller performed exactly 12 additional frame inspections and recorded them.
- Five unit tests passed; Streamlit interface startup passed.
- Raw local output is in demo-result.json; frames and index are in artifacts/demo/.

An optional local Ollama verifier now supports bounded inspect-more decisions and
validated frame references. Eight unit tests cover retrieval and simulated verification.
No installed vision model was used to validate the live Ollama path. Scope: retrieval
plus restricted verification, not a full planning agent. It cannot reliably establish
causal/temporal order. The toy video tests
the plumbing, not usefulness on natural footage. GPU acceleration has not been tested;
current PyTorch is CPU-only. OpenCV was added to the existing shared environment for
verification; independent installation is documented in README.md.

Next: gather short real repair videos with labeled queries, benchmark fixed versus
budgeted sampling, add negative-query calibration, then add a small VLM verifier with
bounded read-only inspection tools. A held-out evaluation must include unanswerable
questions and distinct source videos. There is no paid API dependency or deployed service.
