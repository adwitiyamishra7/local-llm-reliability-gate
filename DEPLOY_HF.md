# Hugging Face Spaces deployment

This project can run on CPU Basic hardware. At the time of writing, Hugging Face lists CPU Basic as 2 vCPU and 16 GB RAM at no hourly hardware charge; account/plan requirements for creating compute Spaces may still apply. See the [Spaces hardware overview](https://huggingface.co/docs/hub/spaces-overview).

## Prepare a Space

1. Create a Space and select CPU Basic. The requested Streamlit metadata block is:

   ```yaml
   ---
   title: Local LLM Reliability Gate
   sdk: streamlit
   app_file: app.py
   python_version: "3.11"
   ---
   ```

   Hugging Face currently marks its built-in Streamlit SDK as deprecated for new Spaces and recommends the Docker SDK with its Streamlit template. This block is included for an existing or SDK-enabled Streamlit Space as requested; follow the [current Streamlit Spaces instructions](https://huggingface.co/docs/hub/spaces-sdks-streamlit) if creating a new Space.

2. Use `requirements-hf.txt` as the Space's root `requirements.txt` (copy its contents or name it `requirements.txt` in the Space repository). It is derived from the project dependencies and contains no CUDA-specific index or wheel. Keep Python at 3.11.

3. Include only these runtime files:

   ```text
   README.md                         # Space YAML metadata
   app.py
   .streamlit/config.toml
   requirements.txt                  # copied from requirements-hf.txt for Space builds
   src/__init__.py
   src/gate/__init__.py
   src/gate/agreement.py
   src/gate/config.py
   src/gate/factual_verifier.py
   src/gate/generation.py
   src/gate/model.py
   src/gate/response_quality.py
   src/gate/reliability_gate.py
   artifacts/classifier/reliability_gate_final_v5.pkl
   ```

   The runtime loader selects and deserializes `reliability_gate_final_v5.pkl`; its calibrated threshold and generation metadata are embedded in the pickle. `selected_threshold_v5.json` and `final_gate_metadata_v5.txt` are used by finalization/validation workflows, not loaded by live inference. V4/legacy pickles, benchmark results, and training scripts are not runtime dependencies. The `.hfignore` file in this repository excludes those non-runtime files while retaining the V5 pickle.

4. If a file exceeds 10 MB, track it with Git LFS before pushing, for example:

   ```bash
   git lfs install
   git lfs track "artifacts/path/to/large-file.pkl"
   git add .gitattributes artifacts/path/to/large-file.pkl
   ```

   The current V5 inference pickle is below 10 MB, so it does not require LFS.

5. Push the prepared Space repository. Qwen3-0.6B (`Qwen/Qwen3-0.6B`) downloads from the Hugging Face Hub on the first run, so initial startup needs network access and may take longer. Later starts can reuse the platform's model cache when available. The app falls back to CPU/float32 when CUDA is unavailable.

The Space UI is served by Streamlit on its standard port. For CPU hardware details and current SDK guidance, see the [Spaces hardware overview](https://huggingface.co/docs/hub/spaces-overview) and [Streamlit Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-streamlit).
