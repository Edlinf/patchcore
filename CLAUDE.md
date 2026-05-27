# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

- Install dependencies from the README:
  ```bash
  pipenv install -r requirements.txt
  ```
  If not using Pipenv, use the same `requirements.txt` with the active Python environment.

- Run the Streamlit demo:
  ```bash
  streamlit run streamlit_app.py
  ```

- Train/evaluate from CLI using explicit arguments:
  ```bash
  python indad/run.py patchcore --dataset rubberring --dataset_dir ./datasets --result_dir ./results --image_size 224
  ```
  Valid methods are `spade`, `padim`, and `patchcore`. `patchcore` defaults to feature indices `2,3`; `spade` uses `1,2,3,-1`; `padim` uses `1,2,3`.

- Train/evaluate from YAML config:
  ```bash
  python indad/run-yml.py --cfg_path config/dataset.yml --output_dir ./outputs
  ```
  The YAML path is expected to contain keys such as `method`, `backbone`, `job_no`, `f_coreset`, `dataset_dir`, `image_size`, `dataset`, and `result_dir`.

- Evaluate a saved model archive:
  ```bash
  python indad/predict.py path/to/model.ts2 --dataset rubberring --dataset_dir ./datasets --results_dir ./results-predict
  ```

- Syntax-check the Python source:
  ```bash
  python -m compileall indad streamlit_app.py
  ```

- Tests: this repository currently has no test files or configured test runner; the README lists unit tests as TODO. If pytest tests are added, run a single test with:
  ```bash
  python -m pytest path/to/test_file.py::test_name
  ```

## Architecture overview

This project implements KNN-based industrial anomaly detection methods: SPADE, PaDiM, and PatchCore. The core model implementations live in `indad/models.py`; the command-line training entry points construct those models, load MVTec-style datasets, run `fit()` on healthy training images, then call `evaluate()` on test images to produce image-level and pixel-level ROC-AUC scores.

Data loading is centralized in `indad/data.py`. `MVTecDataset` wraps separate train/test `ImageFolder` datasets under `dataset_dir/<dataset>/train`, `test`, and `ground_truth`. The custom dataset layout follows the README: healthy training images under `train/good`, test images grouped by defect type and `good`, and optional masks under `ground_truth/<defect_type>` with `_mask.png` suffixes. Resizing can use torchvision transforms or OpenCV; train and test resizing should stay consistent.

`indad/models.py` defines a shared `KNNExtractor` base that builds a frozen timm feature extractor with `features_only=True`. SPADE stores global embedding and feature-map galleries, PaDiM models feature-map distributions with covariance estimates, and PatchCore builds a patch library, optionally applies greedy coreset selection, saves it as a TorchScript archive named `patch_lib.ts`, and writes anomaly-map images during prediction.

The main operational path is PatchCore. `indad/run.py` is the direct CLI, while `indad/run-yml.py` is the config-driven/job-oriented entry point. Both set `torch.hub` to the local `hub/` directory, write result YAML/TXT files via `indad/utils.py`, and export model artifacts whose filenames encode method, job number, resize method, backbone, feature indices, feature-map size, image shape, precision, and hash. `indad/predict.py` parses that filename format to reconstruct and load a PatchCore model for evaluation.

Progress and external job state are tracked through `indad/job_ini.py`, which reads and writes `job.ini` under the dataset directory. Code paths that call `set_exec_progress()` expect `total_stages` to exist in the `[job]` section, so direct invocations against a fresh dataset directory may need job initialization first.

Generated outputs are written under `results/`, `outputs/`, or caller-provided result directories. Training also writes `model.txt` into the dataset directory with the exported model artifact name.
