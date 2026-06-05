# Big Image Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--big-image` inference to `indad/patchcore_predict_simple.py`, splitting a stitched image by fixed geometry, predicting each tile, and stitching tile score maps back into a full-size big-image heatmap.

**Architecture:** Keep all big-image inference inside the standalone predictor script. Add pure geometry split helpers mirroring `D:/python_project/028-grid-dataset-preprocess/prepare.py`, then add a `PatchCorePredictor.predict_big_image()` method that loops over in-memory tiles, resizes each tile score map back to its original tile box, fills a full-size score canvas, and leaves margins/gaps as zero.

**Tech Stack:** Python, PIL, OpenCV, NumPy, PyTorch, Click, pytest.

---

## File Structure

- Modify: `indad/patchcore_predict_simple.py`
  - Add `BigTile` dataclass and `split_big_image_by_geometry` helper.
  - Add `PatchCorePredictor.predict_big_image()`.
  - Add `--big-image`, `--rows`, `--cols`, margin/gap CLI args.
  - Add `tile_scores.csv` output for big-image mode.

- Modify: `tests/test_patchcore_predict_simple.py`
  - Add helper tests for geometry split and full-score stitching behavior.

---

### Task 1: Add geometry split helper tests and implementation

**Files:**
- Modify: `tests/test_patchcore_predict_simple.py`
- Modify: `indad/patchcore_predict_simple.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_patchcore_predict_simple.py`:

```python
import numpy as np
from PIL import Image

from patchcore_predict_simple import split_big_image_by_geometry


def test_split_big_image_by_geometry_matches_prepare_py_layout():
    arr = np.zeros((8, 10, 3), dtype=np.uint8)
    arr[2:4, 1:4] = [10, 0, 0]
    arr[2:4, 5:8] = [20, 0, 0]
    arr[5:7, 1:4] = [30, 0, 0]
    arr[5:7, 5:8] = [40, 0, 0]
    image = Image.fromarray(arr, "RGB")

    tiles = split_big_image_by_geometry(
        image,
        rows=2,
        cols=2,
        top_margin=2,
        bottom_margin=1,
        left_margin=1,
        right_margin=2,
        hori_gap=1,
        vert_gap=1,
    )

    assert len(tiles) == 4
    assert tiles[0].box == (1, 2, 4, 4)
    assert tiles[1].box == (5, 2, 8, 4)
    assert tiles[2].box == (1, 5, 4, 7)
    assert tiles[3].box == (5, 5, 8, 7)
    assert np.asarray(tiles[3].image)[0, 0].tolist() == [40, 0, 0]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py::test_split_big_image_by_geometry_matches_prepare_py_layout -v
```

Expected: FAIL because `split_big_image_by_geometry` is missing.

- [ ] **Step 3: Implement helper**

In `indad/patchcore_predict_simple.py`, import dataclass:

```python
from dataclasses import dataclass
```

Add after `IMAGE_SUFFIXES`:

```python

@dataclass
class BigTile:
    index: int
    row: int
    col: int
    image: Image.Image
    box: tuple
```

Add helper after `normalize_vis_map(...)`:

```python

def split_big_image_by_geometry(image, rows, cols, top_margin, bottom_margin, left_margin, right_margin, hori_gap, vert_gap):
    width, height = image.size
    available_width = width - left_margin - right_margin - (cols - 1) * hori_gap
    available_height = height - top_margin - bottom_margin - (rows - 1) * vert_gap
    if available_width <= 0 or available_height <= 0:
        raise ValueError("invalid big-image geometry parameters")
    tile_w = available_width // cols
    tile_h = available_height // rows

    tiles = []
    index = 0
    for row in range(rows):
        for col in range(cols):
            x1 = left_margin + col * (tile_w + hori_gap)
            y1 = top_margin + row * (tile_h + vert_gap)
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            box = (x1, y1, x2, y2)
            tiles.append(BigTile(index=index, row=row, col=col, image=image.crop(box), box=box))
            index += 1
    return tiles
```

- [ ] **Step 4: Run helper test**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py::test_split_big_image_by_geometry_matches_prepare_py_layout -v
```

Expected: PASS.

---

### Task 2: Add full-score stitching and big-image prediction

**Files:**
- Modify: `indad/patchcore_predict_simple.py`
- Modify: `tests/test_patchcore_predict_simple.py`

- [ ] **Step 1: Add full-score stitching test**

Append to `tests/test_patchcore_predict_simple.py`:

```python
from patchcore_predict_simple import stitch_tile_score


def test_stitch_tile_score_keeps_margins_and_gaps_zero():
    full = np.zeros((8, 10), dtype=np.float32)
    tile_score = torch.ones(2, 3)

    stitch_tile_score(full, tile_score, box=(1, 2, 4, 4))

    assert full[0].sum() == 0
    assert full[:, 0].sum() == 0
    assert np.allclose(full[2:4, 1:4], 1.0)
    assert full[4].sum() == 0
```

- [ ] **Step 2: Implement `stitch_tile_score` and `predict_big_image`**

Add module helper after `save_heatmap_outputs(...)`:

```python

def stitch_tile_score(full_score, tile_score_map, box):
    x1, y1, x2, y2 = box
    tile_score = tile_score_map.detach().cpu().float().numpy()
    tile_score = cv2.resize(tile_score, (x2 - x1, y2 - y1))
    full_score[y1:y2, x1:x2] = tile_score
```

Add method inside `PatchCorePredictor` after `predict_image(...)`:

```python
    def predict_big_image(self, image_path, rows, cols, top_margin, bottom_margin, left_margin, right_margin, hori_gap, vert_gap):
        start = time.time()
        image = Image.open(image_path).convert("RGB")
        full_score = np.zeros((image.height, image.width), dtype=np.float32)
        tiles = split_big_image_by_geometry(
            image,
            rows=rows,
            cols=cols,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            left_margin=left_margin,
            right_margin=right_margin,
            hori_gap=hori_gap,
            vert_gap=vert_gap,
        )
        tile_rows = []
        for tile in tiles:
            sample = self.transform(tile.image).unsqueeze(0)
            score, score_map = self.predict_tensor(sample)
            score_value = float(score.item())
            stitch_tile_score(full_score, score_map, tile.box)
            tile_rows.append({
                "tile_index": tile.index,
                "row": tile.row,
                "col": tile.col,
                "score": score_value,
                "x1": tile.box[0],
                "y1": tile.box[1],
                "x2": tile.box[2],
                "y2": tile.box[3],
            })
        elapsed_ms = (time.time() - start) * 1000
        score = float(full_score.max())
        return image, score, torch.from_numpy(full_score), elapsed_ms, tile_rows
```

- [ ] **Step 3: Run tests**

Run:

```bash
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py -v
```

Expected: PASS.

---

### Task 3: Wire CLI and outputs

**Files:**
- Modify: `indad/patchcore_predict_simple.py`

- [ ] **Step 1: Add tile CSV writer**

Add module helper after `write_scores_csv(...)`:

```python

def write_tile_scores_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "tile_index", "row", "col", "score", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 2: Add CLI options**

Add click options before `cli_interface`:

```python
@click.option("--big-image", is_flag=True)
@click.option("--rows", default=5, type=int)
@click.option("--cols", default=3, type=int)
@click.option("--top-margin", default=626, type=int)
@click.option("--bottom-margin", default=626, type=int)
@click.option("--left-margin", default=3, type=int)
@click.option("--right-margin", default=3, type=int)
@click.option("--hori-gap", default=1, type=int)
@click.option("--vert-gap", default=1, type=int)
```

Update `cli_interface(...)` signature to include those parameters.

- [ ] **Step 3: Branch prediction logic in CLI**

Inside the image loop, replace the current `predict_image` block with:

```python
        if big_image:
            src_image, score, score_map, elapsed_ms, tile_rows = predictor.predict_big_image(
                image_path,
                rows=rows,
                cols=cols,
                top_margin=top_margin,
                bottom_margin=bottom_margin,
                left_margin=left_margin,
                right_margin=right_margin,
                hori_gap=hori_gap,
                vert_gap=vert_gap,
            )
            for tile_row in tile_rows:
                tile_row["path"] = str(image_path)
                all_tile_rows.append(tile_row)
        else:
            src_image, score, score_map, elapsed_ms = predictor.predict_image(image_path)
```

Before the loop, initialize:

```python
    all_tile_rows = []
```

After `write_scores_csv(...)`, add:

```python
    if all_tile_rows:
        write_tile_scores_csv(output_dir / "tile_scores.csv", all_tile_rows)
```

- [ ] **Step 4: Compile/help/tests**

Run:

```bash
conda run -n transfusion python -m py_compile indad/patchcore_predict_simple.py
conda run -n transfusion python indad/patchcore_predict_simple.py --help
conda run -n transfusion python -m pytest tests/test_patchcore_predict_simple.py tests/test_patchcore_normalization.py tests/test_patchcore_archive_loading.py -v
```

Expected: `--big-image`, margin, and gap options appear; tests PASS.

---

### Task 4: Smoke test big-image mode

**Files:**
- No source changes expected unless smoke reveals bug.

- [ ] **Step 1: Run one big image**

Use known model and original big image:

```bash
conda run -n transfusion python indad/patchcore_predict_simple.py \
  --model outputs/patchcore_2604291557639_cv2_resnet18_23_32x96_3x256x768_fp32_76a28de6.ts2 \
  --image D:/python_project/dataset/LabelMe_171023/test/bad/10.jpg \
  --output ./results-predict-big-smoke \
  --big-image \
  --match-mode exact_position \
  --neighbor-radius 0
```

Expected:

```text
results-predict-big-smoke/scores.csv
results-predict-big-smoke/tile_scores.csv
results-predict-big-smoke/heatmaps/*.jpg
```

- [ ] **Step 2: Verify output row counts**

Run:

```bash
python - <<'PY'
from pathlib import Path
import csv
root=Path('results-predict-big-smoke')
print((root/'scores.csv').exists())
print((root/'tile_scores.csv').exists())
print(len(list(csv.DictReader((root/'scores.csv').open(encoding='utf-8')))))
print(len(list(csv.DictReader((root/'tile_scores.csv').open(encoding='utf-8')))))
print(len(list((root/'heatmaps').glob('*.jpg'))))
PY
```

Expected:

```text
True
True
1
15
1
```

- [ ] **Step 3: Commit**

Run:

```bash
git add indad/patchcore_predict_simple.py tests/test_patchcore_predict_simple.py
git commit -m "feat: add big-image PatchCore prediction mode"
```

---

## Self-Review Checklist

- Big-image split uses fixed geometry, not black threshold.
- Defaults match `prepare.py`: rows=5, cols=3, top/bottom=626, left/right=3, gaps=1.
- Tile score maps fill only tile boxes; margins and gaps stay zero.
- Ordinary image/directory prediction remains unchanged when `--big-image` is absent.
- `tile_scores.csv` records tile positions and scores.
