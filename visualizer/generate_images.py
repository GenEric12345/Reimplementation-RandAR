"""
Generate images with RandAR and save confidence-ordered visualization frames.

Mirrors the generation pipeline in tools/search_cfg_weights_Autosyll.py but
runs single-process (no DDP) and takes a fixed --cfg-scale instead of searching.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from RandAR.util import instantiate_from_config, load_safetensors


# ---------------------------------------------------------------------------
# HTML template written into output_dir/index.html so the server can serve it
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RandAR Confidence Visualizer</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0f0f1a;
      color: #e0e0e0;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: #16213e;
      padding: 12px 24px;
      border-bottom: 1px solid #1e3a6e;
      flex-shrink: 0;
    }
    header h1 { font-size: 1.1rem; color: #e94560; letter-spacing: 0.02em; }
    header .meta { font-size: 0.78rem; color: #778; margin-top: 2px; }
    .layout { display: flex; flex: 1; overflow: hidden; }

    /* Sidebar */
    .sidebar {
      width: 200px;
      min-width: 200px;
      background: #12182e;
      border-right: 1px solid #1e3a6e;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .sidebar-header {
      padding: 10px 12px;
      font-size: 0.75rem;
      color: #778;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border-bottom: 1px solid #1e3a6e;
      flex-shrink: 0;
    }
    .thumb-list {
      overflow-y: auto;
      flex: 1;
      padding: 8px;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 6px;
      align-content: start;
    }
    .thumb {
      cursor: pointer;
      border: 2px solid transparent;
      border-radius: 5px;
      overflow: hidden;
      background: #1a2040;
      transition: border-color 0.15s;
    }
    .thumb:hover { border-color: #e94560aa; }
    .thumb.active { border-color: #e94560; }
    .thumb img { width: 100%; display: block; image-rendering: pixelated; }
    .thumb .tlabel {
      font-size: 9px;
      color: #999;
      text-align: center;
      padding: 2px 0 3px;
      background: #1a2040;
    }

    /* Main viewer */
    .main {
      flex: 1;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 24px;
      overflow: auto;
    }
    .viewer { display: flex; flex-direction: column; align-items: center; gap: 14px; width: 100%; max-width: 900px; }
    .class-badge {
      font-size: 0.95rem;
      font-weight: 600;
      color: #e94560;
      background: #1e1030;
      border: 1px solid #e94560;
      border-radius: 20px;
      padding: 4px 16px;
      letter-spacing: 0.03em;
    }
    .image-row {
      display: flex;
      gap: 16px;
      align-items: flex-start;
      width: 100%;
    }
    #main-image {
      width: 360px;
      height: 360px;
      flex-shrink: 0;
      border-radius: 8px;
      border: 2px solid #1e3a6e;
      image-rendering: pixelated;
      background: #111;
      display: block;
    }
    .notes-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }
    .notes-label {
      font-size: 0.72rem;
      color: #778;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .notes-text {
      flex: 1;
      min-height: 330px;
      background: #12182e;
      border: 1px solid #1e3a6e;
      border-radius: 6px;
      padding: 10px 12px;
      color: #e0e0e0;
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 0.85rem;
      line-height: 1.5;
      white-space: pre-wrap;
      margin: 0;
    }
    .controls { width: 100%; }
    .slider-labels { display: flex; justify-content: space-between; font-size: 0.7rem; color: #556; margin-bottom: 6px; }
    input[type=range] {
      width: 100%;
      -webkit-appearance: none;
      height: 6px;
      border-radius: 3px;
      background: #1e3a6e;
      outline: none;
      cursor: pointer;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 18px; height: 18px;
      border-radius: 50%;
      background: #e94560;
      cursor: pointer;
      box-shadow: 0 0 4px #e94560aa;
    }
    .progress-bar {
      height: 3px;
      background: #1e3a6e;
      border-radius: 2px;
      overflow: hidden;
      margin-top: 6px;
    }
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #e94560, #f97316);
      transition: width 0.08s;
      border-radius: 2px;
    }
    .stats-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
      color: #778;
      margin-top: 6px;
    }
    .token-jump-wrap { display: flex; align-items: center; gap: 4px; }
    #token-jump {
      width: 54px;
      background: #1a2040;
      border: 1px solid #334;
      border-radius: 4px;
      color: #ccc;
      font-size: 0.8rem;
      padding: 2px 6px;
      text-align: right;
      outline: none;
    }
    #token-jump:focus { border-color: #e94560; color: #fff; }
    .btn-row { display: flex; gap: 8px; margin-top: 10px; }
    .btn {
      padding: 6px 14px;
      border: 1px solid #e94560;
      background: transparent;
      color: #e94560;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.82rem;
      transition: background 0.15s, color 0.15s;
    }
    .btn:hover { background: #e94560; color: #fff; }
    .btn.active { background: #e94560; color: #fff; }
    #loading { color: #556; font-size: 1rem; }
  </style>
</head>
<body>
  <header>
    <h1 id="exp-title">RandAR Confidence Visualizer</h1>
    <div class="meta" id="exp-meta"></div>
  </header>
  <div class="layout">
    <div class="sidebar">
      <div class="sidebar-header">Images</div>
      <div class="thumb-list" id="thumb-list"></div>
    </div>
    <div class="main">
      <div id="loading">Loading metadata...</div>
      <div class="viewer" id="viewer" style="display:none">
        <div class="class-badge" id="class-badge">Class —</div>
        <div class="image-row">
          <img id="main-image" src="" alt="Generated image">
          <div class="notes-panel">
            <div class="notes-label">Notes / Description</div>
            <pre class="notes-text">
            - The image shown is generated in confidence order (not necessarily in 256 steps)
            - It is generated with strictly causal attention on previously generated tokens (starting with just the class ID) 
            </pre>
          </div>
        </div>
        <div class="controls">
          <div class="slider-labels">
            <span>← Empty (0 tokens)</span>
            <span>Full (256 tokens) →</span>
          </div>
          <input type="range" id="frame-slider" min="0" max="256" value="256">
          <div class="progress-bar">
            <div class="progress-fill" id="progress-fill" style="width:100%"></div>
          </div>
          <div class="stats-row">
            <span id="token-count">256 / 256 tokens revealed</span>
            <div class="token-jump-wrap">
              <span>Jump to token:</span>
              <input type="number" id="token-jump" min="0" max="256" value="256">
              <span id="token-jump-max">/ 256</span>
            </div>
          </div>
          <div class="btn-row">
            <button class="btn" id="btn-play">&#9654; Play</button>
            <button class="btn" id="btn-prev">&#8249; Prev</button>
            <button class="btn" id="btn-next">Next &#8250;</button>
            <button class="btn" id="btn-end" title="Jump to full image">Full</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let meta = null;
    let selectedId = 0;
    let playing = false;
    let playTimer = null;

    async function init() {
      try {
        const r = await fetch('metadata.json');
        meta = await r.json();
      } catch (e) {
        document.getElementById('loading').textContent = 'Failed to load metadata.json';
        return;
      }

      document.getElementById('exp-title').textContent =
        'RandAR Confidence Visualizer — ' + (meta.exp_name || '');
      document.getElementById('exp-meta').textContent =
        'cfg=' + meta.cfg_scale +
        '  |  steps=' + meta.num_inference_steps +
        '  |  ordering_runs=' + meta.ordering_runs +
        '  |  block_size=' + meta.block_size;

      const maxFrame = meta.num_frames - 1;
      const slider = document.getElementById('frame-slider');
      slider.max = maxFrame;
      slider.value = maxFrame;
      const jump = document.getElementById('token-jump');
      jump.max = maxFrame;
      jump.value = maxFrame;
      document.getElementById('token-jump-max').textContent = '/ ' + maxFrame;

      buildThumbs();
      document.getElementById('loading').style.display = 'none';
      document.getElementById('viewer').style.display = 'flex';
      selectImage(0);
    }

    function pad(n, w) { return String(n).padStart(w, '0'); }

    function buildThumbs() {
      const list = document.getElementById('thumb-list');
      list.innerHTML = '';
      meta.images.forEach(img => {
        const div = document.createElement('div');
        div.className = 'thumb';
        div.dataset.id = img.id;
        div.innerHTML =
          '<img src="images/' + pad(img.id, 4) + '/final.png" loading="lazy">' +
          '<div class="tlabel">cls ' + img.class_label + '</div>';
        div.addEventListener('click', () => selectImage(img.id));
        list.appendChild(div);
      });
    }

    function selectImage(id) {
      selectedId = id;
      document.querySelectorAll('.thumb').forEach(t => t.classList.remove('active'));
      const el = document.querySelector('.thumb[data-id="' + id + '"]');
      if (el) { el.classList.add('active'); el.scrollIntoView({ block: 'nearest' }); }

      const img = meta.images[id];
      document.getElementById('class-badge').textContent = 'Class ' + img.class_label;

      const slider = document.getElementById('frame-slider');
      updateFrame(parseInt(slider.value));
    }

    function updateFrame(f) {
      const maxFrame = meta.num_frames - 1;
      f = Math.max(0, Math.min(f, maxFrame));
      const slider = document.getElementById('frame-slider');
      slider.value = f;
      document.getElementById('token-jump').value = f;

      const img = meta.images[selectedId];
      const src = 'images/' + pad(img.id, 4) + '/frames/' + pad(f, 4) + '.png';
      document.getElementById('main-image').src = src;

      const numVis = (meta.num_frames > 1)
        ? Math.round(f * meta.block_size / (meta.num_frames - 1))
        : meta.block_size;
      const pct = (maxFrame > 0) ? (f / maxFrame * 100).toFixed(1) : 100;
      document.getElementById('token-count').textContent =
        numVis + ' / ' + meta.block_size + ' tokens revealed';
      document.getElementById('progress-fill').style.width = pct + '%';
    }

    document.getElementById('frame-slider').addEventListener('input', e => {
      updateFrame(parseInt(e.target.value));
    });

    document.getElementById('token-jump').addEventListener('change', e => {
      const v = parseInt(e.target.value);
      if (!isNaN(v)) updateFrame(v);
    });
    document.getElementById('token-jump').addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); updateFrame(parseInt(e.target.value)); }
    });

    document.getElementById('btn-prev').addEventListener('click', () => {
      const s = document.getElementById('frame-slider');
      const v = parseInt(s.value);
      if (v > 0) updateFrame(v - 1);
    });

    document.getElementById('btn-next').addEventListener('click', () => {
      const s = document.getElementById('frame-slider');
      const v = parseInt(s.value);
      if (v < meta.num_frames - 1) updateFrame(v + 1);
    });

    document.getElementById('btn-end').addEventListener('click', () => {
      updateFrame(meta.num_frames - 1);
    });

    document.getElementById('btn-play').addEventListener('click', () => {
      playing = !playing;
      const btn = document.getElementById('btn-play');
      btn.textContent = playing ? '⏸ Pause' : '▶ Play';
      btn.classList.toggle('active', playing);

      if (playing) {
        const slider = document.getElementById('frame-slider');
        if (parseInt(slider.value) >= meta.num_frames - 1) updateFrame(0);
        playTimer = setInterval(() => {
          const s = document.getElementById('frame-slider');
          const next = parseInt(s.value) + 1;
          if (next >= meta.num_frames) {
            playing = false;
            document.getElementById('btn-play').textContent = '▶ Play';
            document.getElementById('btn-play').classList.remove('active');
            clearInterval(playTimer);
          } else {
            updateFrame(next);
          }
        }, 120);
      } else {
        clearInterval(playTimer);
      }
    });

    // Keyboard shortcuts — disabled when focus is in textarea or jump input
    document.addEventListener('keydown', e => {
      if (!meta) return;
      if (e.target.tagName === 'TEXTAREA' || e.target.id === 'token-jump') return;
      const slider = document.getElementById('frame-slider');
      const v = parseInt(slider.value);
      if (e.key === 'ArrowRight') { if (v < meta.num_frames - 1) updateFrame(v + 1); }
      else if (e.key === 'ArrowLeft') { if (v > 0) updateFrame(v - 1); }
      else if (e.key === 'ArrowUp') { if (selectedId > 0) selectImage(selectedId - 1); }
      else if (e.key === 'ArrowDown') { if (selectedId < meta.images.length - 1) selectImage(selectedId + 1); }
      else if (e.key === ' ') { e.preventDefault(); document.getElementById('btn-play').click(); }
    });

    init();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Core generation logic (mirrors search_cfg_weights_Autosyll.py)
# ---------------------------------------------------------------------------

def generate_ordering(c_indices, cfg_scales, runs, gpt_model, args):
    """Compute confidence-based token ordering by averaging entropy across multiple runs.

    Identical to generate_ordering() in tools/search_cfg_weights_Autosyll.py.
    Lower entropy → higher confidence; tokens sorted ascending so confident
    positions come first in the returned order.
    """
    entropys = []
    for _ in range(runs):
        indices, entropy = gpt_model.generate_with_entropy(
            cond=c_indices,
            token_order=None,
            cfg_scales=cfg_scales,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        entropys.append(entropy)
    avg_entropy = torch.stack(entropys).mean(dim=0)
    token_order = torch.argsort(avg_entropy, dim=-1)  # ascending: confident first
    del entropys
    torch.cuda.empty_cache()
    return token_order

def generate_masked_frames(result_indices, token_order, tokenizer, num_frames, image_size_eval):
    bs, block_size = result_indices.shape
    grid_size = int(block_size ** 0.5)

    full_images = tokenizer.decode_codes_to_img(result_indices, image_size_eval)
    frames = []

    H = W = image_size_eval
    patch_h = H // grid_size
    patch_w = W // grid_size

    for frame_idx in range(num_frames):
        num_visible = round(frame_idx * block_size / (num_frames - 1))

        mask_tokens = torch.zeros(bs, block_size, device=result_indices.device, dtype=torch.bool)
        if num_visible > 0:
            positions = token_order[:, :num_visible]
            mask_tokens.scatter_(1, positions, True)

        mask_grid = mask_tokens.view(bs, grid_size, grid_size)
        mask_pixels = mask_grid.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)
        mask_pixels = mask_pixels[..., None].cpu().numpy()

        frame = full_images.copy()
        frame = frame * mask_pixels  # unrevealed regions black
        frames.append(frame.astype("uint8"))

    return frames

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(args.global_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.global_seed)

    # Load config
    config = OmegaConf.load(args.config)

    # Load tokenizer (VQ model) — mirrors Autosyll
    print("Loading tokenizer...")
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    tokenizer.load_state_dict(state_dict)

    # Load GPT model — mirrors Autosyll
    print("Loading GPT model...")
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    gpt_model = instantiate_from_config(config.ar_model).to(device=device, dtype=precision)
    model_weight = load_safetensors(args.gpt_ckpt)
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()

    block_size = gpt_model.block_size
    num_frames = block_size + 1  # one frame per token (0 through block_size tokens revealed)

    # Determine class labels
    if args.class_labels:
        class_list = [int(x.strip()) for x in args.class_labels.split(",")]
    else:
        rng = torch.Generator()
        rng.manual_seed(args.global_seed)
        class_list = torch.randint(0, args.num_classes, (args.num_images,), generator=rng).tolist()

    num_images = len(class_list)
    batch_size = min(args.batch_size, num_images)

    # Build ckpt name string (mirrors Autosyll)
    ckpt_string_name = (
        os.path.basename(args.gpt_ckpt)
        .replace(".pth", "")
        .replace(".pt", "")
        .replace(".safetensors", "")
    )

    # Output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg_scales = (1.0, args.cfg_scale)

    metadata = {
        "exp_name": args.exp_name,
        "ckpt": ckpt_string_name,
        "cfg_scale": args.cfg_scale,
        "num_frames": num_frames,
        "block_size": block_size,
        "image_size_eval": args.image_size_eval,
        "num_inference_steps": args.num_inference_steps,
        "ordering_runs": args.ordering_runs,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_images": num_images,
        "images": [],
    }

    image_idx = 0
    num_batches = (num_images + batch_size - 1) // batch_size

    for batch_i, batch_start in enumerate(
        tqdm(range(0, num_images, batch_size), desc="Batches")
    ):
        batch_classes = class_list[batch_start : batch_start + batch_size]
        actual_bs = len(batch_classes)
        c_indices = torch.tensor(batch_classes, device=device, dtype=torch.long)

        print(f"\n[Batch {batch_i+1}/{num_batches}] Classes: {batch_classes}")

        # Step 1: Confidence-based token ordering (from Autosyll)
        print(f"  Computing token ordering ({args.ordering_runs} runs)...")
        token_order = generate_ordering(
            c_indices, cfg_scales, args.ordering_runs, gpt_model, args
        )

        # Step 2: Final generation with confidence ordering
        print("  Generating final images...")
        result_indices = gpt_model.generate(
            cond=c_indices,
            token_order=token_order,
            cfg_scales=cfg_scales,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        # Step 3: Build visualization frames
        print(f"  Rendering {num_frames} frames...")
        frames = generate_masked_frames(
            result_indices, token_order, tokenizer, num_frames, args.image_size_eval
        )

        # Step 4: Save to disk
        for local_idx in range(actual_bs):
            img_dir = os.path.join(output_dir, "images", f"{image_idx:04d}")
            frames_dir = os.path.join(img_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            # Final image = last frame
            final_img = frames[-1][local_idx]
            Image.fromarray(final_img).save(os.path.join(img_dir, "final.png"))

            # All frames
            for frame_idx, frame_batch in enumerate(frames):
                Image.fromarray(frame_batch[local_idx]).save(
                    os.path.join(frames_dir, f"{frame_idx:04d}.png")
                )

            metadata["images"].append(
                {
                    "id": image_idx,
                    "class_label": batch_classes[local_idx],
                    "token_order": token_order[local_idx].cpu().tolist(),
                }
            )
            image_idx += 1

    # Write metadata and viewer HTML
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(_HTML_TEMPLATE)

    print(f"\nDone! {num_images} image(s) saved to: {output_dir}")
    print(f"Serve with:  python visualizer/server.py --output-dir {output_dir}")
    print("Then open:   http://localhost:8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RandAR images with confidence-ordered visualization frames."
    )
    # Model / checkpoint
    parser.add_argument("--config", type=str, required=True,
                        help="Path to OmegaConf config yaml (e.g. configs/randar/randar_xl_0.7b_llamagen.yaml)")
    parser.add_argument("--gpt-ckpt", type=str, required=True,
                        help="Path to GPT safetensors checkpoint")
    parser.add_argument("--vq-ckpt", type=str, required=True,
                        help="Path to VQ tokenizer checkpoint (.pt)")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["none", "fp16", "bf16"],
                        help="Model precision (default: bf16)")

    # Image / generation settings (mirrors Autosyll)
    parser.add_argument("--cfg-scale", type=float, required=True,
                        help="Classifier-free guidance scale (e.g. 4.0)")
    parser.add_argument("--num-inference-steps", type=int, default=88,
                        help="Parallel decoding steps (default: 88)")
    parser.add_argument("--ordering-runs", type=int, default=8,
                        help="Number of runs to average for confidence ordering (default: 8)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling (default: 0 = disabled)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p nucleus sampling (default: 1.0)")
    parser.add_argument("--image-size", type=int, default=256,
                        choices=[128, 256, 384, 512])
    parser.add_argument("--image-size-eval", type=int, default=256,
                        choices=[128, 256, 384, 512])
    parser.add_argument("--downsample-size", type=int, default=16, choices=[8, 16])
    parser.add_argument("--num-classes", type=int, default=1000)

    # What to generate
    parser.add_argument("--class-labels", type=str, default=None,
                        help="Comma-separated class indices to generate (e.g. '207,388,985'). "
                             "If omitted, --num-images random classes are drawn.")
    parser.add_argument("--num-images", type=int, default=4,
                        help="Number of images when --class-labels is not given (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Images per GPU forward pass (default: 1)")
    parser.add_argument("--global-seed", type=int, default=0)

    # Output
    parser.add_argument("--exp-name", type=str, default="randar",
                        help="Experiment name shown in the viewer")
    parser.add_argument("--output-dir", type=str, default="visualizer/output",
                        help="Directory to save images and viewer HTML")

    args = parser.parse_args()
    main(args)
