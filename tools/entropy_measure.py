
""" Measure and visualize per-cell decoding confidence (entropy) under confidence-first
    decoding, as a function of decode step.

    For each generated image, this records the entropy of the model's predictive
    distribution at the moment each raster cell was committed (via
    gpt_model.generate_with_confidence), and logs an aggregated
    entropy-vs-decode-step plot to Weights & Biases. No FID computation and no
    CFG-scale search are performed here; images and per-image entropy/step
    arrays are also saved to disk for later inspection.

    Structured as a single-GPU/single-process script (mirrors
    visualizer/generate_images.py) since this is a qualitative/diagnostic run
    rather than a large-scale DDP evaluation.
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import sys
import json
import argparse
import numpy as np
import wandb
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append("./")
from RandAR.util import instantiate_from_config, load_safetensors


def main(args):
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(args.global_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.global_seed)

    # Load config
    config = OmegaConf.load(args.config)

    # Load tokenizer (VQ model)
    print("Loading tokenizer...")
    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    tokenizer.load_state_dict(state_dict)

    # Load GPT model
    print("Loading GPT model...")
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    gpt_model = instantiate_from_config(config.ar_model).to(device=device, dtype=precision)
    model_weight = load_safetensors(args.gpt_ckpt)
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()

    block_size = gpt_model.block_size

    # Determine class labels
    if args.class_labels:
        class_list = [int(x.strip()) for x in args.class_labels.split(",")]
    else:
        rng = torch.Generator()
        rng.manual_seed(args.global_seed)
        class_list = torch.randint(0, args.num_classes, (args.num_images,), generator=rng).tolist()

    num_images = len(class_list)
    batch_size = min(args.batch_size, num_images)

    # Build ckpt name string
    ckpt_string_name = (
        os.path.basename(args.gpt_ckpt)
        .replace(".pth", "")
        .replace(".pt", "")
        .replace(".safetensors", "")
    )

    # Output directory
    output_dir = args.output_dir
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    cfg_scales = (1.0, args.cfg_scale)

    # Weights & Biases setup
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.exp_name,
        config=vars(args),
    )

    metadata = {
        "exp_name": args.exp_name,
        "ckpt": ckpt_string_name,
        "cfg_scale": args.cfg_scale,
        "block_size": block_size,
        "image_size_eval": args.image_size_eval,
        "num_inference_steps": args.num_inference_steps,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_images": num_images,
        "images": [],
    }

    rows = []  # (image_id, class_label, raster_pos, step, entropy)

    image_idx = 0
    num_batches = (num_images + batch_size - 1) // batch_size

    for batch_i, batch_start in enumerate(
        tqdm(range(0, num_images, batch_size), desc="Batches")
    ):
        batch_classes = class_list[batch_start : batch_start + batch_size]
        actual_bs = len(batch_classes)
        c_indices = torch.tensor(batch_classes, device=device, dtype=torch.long)

        print(f"\n[Batch {batch_i+1}/{num_batches}] Classes: {batch_classes}")

        result_indices, result_entropy, result_step = gpt_model.generate_with_confidence(
            cond=c_indices,
            token_order=None,
            cfg_scales=cfg_scales,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            probe_candidate_multiplier=args.probe_candidate_multiplier,
            probe_min_candidates=args.probe_min_candidates,
        )

        images = tokenizer.decode_codes_to_img(result_indices, args.image_size_eval)

        entropy_cpu = result_entropy.float().cpu()
        step_cpu = result_step.cpu()

        for local_idx in range(actual_bs):
            Image.fromarray(images[local_idx]).save(
                os.path.join(images_dir, f"{image_idx:04d}.png")
            )

            entropy_list = entropy_cpu[local_idx].tolist()
            step_list = step_cpu[local_idx].tolist()

            with open(os.path.join(images_dir, f"{image_idx:04d}_entropy_step.json"), "w") as f:
                json.dump(
                    {
                        "entropy": entropy_list,
                        "step": step_list,
                        "class_label": batch_classes[local_idx],
                    },
                    f,
                )

            for raster_pos in range(block_size):
                rows.append(
                    (
                        image_idx,
                        batch_classes[local_idx],
                        raster_pos,
                        step_list[raster_pos],
                        entropy_list[raster_pos],
                    )
                )

            metadata["images"].append(
                {"id": image_idx, "class_label": batch_classes[local_idx]}
            )
            image_idx += 1

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # Aggregate and log the entropy-vs-step plots
    table = wandb.Table(columns=["image_id", "class_label", "raster_pos", "step", "entropy"], data=rows)
    wandb.log({
        "entropy_vs_step": wandb.plot.scatter(
            table, "step", "entropy", title="Per-cell entropy vs. decode step"
        )
    })

    steps_arr = np.array([r[3] for r in rows])
    entropy_arr = np.array([r[4] for r in rows])
    max_step = int(steps_arr.max()) if len(steps_arr) > 0 else -1
    step_summary = []
    for step in range(max_step + 1):
        step_entropy = entropy_arr[steps_arr == step]
        if len(step_entropy) == 0:
            continue
        step_summary.append((step, float(step_entropy.mean()), float(step_entropy.std())))

    summary_table = wandb.Table(columns=["step", "mean_entropy", "std_entropy"], data=step_summary)
    wandb.log({
        "mean_entropy_by_step": wandb.plot.line(
            summary_table, "step", "mean_entropy", title="Mean confidence (entropy) by decode step"
        )
    })

    wandb.finish()

    print(f"\nDone! {num_images} image(s) saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model / checkpoint
    parser.add_argument("--config", type=str, default="configs/randar/randar_l_0.3b_llamagen.yaml")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])

    # Image / generation settings
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="classifier-free guidance scale")
    parser.add_argument("--image-size-eval", type=int, choices=[128, 256, 384, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-inference-steps", type=int, default=88)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--probe-candidate-multiplier", type=int, default=4)
    parser.add_argument("--probe-min-candidates", type=int, default=32)

    # What to generate
    parser.add_argument("--class-labels", type=str, default=None,
                        help="Comma-separated class indices to generate (e.g. '207,388,985'). "
                             "If omitted, --num-images random classes are drawn.")
    parser.add_argument("--num-images", type=int, default=128,
                        help="Number of images when --class-labels is not given (default: 128)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)

    # Output
    parser.add_argument("--output-dir", type=str, default="./results/entropy_measure")

    # Weights & Biases
    parser.add_argument("--wandb-project", type=str, default="RandAR_Visuals")
    parser.add_argument("--wandb-entity", type=str, default="ericyee07")
    parser.add_argument("--wandb-offline", action="store_true")

    args = parser.parse_args()
    main(args)
