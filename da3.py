import argparse
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(input_path: Path, limit: int, tail: bool) -> list[str]:
    if input_path.is_file():
        return [str(input_path)]

    images = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if limit > 0:
        images = images[-limit:] if tail else images[:limit]
    return [str(path) for path in images]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 on a few local images.")
    parser.add_argument("--input", default="data", help="Image file or directory.")
    parser.add_argument("--limit", type=int, default=5, help="Max images from a directory. Use 0 for all.")
    parser.add_argument("--tail", action="store_true", help="Use the last N images from a directory.")
    parser.add_argument("--model-dir", default=".", help="Directory containing config.json and model.safetensors.")
    parser.add_argument("--output", default="output/da3_test", help="Export directory.")
    parser.add_argument(
        "--export-format",
        default="mini_npz-depth_vis",
        help="DA3 export format, e.g. mini_npz, depth_vis, mini_npz-depth_vis, glb.",
    )
    parser.add_argument("--process-res", type=int, default=504, help="Inference resize resolution.")
    parser.add_argument(
        "--as-batch",
        action="store_true",
        help="Run all images together. Without this, images run one by one to avoid center-cropping mixed sizes.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input)
    export_dir = Path(args.output)

    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing model.safetensors in {model_dir.resolve()}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir.resolve()}")

    images = collect_images(input_path, args.limit, args.tail)
    if not images:
        raise FileNotFoundError(f"No images found at {input_path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DA3 from {model_dir.resolve()} on {device}")
    print(f"Running {len(images)} image(s)")

    model = DepthAnything3.from_pretrained(str(model_dir)).to(device)
    if args.as_batch:
        prediction = model.inference(
            image=images,
            export_dir=str(export_dir),
            export_format=args.export_format,
            process_res=args.process_res,
        )
    else:
        prediction = None
        for index, image in enumerate(images, start=1):
            image_output = export_dir / Path(image).stem
            print(f"[{index}/{len(images)}] {image} -> {image_output}")
            prediction = model.inference(
                image=[image],
                export_dir=str(image_output),
                export_format=args.export_format,
                process_res=args.process_res,
            )

    print("Done")
    if prediction is not None:
        print(f"last depth: {prediction.depth.shape} {prediction.depth.dtype}")
    if prediction is not None and getattr(prediction, "conf", None) is not None:
        print(f"conf: {prediction.conf.shape} {prediction.conf.dtype}")
    print(f"export_dir: {export_dir.resolve()}")


if __name__ == "__main__":
    main()
