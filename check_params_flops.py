import argparse

from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import get_flops, get_num_params


def main():
    parser = argparse.ArgumentParser(description="Check YOLO inference parameters and GFLOPs.")
    parser.add_argument("--model", required=True, help="Path to a .pt or .yaml model")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640], help="Size or height width")
    args = parser.parse_args()

    if len(args.imgsz) not in (1, 2) or min(args.imgsz) <= 0:
        parser.error("--imgsz requires one or two positive integers")

    model = YOLO(args.model, verbose=False)
    model.model.cpu().float().eval()

    imgsz = check_imgsz(args.imgsz, stride=model.model.stride, min_dim=2)
    params_loaded = get_num_params(model.model)

    # Match optimized inference: fuse Conv/BN and remove unused YOLO26 branches.
    model.fuse(verbose=False)
    params_infer = get_num_params(model.model)
    gflops = get_flops(model.model, imgsz=imgsz)

    if gflops <= 0:
        raise RuntimeError(
            "FLOPs profiling failed. Check model compatibility and install "
            "ultralytics-thop>=2.1.6 in the active Python environment."
        )

    print(f"\nModel           : {args.model}")
    print(f"Input (BCHW)    : 1 x 3 x {imgsz[0]} x {imgsz[1]}")
    print(f"Params loaded   : {params_loaded:,} ({params_loaded / 1e6:.4f} M)")
    print(f"Params inference: {params_infer:,} ({params_infer / 1e6:.4f} M)")
    print(f"GFLOPs / image  : {gflops:.4f}")


if __name__ == "__main__":
    main()