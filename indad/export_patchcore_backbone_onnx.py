import os
from pathlib import Path

import click
import torch
import timm


class PatchCoreBackboneWrapper(torch.nn.Module):
    def __init__(self, backbone: str, out_indices):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            out_indices=tuple(out_indices),
            features_only=True,
            pretrained=True,
        )
        self.model.eval()

    def forward(self, x):
        features = self.model(x)
        return tuple(features)


def parse_pair(value: str):
    sep = "," if "," in value else "x"
    parts = [p.strip() for p in value.split(sep)]
    if len(parts) != 2:
        raise ValueError(f"expected pair like 384,128 or 128x384, got {value}")
    return [int(parts[0]), int(parts[1])]


def parse_indices(value: str):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


@click.command()
@click.option("--backbone", default="resnet18")
@click.option("--out-indices", default="2,3")
@click.option("--image-size", default="384,128", help="Input image size as width,height")
@click.option("--batch-size", default=15, type=int)
@click.option("--output", required=True, type=Path)
@click.option("--opset", default=17, type=int)
def cli_interface(backbone, out_indices, image_size, batch_size, output, opset):
    project_root = Path(__file__).resolve().parents[1]
    torch.hub.set_dir(str(project_root / "hub"))

    width, height = parse_pair(image_size)
    indices = parse_indices(out_indices)
    model = PatchCoreBackboneWrapper(backbone, indices)
    dummy = torch.randn(batch_size, 3, height, width)

    output.parent.mkdir(parents=True, exist_ok=True)
    output_names = [f"feat{i}" for i in range(len(indices))]
    dynamic_axes = {"input": {0: "batch"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    print(f"exported {output}")
    print(f"input_shape={(batch_size, 3, height, width)}")
    print(f"output_names={output_names}")


if __name__ == "__main__":
    cli_interface()
