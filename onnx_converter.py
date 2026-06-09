import os
import pickle
import argparse
import numpy as np
import torch
import timm
from PIL import Image
from src.transform import get_tfs


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a trained PyTorch model to ONNX and verify it")

    parser.add_argument('-dn', '--ds_nomi', type=str, required=True, help="Name of the dataset (used to find the checkpoint and class names)")
    parser.add_argument('--model_name', type=str, default="rexnet_150", help="Model architecture from timm (must match training)")
    parser.add_argument('--save_dir', type=str, default="saved_models", help="Directory with the .pth checkpoints")
    parser.add_argument('--cls_root', type=str, default="saved_cls_names", help="Directory with the saved class names (.pkl)")
    parser.add_argument('--onnx_dir', type=str, default="onnx_models", help="Directory to save the exported .onnx model")
    parser.add_argument('--image_size', type=int, default=224, help="Input image size the model was trained with")
    parser.add_argument('--opset', type=int, default=18, help="ONNX opset version")
    parser.add_argument('--test_image', type=str, default=None, help="Optional image path to run a PyTorch-vs-ONNX sanity check")
    return parser.parse_args()


class ONNXConverter:
    def __init__(self, ds_nomi, model_name, save_dir, cls_root, onnx_dir, image_size=224, opset=12):
        self.ds_nomi = ds_nomi
        self.model_name = model_name
        self.image_size = image_size
        self.opset = opset

        # Paths
        self.ckpt_path = os.path.join(save_dir, f"{ds_nomi}_best_model.pth")
        self.cls_path = os.path.join(cls_root, f"{ds_nomi}_cls_names.pkl")
        os.makedirs(onnx_dir, exist_ok=True)
        self.onnx_path = os.path.join(onnx_dir, f"{ds_nomi}.onnx")

        # Class names (dict {name: idx}) -> list of names ordered by index
        with open(self.cls_path, "rb") as fp:
            cls_names = pickle.load(fp)
        self.class_names = list(cls_names.keys())
        self.num_classes = len(self.class_names)

        # Build the model and load the trained weights (CPU is enough for export)
        self.device = "cpu"
        self.model = timm.create_model(model_name, num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.eval().to(self.device)   # eval() freezes dropout/batchnorm

    def convert(self):
        # Dummy input drives the tracing -> shape must match the real input
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            self.onnx_path,
            export_params=True,                 # bake the weights into the file
            opset_version=self.opset,
            input_names=['input'],
            output_names=['output'],
            # batch dimension is dynamic -> the .onnx accepts any batch size
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )
        print(f"ONNX model saved -> {self.onnx_path}")

    def verify(self):
        import onnx
        onnx_model = onnx.load(self.onnx_path)
        onnx.checker.check_model(onnx_model)    # raises if the graph is invalid
        print("ONNX graph check passed.")

    def test_inference(self, image_path):
        import onnxruntime as ort

        # Same preprocessing as training (Resize -> ToTensor -> Normalize)
        tfs = get_tfs(im_size=self.image_size)
        image = Image.open(image_path).convert("RGB")
        input_tensor = tfs(image).unsqueeze(0)              # (1, 3, H, W)
        input_numpy = input_tensor.numpy()                  # onnxruntime needs numpy, not torch

        # PyTorch prediction
        with torch.no_grad():
            torch_logits = self.model(input_tensor.to(self.device)).cpu().numpy()
        torch_pred = int(np.argmax(torch_logits, axis=1)[0])

        # ONNX prediction
        session = ort.InferenceSession(self.onnx_path)
        input_name = session.get_inputs()[0].name
        onnx_logits = session.run(None, {input_name: input_numpy})[0]
        onnx_pred = int(np.argmax(onnx_logits, axis=1)[0])

        print(f"PyTorch -> {self.class_names[torch_pred]}  |  ONNX -> {self.class_names[onnx_pred]}")
        # Outputs should be numerically almost identical
        max_diff = np.abs(torch_logits - onnx_logits).max()
        print(f"Max logits difference (PyTorch vs ONNX): {max_diff:.6f}")
        print("MATCH ✓" if torch_pred == onnx_pred else "MISMATCH ✗ — check preprocessing/opset")


def main():
    args = parse_args()
    converter = ONNXConverter(
        ds_nomi=args.ds_nomi,
        model_name=args.model_name,
        save_dir=args.save_dir,
        cls_root=args.cls_root,
        onnx_dir=args.onnx_dir,
        image_size=args.image_size,
        opset=args.opset,
    )
    converter.convert()
    converter.verify()
    if args.test_image:
        converter.test_inference(args.test_image)


if __name__ == "__main__":
    main()
