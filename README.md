# 🧠 Vision Classifier Studio

A modular PyTorch image-classification pipeline with training metrics, learning curves,
Grad-CAM explainability, an ONNX exporter and an interactive **Streamlit** demo.

## Features
- 🏋️ Training / validation loop with accuracy, F1 and early stopping (`src/train.py`)
- 📈 Learning-curve plots (`src/plot.py`)
- 🔥 Grad-CAM++ heatmaps showing where the model looks (`src/infer.py`)
- 🔄 PyTorch → ONNX export with a PyTorch-vs-ONNX sanity check (`onnx_converter.py`)
- 🖥️ Interactive Streamlit demo (`demo.py`)

## Project layout
```
main.py             # train + evaluate entry point
demo.py             # Streamlit web demo
onnx_converter.py   # export a trained model to ONNX
src/                # train, infer (Grad-CAM), plot, transforms, vis
data/               # dataset download + parsing
saved_models/       # trained checkpoints  ({ds}_best_model.pth)
saved_cls_names/    # pickled class names  ({ds}_cls_names.pkl)
demo_ims/           # sample images shown in the demo
```

## Run the demo locally
```bash
pip install -r requirements.txt
streamlit run demo.py
```

## Train a model
```bash
python main.py -dn apple_disease --device cuda --batch_size 64
```

## Export to ONNX
```bash
python onnx_converter.py -dn apple_disease --test_image path/to/image.jpg
```

---
Built with PyTorch · timm · pytorch-grad-cam · Streamlit
