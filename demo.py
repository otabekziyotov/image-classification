import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import streamlit as st
import torch
import random
import pickle
import pandas as pd
from PIL import Image
from glob import glob
from src.infer import ModelInferenceVisualizer
from data.parse import CustomDataset

st.set_page_config(page_title="Vision Classifier Studio", page_icon="🧠", layout="wide")


# ---------- Styling ----------
def inject_css():
    st.markdown(
        """
        <style>
            /* Hero header */
            .hero {
                background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                padding: 2.2rem 2rem; border-radius: 18px; margin-bottom: 1.5rem;
                box-shadow: 0 10px 30px rgba(37,117,252,0.25);
            }
            .hero h1 { color: #fff; margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -1px; }
            .hero p  { color: rgba(255,255,255,0.85); margin: .4rem 0 0; font-size: 1.05rem; }

            /* Image card */
            .card-title {
                text-align: center; font-weight: 700; font-size: 1.1rem;
                color: #cbd5e1; margin-bottom: .6rem; letter-spacing: .3px;
            }
            div[data-testid="stImage"] img { border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.35); }

            /* Badges */
            .badge {
                display: inline-block; padding: .5rem 1.1rem; border-radius: 999px;
                font-weight: 700; font-size: 1.05rem; margin: .2rem 0;
            }
            .badge-pred { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid #22c55e; }
            .badge-gt   { background: rgba(148,163,184,0.15); color: #cbd5e1; border: 1px solid #64748b; }
            .badge-ok   { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid #22c55e; }
            .badge-bad  { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid #ef4444; }

            .verdict {
                text-align: center; font-size: 1.25rem; margin-top: 1rem;
                padding: 1rem; border-radius: 14px; background: rgba(255,255,255,0.04);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(model_path, model_name, num_classes, device):
    import timm
    model = timm.create_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.eval().to(device)


@st.cache_data
def load_class_names(pkl_path):
    with open(pkl_path, "rb") as fp:
        return pickle.load(fp)


class StreamlitApp:
    def __init__(self, ds_nomi, model_name, dataset_root):
        self.ds_nomi = ds_nomi
        self.model_name = model_name
        self.dataset_root = dataset_root

    def _class_of(self, path):
        """Class name from an image path (dataset-specific rule)."""
        if self.ds_nomi in ["lentils", "apple_disease"]:
            return os.path.basename(path).split("_")[0]
        return os.path.basename(os.path.dirname(path))

    def _group_by_class(self, paths):
        """Group saved sample paths into {class_name: [paths]} by the '___{class}' suffix."""
        pools = {}
        for p in paths:
            cname = os.path.basename(p).split("___")[-1].rsplit(".", 1)[0]
            pools.setdefault(cname, []).append(p)
        return pools

    def prepare_samples(self, save_dir, class_names, per_class=8):
        """Build a POOL of several images per class on first run, so the demo can
        show a different random image per class on every refresh (works on Streamlit
        Cloud too, since the pool is committed and no dataset is needed at runtime)."""
        os.makedirs(save_dir, exist_ok=True)
        pools = self._group_by_class(glob(os.path.join(save_dir, "*.png")))
        # Every class already has at least one image -> reuse the committed pool
        if all(len(pools.get(c, [])) > 0 for c in class_names):
            return pools

        ds = (
            CustomDataset(self.dataset_root, data_type="test", ds_nomi=self.ds_nomi)
            if self.ds_nomi in ["facial_expression", "pokemon"]
            else CustomDataset(self.dataset_root, ds_nomi=self.ds_nomi)
        )
        if len(ds.rasm_yolaklari) == 0:
            st.error(
                f"No images found for '{self.ds_nomi}' under '{self.dataset_root}'. "
                f"Set the correct 'Dataset root' in the sidebar (e.g. /content/datasets)."
            )
            st.stop()

        # Group every dataset image by class, then save up to `per_class` random ones each.
        # Filename keeps the class as the last '___' segment so GT parsing still works.
        by_class = {}
        for path in ds.rasm_yolaklari:
            c = self._class_of(path)
            if c in class_names:
                by_class.setdefault(c, []).append(path)

        n = 0
        for c, paths in by_class.items():
            for p in random.sample(paths, min(per_class, len(paths))):
                n += 1
                with Image.open(p).convert("RGB") as im:
                    im.thumbnail((384, 384))  # keep the committed pool small
                    im.save(os.path.join(save_dir, f"sample_{n}___{c}.png"))
        return self._group_by_class(glob(os.path.join(save_dir, "*.png")))

    def render_class_gallery(self, pools, per_row=6):
        """Show ONE random example per class (re-rolled on every refresh)."""
        classes = sorted([c for c, paths in pools.items() if paths])
        with st.expander(f"📚 Classes this model recognizes ({len(classes)})", expanded=True):
            for i in range(0, len(classes), per_row):
                cols = st.columns(per_row)
                for col, c in zip(cols, classes[i:i + per_row]):
                    with col:
                        st.image(random.choice(pools[c]), use_container_width=True)
                        st.markdown(
                            f"<div style='text-align:center;font-weight:600;color:#cbd5e1'>{c}</div>",
                            unsafe_allow_html=True,
                        )

    def render_results(self, result, class_names):
        predicted_class = list(class_names.keys())[result["pred"]]
        gt = result["gt"]
        confidence = result["confidence"]
        known_gt = gt in class_names  # uploaded images have an unknown GT

        st.markdown("### 🔍 Inference Result")

        # --- Image row: original + Grad-CAM ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card-title'>📷 Original Image</div>", unsafe_allow_html=True)
            st.image(result["original_im"], use_container_width=True)
        with c2:
            st.markdown("<div class='card-title'>🔥 Grad-CAM — where the model looks</div>", unsafe_allow_html=True)
            st.image(result["gradcam"], use_container_width=True)

        st.divider()

        # --- Prediction panel ---
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f"<span class='badge badge-pred'>PRED &nbsp;→&nbsp; {predicted_class}</span>", unsafe_allow_html=True)
            if known_gt:
                ok = (predicted_class == gt)
                st.markdown(
                    f"<span class='badge {'badge-ok' if ok else 'badge-bad'}'>GT &nbsp;→&nbsp; {gt} &nbsp;{'✓' if ok else '✗'}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<span class='badge badge-gt'>GT &nbsp;→&nbsp; unknown (uploaded)</span>", unsafe_allow_html=True)

            st.metric(label="Confidence", value=f"{confidence:.1f}%")
            st.progress(min(int(confidence), 100))

        with right:
            st.markdown("<div class='card-title'>Class probabilities</div>", unsafe_allow_html=True)
            probs_dict = result.get("probs_dict")
            if probs_dict:
                df = (
                    pd.DataFrame({"class": list(probs_dict.keys()), "probability": list(probs_dict.values())})
                    .sort_values("probability", ascending=False)
                    .set_index("class")
                )
                st.bar_chart(df, height=260, color="#2575fc")
            else:
                st.image(result["probs"], use_container_width=True)

        # --- Verdict ---
        st.markdown(
            f"""
            <div class='verdict'>
                The model is <b style='color:#2575fc;'>{confidence:.2f}%</b> confident this image is a
                <b style='color:#22c55e;'>{predicted_class.upper()}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    def run(self):
        inject_css()

        st.markdown(
            """
            <div class='hero'>
                <h1>🧠 Vision Classifier Studio</h1>
                <p>Upload or pick an image — see the prediction, confidence and a Grad-CAM heatmap of the model's attention.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Load class names and model
        class_names = load_class_names(f"saved_cls_names/{self.ds_nomi}_cls_names.pkl")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(
            model_path=f"saved_models/{self.ds_nomi}_best_model.pth",
            model_name=self.model_name,
            num_classes=len(class_names),
            device=device,
        )

        model_inference = ModelInferenceVisualizer(
            model=model,
            device=device,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            outputs_dir=None,
            ds_nomi=self.ds_nomi,
            class_names=class_names,
        )

        # Build a pool of images per class, then show a random one per class as a gallery
        pools = self.prepare_samples(os.path.join("demo_ims", self.ds_nomi), class_names)
        self.render_class_gallery(pools)

        all_samples = sorted(p for paths in pools.values() for p in paths)

        # Image source: pick a sample OR upload your own (tabs)
        tab_sample, tab_upload = st.tabs(["🖼️ Sample image", "⬆️ Upload your own"])
        with tab_sample:
            selected_image = st.selectbox(
                "Pick a sample image", all_samples,
                format_func=lambda p: os.path.basename(p).split("___")[-1].rsplit(".", 1)[0] + "  ·  " + os.path.basename(p),
            )
        with tab_upload:
            uploaded_image = st.file_uploader("Drop a JPG / PNG", type=["jpg", "png", "jpeg"])

        im_path = uploaded_image if uploaded_image else selected_image

        if im_path:
            with st.spinner("🧪 Running inference & Grad-CAM..."):
                result = model_inference.demo(im_path)
            self.render_results(result, class_names)
        else:
            st.info("Select a sample or upload an image to begin.")


def sidebar_controls():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        available_datasets = [os.path.basename(p).split("_best_model")[0] for p in glob("saved_models/*.pth")]
        if not available_datasets:
            st.error("No trained models in 'saved_models/'. Train one with main.py first.")
            st.stop()

        ds_nomi = st.selectbox("📦 Dataset", options=available_datasets, index=0)
        model_name = st.text_input("🏗️ Model architecture", value="rexnet_150")
        dataset_root = st.text_input("📁 Dataset root", value="/content/datasets")

        # Quick stats
        try:
            cls = load_class_names(f"saved_cls_names/{ds_nomi}_cls_names.pkl")
            st.divider()
            st.markdown(f"**Classes ({len(cls)}):**")
            st.caption(", ".join(list(cls.keys())))
        except Exception:
            pass

        st.divider()
        st.caption("Built with PyTorch · timm · Grad-CAM · Streamlit")
        return ds_nomi, model_name, dataset_root


if __name__ == "__main__":
    ds_nomi, model_name, dataset_root = sidebar_controls()
    app = StreamlitApp(ds_nomi=ds_nomi, model_name=model_name, dataset_root=dataset_root)
    app.run()
