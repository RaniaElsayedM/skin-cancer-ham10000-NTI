import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
import os
import json
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Flatten
import tensorflow as tf

# Optional: try to support PyTorch if available
try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Optional: TensorFlow / Keras support
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="CV Starter • Nesma", page_icon="🧠", layout="wide")
st.title("🧠 Computer Vision Starter App")
st.caption("Local model friendly. Supports **classification** and **segmentation**. Works with TensorFlow/Keras; PyTorch if installed.")

with st.sidebar:
    st.header("⚙️ Settings")
    task = st.radio("Choose task", ["Classification", "Segmentation"], horizontal=False)
    framework = st.selectbox(
        "Framework",
        [opt for opt, ok in [("TensorFlow / Keras", TF_AVAILABLE), ("PyTorch", TORCH_AVAILABLE)] if ok] or ["TensorFlow / Keras"],
        help="Only shows frameworks detected in your environment."
    )
    image_size = st.number_input("Model input size (square)", min_value=64, max_value=1024, value=256, step=32)
    normalize_01 = st.checkbox("Normalize to [0,1]", True)
    keep_ratio = st.checkbox("Keep aspect ratio (pad)", True)

st.divider()

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img

@st.cache_data(show_spinner=False)
def preprocess_image(img: Image.Image, size: int, keep_aspect: bool = True, normalize: bool = True):
    if keep_aspect:
        img_resized = ImageOps.pad(img, (size, size), method=Image.BICUBIC, color=(0, 0, 0))
    else:
        img_resized = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img_resized).astype("float32")
    if normalize:
        arr = arr / 255.0
    # Add batch dimension
    arr_b = np.expand_dims(arr, axis=0)
    return img_resized, arr_b

@st.cache_data(show_spinner=False)
def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Convert single-channel [H,W] mask to a RGBA overlay for visualization."""
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    mask_min, mask_max = float(mask.min()), float(mask.max())
    if mask_max > mask_min:
        mask_norm = (mask - mask_min) / (mask_max - mask_min)
    else:
        mask_norm = np.zeros_like(mask)
    mask_u8 = (mask_norm * 255).astype(np.uint8)
    overlay = Image.fromarray(mask_u8, mode="L").convert("RGBA")
    # color = red overlay with original alpha
    r, g, b, a = overlay.split()
    overlay = Image.merge("RGBA", (Image.new("L", overlay.size, 255), Image.new("L", overlay.size, 0), Image.new("L", overlay.size, 0), a))
    return overlay

# -------------------------
# Labels support (for classification)
# -------------------------
@st.cache_data(show_spinner=False)
def parse_labels_file(content: bytes, filename: str):
    try:
        if filename.lower().endswith((".json",)):
            mapping = json.loads(content.decode("utf-8"))
            # Expect dict {"0": "class_name", ...} or {0: "class_name", ...}
            clean = {}
            for k, v in mapping.items():
                try:
                    clean[int(k)] = str(v)
                except Exception:
                    pass
            return clean
        else:
            # txt: one class per line, index = line number
            lines = [l.strip() for l in content.decode("utf-8").splitlines() if l.strip()]
            return {i: name for i, name in enumerate(lines)}
    except Exception:
        return {}

# -------------------------
# Model loading (path or upload)
# -------------------------
@st.cache_resource(show_spinner=True)

def load_sequential_model_fix(path):
    model = load_model(path, compile=False)

    # لو أي Flatten بياخد list -> نصلحه
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Flatten):
            inp = layer.input
            if isinstance(inp, list):  # يعني Flatten بيتوصّل بـ list
                new_inp = inp[0]
                x = Flatten()(new_inp)
                for l in model.layers[i+1:]:
                    x = l(x)
                model = tf.keras.Model(inputs=new_inp, outputs=x)
                print("✅ Flatten layer fixed.")
                return model
    return model



def load_tf_model_from_path(path):
    model = load_model(path, compile=False)

    # جرب تصلح Flatten لو بياخد list
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Flatten):
            inp = layer.input
            if isinstance(inp, list):  # لو بياخد list
                new_inp = inp[0]       # ناخد أول عنصر
                x = Flatten(name=layer.name)(new_inp)
                for l in model.layers[i+1:]:
                    x = l(x)
                model = Model(inputs=new_inp, outputs=x)
                return model  # خلاص عملنا التصليح

    # لو ما فيش Flatten بياخد list، رجّع الموديل زي ما هو
    return model

     


@st.cache_resource(show_spinner=True)
def load_tf_model_from_bytes(model_bytes: bytes, custom_objects: dict | None = None):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow/Keras not installed in this environment.")
    tmp_path = os.path.join(st.secrets.get("TMP_DIR", "."), "_uploaded_model.h5")
    with open(tmp_path, "wb") as f:
        f.write(model_bytes)
    model = keras_load_model(tmp_path, compile=False, custom_objects=custom_objects)
    return model

@st.cache_resource(show_spinner=True)
def load_torch_model_from_path(path: str):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed in this environment.")
    model = torch.jit.load(path, map_location="cpu") if _is_torchscript_path(path) else torch.load(path, map_location="cpu")
    model.eval()
    return model

@st.cache_resource(show_spinner=True)
def load_torch_model_from_bytes(model_bytes: bytes):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed in this environment.")
    buffer = io.BytesIO(model_bytes)
    buffer.seek(0)
    model = torch.jit.load(buffer, map_location="cpu") if _is_torchscript(buffer) else torch.load(buffer, map_location="cpu")
    model.eval()
    return model

def _is_torchscript(buff: io.BytesIO) -> bool:
    # naive check: TorchScript uses zip archive magic 'PK' at start
    head = buff.getvalue()[:4]
    buff.seek(0)
    return head == b'PK'

def _is_torchscript_path(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            return f.read(4) == b'PK'
    except Exception:
        return False

# -------------------------
# UI: model source + images + labels
# -------------------------
col_model, col_images = st.columns([1, 2])
with col_model:
    st.subheader("🧩 Model")

    use_local_path = st.checkbox("Load model from LOCAL PATH (recommended for large files)", value=True, help="When you run Streamlit locally, you can point to a file or folder on your machine.")
    model_path = None
    model_file = None

    if use_local_path:
        default_path = st.session_state.get("model_path", "")
        model_path = st.text_input("Model path (.h5/.keras/.pb dir, or .pt/.pth)", value=default_path, placeholder=r"e.g. C:\models\my_model.h5 or /home/user/models/saved_model/")
        if model_path:
            st.session_state["model_path"] = model_path
    else:
        model_file = st.file_uploader(
            "Or upload model file",
            type=["h5", "keras", "pb", "pth", "pt", "ptl"],
            help="Keras (.h5/.keras/SavedModel dir zipped) or PyTorch (.pt/.pth)."
        )

    st.markdown("**(Optional)** Class labels file (TXT: one per line) or JSON mapping {index: name}")
    labels_file = st.file_uploader("Upload labels", type=["txt", "json"], accept_multiple_files=False, key="labels")

with col_images:
    st.subheader("🖼️ Images")
    allow_multiple = task == "Classification"
    image_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=allow_multiple)

run = st.button("🚀 Run", type="primary", disabled=((not use_local_path and model_file is None) and not model_path) or not image_files)

# -------------------------
# Pipeline
# -------------------------
if run:
    if framework.startswith("TensorFlow") and not TF_AVAILABLE:
        st.error("TensorFlow/Keras not available. Install tensorflow first.")
        st.stop()
    if framework == "PyTorch" and not TORCH_AVAILABLE:
        st.error("PyTorch not available. Install torch first.")
        st.stop()

    # Parse labels if provided
    idx_to_name = None
    if labels_file is not None:
        idx_to_name = parse_labels_file(labels_file.read(), labels_file.name)

    # Load model
    with st.spinner("Loading model… (cached)"):
        if framework.startswith("TensorFlow"):
            if use_local_path and model_path:
                model = load_tf_model_from_path(model_path)
            elif model_file is not None:
                model = load_tf_model_from_bytes(model_file.read())
            else:
                st.error("Please provide a model path or upload a file.")
                st.stop()
        else:
            if use_local_path and model_path:
                model = load_torch_model_from_path(model_path)
            elif model_file is not None:
                model = load_torch_model_from_bytes(model_file.read())
            else:
                st.error("Please provide a model path or upload a file.")
                st.stop()

    if task == "Classification":
        st.subheader("🔮 Predictions")
        cols = st.columns(3)
        for idx, f in enumerate(image_files):
            img = load_image(f.read())
            img_resized, arr_b = preprocess_image(img, image_size, keep_aspect=keep_ratio, normalize=normalize_01)

            if framework.startswith("TensorFlow"):
                preds = model.predict(arr_b)
                probs = preds[0]
            else:
                # PyTorch path
                to_tensor = transforms.ToTensor() if TORCH_AVAILABLE else None
                x = to_tensor(img_resized).unsqueeze(0)
                if normalize_01:
                    x = x  # ToTensor already scales to [0,1]
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # Top-3
            topk = int(min(3, len(probs)))
            top_idx = np.argsort(-probs)[:topk]

            with cols[idx % 3]:
                st.image(img, caption=f.name, use_container_width=True)
                st.markdown("**Top predictions:**")
                for i in top_idx:
                    label = idx_to_name.get(int(i), f"Class {int(i)}") if idx_to_name else f"Class {int(i)}"
                    st.write(f"{label} — {float(probs[i]):.2%}")

    else:  # Segmentation
        st.subheader("🗺️ Segmentation Results")
        if len(image_files) > 1:
            st.info("Segmentation runs on the first image only. Upload one image for best results.")
        f = image_files[0]
        img = load_image(f.read())
        orig_size = img.size
        img_resized, arr_b = preprocess_image(img, image_size, keep_aspect=keep_ratio, normalize=normalize_01)

        if framework.startswith("TensorFlow"):
            pred = model.predict(arr_b)
            mask_pred = pred[0]
        else:
            to_tensor = transforms.ToTensor() if TORCH_AVAILABLE else None
            x = to_tensor(img_resized).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
                # try to support either logits [B,C,H,W] or mask [B,1,H,W]
                if out.ndim == 4 and out.shape[1] > 1:
                    out = torch.softmax(out, dim=1)[:, 1:2, ...]  # take class-1 prob as foreground
                mask_pred = out[0].permute(1, 2, 0).cpu().numpy()

        # Resize mask back to original size
        mask_img = Image.fromarray((np.clip(mask_pred, 0, 1) * 255).astype(np.uint8).squeeze(), mode="L")
        mask_big = mask_img.resize(orig_size, Image.NEAREST)

        # Compose overlay
        overlay = colorize_mask(np.array(mask_big))
        composite = Image.alpha_composite(img.convert("RGBA"), overlay.putalpha(120) or overlay)

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.image(img, caption="Original", use_container_width=True)
        with c2:
            st.image(mask_big, caption="Mask", use_container_width=True)
        with c3:
            st.image(composite, caption="Overlay", use_container_width=True)

        # Download buttons
        buf_mask = io.BytesIO()
        mask_big.save(buf_mask, format="PNG")
        st.download_button("⬇️ Download mask (PNG)", data=buf_mask.getvalue(), file_name="mask.png", mime="image/png")

st.divider()
st.markdown(
    """
**How to use**
1) Choose your **task** and **framework** from the sidebar.
2) Provide your **model path** (recommended for large local files), or upload the file if small.
3) (Optional) Upload **labels**: TXT (one class per line) or JSON mapping `{index: name}`.
4) Upload **image(s)**, set the **input size**, then click **Run**.

**Notes**
- For Keras, you can also point to a **SavedModel directory**.
- If your model needs custom objects (loss/metrics), loading with `compile=False` usually works for inference.
- PyTorch: TorchScript (`.pt` exported via `torch.jit.trace/script`) loads most reliably. Plain `.pth` pickles may require the original model class code.
- This app caches the loaded model, so it won't reload every time you click Run.
- If predictions look wrong, double-check **input size** and **normalization**.
"""
)
