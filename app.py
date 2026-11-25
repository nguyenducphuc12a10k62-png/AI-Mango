"""Chú Năm Vựa Xoài — compact local UI for MangoAI V7.1."""
from pathlib import Path
from hashlib import sha256
from io import BytesIO
import base64
import warnings
import html
import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from predict import load_models, predict
from features import FEATURE_VERSION, normalize_fruit, quality_interior, auto_fruit_box, resize

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / 'assets' / 'logo.png'

st.set_page_config(
    page_title='Chú Năm Vựa Xoài',
    page_icon=str(LOGO) if LOGO.exists() else '🥭',
    layout='wide',
    initial_sidebar_state='collapsed'
)

CSS = """<style>
:root {color-scheme:light;}
.stApp {background:#f7f8ed;color:#203e2c;}
[data-testid="stHeader"] {background:transparent;}
.stMainBlockContainer,.block-container {max-width:1340px;padding:2rem 1.65rem .75rem!important;}
[data-testid="stVerticalBlock"] {gap:.55rem;}
[data-testid="stHorizontalBlock"] {gap:16px;align-items:stretch;}
[data-testid="stColumn"] {min-width:0;}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {height:100%;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#d6e0d0!important;border-radius:16px!important;background:#fafbf5;}
[data-testid="stFileUploaderDropzone"] {background:#f1f5eb;padding:.55rem!important;}
[data-testid="stCameraInput"] video,[data-testid="stCameraInput"] img,[data-testid="stCameraInput"] canvas {max-height:clamp(120px,30vh,240px)!important;max-width:100%;object-fit:contain!important;}
[data-testid="stCameraInput"] {min-width:0;}
[data-testid="stCameraInput"] button {min-height:2.25rem;}
[data-testid="stTabs"] button {font-size:.83rem;font-weight:650;padding:.4rem .3rem;}
[data-testid="stTabs"] [role="tabpanel"] {padding-top:8px;}
.stButton>button {border-radius:11px;font-weight:750;min-height:2.65rem;}
.stButton>button[kind="primary"] {background:#2a6845;border-color:#2a6845;color:white;}
.cn-brand-wrap {padding-bottom:14px;}
.cn-brand {display:flex;align-items:center;gap:16px;padding:10px 20px;border-radius:20px;background:linear-gradient(115deg,#17452e,#28603a);color:white;box-shadow:0 8px 24px #254b2014;}
.cn-brand img {width:78px;height:64px;object-fit:contain;flex-shrink:0;}
.cn-brand h1 {color:#fff!important;font-size:clamp(1.45rem,2.5vw,2rem);line-height:1.15;margin:0!important;padding:0!important;font-weight:850;letter-spacing:-.035em;}
.cn-slogan {font-size:.94rem;color:#ffe7a1;margin-top:4px;font-weight:650;}
.cn-tag {margin-left:auto;font-size:.72rem;font-weight:700;letter-spacing:.06em;color:#e6f2db;border:1px solid #91b68a;padding:7px 11px;border-radius:30px;white-space:nowrap;}
.cn-heading {display:block;}
.cn-step {color:#6a794f;font-size:.72rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;line-height:1.3;}
.cn-title {font-size:1.14rem;font-weight:800;color:#244b33;line-height:1.35;margin:5px 0 0;}
/* Heading + state are in ONE HTML element. Padding applies to all states. */
.cn-status-area {box-sizing:border-box;padding-top:32px;padding-bottom:2px;}
.cn-result {border:1px solid #cfe4ce;background:#ecf6e9;border-radius:13px;padding:11px 14px;}
.cn-result.bad {background:#fff1e8;border-color:#edd1be;}
.cn-small {font-size:.73rem;font-weight:750;color:#526750;line-height:1.35;}
.cn-headline {font-size:1.18rem;line-height:1.3;font-weight:850;color:#21593a;margin:4px 0;}
.cn-result.bad .cn-headline {color:#943b24;}
.cn-state {font-size:.9rem;color:#2e4932;font-weight:650;line-height:1.35;}
.cn-metrics {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding-top:18px;padding-bottom:8px;}
.cn-metric {min-width:0;border:1px solid #dfe6d9;background:#f0f4eb;border-radius:10px;padding:9px 12px;}
.cn-metric-label {font-size:.78rem;color:#4d614c;font-weight:650;line-height:1.25;}
.cn-metric-value {font-size:1.65rem;line-height:1.2;color:#244c34;font-weight:800;letter-spacing:-.03em;margin-top:5px;}
.cn-empty,.cn-error {box-sizing:border-box;min-height:clamp(150px,31vh,250px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;border-radius:12px;text-align:center;padding:18px;}
.cn-empty {background:#f2f5eb;border:1px dashed #c9d6bc;color:#63745a;}
.cn-empty strong {font-size:1.02rem;color:#335b3d;}
.cn-error {background:#fff7e3;border:1px solid #ecdbad;color:#76591f;}
.cn-error strong {font-size:1.04rem;}
.cn-error span,.cn-empty span {font-size:.87rem;line-height:1.45;}
.cn-preview {box-sizing:border-box;width:100%;height:clamp(140px,30vh,240px);display:flex;align-items:center;justify-content:center;background:#f0f3eb;border-radius:12px;overflow:hidden;}
.cn-preview.mask {background:#101410;}
.cn-preview img {display:block;max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;}
.cn-image-label {text-align:center;font-size:.75rem;color:#62735c;margin-top:5px;}
.cn-input-empty {min-height:clamp(140px,30vh,240px);}
@media(min-width:769px) and (max-height:800px) {
  .stMainBlockContainer,.block-container {padding-top:1.5rem!important;}
  .cn-brand img {height:56px;width:68px;}
  .cn-brand {padding:8px 18px;}
  .cn-brand-wrap {padding-bottom:12px;}
  .cn-preview {height:clamp(135px,28vh,210px);}
  .cn-metric {padding:7px 10px;}
}
@media(max-width:768px){
  .stMainBlockContainer,.block-container{padding:2.5rem .75rem .75rem!important;}
  [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important;min-width:0!important;}
  .cn-brand{padding:10px 12px;gap:9px;}
  .cn-brand img{width:54px;height:54px;}
  .cn-tag{display:none;}
  .cn-brand h1{font-size:1.35rem;}
  .cn-slogan{font-size:.8rem;}
  .cn-metrics{gap:6px;}
  .cn-metric{padding:7px;}
  .cn-metric-label{font-size:.7rem;}
  .cn-metric-value{font-size:1.28rem;}
  .cn-preview{height:230px;}
  
  .cn-preview{height:230px;}
}

/* ================================ */
/* FIX CAMERA BUTTON                */
/* ================================ */

[data-testid="stCameraInput"] button {
    background-color: #2A6845 !important;
    color: #FFFFFF !important;
    border: 1px solid #2A6845 !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    font-weight: 650 !important;
}

[data-testid="stCameraInput"] button * {
    color: #FFFFFF !important;
}

[data-testid="stCameraInput"] button svg {
    fill: currentColor !important;
}

[data-testid="stCameraInput"] button:hover {
    background-color: #194B32 !important;
}

/* ================================ */
/* FIX RADIO LABEL                  */
/* ================================ */

[data-testid="stRadio"] label,
[data-testid="stRadio"] label p {
    color: #203E2C !important;
}

/* ================================ */
/* FIX UPLOAD LABEL                 */
/* ================================ */

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small {
    color: #203E2C !important;
}

</style>"""

st.markdown(CSS, unsafe_allow_html=True)

}
</style>"""

logo_markup = (f'<img alt="Logo Chú Năm" src="data:image/png;base64,{base64.b64encode(LOGO.read_bytes()).decode()}"/>' if LOGO.exists() else '<span style="font-size:3rem">🥭</span>')
st.markdown(f'<div class="cn-brand-wrap"><div class="cn-brand">{logo_markup}<div><h1>Chú Năm Vựa Xoài</h1><div class="cn-slogan">Chọn xoài đừng để xoài &quot;Thâm&quot;</div></div><span class="cn-tag">PHÂN LOẠI XOÀI · AI</span></div></div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def models():
    return load_models()


def decode(raw):
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(BytesIO(raw)) as im:
            if im.width * im.height > 24_000_000:
                raise ValueError('Ảnh quá lớn (tối đa 24 megapixel).')
            im = ImageOps.exif_transpose(im).convert('RGB')
            return im.copy()


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def auto_preview(pil):
    preview = resize(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
    auto = auto_fruit_box(preview)
    if auto is not None:
        (x0, y0, x1, y1), _ = auto
        cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (40, 220, 35), 2)
    return rgb(preview), auto is not None


def analyze(pil, bundles):
    p, f, img, mask, defect = predict(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), bundles)
    overlay = img.copy()
    overlay[defect > 0] = (0, 0, 255)
    _, roi_mask, (x0, y0, x1, y1) = normalize_fruit(img, mask)
    _, inner, _, _ = quality_interior(roi_mask)
    full = np.zeros_like(mask)
    full[y0:y1, x0:x1] = cv2.resize(inner.astype(np.uint8) * 255, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    full[mask == 0] = 0
    contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_view = overlay.copy()
    cv2.drawContours(roi_view, contours, -1, (255, 255, 0), 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inside = mask > 0
    blur = cv2.Laplacian(gray, cv2.CV_64F)[inside].var()
    clipped = np.mean((gray[inside] < 15) | (gray[inside] > 245))
    return {
        'p': p,
        'f': f,
        'images': [rgb(img), mask, rgb(overlay), rgb(roi_view)],
        'warn': bool(blur < 40 or clipped > .20),
    }




def heading(step, title):
    return f'<div class="cn-heading"><div class="cn-step">{step}</div><div class="cn-title">{title}</div></div>'


def preview_html(image, label=None, mask=False):
    """Responsive image, centered, no stretching or cropping."""
    im = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
    im = im.convert('RGB')
    im.thumbnail((1000, 800), Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    # Scale small images up only for display; no effect on inference.
    buf = BytesIO()
    im.save(buf, format='PNG' if mask else 'JPEG', **({} if mask else {'quality': 92}))
    mime = 'png' if mask else 'jpeg'
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    caption = f'<div class="cn-image-label">{html.escape(label)}</div>' if label else ''
    return f'<div class="cn-preview{" mask" if mask else ""}"><img src="data:image/{mime};base64,{data}" alt="{html.escape(label or "Ảnh kết quả phân tích")}"></div>{caption}'


def result_html(result=None, error=None):
    """Render all states with the same explicit 32px heading-to-content gap."""
    if error:
        state = ('<div class="cn-error" role="alert"><strong>Không thể phân tích ảnh</strong>'
                 f'<span>{html.escape(error)}</span></div>')
    elif result:
        p, f = result['p'], result['f']
        good = p['grading'] == 'Class_I'
        ripe = p['ripeness'] == 'Ripe'
        label = 'Đạt chất lượng xuất khẩu' if good else 'Không đạt chất lượng xuất khẩu'
        values = [('Màu xanh', f['green_ratio']), ('Màu vàng', f['yellow_ratio']), ('Nghi khuyết điểm', f['defect_ratio'])]
        metrics = ''.join(f'<div class="cn-metric"><div class="cn-metric-label">{name}</div><div class="cn-metric-value">{value:.1%}</div></div>' for name, value in values)
        state = (f'<div class="cn-result{"" if good else " bad"}"><div class="cn-small">PHÂN LOẠI THEO TIÊU CHÍ ĐỒ ÁN</div>'
                 f'<div class="cn-headline">{label}</div><div class="cn-state">Độ chín: {"Chín" if ripe else "Chưa chín"} · Loại {"I" if good else "II"}</div></div>'
                 f'<div class="cn-metrics">{metrics}</div>')
    else:
        state = '<div class="cn-empty"><strong>Chưa có kết quả phân loại</strong><span>Chọn ảnh bên trái và bấm Phân tích quả xoài.</span></div>'
    return heading('02 / Kết quả', 'Kiểm tra ngoại quan') + f'<div class="cn-status-area">{state}</div>'


left, right = st.columns([0.38, 0.62], gap='medium')
pil = None
key = None
run = False

with left:
    with st.container(border=True):
        st.markdown(heading('01 / Đầu vào', 'Chọn ảnh quả xoài'), unsafe_allow_html=True)
        mode = st.radio('Nguồn ảnh', ['Tải ảnh', 'Camera'], horizontal=True, label_visibility='collapsed')
        is_camera = mode == 'Camera'
        if is_camera:
            file = st.camera_input('Chụp một quả xoài', label_visibility='collapsed', key='mango_camera')
        else:
            file = st.file_uploader('Chọn ảnh JPG / PNG / WEBP', type=['jpg', 'jpeg', 'png', 'webp'], label_visibility='collapsed', key='mango_upload')

        if file is not None:
            raw = file.getvalue()
            try:
                if len(raw) > 10 * 1024 * 1024:
                    raise ValueError('Ảnh tối đa 10 MB.')
                pil = decode(raw)
            except Exception:
                st.error('Ảnh không hợp lệ hoặc quá lớn. Chọn ảnh JPG/PNG/WEBP dưới 10 MB, tối đa 24 MP.')

        # Placeholders keep the main button above optional manual controls.
        # Native camera already displays the photo, so no duplicate main preview.
        preview_slot = st.container()
        action_slot = st.container()
        if pil is not None:
            lr, tb = (0, 100), (0, 100)
            with st.expander('Chỉnh khung thủ công (tùy chọn)', expanded=False):
                ident = sha256(raw).hexdigest()[:20]
                lr = st.slider('Ngang (%)', 0, 100, (0, 100), key='cropx_' + ident)
                tb = st.slider('Dọc (%)', 0, 100, (0, 100), key='cropy_' + ident)
                if lr[1] - lr[0] < 10 or tb[1] - tb[0] < 10:
                    st.warning('Khung quá nhỏ. Hãy chọn lại.')
                    pil = None
                else:
                    w, h = pil.size
                    pil = pil.crop((int(w * lr[0] / 100), int(h * tb[0] / 100), int(w * lr[1] / 100), int(h * tb[1] / 100)))
                    key = sha256(raw + repr((mode, lr, tb, FEATURE_VERSION)).encode()).hexdigest()
                    if is_camera:
                        preview, found = auto_preview(pil)
                        # Only visible when user explicitly expands manual editing.
                        st.markdown(preview_html(preview, 'Khung tự động' if found else 'Ảnh đã chọn'), unsafe_allow_html=True)
            if pil is not None and not is_camera:
                preview, found = auto_preview(pil)
                preview_slot.markdown(preview_html(preview, 'Khung tự động' if found else 'Ảnh đầu vào'), unsafe_allow_html=True)
        elif not is_camera and file is None:
            preview_slot.markdown('<div class="cn-empty cn-input-empty"><strong>Ảnh của bạn sẽ xuất hiện ở đây</strong><span>1 quả xoài · nền trơn · đủ sáng</span></div>', unsafe_allow_html=True)
        run = action_slot.button('Phân tích quả xoài', type='primary', use_container_width=True, disabled=pil is None)

if st.session_state.get('analysis_key') != key:
    st.session_state.pop('analysis', None)
    st.session_state.pop('analysis_error', None)
    st.session_state['analysis_key'] = key

with right:
    with st.container(border=True):
        if run and pil is not None:
            st.session_state.pop('analysis', None)
            st.session_state.pop('analysis_error', None)
            with st.spinner('Đang phân vùng và phân loại...'):
                try:
                    st.session_state['analysis'] = analyze(pil, models())
                except (FileNotFoundError, ValueError) as e:
                    if isinstance(e, FileNotFoundError) or 'version mismatch' in str(e).lower():
                        st.session_state['analysis_error'] = 'Chưa có model đúng phiên bản. Hãy huấn luyện lại V7.1 trước khi chạy web.'
                    else:
                        st.session_state['analysis_error'] = 'Hãy chỉnh khung hoặc chụp lại một quả xoài rõ nét.'
                except Exception:
                    st.session_state['analysis_error'] = 'Hãy thử ảnh khác hoặc kiểm tra model.'
        result = st.session_state.get('analysis')
        error = st.session_state.get('analysis_error')
        st.markdown(result_html(result, error), unsafe_allow_html=True)
        if result and not error:
            tabs = st.tabs(['Khuyết điểm', 'ROI trung tâm', 'Mask quả', 'Ảnh xử lý'])
            for tab, idx in zip(tabs, [2, 3, 1, 0]):
                with tab:
                    st.markdown(preview_html(result['images'][idx], mask=idx == 1), unsafe_allow_html=True)
