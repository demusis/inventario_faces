from __future__ import annotations

"""Shared helpers for rendering the technical help shown in the config dialog."""

from collections.abc import Sequence

INSIGHTFACE_URL = "https://github.com/deepinsight/insightface"
ARCFACE_URL = (
    "https://openaccess.thecvf.com/content_CVPR_2019/html/"
    "Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html"
)
SCRFD_URL = "https://arxiv.org/abs/2105.04714"
ONNXRUNTIME_EP_URL = "https://onnxruntime.ai/docs/execution-providers/"
OPENCV_VIDEOCAPTURE_URL = "https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html"
MEDIAINFO_URL = "https://mediaarea.net/en/MediaInfo"
FAISS_URL = "https://github.com/facebookresearch/faiss"
ABNT_ACCESS_DATE = "Acesso em: 4 abr. 2026."

ReferenceItem = tuple[str, str]

_ABNT_REFERENCE_TEMPLATES = {
    INSIGHTFACE_URL: (
        "INSIGHTFACE. <i>InsightFace: an open source 2D and 3D deep face analysis library</i>. "
        "Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. "
        f"{ABNT_ACCESS_DATE}"
    ),
    ARCFACE_URL: (
        "DENG, Jiankang et al. <i>ArcFace: additive angular margin loss for deep face recognition</i>. "
        "In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. "
        "Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. "
        f"{ABNT_ACCESS_DATE}"
    ),
    SCRFD_URL: (
        "GUO, Jia et al. <i>Sample and computation redistribution for efficient face detection</i>. "
        "arXiv, 2021. Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. "
        f"{ABNT_ACCESS_DATE}"
    ),
    ONNXRUNTIME_EP_URL: (
        "MICROSOFT. <i>ONNX Runtime: execution providers</i>. "
        "Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. "
        f"{ABNT_ACCESS_DATE}"
    ),
    OPENCV_VIDEOCAPTURE_URL: (
        "OPENCV. <i>OpenCV 4.x documentation: VideoCapture class reference</i>. "
        "Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. "
        f"{ABNT_ACCESS_DATE}"
    ),
    MEDIAINFO_URL: (
        "MEDIAAREA. <i>MediaInfo</i>. "
        "Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. "
        f"{ABNT_ACCESS_DATE}"
    ),
    FAISS_URL: (
        "META AI. <i>FAISS</i>. "
        "Disponível em: &lt;<a href='{url}'>{url}</a>&gt;. "
        f"{ABNT_ACCESS_DATE}"
    ),
}

__all__ = [
    "ABNT_ACCESS_DATE",
    "ARCFACE_URL",
    "FAISS_URL",
    "INSIGHTFACE_URL",
    "MEDIAINFO_URL",
    "ONNXRUNTIME_EP_URL",
    "OPENCV_VIDEOCAPTURE_URL",
    "SCRFD_URL",
    "ReferenceItem",
    "abnt_reference_html",
    "build_config_help_html",
]


def build_config_help_html(
    *,
    definition: str,
    operational_effect: str,
    recommendation: str,
    caveat: str | None = None,
    references: Sequence[ReferenceItem] | None = None,
) -> str:
    """Build the HTML block used in the lower technical-help panel."""

    reference_items = tuple(references or ())
    references_html = "".join(
        f"<li>{abnt_reference_html(label, url)}</li>" for label, url in reference_items
    )
    caveat_html = f"<p><b>Observação.</b> {caveat}</p>" if caveat else ""
    references_block = (
        f"<p><b>Referências.</b></p><ul>{references_html}</ul>" if reference_items else ""
    )
    return (
        "<div style='font-family: Segoe UI, sans-serif; font-size: 10pt;'>"
        f"<p><b>Definição.</b> {definition}</p>"
        f"<p><b>Efeito operacional.</b> {operational_effect}</p>"
        f"<p><b>Recomendação técnica.</b> {recommendation}</p>"
        f"{caveat_html}"
        f"{references_block}"
        "</div>"
    )


def abnt_reference_html(label: str, url: str) -> str:
    """Render a single ABNT-style reference entry for the config help panel."""

    template = _ABNT_REFERENCE_TEMPLATES.get(
        url,
        f"{label}. Disponível em: &lt;<a href='{{url}}'>{{url}}</a>&gt;. {ABNT_ACCESS_DATE}",
    )
    return template.format(url=url)
