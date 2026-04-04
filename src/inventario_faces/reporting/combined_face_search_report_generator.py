from __future__ import annotations

from inventario_faces.domain.entities import FaceSearchResult, ReportArtifacts
from inventario_faces.domain.protocols import FaceSearchReportGenerator


class CombinedFaceSearchReportGenerator:
    def __init__(
        self,
        latex_generator: FaceSearchReportGenerator,
        docx_generator: FaceSearchReportGenerator,
    ) -> None:
        self._latex_generator = latex_generator
        self._docx_generator = docx_generator

    def generate(self, result: FaceSearchResult) -> ReportArtifacts:
        latex_artifacts = self._latex_generator.generate(result)
        docx_artifacts = self._docx_generator.generate(result)
        return ReportArtifacts(
            tex_path=latex_artifacts.tex_path,
            pdf_path=latex_artifacts.pdf_path,
            docx_path=docx_artifacts.docx_path,
        )
