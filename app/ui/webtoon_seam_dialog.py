import os

from PySide6 import QtWidgets
from PySide6.QtCore import QCoreApplication


def _t(source: str) -> str:
    """Translate a string under the "Webtoon" context."""
    return QCoreApplication.translate("Webtoon", source)


class SeamAdjustDialog(QtWidgets.QDialog):
    """Lets the user nudge each chunk-boundary of a stitched webtoon.

    In lightweight stitch mode the long image is cut into fixed-height chunks
    (one per navigable "page"). OCR/translation run per chunk, so a speech
    bubble crossing a chunk boundary gets its text split across two chunks and
    loses context. Each boundary (between chunk i and i+1) has a +-px offset;
    moving it re-cuts the chunks so the bubble lands on a single chunk. No
    content is lost or duplicated -- the offsets only move where the cut
    happens.

    The chunks are numbered from 1 and labelled by their file name (exactly
    as they appear in the page list), so no mental +1 offset is needed.
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle(_t("Adjust chunk seams"))
        self.setMinimumWidth(480)

        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            _t(
                "+ moves the cut down: the upper chunk grows.\n"
                "− moves the cut up: the lower chunk grows."
            )
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)

        chunk_files = list(getattr(controller.main, "image_files", []) or [])
        offsets = controller._chunk_boundary_offsets or []
        self.spinboxes: list[QtWidgets.QSpinBox] = []
        for i in range(len(offsets)):
            fname_a = os.path.basename(chunk_files[i]) if i < len(chunk_files) else f"chunk{i + 1}"
            fname_b = (
                os.path.basename(chunk_files[i + 1])
                if i + 1 < len(chunk_files)
                else f"chunk{i + 2}"
            )
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(
                _t("Boundary {n}: {a} ↔ {b}").format(n=i + 1, a=fname_a, b=fname_b)
            )
            sb = QtWidgets.QSpinBox()
            sb.setRange(-1000, 1000)
            sb.setValue(int(offsets[i]))
            sb.setSuffix(_t(" px"))
            sb.setToolTip(
                _t("+ moves the cut down (upper chunk grows); − moves it up (lower chunk grows).")
            )
            row.addWidget(label, 1)
            row.addWidget(sb, 0)
            inner_layout.addLayout(row)
            self.spinboxes.append(sb)

        if not self.spinboxes:
            inner_layout.addWidget(
                QtWidgets.QLabel(_t("No chunk boundaries to adjust in this stitched webtoon."))
            )

        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        btn_layout = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton(_t("Reset all"))
        cancel_btn = QtWidgets.QPushButton(_t("Cancel"))
        ok_btn = QtWidgets.QPushButton(_t("OK"))
        ok_btn.setDefault(True)
        reset_btn.clicked.connect(self._reset_all)
        cancel_btn.clicked.connect(self.reject)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        self.resize(500, min(560, 90 + len(self.spinboxes) * 36 + 80))

    def _reset_all(self):
        for sb in self.spinboxes:
            sb.setValue(0)

    def apply_offsets(self):
        """Re-cut the chunks with the chosen offsets, then redraw once."""
        values = [sb.value() for sb in self.spinboxes]
        self.controller.apply_chunk_offsets(values)
        self.controller.refresh_seam_guides()
