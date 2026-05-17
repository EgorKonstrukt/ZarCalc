from __future__ import annotations
from typing import List, Tuple
from PyQt5.QtCore import Qt, pyqtSignal, QLocale
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QComboBox, QApplication, QMessageBox,
)

_SEP_MAP = {0: "\t", 1: ",", 2: ";", 3: " "}
_MIN_ROWS = 12


def _btn(label: str, bg: str) -> QPushButton:
    b = QPushButton(label)
    b.setCursor(Qt.PointingHandCursor)
    b.setFixedHeight(24)
    return b


class DataTable(QWidget):
    """Editable two-column (x, y) table with clipboard paste support."""

    data_changed = pyqtSignal()

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        lbl = QLabel("Data points (x, y):")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self._sep = QComboBox()
        self._sep.addItems(["Tab", "Comma", "Semicolon", "Space"])
        self._sep.setFixedWidth(88)
        self._sep.setToolTip("Column separator used for paste / import")
        hdr.addWidget(QLabel("Sep:"))
        hdr.addWidget(self._sep)
        self._btn_paste = _btn("Paste", "#3b82f6")
        self._btn_paste.setToolTip("Paste two-column data from clipboard")
        self._btn_paste.clicked.connect(self._paste)
        self._btn_clear = _btn("Clear", "#ef4444")
        self._btn_clear.clicked.connect(self._clear)
        hdr.addWidget(self._btn_paste)
        hdr.addWidget(self._btn_clear)
        lay.addLayout(hdr)
        self._table = QTableWidget(_MIN_ROWS, 2)
        self._table.setHorizontalHeaderLabels(["x", "y"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self._table.setAlternatingRowColors(True)
        self._table.itemChanged.connect(self._changed)
        lay.addWidget(self._table)
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        self._btn_add = _btn("+ Row", "#6b7280")
        self._btn_add.clicked.connect(self._add_row)
        self._btn_del = _btn("- Row", "#6b7280")
        self._btn_del.clicked.connect(self._del_row)
        self._count = QLabel("0 pts")
        ctrl.addWidget(self._btn_add)
        ctrl.addWidget(self._btn_del)
        ctrl.addStretch()
        ctrl.addWidget(self._count)
        lay.addLayout(ctrl)

    def _changed(self, _item=None) -> None:
        self._count.setText(f"{len(self.get_points())} pts")
        self.data_changed.emit()

    def _add_row(self) -> None:
        self._table.setRowCount(self._table.rowCount() + 1)

    def _del_row(self) -> None:
        rows = {i.row() for i in self._table.selectedIndexes()}
        if not rows:
            rows = {self._table.rowCount() - 1}
        for r in sorted(rows, reverse=True):
            self._table.removeRow(r)
        if self._table.rowCount() < _MIN_ROWS:
            self._table.setRowCount(_MIN_ROWS)
        self.data_changed.emit()

    def _clear(self) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(_MIN_ROWS)
        for r in range(_MIN_ROWS):
            self._table.setItem(r, 0, QTableWidgetItem(""))
            self._table.setItem(r, 1, QTableWidgetItem(""))
        self._table.blockSignals(False)
        self._changed()

    def _paste(self) -> None:
        text = QApplication.clipboard().text()
        if not text.strip():
            return
        sep = _SEP_MAP[self._sep.currentIndex()]
        rows: List[Tuple[str, str]] = []
        for line in text.strip().splitlines():
            parts = line.strip().split(sep)
            if len(parts) < 2:
                parts = line.strip().split()
            if len(parts) >= 2:
                rows.append((parts[0].strip(), parts[1].strip()))
        if not rows:
            QMessageBox.warning(self, "Paste", "No two-column data found.")
            return
        self._table.blockSignals(True)
        needed = max(len(rows), _MIN_ROWS)
        self._table.setRowCount(needed)
        for r, (xv, yv) in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(xv))
            self._table.setItem(r, 1, QTableWidgetItem(yv))
        for r in range(len(rows), needed):
            self._table.setItem(r, 0, QTableWidgetItem(""))
            self._table.setItem(r, 1, QTableWidgetItem(""))
        self._table.blockSignals(False)
        self._changed()

    def get_points(self) -> List[Tuple[float, float]]:
        pts = []
        for r in range(self._table.rowCount()):
            xi = self._table.item(r, 0)
            yi = self._table.item(r, 1)
            if not xi or not yi:
                continue
            xs = xi.text().replace(",", ".").strip()
            ys = yi.text().replace(",", ".").strip()
            if not xs or not ys:
                continue
            try:
                pts.append((float(xs), float(ys)))
            except ValueError:
                continue
        return sorted(pts, key=lambda p: p[0])

    def set_points(self, pts: List[Tuple[float, float]]) -> None:
        self._table.blockSignals(True)
        needed = max(len(pts), _MIN_ROWS)
        self._table.setRowCount(needed)
        for r, (x, y) in enumerate(pts):
            self._table.setItem(r, 0, QTableWidgetItem(str(x)))
            self._table.setItem(r, 1, QTableWidgetItem(str(y)))
        for r in range(len(pts), needed):
            self._table.setItem(r, 0, QTableWidgetItem(""))
            self._table.setItem(r, 1, QTableWidgetItem(""))
        self._table.blockSignals(False)
        self._changed()

    def to_state(self) -> list:
        return [list(p) for p in self.get_points()]

    def apply_state(self, pts: list) -> None:
        self.set_points([tuple(p) for p in pts])