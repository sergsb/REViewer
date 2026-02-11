#!/usr/bin/env python3
"""REViewer — Results Explorer & Viewer for molecular generative models (such as REINVENT 4)."""

import sys
from enum import Enum, auto
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QStatusBar, QProgressBar,
    QComboBox, QSpinBox, QGroupBox, QMessageBox, QScrollArea,
    QFrame, QDoubleSpinBox, QSlider, QDialog, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPalette, QColor, QFont, QPainter, QPen, QBrush, QLinearGradient

import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem, PlotWidget
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
import umap

try:
    from reviewer import __version__, __app_name__
except ImportError:
    __version__ = "0.1.0"
    __app_name__ = "REViewer"


class AppState(Enum):
    SETUP = auto()
    VISUALIZATION = auto()
    SELECTING = auto()


# ---------------------------------------------------------------------------
# Background UMAP worker
# ---------------------------------------------------------------------------

class UMAPWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(np.ndarray, list)
    error = pyqtSignal(str)

    def __init__(self, smiles_list, n_neighbors=15, min_dist=0.1):
        super().__init__()
        self.smiles_list = smiles_list
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def run(self):
        try:
            self.progress.emit(5, "Parsing SMILES...")
            fps, valid_indices = [], []
            total = len(self.smiles_list)
            for i, smi in enumerate(self.smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    arr = np.zeros((2048,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fps.append(arr)
                    valid_indices.append(i)
                if (i + 1) % 200 == 0:
                    self.progress.emit(5 + int(25 * (i + 1) / total),
                                       f"Fingerprints: {i+1}/{total}")
            self.progress.emit(30, f"Valid molecules: {len(fps)}")
            if len(fps) < 2:
                self.error.emit("Not enough valid molecules for UMAP")
                return
            fp_matrix = np.array(fps)
            self.progress.emit(35, "Running UMAP...")
            reducer = umap.UMAP(
                n_neighbors=min(self.n_neighbors, len(fps) - 1),
                min_dist=self.min_dist, n_components=2,
                metric='jaccard', random_state=42, verbose=False,
            )
            embedding = reducer.fit_transform(fp_matrix)
            self.progress.emit(95, "Finalizing...")
            full = np.full((len(self.smiles_list), 2), np.nan)
            for i, idx in enumerate(valid_indices):
                full[idx] = embedding[i]
            self.progress.emit(100, "Complete!")
            self.finished.emit(full, valid_indices)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Molecule display widgets
# ---------------------------------------------------------------------------

IMG_RENDER_W, IMG_RENDER_H = 450, 300   # render resolution 3:2
IMG_DISPLAY_W, IMG_DISPLAY_H = 180, 120  # display size 3:2

class MoleculeCard(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "MoleculeCard{background:white;border:1px solid #c0c0c0;border-radius:6px}"
            "MoleculeCard:hover{border:2px solid #3498db}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 4)
        lay.setSpacing(2)
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img.setFixedSize(IMG_DISPLAY_W, IMG_DISPLAY_H)
        lay.addWidget(self.img)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("font-size:10px;color:#444;border:none")
        lay.addWidget(self.info)

    def set_molecule(self, smiles, info_dict=None):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.img.setText("Invalid")
            return
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(IMG_RENDER_W, IMG_RENDER_H)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2.1
        opts.padding = 0.1
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        qi = QImage()
        qi.loadFromData(png)
        pm = QPixmap.fromImage(qi).scaled(
            IMG_DISPLAY_W, IMG_DISPLAY_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.img.setPixmap(pm)
        if info_dict:
            self.info.setText("\n".join(f"{k}: {v}" for k, v in info_dict.items()))
        else:
            self.info.setText(smiles[:50])


class MoleculePanel(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(180)
        self.setMaximumWidth(200)
        self.setStyleSheet("QScrollArea{background:#f5f5f5;border:none}")
        self.container = QWidget()
        self.lay = QVBoxLayout(self.container)
        self.lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.lay.setSpacing(4)
        self.lay.setContentsMargins(4, 4, 4, 4)
        self.setWidget(self.container)
        self.cards = []
        self.placeholder = QLabel("Hover over points\nto see molecules")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color:#888;font-size:12px;padding:20px")
        self.lay.addWidget(self.placeholder)

    def show_molecules(self, smiles_list, info_list=None):
        for c in self.cards:
            c.setParent(None); c.deleteLater()
        self.cards = []
        self.placeholder.setVisible(not smiles_list)
        for i, smi in enumerate(smiles_list[:4]):
            card = MoleculeCard()
            card.set_molecule(smi, info_list[i] if info_list else None)
            self.lay.addWidget(card)
            self.cards.append(card)

    def clear(self):
        for c in self.cards:
            c.setParent(None); c.deleteLater()
        self.cards = []
        self.placeholder.setVisible(True)


# ---------------------------------------------------------------------------
# Scatter plot with lasso selection
# ---------------------------------------------------------------------------

class ScatterPlot(PlotWidget):
    points_hovered = pyqtSignal(list)
    selection_changed = pyqtSignal(list)

    SEL_LASSO = 'lasso'
    SEL_RECT = 'rect'

    def __init__(self):
        super().__init__()
        self.setBackground('w')
        self.getPlotItem().getViewBox().setBackgroundColor('w')
        self.scatter = ScatterPlotItem(size=8, pen=pg.mkPen(None),
                                       brush=pg.mkBrush(70, 130, 180, 180))
        self.addItem(self.scatter)
        self.sel_scatter = ScatterPlotItem(size=12, pen=pg.mkPen('r', width=2),
                                           brush=pg.mkBrush(255, 100, 100, 200))
        self.addItem(self.sel_scatter)
        self.hover_scatter = ScatterPlotItem(size=14, pen=pg.mkPen('#ff6600', width=2),
                                             brush=pg.mkBrush(255, 165, 0, 180))
        self.addItem(self.hover_scatter)
        self.data_x = np.array([])
        self.data_y = np.array([])
        self.valid_mask = np.array([])
        self.range_mask = None
        self.selected_indices = []
        self.pt_size = 8
        self.sel_mode = False
        self.sel_tool = self.SEL_LASSO
        self.shape_item = None
        self.lasso_pts = []
        self.rect_origin = None
        self.drawing = False
        self.setLabel('bottom', 'UMAP 1')
        self.setLabel('left', 'UMAP 2')
        self.setTitle('Molecular Space')
        self.vb = self.getPlotItem().getViewBox()
        self.scene().sigMouseMoved.connect(self._on_move)
        self.scene().sigMouseClicked.connect(self._on_click)

    @staticmethod
    def _point_size(n):
        for limit, sz in [(100, 12), (500, 10), (1000, 8), (5000, 6), (10000, 4)]:
            if n < limit:
                return sz
        return 3

    def set_data(self, x, y, mask, brushes=None):
        self.data_x, self.data_y, self.valid_mask = x, y, mask
        vx, vy = x[mask], y[mask]
        self.pt_size = self._point_size(len(vx))
        kw = dict(x=vx, y=vy, size=self.pt_size)
        if brushes:
            kw['brush'] = brushes
        self.scatter.setData(**kw)
        self.clear_selection()

    def update_colors(self, brushes):
        vx, vy = self.data_x[self.valid_mask], self.data_y[self.valid_mask]
        self.scatter.setData(x=vx, y=vy, brush=brushes, size=self.pt_size)

    def set_range_mask(self, mask):
        """Set boolean mask (over full data) for property-range filtering of hover."""
        self.range_mask = mask

    def _on_move(self, pos):
        if len(self.data_x) == 0:
            return
        mp = self.vb.mapSceneToView(pos)
        mx, my = mp.x(), mp.y()
        if self.sel_mode and self.drawing:
            if self.sel_tool == self.SEL_LASSO:
                self.lasso_pts.append([mx, my])
                self._draw_lasso()
            elif self.sel_tool == self.SEL_RECT:
                self._draw_rect(mx, my)
            return
        if not self.sel_mode:
            vr = self.viewRange()
            xr = vr[0][1] - vr[0][0]
            yr = vr[1][1] - vr[1][0]
            if xr <= 0 or yr <= 0:
                return
            d = np.sqrt(((self.data_x - mx) / xr) ** 2 + ((self.data_y - my) / yr) ** 2)
            d[~self.valid_mask] = np.inf
            if self.range_mask is not None:
                d[~self.range_mask] = np.inf
            nearby = np.where(d < 0.02)[0]
            if len(nearby) > 0:
                nearby = nearby[np.argsort(d[nearby])][:4]
                idx = nearby.tolist()
                self.hover_scatter.setData(
                    x=self.data_x[idx], y=self.data_y[idx],
                    size=self.pt_size + 6)
                self.points_hovered.emit(idx)
            else:
                self.hover_scatter.setData(x=[], y=[])
                self.points_hovered.emit([])

    def _on_click(self, event):
        if not self.sel_mode:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            p = self.vb.mapSceneToView(event.scenePos())
            if not self.drawing:
                self.drawing = True
                if self.sel_tool == self.SEL_LASSO:
                    self.lasso_pts = [[p.x(), p.y()]]
                else:
                    self.rect_origin = (p.x(), p.y())
            else:
                self.drawing = False
                if self.sel_tool == self.SEL_LASSO:
                    if len(self.lasso_pts) > 3:
                        self.lasso_pts.append(self.lasso_pts[0])
                        self._finish_lasso()
                else:
                    self._finish_rect(p.x(), p.y())
                self._clear_shape()
        elif event.button() == Qt.MouseButton.RightButton:
            self.drawing = False
            self.lasso_pts = []
            self.rect_origin = None
            self._clear_shape()

    # --- Drawing helpers ---

    def _draw_lasso(self):
        if len(self.lasso_pts) < 2:
            return
        self._clear_shape()
        pts = np.array(self.lasso_pts)
        self.shape_item = pg.PlotDataItem(pts[:, 0], pts[:, 1],
                                          pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
        self.addItem(self.shape_item)

    def _draw_rect(self, mx, my):
        if self.rect_origin is None:
            return
        self._clear_shape()
        ox, oy = self.rect_origin
        xs = [ox, mx, mx, ox, ox]
        ys = [oy, oy, my, my, oy]
        self.shape_item = pg.PlotDataItem(xs, ys,
                                          pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
        self.addItem(self.shape_item)

    def _clear_shape(self):
        if self.shape_item:
            self.removeItem(self.shape_item)
            self.shape_item = None

    # --- Selection logic ---

    def start_selection(self, tool=None):
        if tool:
            self.sel_tool = tool
        self.sel_mode = True
        self.drawing = False
        self.lasso_pts = []
        self.rect_origin = None
        self.vb.setMouseEnabled(x=False, y=False)
        label = 'LASSO' if self.sel_tool == self.SEL_LASSO else 'RECTANGLE'
        self.setTitle(f'{label} — Click to start, click to finish')

    def stop_selection(self):
        self.sel_mode = False
        self.drawing = False
        self.vb.setMouseEnabled(x=True, y=True)
        self.setTitle('Molecular Space')
        self._clear_shape()

    def _apply_selection(self, indices):
        self.selected_indices = indices
        if indices:
            self.sel_scatter.setData(x=self.data_x[indices], y=self.data_y[indices],
                                     size=self.pt_size + 4)
        else:
            self.sel_scatter.setData(x=[], y=[])
        self.selection_changed.emit(indices)

    def _finish_lasso(self):
        if len(self.lasso_pts) < 3:
            return
        try:
            from matplotlib.path import Path as MplPath
            path = MplPath(self.lasso_pts)
            sel = [i for i in range(len(self.data_x))
                   if self.valid_mask[i] and path.contains_point((self.data_x[i], self.data_y[i]))]
            self._apply_selection(sel)
        except Exception as e:
            print(f"Selection error: {e}")

    def _finish_rect(self, mx, my):
        if self.rect_origin is None:
            return
        ox, oy = self.rect_origin
        x0, x1 = min(ox, mx), max(ox, mx)
        y0, y1 = min(oy, my), max(oy, my)
        sel = [i for i in range(len(self.data_x))
               if self.valid_mask[i]
               and x0 <= self.data_x[i] <= x1
               and y0 <= self.data_y[i] <= y1]
        self._apply_selection(sel)

    def clear_selection(self):
        self.selected_indices = []
        self.sel_scatter.setData(x=[], y=[])
        self.hover_scatter.setData(x=[], y=[])
        self.lasso_pts = []
        self.rect_origin = None
        self.drawing = False
        self.selection_changed.emit([])


# ---------------------------------------------------------------------------
# Sampling strategy dialog
# ---------------------------------------------------------------------------

class SamplingDialog(QDialog):
    """Dialog to choose sampling strategy when dataset > MAX_MOLECULES.
    Also detects 'found' column and offers pre-filtering to in-space molecules."""

    def __init__(self, total, max_n, columns, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset too large — choose sampling")
        self.setMinimumWidth(460)
        self.total = total
        self.max_n = max_n
        self.has_found = 'found' in columns
        self.n_found = int((df['found'] == 1).sum()) if self.has_found else 0

        lo = QVBoxLayout(self)
        lo.setSpacing(10)
        lo.setContentsMargins(20, 15, 20, 15)

        lo.addWidget(QLabel(
            f"Dataset has <b>{total:,}</b> molecules after deduplication.<br>"
            f"Maximum for UMAP is <b>{max_n:,}</b>."))

        # --- Found pre-filter ---
        if self.has_found:
            found_box = QGroupBox("Chemical Space Filter")
            found_box.setStyleSheet(
                "QGroupBox{font-weight:bold;border:1px solid #3498db;"
                "border-radius:4px;margin-top:6px;padding-top:8px}"
                "QGroupBox::title{color:#3498db}")
            fl = QVBoxLayout(found_box)
            self.radio_all_mols = QRadioButton(f"All molecules ({total:,})")
            self.radio_found_only = QRadioButton(
                f"Only in-space / found=1 ({self.n_found:,})")
            self.radio_all_mols.setChecked(True)
            self.radio_all_mols.toggled.connect(self._on_found_toggle)
            fl.addWidget(self.radio_all_mols)
            fl.addWidget(self.radio_found_only)
            if self.n_found <= max_n:
                hint = QLabel(f"  → in-space fits without sampling ({self.n_found:,} ≤ {max_n:,})")
                hint.setStyleSheet("color:#27ae60;font-size:10px")
                fl.addWidget(hint)
            lo.addWidget(found_box)
        else:
            warn = QLabel("No 'found' column — raw REINVENT data, "
                          "cannot filter by chemical space.")
            warn.setWordWrap(True)
            warn.setStyleSheet("background:#fff3cd;color:#856404;border:1px solid #ffc107;"
                               "border-radius:4px;padding:6px;font-size:11px")
            lo.addWidget(warn)

        # --- Sampling strategy (may be hidden if found-only fits) ---
        self.sampling_box = QGroupBox("Sampling Strategy")
        self.sampling_box.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #888;"
            "border-radius:4px;margin-top:6px;padding-top:8px}"
            "QGroupBox::title{color:#555}")
        sl = QVBoxLayout(self.sampling_box)

        self.radio_random = QRadioButton("Random sample")
        self.radio_random.setChecked(True)
        self.radio_random.toggled.connect(self._on_strategy_toggle)
        sl.addWidget(self.radio_random)

        self.radio_proportion = QRadioButton("Proportional (preserve in-space ratio)")
        self.radio_proportion.toggled.connect(self._on_strategy_toggle)
        sl.addWidget(self.radio_proportion)
        if not self.has_found:
            self.radio_proportion.setEnabled(False)
            self.radio_proportion.setToolTip("Requires 'found' column")

        # Proportion slider
        self.prop_row = QWidget()
        prl = QHBoxLayout(self.prop_row)
        prl.setContentsMargins(24, 0, 0, 0)
        prl.addWidget(QLabel("In-space %:"))
        self.prop_slider = QSlider(Qt.Orientation.Horizontal)
        self.prop_slider.setRange(1, 99)
        default_pct = round(100.0 * self.n_found / max(total, 1)) if self.has_found else 50
        self.prop_slider.setValue(max(1, min(99, default_pct)))
        prl.addWidget(self.prop_slider)
        self.prop_val_lbl = QLabel(f"{self.prop_slider.value()}%")
        self.prop_val_lbl.setMinimumWidth(36)
        self.prop_slider.valueChanged.connect(
            lambda v: self.prop_val_lbl.setText(f"{v}%"))
        prl.addWidget(self.prop_val_lbl)
        sl.addWidget(self.prop_row)
        self.prop_row.setVisible(False)

        self.radio_activity = QRadioButton("Activity-biased (keep most active)")
        self.radio_activity.toggled.connect(self._on_strategy_toggle)
        sl.addWidget(self.radio_activity)

        # Activity options
        self.act_box = QWidget()
        al = QHBoxLayout(self.act_box)
        al.setContentsMargins(24, 0, 0, 0)
        al.addWidget(QLabel("Activity column:"))
        self.act_combo = QComboBox()
        self.act_combo.setMinimumWidth(160)
        for c in columns:
            self.act_combo.addItem(c)
        for i, c in enumerate(columns):
            cl = c.lower()
            if any(kw in cl for kw in ['activity', 'score', 'pchembl', 'pic50', 'potency']):
                self.act_combo.setCurrentIndex(i)
                break
        al.addWidget(self.act_combo)
        al.addStretch()
        sl.addWidget(self.act_box)
        self.act_box.setVisible(False)

        self.top_pct_lbl = QLabel("Top active to keep: 30%  — random fill rest")
        self.top_pct_lbl.setStyleSheet("color:#666;font-size:11px;margin-left:24px")
        sl.addWidget(self.top_pct_lbl)
        self.top_pct_lbl.setVisible(False)

        self.top_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_slider.setRange(10, 80)
        self.top_slider.setValue(30)
        self.top_slider.setStyleSheet("margin-left:24px")
        self.top_slider.valueChanged.connect(
            lambda v: self.top_pct_lbl.setText(f"Top active to keep: {v}%  — random fill rest"))
        sl.addWidget(self.top_slider)
        self.top_slider.setVisible(False)

        lo.addWidget(self.sampling_box)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok = QPushButton("Continue")
        ok.setStyleSheet("QPushButton{background:#27ae60;color:white;padding:8px 20px;"
                         "border:none;border-radius:4px;font-weight:bold}")
        ok.clicked.connect(self.accept)
        btn_row.addWidget(ok)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)
        lo.addLayout(btn_row)

        # initial state
        self._on_found_toggle()

    def _on_found_toggle(self):
        """Show/hide sampling box based on whether found-only fits."""
        if self.has_found and hasattr(self, 'radio_found_only') and self.radio_found_only.isChecked():
            needs_sampling = self.n_found > self.max_n
        else:
            needs_sampling = True
        self.sampling_box.setVisible(needs_sampling)

    def _on_strategy_toggle(self):
        biased = self.radio_activity.isChecked()
        self.act_box.setVisible(biased)
        self.top_pct_lbl.setVisible(biased)
        self.top_slider.setVisible(biased)
        self.prop_row.setVisible(self.radio_proportion.isChecked())

    def apply_sampling(self, df):
        """Apply pre-filter and sampling strategy, return sampled DataFrame."""
        # step 1: pre-filter by found if requested
        if self.has_found and self.radio_found_only.isChecked():
            df = df[df['found'] == 1].reset_index(drop=True)
            if len(df) <= self.max_n:
                return df  # fits without sampling

        # step 2: apply sampling strategy
        if self.radio_random.isChecked():
            return df.sample(n=self.max_n, random_state=42).reset_index(drop=True)

        if self.radio_proportion.isChecked() and 'found' in df.columns:
            target_pct = self.prop_slider.value() / 100.0
            found_mask = (df['found'] == 1)
            n_in_total = int(found_mask.sum())
            n_out_total = len(df) - n_in_total
            n_in_keep = min(n_in_total, int(self.max_n * target_pct))
            n_out_keep = min(n_out_total, self.max_n - n_in_keep)
            # adjust if we couldn't get enough of one side
            n_in_keep = min(n_in_total, self.max_n - n_out_keep)
            in_sample = df[found_mask].sample(n=n_in_keep, random_state=42)
            out_sample = df[~found_mask].sample(n=n_out_keep, random_state=42)
            return pd.concat([in_sample, out_sample]).reset_index(drop=True)

        # Activity-biased sampling
        col = self.act_combo.currentText()
        top_pct = self.top_slider.value() / 100.0
        n_top = int(self.max_n * top_pct)
        n_random = self.max_n - n_top

        vals = pd.to_numeric(df[col], errors='coerce')
        ranked = vals.sort_values(ascending=False, na_position='last')
        top_idx = ranked.index[:n_top].tolist()

        remaining = df.index.difference(top_idx)
        if len(remaining) > 0 and n_random > 0:
            random_idx = df.loc[remaining].sample(
                n=min(n_random, len(remaining)), random_state=42).index.tolist()
        else:
            random_idx = []

        return df.loc[top_idx + random_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Setup wizard dialog
# ---------------------------------------------------------------------------

class SetupWizard(QDialog):
    setup_complete = pyqtSignal(pd.DataFrame, str, np.ndarray, list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{__app_name__} — Setup")
        self.setMinimumSize(500, 400)
        self.setModal(True)
        self.df = None
        self.file_path = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        lo = QVBoxLayout(self)
        lo.setSpacing(15)
        lo.setContentsMargins(30, 20, 30, 20)

        title = QLabel(f"{__app_name__} — Setup")
        title.setStyleSheet("font-size:22px;font-weight:bold;color:#2c3e50")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lo.addWidget(title)

        # --- File ---
        fg = QGroupBox("Step 1 — Load CSV")
        fl = QVBoxLayout(fg)
        self.load_btn = QPushButton("Open CSV File...")
        self.load_btn.clicked.connect(self._load)
        fl.addWidget(self.load_btn)
        self.file_lbl = QLabel("No file loaded")
        self.file_lbl.setStyleSheet("color:#888")
        fl.addWidget(self.file_lbl)
        row = QHBoxLayout()
        row.addWidget(QLabel("SMILES column:"))
        self.smiles_cb = QComboBox()
        self.smiles_cb.setEnabled(False)
        self.smiles_cb.setMinimumWidth(180)
        row.addWidget(self.smiles_cb)
        row.addStretch()
        fl.addLayout(row)
        lo.addWidget(fg)

        # --- UMAP ---
        ug = QGroupBox("Step 2 — UMAP")
        ul = QVBoxLayout(ug)
        pr = QHBoxLayout()
        pr.addWidget(QLabel("n_neighbors:"))
        self.nn = QSpinBox()
        self.nn.setRange(2, 200)
        self.nn.setValue(15)
        pr.addWidget(self.nn)
        pr.addSpacing(15)
        pr.addWidget(QLabel("min_dist:"))
        self.md = QDoubleSpinBox()
        self.md.setRange(0.0, 1.0)
        self.md.setValue(0.1)
        self.md.setSingleStep(0.05)
        pr.addWidget(self.md)
        pr.addStretch()
        ul.addLayout(pr)
        self.go_btn = QPushButton("Calculate UMAP & Start")
        self.go_btn.setEnabled(False)
        self.go_btn.setStyleSheet("QPushButton{background:#27ae60;color:white;padding:10px;"
                                  "border:none;border-radius:5px;font-weight:bold}"
                                  "QPushButton:disabled{background:#bdc3c7}")
        self.go_btn.clicked.connect(self._calc)
        ul.addWidget(self.go_btn)
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        ul.addWidget(self.pbar)
        self.plbl = QLabel("")
        self.plbl.setStyleSheet("color:#666;font-size:11px")
        self.plbl.setVisible(False)
        ul.addWidget(self.plbl)
        lo.addWidget(ug)
        lo.addStretch()

    def _load(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv);;All (*)")
        if not fp:
            return
        try:
            self.df = pd.read_csv(fp)
            self.file_path = fp
            self.file_lbl.setText(f"✓ {Path(fp).name}  ({len(self.df)} rows)")
            self.file_lbl.setStyleSheet("color:#27ae60;font-weight:bold")
            self.smiles_cb.clear()
            self.smiles_cb.setEnabled(True)
            for c in self.df.columns:
                self.smiles_cb.addItem(c)
            for i, c in enumerate(self.df.columns):
                if 'smiles' in c.lower():
                    self.smiles_cb.setCurrentIndex(i)
                    break
            self.go_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    MAX_MOLECULES = 10000

    def _calc(self):
        if self.df is None:
            return
        scol = self.smiles_cb.currentText()
        # deduplicate
        n0 = len(self.df)
        self.df = self.df.drop_duplicates(subset=[scol], keep='first').reset_index(drop=True)
        # sample if too large
        if len(self.df) > self.MAX_MOLECULES:
            dlg = SamplingDialog(len(self.df), self.MAX_MOLECULES,
                                 self.df.columns.tolist(), self.df, parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            self.df = dlg.apply_sampling(self.df)
        info = f"✓ {Path(self.file_path).name}  ({len(self.df)} molecules"
        if len(self.df) < n0:
            info += f", was {n0}"
        info += ")"
        self.file_lbl.setText(info)
        self.file_lbl.setStyleSheet("color:#27ae60;font-weight:bold")
        slist = self.df[scol].astype(str).tolist()
        self.go_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.pbar.setVisible(True)
        self.plbl.setVisible(True)
        self.worker = UMAPWorker(slist, self.nn.value(), self.md.value())
        self.worker.progress.connect(lambda v, m: (self.pbar.setValue(v), self.plbl.setText(m)))
        self.worker.finished.connect(lambda emb, vi: (
            self.setup_complete.emit(self.df, self.file_path, emb, vi, scol),
            self.accept(),
        ))
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_err(self, msg):
        self.go_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.pbar.setVisible(False)
        self.plbl.setVisible(False)
        QMessageBox.critical(self, "UMAP Error", msg)


# ---------------------------------------------------------------------------
# Dual-handle range slider with colored track
# ---------------------------------------------------------------------------

class RangeSlider(QWidget):
    """Dual-handle slider: blue handle (low), red handle (high), gradient between."""
    range_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedHeight(40)
        self.setMinimumWidth(250)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lo = 0.0   # normalized 0..1
        self.hi = 1.0
        self._drag = None  # 'lo', 'hi', or None
        self._margin = 12  # px margin for handles

    def set_range(self, lo, hi):
        self.lo = max(0.0, min(lo, 1.0))
        self.hi = max(0.0, min(hi, 1.0))
        if self.lo > self.hi:
            self.lo, self.hi = self.hi, self.lo
        self.update()

    def _x(self, t):
        """Normalized t (0..1) to pixel x."""
        return self._margin + t * (self.width() - 2 * self._margin)

    def _t(self, x):
        """Pixel x to normalized t (0..1)."""
        w = self.width() - 2 * self._margin
        return max(0.0, min(1.0, (x - self._margin) / w)) if w > 0 else 0.0

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = 16
        tl, tr = self._x(0), self._x(1)
        lx, hx = self._x(self.lo), self._x(self.hi)
        # gray track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(210, 210, 210)))
        p.drawRoundedRect(int(tl), cy - 3, int(tr - tl), 6, 3, 3)
        # colored range
        if hx > lx:
            grad = QLinearGradient(lx, 0, hx, 0)
            grad.setColorAt(0, QColor(52, 152, 219))
            grad.setColorAt(1, QColor(231, 76, 60))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(int(lx), cy - 3, int(hx - lx), 6, 3, 3)
        # handles
        for val, color in [(self.lo, QColor(52, 152, 219)), (self.hi, QColor(231, 76, 60))]:
            x = self._x(val)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(QBrush(color))
            p.drawEllipse(int(x) - 7, cy - 7, 14, 14)
        # value labels
        p.setPen(QColor(80, 80, 80))
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(int(lx) - 15, self.height() - 2, f"{self.lo:.2f}")
        p.drawText(int(hx) - 15, self.height() - 2, f"{self.hi:.2f}")
        p.end()

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        x = ev.pos().x()
        dl = abs(x - self._x(self.lo))
        dh = abs(x - self._x(self.hi))
        if dl <= dh:
            self._drag = 'lo'
        else:
            self._drag = 'hi'
        self._move(x)

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._move(ev.pos().x())

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def _move(self, x):
        t = self._t(x)
        eps = 0.005
        if self._drag == 'lo':
            self.lo = min(t, self.hi - eps)
        elif self._drag == 'hi':
            self.hi = max(t, self.lo + eps)
        self.update()
        self.range_changed.emit()


# ---------------------------------------------------------------------------
# Property toolbar (above the map)
# ---------------------------------------------------------------------------

class PropertyBar(QWidget):
    """Horizontal bar: property combo + dual-handle RangeSlider."""
    combo_changed = pyqtSignal()
    slider_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedHeight(50)
        self.is_numeric = False
        self.data_min = 0.0
        self.data_max = 1.0

        lo = QHBoxLayout(self)
        lo.setContentsMargins(8, 4, 8, 4)
        lo.setSpacing(10)

        lo.addWidget(QLabel("Color by:"))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(140)
        self.combo.addItem("None")
        self.combo.currentTextChanged.connect(self._on_combo)
        lo.addWidget(self.combo)

        self.slider = RangeSlider()
        self.slider.range_changed.connect(self._on_slider)
        lo.addWidget(self.slider, stretch=1)

        self.range_lbl = QLabel("")
        self.range_lbl.setMinimumWidth(110)
        self.range_lbl.setStyleSheet("color:#555;font-size:10px")
        lo.addWidget(self.range_lbl)

        self.slider.setVisible(False)
        self.range_lbl.setVisible(False)

    def set_columns(self, cols):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("None")
        for c in cols:
            self.combo.addItem(c)
        self.combo.blockSignals(False)

    def configure(self, values):
        if values is None:
            self.is_numeric = False
            self.slider.setVisible(False)
            self.range_lbl.setVisible(False)
            return
        try:
            nv = pd.to_numeric(values, errors='coerce')
            valid = nv[~np.isnan(nv)]
            if len(valid) > 0 and len(valid) / len(values) > 0.5:
                self.is_numeric = True
                self.data_min = float(np.nanmin(valid))
                self.data_max = float(np.nanmax(valid))
                self.slider.set_range(0.0, 1.0)
                self._update_range_label()
                self.slider.setVisible(True)
                self.range_lbl.setVisible(True)
                return
        except Exception:
            pass
        self.is_numeric = False
        self.slider.setVisible(False)
        self.range_lbl.setVisible(False)

    def _update_range_label(self):
        self.range_lbl.setText(f"{self.get_low():.2f} — {self.get_high():.2f}")

    def get_low(self):
        return self.data_min + self.slider.lo * (self.data_max - self.data_min)

    def get_high(self):
        return self.data_min + self.slider.hi * (self.data_max - self.data_min)

    def _on_combo(self, _text):
        self.combo_changed.emit()

    def _on_slider(self):
        self._update_range_label()
        self.slider_changed.emit()

    def current_property(self):
        return self.combo.currentText()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MolExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{__app_name__} — Results Explorer & Viewer")
        self.setGeometry(100, 100, 1500, 900)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        self.setPalette(pal)

        self.state = AppState.SETUP
        self.df = None
        self.df_full = None
        self.scol = None
        self.has_found = False
        self.smiles_list = []
        self.embedding = None
        self.valid_indices = []
        self.valid_mask = np.array([])

        self._build_ui()
        QTimer.singleShot(100, self._show_wizard)

    # --- UI setup ---

    def _build_ui(self):
        cw = QWidget()
        cw.setStyleSheet("background:white")
        self.setCentralWidget(cw)
        main = QHBoxLayout(cw)
        main.setContentsMargins(8, 8, 8, 8)

        # Left panel
        left = QWidget()
        left.setMaximumWidth(260)
        left.setMinimumWidth(240)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        info_lbl = QLabel(
            f"<b>{__app_name__}</b> v{__version__}<br>"
            "<span style='font-size:10px'><b> Molecular Generators Results Explorer & Viewer</b><br><br>"
            "This program is designed for "
            "analysis of the chemical space generated by molecular generators. "
            "Currently, output files from REINVENT 4 are supported.<br><br>"
            "</span>")
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet(
            "background:#eaf2f8;color:#2c3e50;border:1px solid #aed6f1;"
            "border-radius:4px;padding:8px;font-size:11px")
        ll.addWidget(info_lbl)

        self.file_lbl = QLabel("")
        self.file_lbl.setWordWrap(True)
        self.file_lbl.setStyleSheet("color:#27ae60;font-size:11px;padding:4px")
        ll.addWidget(self.file_lbl)

        # Space mode panel — warning or toggle
        self.space_warn = QLabel(
            "Raw REINVENT data\nNo 'found' column — cannot\n"
            "determine which molecules\nexist in target chemical space.")
        self.space_warn.setWordWrap(True)
        self.space_warn.setStyleSheet(
            "background:#fff3cd;color:#856404;border:1px solid #ffc107;"
            "border-radius:4px;padding:8px;font-size:11px")
        ll.addWidget(self.space_warn)
        self.space_warn.setVisible(False)

        self.space_toggle_box = QGroupBox("Chemical Space")
        self.space_toggle_box.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #3498db;"
            "border-radius:4px;margin-top:8px;padding-top:8px}"
            "QGroupBox::title{color:#3498db}")
        stl = QVBoxLayout(self.space_toggle_box)
        self.radio_all = QRadioButton("All molecules")
        self.radio_all.setChecked(True)
        self.radio_inspace = QRadioButton("Only in-space (found=1)")
        self.radio_all.toggled.connect(self._on_space_mode_changed)
        stl.addWidget(self.radio_all)
        stl.addWidget(self.radio_inspace)
        self.space_count_lbl = QLabel("")
        self.space_count_lbl.setStyleSheet("color:#555;font-size:10px")
        stl.addWidget(self.space_count_lbl)
        ll.addWidget(self.space_toggle_box)
        self.space_toggle_box.setVisible(False)

        # Selection group
        sg = QGroupBox("Selection")
        sg.setStyleSheet("QGroupBox{font-weight:bold;border:1px solid #27ae60;"
                         "border-radius:4px;margin-top:8px;padding-top:8px}"
                         "QGroupBox::title{color:#27ae60}")
        sl = QVBoxLayout(sg)

        tool_row = QHBoxLayout()
        self.lasso_btn = QPushButton("Lasso")
        self.lasso_btn.setCheckable(True)
        self.lasso_btn.setStyleSheet("QPushButton{background:#27ae60;color:white;padding:6px;"
                                     "border:none;border-radius:4px}"
                                     "QPushButton:checked{background:#e74c3c}")
        self.lasso_btn.clicked.connect(lambda: self._toggle_sel(ScatterPlot.SEL_LASSO))
        tool_row.addWidget(self.lasso_btn)
        self.rect_btn = QPushButton("Rectangle")
        self.rect_btn.setCheckable(True)
        self.rect_btn.setStyleSheet("QPushButton{background:#27ae60;color:white;padding:6px;"
                                    "border:none;border-radius:4px}"
                                    "QPushButton:checked{background:#e74c3c}")
        self.rect_btn.clicked.connect(lambda: self._toggle_sel(ScatterPlot.SEL_RECT))
        tool_row.addWidget(self.rect_btn)
        sl.addLayout(tool_row)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_sel)
        sl.addWidget(self.clear_btn)
        self.sel_lbl = QLabel("Selected: 0")
        self.sel_lbl.setStyleSheet("font-weight:bold;color:#e74c3c")
        sl.addWidget(self.sel_lbl)
        self.sel_cond_lbl = QLabel("")
        self.sel_cond_lbl.setWordWrap(True)
        self.sel_cond_lbl.setStyleSheet("color:#555;font-size:10px")
        sl.addWidget(self.sel_cond_lbl)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        sl.addWidget(self.export_btn)
        ll.addWidget(sg)
        self.sel_group = sg
        ll.addStretch()

        # Center
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)

        self.prop_bar = PropertyBar()
        self.prop_bar.combo_changed.connect(self._on_combo_changed)
        self.prop_bar.slider_changed.connect(self._update_colors)
        cl.addWidget(self.prop_bar)

        self.plot = ScatterPlot()
        self.plot.points_hovered.connect(self._on_hover)
        self.plot.selection_changed.connect(self._on_sel_changed)
        cl.addWidget(self.plot, stretch=1)

        # Right
        self.mol_panel = MoleculePanel()

        main.addWidget(left)
        main.addWidget(center, stretch=1)
        main.addWidget(self.mol_panel)

        self.sbar = QStatusBar()
        self.setStatusBar(self.sbar)
        self.sbar.showMessage("Ready")

        # initial visibility
        self.prop_bar.setVisible(False)
        self.sel_group.setVisible(False)
        self.mol_panel.setVisible(False)

    # --- State transitions ---

    def _set_state(self, s):
        self.state = s
        vis = s in (AppState.VISUALIZATION, AppState.SELECTING)
        self.prop_bar.setVisible(vis)
        self.sel_group.setVisible(vis)
        self.mol_panel.setVisible(vis)

    # --- Wizard ---

    def _show_wizard(self):
        self.hide()
        w = SetupWizard(self)
        w.setup_complete.connect(self._on_setup)
        result = w.exec()
        if result == QDialog.DialogCode.Rejected and self.df is None:
            sys.exit(0)
        self.show()

    def _on_setup(self, df, fpath, embedding, vi, scol):
        self.scol = scol
        # store full (unfiltered) data
        self.df_full = df
        self.embedding_full = embedding
        self.valid_mask_full = ~np.isnan(embedding[:, 0])
        self.has_found = 'found' in df.columns

        # configure space panel
        if self.has_found:
            n_in = int((df['found'] == 1).sum())
            self.space_warn.setVisible(False)
            self.space_toggle_box.setVisible(True)
            self.space_count_lbl.setText(
                f"{n_in} in-space / {len(df)} total")
            self.radio_all.setChecked(True)
        else:
            self.space_warn.setVisible(True)
            self.space_toggle_box.setVisible(False)

        # apply current mode (defaults to "all")
        self._apply_space_filter()

        self.file_lbl.setText(f"✓ {Path(fpath).name}\n{len(self.df)} molecules")
        self.prop_bar.set_columns(self.df.columns.tolist())

        brushes = self._make_brushes()
        self.plot.set_data(self.embedding[:, 0], self.embedding[:, 1],
                           self.valid_mask, brushes)

        self._set_state(AppState.VISUALIZATION)
        self.sbar.showMessage(f"Visualizing {int(self.valid_mask.sum())} molecules")

    def _apply_space_filter(self):
        """Set self.df / embedding / masks based on space mode toggle."""
        if self.has_found and self.radio_inspace.isChecked():
            mask = (self.df_full['found'] == 1).values
            self.df = self.df_full[mask].reset_index(drop=True)
            self.embedding = self.embedding_full[mask]
        else:
            self.df = self.df_full
            self.embedding = self.embedding_full
        self.smiles_list = self.df[self.scol].astype(str).tolist()
        self.valid_mask = ~np.isnan(self.embedding[:, 0])

    def _on_space_mode_changed(self):
        """Called when All / Only in-space radio toggles."""
        if self.df_full is None:
            return
        self._apply_space_filter()

        self.file_lbl.setText(
            f"{'In-space' if self.radio_inspace.isChecked() else 'All'}: "
            f"{len(self.df)} molecules")
        self.prop_bar.set_columns(self.df.columns.tolist())

        # re-configure property bar if a property is selected
        col = self.prop_bar.current_property()
        if col != "None" and col in self.df.columns:
            self.prop_bar.configure(self.df[col].values)

        brushes = self._make_brushes()
        self.plot.set_data(self.embedding[:, 0], self.embedding[:, 1],
                           self.valid_mask, brushes)
        self._push_range_mask()
        self.sbar.showMessage(f"Showing {int(self.valid_mask.sum())} molecules")

    # --- Color logic ---

    def _on_combo_changed(self):
        """Called when property combo changes — reconfigure sliders."""
        if self.embedding is None or not self.valid_mask.any():
            return
        col = self.prop_bar.current_property()
        if col != "None" and col in self.df.columns:
            self.prop_bar.configure(self.df[col].values)
        else:
            self.prop_bar.configure(None)
        self._update_colors()

    def _update_colors(self):
        """Called on slider move or after combo change — just recolor."""
        if self.embedding is None or not self.valid_mask.any():
            return
        brushes = self._make_brushes()
        if brushes:
            self.plot.update_colors(brushes)
        self._push_range_mask()

    def _push_range_mask(self):
        """Send current property-range mask to the plot for hover filtering."""
        rmask = self._get_range_mask()
        if rmask is not None:
            self.plot.set_range_mask(rmask.values)
        else:
            self.plot.set_range_mask(None)

    def _make_brushes(self):
        if not self.valid_mask.any():
            return None
        col = self.prop_bar.current_property()
        n_valid = int(self.valid_mask.sum())
        default_brush = pg.mkBrush(70, 130, 180, 180)
        if col == "None" or col not in self.df.columns:
            return [default_brush] * n_valid
        try:
            vals = self.df[col].values[self.valid_mask]
            if np.issubdtype(vals.dtype, np.number) or self.prop_bar.is_numeric:
                vals = pd.to_numeric(vals, errors='coerce').astype(float)
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                lo, hi = self.prop_bar.get_low(), self.prop_bar.get_high()
                rng = max(vmax - vmin, 1e-9)
                t = (vals - vmin) / rng
                r = (50 + 205 * t).astype(int)
                g = (50 + 150 * (1 - np.abs(2 * t - 1))).astype(int)
                b = (255 * (1 - t)).astype(int)
                nan_mask = np.isnan(vals)
                out_mask = (vals < lo) | (vals > hi)
                brushes = []
                for i in range(n_valid):
                    if nan_mask[i]:
                        brushes.append(pg.mkBrush(160, 160, 160, 120))
                    elif out_mask[i]:
                        brushes.append(pg.mkBrush(190, 190, 190, 100))
                    else:
                        brushes.append(pg.mkBrush(int(r[i]), int(g[i]), int(b[i]), 180))
                return brushes
            else:
                uvals = pd.unique(vals)
                cmap = {v: i for i, v in enumerate(uvals)}
                n = max(len(uvals), 1)
                brushes = []
                for v in vals:
                    hue = int(360 * cmap.get(v, 0) / n)
                    c = QColor.fromHsv(hue, 200, 200)
                    brushes.append(pg.mkBrush(c.red(), c.green(), c.blue(), 180))
                return brushes
        except Exception as e:
            print(f"Color error: {e}")
            return [default_brush] * n_valid

    # --- Selection ---

    def _toggle_sel(self, tool):
        # figure out which button was toggled
        is_lasso = (tool == ScatterPlot.SEL_LASSO)
        btn = self.lasso_btn if is_lasso else self.rect_btn
        other = self.rect_btn if is_lasso else self.lasso_btn

        if btn.isChecked():
            other.setChecked(False)
            self.plot.start_selection(tool)
            self._set_state(AppState.SELECTING)
        else:
            self.plot.stop_selection()
            self._set_state(AppState.VISUALIZATION)

    def _clear_sel(self):
        self.plot.clear_selection()
        self.lasso_btn.setChecked(False)
        self.rect_btn.setChecked(False)
        self.plot.stop_selection()
        self.sel_cond_lbl.setText("")
        self._set_state(AppState.VISUALIZATION)

    def _get_range_mask(self):
        """Return boolean mask over df for rows within property range, or None."""
        col = self.prop_bar.current_property()
        if col == "None" or col not in self.df.columns or not self.prop_bar.is_numeric:
            return None
        lo, hi = self.prop_bar.get_low(), self.prop_bar.get_high()
        if lo <= self.prop_bar.data_min and hi >= self.prop_bar.data_max:
            return None  # full range, no filtering
        vals = pd.to_numeric(self.df[col], errors='coerce')
        return (vals >= lo) & (vals <= hi)

    def _on_sel_changed(self, indices):
        if not indices:
            self.sel_lbl.setText("Selected: 0")
            self.sel_cond_lbl.setText("")
            self.export_btn.setEnabled(False)
            return
        # apply property range filter
        range_mask = self._get_range_mask()
        if range_mask is not None:
            indices = [i for i in indices if range_mask.iloc[i]]
            self.plot.selected_indices = indices
            # update scatter highlight to filtered set
            if indices:
                self.plot.sel_scatter.setData(
                    x=self.plot.data_x[indices], y=self.plot.data_y[indices],
                    size=self.plot.pt_size + 4)
            else:
                self.plot.sel_scatter.setData(x=[], y=[])

        self.sel_lbl.setText(f"Selected: {len(indices)}")
        self.export_btn.setEnabled(len(indices) > 0)

        # condition label
        col = self.prop_bar.current_property()
        if range_mask is not None:
            lo, hi = self.prop_bar.get_low(), self.prop_bar.get_high()
            self.sel_cond_lbl.setText(
                f"with {col} in [{lo:.2f}, {hi:.2f}]")
        else:
            self.sel_cond_lbl.setText("")

    # --- Hover ---

    def _on_hover(self, indices):
        if not indices or self.df is None:
            self.mol_panel.clear()
            return
        indices = indices[:4]
        smiles = [self.smiles_list[i] for i in indices]
        infos = []
        show_cols = [c for c in ['step', 'Score', 'pChEMBL', 'Off-targets Score (raw)']
                     if c in self.df.columns]
        for i in indices:
            d = {"idx": i}
            for c in show_cols:
                v = self.df.iloc[i][c]
                if pd.notna(v):
                    d[c] = f"{v:.3f}" if isinstance(v, float) else str(v)
            infos.append(d)
        self.mol_panel.show_molecules(smiles, infos)

    # --- Export ---

    def _export(self):
        idx = self.plot.selected_indices
        if not idx:
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Save", "", "CSV (*.csv);;All (*)")
        if not fp:
            return
        try:
            sub = self.df.iloc[idx].copy()
            sub.to_csv(fp, index=False)
            self.sbar.showMessage(f"Exported {len(idx)} molecules to {Path(fp).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    pal.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    app.setPalette(pal)
    w = MolExplorer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
