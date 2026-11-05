"""Dos nubes PCA 3D con rotación, zoom, historial y muestra actual."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QCheckBox, QSplitter)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

COLORES = ['#35b8ff', '#ffad42', '#45d6a0', '#be89ff', '#fb7185']


class Grafico(QWidget):
    def __init__(self, titulo):
        super().__init__()
        self.titulo = titulo
        self.datos_base = None
        self.historial = []
        self.marcadores = []
        self.anotacion = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.fig = Figure(figsize=(6, 5), facecolor='#101f30')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d', computed_zorder=False)
        self.fig.subplots_adjust(left=.03, right=.96, top=.94, bottom=.08)
        self.ax.set_facecolor('#101f30')
        self.ax.tick_params(colors='#afc3d9', labelsize=8)
        for eje in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            eje.label.set_color('#dbeafe')
            eje.set_pane_color((.055, .10, .16, 1))
        self.ax.set_title(titulo + ' · cargando base…', color='#e0eeff', fontsize=13)
        self.ax.set_xlabel('PC1'); self.ax.set_ylabel('PC2'); self.ax.set_zlabel('PC3')
        self.ax.view_init(elev=22, azim=-55)
        self.barra = NavigationToolbar2QT(self.canvas, self)
        self.barra.setStyleSheet('QToolBar { background: #dce6f1; color: #10213b; } QToolButton { background: #dce6f1; color: #10213b; }')
        layout.addWidget(self.barra)
        layout.addWidget(self.canvas, 1)
        fila = QHBoxLayout()
        self.seguir = QCheckBox('Incluir muestra nueva en la vista')
        self.seguir.setChecked(True)
        self.mostrar_historial = QCheckBox('Historial')
        self.mostrar_historial.setChecked(True)
        self.mostrar_historial.toggled.connect(self.redibujar_puntos)
        centrar = QPushButton('Centrar todo')
        centrar.clicked.connect(self.centrar)
        fila.addWidget(self.seguir); fila.addWidget(self.mostrar_historial); fila.addWidget(centrar)
        layout.addLayout(fila)
        self.estado = QLabel('Esperando PCA guardado…')
        self.estado.setWordWrap(True)
        self.estado.setMinimumHeight(42)
        self.estado.setTextFormat(Qt.PlainText)
        self.estado.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.estado)
        self.canvas.mpl_connect('scroll_event', self.zoom)

    def cargar_base(self, datos):
        self.datos_base = datos
        Z = np.asarray(datos['coordenadas'])
        grupos = np.asarray(datos['grupos'])
        for i, grupo in enumerate(sorted(set(grupos))):
            z = Z[grupos == grupo]
            self.ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=20, alpha=.48,
                            color=COLORES[i % len(COLORES)], label=f'{grupo} ({len(z)})', depthshade=False, zorder=1)
        for i, setter in enumerate((self.ax.set_xlabel, self.ax.set_ylabel, self.ax.set_zlabel)):
            setter(f"PC{i+1} ({100*datos['varianza'][i]:.1f} %)")
        total = sum(datos['varianza']) * 100
        self.ax.set_title(f'{self.titulo} · {len(Z)} muestras base · 3 PC: {total:.1f} %', color='#e0eeff', fontsize=11)
        self.ax.legend(loc='upper left', fontsize=8, facecolor='#e3edf7', framealpha=.9)
        self.estado.setText('Base lista. La estrella indica la última muestra; verde = aceptada, rojo = rechazada.')
        self.centrar()

    def agregar(self, datos):
        if datos.get('sin_vector'):
            self.estado.setText(datos.get('motivo', 'Sin vector para proyectar.') + ' Se conserva la última muestra válida.')
            if self.anotacion is not None:
                self.anotacion.set_text('  Última válida')
                self.canvas.draw_idle()
            return
        # Una huella puede consultarse varias veces; la misma muestra no se duplica.
        clave = datos.get('muestra_id')
        if clave:
            self.historial = [p for p in self.historial if p.get('muestra_id') != clave]
        self.historial.append(datos)
        self.historial = self.historial[-100:]
        self.redibujar_puntos()
        coords = datos['coordenadas']
        distancia, limite = datos.get('distancia'), datos.get('limite')
        valores = '' if distancia is None or limite is None else f' | Distancia original: {distancia:.4f} / límite: {limite:.4f}'
        self.estado.setText(f"{datos.get('etiqueta', 'Muestra actual')}{valores}\n"
                            + 'PCA sólo visualiza; la decisión se conserva en el espacio original.')
        if self.seguir.isChecked():
            limites = (self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim())
            if any(not lo <= valor <= hi for valor, (lo, hi) in zip(coords, limites)):
                self.centrar()
        self.canvas.draw_idle()

    def redibujar_puntos(self, *_):
        for m in self.marcadores:
            m.remove()
        self.marcadores = []
        if self.anotacion is not None:
            self.anotacion.remove()
            self.anotacion = None
        if not self.historial:
            self.canvas.draw_idle()
            return
        if self.mostrar_historial.isChecked() and len(self.historial) > 1:
            z = np.asarray([p['coordenadas'] for p in self.historial[:-1]])
            self.marcadores.append(self.ax.scatter(z[:,0], z[:,1], z[:,2], marker='x',
                                                   s=45, color='#d2dae6', alpha=.65, depthshade=False, zorder=5))
        ultimo = self.historial[-1]
        x, y, z = ultimo['coordenadas']
        color = '#55f2b5' if ultimo.get('aceptada') else '#ff536e'
        self.marcadores.append(self.ax.scatter([x], [y], [z], marker='*', s=340,
                                               c=color, edgecolors='#ffffff', linewidths=1.1, depthshade=False, zorder=10))
        self.anotacion = self.ax.text(x, y, z, '  Actual', color='#ffffff', fontsize=9, zorder=11)
        self.canvas.draw_idle()

    def centrar(self):
        if not self.datos_base:
            return
        arrays = [np.asarray(self.datos_base['coordenadas'])]
        if self.historial:
            arrays.append(np.asarray([p['coordenadas'] for p in self.historial]))
        Z = np.concatenate(arrays)
        for i, setter in enumerate((self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim)):
            lo, hi = Z[:, i].min(), Z[:, i].max()
            margen = max((hi-lo)*.10, 1.)
            setter(lo-margen, hi+margen)
        self.canvas.draw_idle()

    def zoom(self, evento):
        if evento.inaxes != self.ax:
            return
        factor = .85 if evento.button == 'up' else 1/.85
        for getter, setter in ((self.ax.get_xlim, self.ax.set_xlim),
                                (self.ax.get_ylim, self.ax.set_ylim),
                                (self.ax.get_zlim, self.ax.set_zlim)):
            lo, hi = getter(); centro = (lo+hi)/2; radio = (hi-lo)*factor/2
            setter(centro-radio, centro+radio)
        self.canvas.draw_idle()


class VentanaPCA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PCA 3D interactivo · Huellas y audio')
        self.resize(1350, 780)
        self.setMinimumSize(960, 600)
        self.setStyleSheet('QWidget { background: #101f30; color: #dceaff; } QPushButton { padding: 5px; }')
        raiz = QWidget(); lay = QVBoxLayout(raiz)
        ayuda = QLabel('Arrastrá con el botón izquierdo para rotar · rueda para acercar/alejar · Centrar todo para recuperar la vista')
        ayuda.setWordWrap(True); lay.addWidget(ayuda)
        split = QSplitter(Qt.Horizontal)
        self.graficos = {'huellas': Grafico('Huellas'), 'audio': Grafico('Audio')}
        for grafico in self.graficos.values():
            split.addWidget(grafico)
        split.setSizes([650, 650]); lay.addWidget(split, 1)
        self.setCentralWidget(raiz)
