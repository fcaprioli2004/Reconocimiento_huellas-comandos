from pathlib import Path
import time
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QButtonGroup, QSplitter, QComboBox,
    QProgressBar, QFileDialog, QSizePolicy)
from interfaz.componentes import Imagen, Onda, Emblema
from interfaz.estilos import ESTILO


def etiqueta(texto, nombre=None):
    w = QLabel(texto)
    if nombre:
        w.setObjectName(nombre)
    w.setTextFormat(Qt.PlainText)
    return w


def panel():
    w = QFrame()
    w.setObjectName('panel')
    lay = QVBoxLayout(w)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(6)
    return w, lay


class VentanaPrincipal(QMainWindow):
    def __init__(self, controlador, anguli=None):
        super().__init__()
        self.c = controlador
        self.anguli = anguli
        self.permitir_cierre = False
        self.vistas = {}
        self.vista_actual = 'Transformada'
        self.audios = {}
        self.datos_comando = None
        self.setWindowTitle('Huellas + Voz · Inteligencia Artificial I')
        self.setMinimumSize(1000, 640)
        disponible = self.screen().availableGeometry()
        self.resize(min(1280, int(disponible.width() * .95)),
                    min(780, int(disponible.height() * .90)))
        self.setStyleSheet(ESTILO)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(Emblema())
        header.addWidget(etiqueta('Huellas + Voz', 'titulo'))
        self.subtitulo = etiqueta('  │  Reconocimiento de huellas sintéticas', 'muted')
        header.addWidget(self.subtitulo)
        header.addStretch()
        self.sistema = etiqueta('●  Sistema listo', 'estado')
        header.addWidget(self.sistema)
        self.salir = QPushButton('Salir')
        self.salir.clicked.connect(lambda: self.c.accion('salir'))
        header.addWidget(self.salir)
        layout.addLayout(header)
        superior, sup = panel()
        fila = QHBoxLayout()
        fila.addWidget(etiqueta('1 · Generar huella', 'seccion'))
        self.identidades = QButtonGroup(self)
        self.identidades.setExclusive(True)
        self.opciones = []
        for i, nombre in enumerate(['Identidad 1', 'Identidad 2', 'Identidad 3', 'Intruso']):
            boton = QPushButton(nombre)
            boton.setCheckable(True)
            boton.setChecked(i == 0)
            self.identidades.addButton(boton, i)
            self.opciones.append(boton)
            fila.addWidget(boton, 1)
        self.generar = QPushButton('Generar')
        self.generar.setObjectName('principal')
        self.generar.clicked.connect(self.generar_huella)
        fila.addWidget(self.generar)
        sup.addLayout(fila)
        self.confirmacion = etiqueta('Seleccioná una identidad sintética y presioná Generar.', 'muted')
        self.confirmacion.setAlignment(Qt.AlignCenter)
        sup.addWidget(self.confirmacion)
        layout.addWidget(superior)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(10)
        split.setChildrenCollapsible(False)
        izquierda, izq = panel()
        izquierda.setMinimumWidth(300)
        cab = QHBoxLayout()
        self.titulo_huella = etiqueta('Huella transformada', 'seccion')
        self.titulo_huella.setStyleSheet('font-size: 16px; font-weight: 700;')
        self.titulo_huella.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        cab.addWidget(self.titulo_huella, 1)
        self.config_anguli = QPushButton('Anguli…')
        self.config_anguli.clicked.connect(self.elegir_anguli)
        self.config_anguli.setToolTip('Opcional: elegir el ejecutable sin modificar config_anguli.json')
        cab.addWidget(self.config_anguli)
        izq.addLayout(cab)
        self.imagen = Imagen()
        izq.addWidget(self.imagen, 1)
        self.transformaciones = etiqueta('Transformaciones: —', 'muted')
        self.transformaciones.setWordWrap(True)
        self.transformaciones.setMaximumHeight(36)
        self.transformaciones.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        izq.addWidget(self.transformaciones)
        tabs = QGridLayout()
        tabs.setSpacing(5)
        self.grupo_vistas = QButtonGroup(self)
        self.botones_vistas = {}
        for i, vista in enumerate(('Transformada', 'Normalizada', 'Binaria', 'Esqueleto')):
            b = QPushButton(vista)
            b.setCheckable(True)
            b.setEnabled(False)
            b.clicked.connect(lambda checked=False, nombre=vista: self.ver(nombre))
            self.grupo_vistas.addButton(b)
            self.botones_vistas[vista] = b
            tabs.addWidget(b, i // 2, i % 2)
        izq.addLayout(tabs)
        split.addWidget(izquierda)
        derecha = QWidget()
        derecha.setMinimumWidth(600)
        der = QVBoxLayout(derecha)
        der.setContentsMargins(0, 0, 0, 0)
        der.setSpacing(8)
        audio, aud = panel()
        cab_audio = QHBoxLayout()
        cab_audio.addWidget(etiqueta('2 · Dar un comando', 'seccion'))
        cab_audio.addStretch()
        cab_audio.addWidget(etiqueta('Decí: reconocer, identificar o salir', 'muted'))
        aud.addLayout(cab_audio)
        micro = QHBoxLayout()
        self.microfonos = QComboBox()
        self.microfonos.addItem('Micrófono predeterminado del sistema', None)
        self.microfonos.setMinimumWidth(0)
        self.microfonos.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.actualizar = QPushButton('Actualizar')
        self.actualizar.clicked.connect(self.c.listar_dispositivos)
        micro.addWidget(self.microfonos, 1)
        micro.addWidget(self.actualizar)
        self.grabar = QPushButton('Grabar')
        self.grabar.setObjectName('principal')
        self.grabar.clicked.connect(lambda: self.c.grabar(self.microfonos.currentData()))
        micro.addWidget(self.grabar)
        aud.addLayout(micro)
        grab = QHBoxLayout()
        self.luz = etiqueta('●')
        self.luz.setStyleSheet('font-size: 24px; color: #52647a;')
        grab.addWidget(self.luz)
        self.estado_audio = etiqueta('MICRÓFONO INACTIVO')
        self.estado_audio.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        grab.addWidget(self.estado_audio, 1)
        self.grupo_audio = QButtonGroup(self)
        self.botones_audio = {}
        for clave, texto in [('real', 'Captura real'), ('normalizada', 'Normalizada')]:
            b = QPushButton(texto)
            b.setCheckable(True)
            b.setEnabled(False)
            b.setChecked(clave == 'real')
            b.clicked.connect(lambda checked=False, tipo=clave: self.ver_audio(tipo))
            self.grupo_audio.addButton(b)
            self.botones_audio[clave] = b
            grab.addWidget(b)
        aud.addLayout(grab)
        self.onda = Onda()
        aud.addWidget(self.onda, 1)
        self.progreso = QProgressBar()
        self.progreso.setFixedHeight(3)
        self.progreso.setTextVisible(False)
        self.progreso.setRange(0, 1)
        self.progreso.setValue(0)
        aud.addWidget(self.progreso)
        der.addWidget(audio, 1)
        resultado, res = panel()
        cab_resultado = QHBoxLayout()
        titulo_resultado = etiqueta('Último comando y resultado', 'seccion')
        titulo_resultado.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        cab_resultado.addWidget(titulo_resultado, 1)
        self.reconocer = QPushButton('Reconocer')
        self.identificar = QPushButton('Identificar')
        self.reconocer.clicked.connect(lambda: self.accion_manual('reconocer'))
        self.identificar.clicked.connect(lambda: self.accion_manual('identificar'))
        cab_resultado.addWidget(self.reconocer)
        cab_resultado.addWidget(self.identificar)
        res.addLayout(cab_resultado)
        fila_comando = QHBoxLayout()
        self.comando = etiqueta('Esperando comando', 'comando')
        self.comando.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        fila_comando.addWidget(self.comando, 1)
        self.metricas_voz = etiqueta('Distancia voz: —\nLímite voz: —', 'metricas_voz')
        self.metricas_voz.setMinimumWidth(175)
        fila_comando.addWidget(self.metricas_voz)
        res.addLayout(fila_comando)
        self.respuesta = etiqueta('SIN HUELLA', 'resultado')
        self.respuesta.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        res.addWidget(self.respuesta)
        self.identidad = etiqueta('', 'identidad')
        self.identidad.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        res.addWidget(self.identidad)
        self.metricas = etiqueta('Distancia huella: —       │       Límite: —', 'muted')
        res.addWidget(self.metricas)
        self.detalle = etiqueta('Los resultados provienen de los modelos originales.', 'muted')
        self.detalle.setWordWrap(True)
        self.detalle.setFixedHeight(34)
        self.detalle.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.detalle.setTextInteractionFlags(Qt.TextSelectableByMouse)
        res.addWidget(self.detalle)
        res.addStretch(1)
        der.addWidget(resultado, 1)
        split.addWidget(derecha)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setSizes([320, 900])
        layout.addWidget(split, 1)
        self.pie = etiqueta('', 'muted')
        self.pie.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.pie.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pie)
        self.c.cambio.connect(self.refrescar)
        self.c.muestra.connect(self.cargar_muestra)
        self.c.resultado.connect(self.cargar_resultado)
        self.c.onda.connect(self.cargar_audios)
        self.c.comando_audio.connect(self.cargar_comando_audio)
        self.c.dispositivos.connect(self.cargar_dispositivos)
        self.c.cerrado.connect(self.cierre_completo)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refrescar_tiempo)
        self.timer.start(100)
        self.refrescar()

    def cargar_audios(self, datos):
        if not datos:
            self.audios = {}
            self.datos_comando = None
            self.metricas_voz.setText('Distancia voz: —\nLímite voz: —')
            self.metricas_voz.setToolTip('')
            self.onda.cargar([])
        elif isinstance(datos, dict):
            if 'real' in datos:
                self.audios = dict(datos)
                self.botones_audio['real'].setChecked(True)
            else:
                self.audios.update(datos)
        else:
            # Compatible con la envolvente de captura anterior.
            self.audios = {'real': datos}
        for tipo, boton in self.botones_audio.items():
            boton.setEnabled(bool(self.audios.get(tipo)))
        elegido = next((tipo for tipo, b in self.botones_audio.items() if b.isChecked()), 'real')
        if self.audios.get(elegido):
            self.ver_audio(elegido)

    def ver_audio(self, tipo):
        if self.audios.get(tipo):
            self.botones_audio[tipo].setChecked(True)
            self.onda.cargar(self.audios[tipo], tipo)

    def cargar_comando_audio(self, datos):
        self.datos_comando = datos
        info = datos.get('diagnostico', {})
        candidata = info.get('propuesta')
        distancia = info.get('distancia')
        limite = datos.get('umbrales', {}).get(candidata)
        def formato(valor):
            return '—' if valor is None else f'{valor:.4f}'
        self.metricas_voz.setText(f'Distancia voz: {formato(distancia)}\nLímite voz: {formato(limite)}')
        aceptado = datos.get('comando') in ('reconocer', 'identificar', 'salir')
        self.metricas_voz.setStyleSheet('color: ' + ('#a7bed9' if aceptado else '#ff6576') + ';')
        self.metricas_voz.setToolTip(
            f"Candidata: {candidata or 'ninguna'}\nDistancia: {distancia}\nLímite: {limite}\n"
            f"Apoyo: {info.get('apoyo')}\nApoyo mínimo: {datos.get('apoyo_minimo')}\n"
            + ('Comando aceptado' if aceptado else 'Comando no reconocido; no se ejecutó la candidata'))
        comando = datos.get('comando', 'no_reconocido')
        self.comando.setText('No reconocido' if comando == 'no_reconocido' else '“' + comando.capitalize() + '”')

    def accion_manual(self, comando):
        self.datos_comando = None
        self.metricas_voz.setText('Distancia voz: —\nLímite voz: —')
        self.metricas_voz.setToolTip('Consulta mediante botón, sin una nueva captura de voz.')
        self.c.accion(comando)

    def generar_huella(self):
        identidad = ('1', '2', '3', 'intruso')[self.identidades.checkedId()]
        self.c.generar(identidad, self.anguli)

    def elegir_anguli(self):
        ruta, _ = QFileDialog.getOpenFileName(self, 'Seleccionar Anguli.exe', '', 'Ejecutable (*.exe);;Todos (*)')
        if ruta:
            self.anguli = ruta
            self.config_anguli.setToolTip(ruta)

    def cargar_dispositivos(self, datos):
        anterior = self.microfonos.currentData()
        self.microfonos.clear()
        self.microfonos.addItem('Micrófono predeterminado del sistema', None)
        for d in datos:
            self.microfonos.addItem(f"{d['id']} · {d['nombre']} (API {d['hostapi']})", d['id'])
        indice = self.microfonos.findData(anterior)
        if indice >= 0:
            self.microfonos.setCurrentIndex(indice)

    def cargar_muestra(self, muestra):
        self.vistas = muestra.get('vistas', {}) if muestra else {}
        self.imagen.cargar()
        for nombre, b in self.botones_vistas.items():
            b.setEnabled(nombre in self.vistas)
            b.setChecked(False)
        if muestra:
            self.ver('Transformada')
            aplicadas = muestra.get('transformaciones', {}).get('aplicadas', [])
            self.transformaciones.setText('Transformaciones: ' + (', '.join(x['nombre'] for x in aplicadas) or 'ninguna'))
            self.transformaciones.setToolTip(str(muestra.get('transformaciones', {})))
        else:
            self.transformaciones.setText('Transformaciones: —')
            self.titulo_huella.setText('Huella transformada')

    def ver(self, nombre):
        ruta = self.vistas.get(nombre)
        if ruta:
            self.vista_actual = nombre
            self.imagen.cargar(ruta)
            self.titulo_huella.setText('Huella ' + nombre.lower())
            self.botones_vistas[nombre].setChecked(True)

    def cargar_resultado(self, r):
        if not r:
            self.comando.setText('Esperando comando')
            self.datos_comando = None
            self.metricas_voz.setText('Distancia voz: —\nLímite voz: —')
            self.metricas_voz.setToolTip('')
            self.respuesta.setText('SIN HUELLA')
            self.respuesta.setStyleSheet('color: #9db4d0;')
            self.identidad.setText('')
            self.metricas.setText('Distancia huella: —       │       Límite: —')
            return
        self.comando.setText('“' + r['comando'].capitalize() + '”')
        color = '#47e9be' if r['autorizada'] else '#ff5365'
        self.respuesta.setText('AUTORIZADO' if r['autorizada'] else 'NO AUTORIZADO')
        self.respuesta.setStyleSheet('color: ' + color + ';')
        ident = r.get('identidad')
        self.identidad.setText(ident.replace('Huellas_', 'Identidad ') if ident else '')
        self.metricas.setText(f"Distancia huella: {r['distancia']:.4f}       │       Límite: {r['limite']:.4f}")
        self.metricas.setToolTip(f"Distancia: {r['distancia']}\nLímite: {r['limite']}")

    def refrescar_tiempo(self):
        c = self.c
        if c.fase == 'grabando':
            segundos = max(0, time.monotonic() - c.inicio_grabacion)
            # El fin real se confirma con el evento capturado, no con este reloj.
            self.estado_audio.setText(f'GRABANDO  {segundos:04.1f} s')

    def refrescar(self):
        c = self.c
        libre = not c.ocupado and not c.cerrando
        self.generar.setEnabled(libre)
        self.config_anguli.setEnabled(libre)
        self.actualizar.setEnabled(libre)
        self.microfonos.setEnabled(libre)
        for b in self.opciones:
            b.setEnabled(libre)
        for b in (self.grabar, self.reconocer, self.identificar):
            b.setEnabled(libre and c.actual is not None)
        self.salir.setEnabled(not c.cerrando)
        self.sistema.setText('●  ' + ('Sistema ocupado' if c.ocupado else 'Sistema listo'))
        self.pie.setText(f"{'✓ Huella generada' if c.actual else '○ Sin huella'}       │       {c.estado}       │       {'Esperando comando' if libre and c.actual else 'Huellas + Voz'}")
        if c.fase == 'generando':
            self.confirmacion.setText('Generando y procesando…')
        elif c.actual:
            self.confirmacion.setText('✓  Huella generada · Lista para recibir un comando')
        else:
            self.confirmacion.setText('Seleccioná una identidad sintética y presioná Generar.')
        grabando = c.fase == 'grabando'
        self.luz.setStyleSheet('font-size: 24px; color: ' + ('#ff4159' if grabando else '#52647a') + ';')
        self.onda.animar(grabando)
        self.estado_audio.setStyleSheet('color: ' + ('#ff4159' if grabando else '#b3cae3') + '; font-weight: 700;')
        self.estado_audio.setText({'preparando': 'PREPARANDO MICRÓFONO…',
            'procesando_audio': 'PROCESANDO COMANDO...', 'cerrando': 'CERRANDO…'}.get(c.fase, 'MICRÓFONO INACTIVO'))
        self.progreso.setRange(0, 0 if c.ocupado else 1)
        self.detalle.setText(c.detalle)
        self.detalle.setToolTip(c.detalle)
        if c.estado in ('SIN HUELLA', 'COMANDO NO RECONOCIDO', 'ERROR', 'PROCESANDO', 'PROCESANDO COMANDO...'):
            # Una consulta nueva no debe presentar un resultado viejo como actual.
            self.respuesta.setText(c.estado)
            self.respuesta.setStyleSheet('color: ' + ('#ff6576' if c.estado == 'ERROR' else '#adc4df') + ';')
            self.identidad.setText('')
            self.metricas.setText('Distancia huella: —       │       Límite: —')
        elif c.estado == 'LISTO' and self.comando.text() == 'Esperando comando':
            self.respuesta.setText('LISTA PARA CONSULTAR')
        self.refrescar_tiempo()

    def closeEvent(self, evento):
        if self.permitir_cierre:
            evento.accept()
        else:
            evento.ignore()
            self.c.accion('salir')

    def cierre_completo(self):
        self.permitir_cierre = True
        self.timer.stop()
        QTimer.singleShot(0, self.close)
