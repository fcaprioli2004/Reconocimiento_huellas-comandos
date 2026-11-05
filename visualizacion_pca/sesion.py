import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import uuid
from PySide6.QtCore import QObject
from visualizacion_pca.backend import Backend
from visualizacion_pca.ventana import VentanaPCA


class SesionPCA(QObject):
    def __init__(self, controlador, raiz, python_audio=None, python_huellas=None,
                 modelos=None, datos=None, parent=None):
        super().__init__(parent)
        self.c = controlador
        self.raiz = Path(raiz)
        self.ventana = VentanaPCA()
        self.backend = Backend(raiz, python_audio, python_huellas, self)
        nombre = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
        self.salida = self.raiz / 'salidas_integracion/pca' / nombre
        self.salida.mkdir(parents=True, exist_ok=False)
        self.backend.base.connect(self.cargar_base)
        self.backend.punto.connect(self.punto)
        self.backend.error.connect(self.error)
        self.c.comando_audio.connect(self.audio)
        self.c.resultado.connect(self.huella)
        self.backend.iniciar(modelos, datos)

    def guardar(self, nombre, datos):
        try:
            (self.salida / nombre).write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError:
            logging.exception('No se pudo guardar evidencia PCA')

    def cargar_base(self, dominio, datos):
        grafico = self.ventana.graficos[dominio]
        grafico.cargar_base(datos)
        self.guardar(f'base_{dominio}.json', datos)
        try:
            grafico.fig.savefig(self.salida / f'base_{dominio}_3d.png', dpi=160, facecolor=grafico.fig.get_facecolor())
        except OSError:
            logging.exception('No se pudo guardar figura PCA')

    def audio(self, datos):
        comando = datos.get('comando', 'no_reconocido')
        info = datos.get('diagnostico', {})
        candidato = info.get('propuesta')
        etiqueta = comando.capitalize() if comando != 'no_reconocido' else 'Comando rechazado'
        if comando == 'no_reconocido' and candidato:
            etiqueta += f' · candidata: {candidato}'
        self.backend.encolar('audio', 'proyectar', vector=datos.get('vector_caracteristicas'),
            muestra_id=uuid.uuid4().hex, aceptada=comando != 'no_reconocido', etiqueta=etiqueta,
            distancia=info.get('distancia'), limite=datos.get('umbrales', {}).get(candidato))

    def huella(self, datos):
        if not datos or self.c.actual is None:
            return
        actual = self.c.actual
        autorizada = bool(datos['autorizada'])
        etiqueta = 'AUTORIZADO' if autorizada else 'NO AUTORIZADO'
        if datos.get('identidad'):
            etiqueta += ' · ' + datos['identidad'].replace('Huellas_', 'Identidad ')
        self.backend.encolar('huellas', 'proyectar', imagen=actual['imagen_a_procesar'],
            muestra_id=actual['archivo_resultado'], aceptada=autorizada, etiqueta=etiqueta,
            distancia=datos['distancia'], limite=datos['limite'])

    def punto(self, dominio, datos):
        self.ventana.graficos[dominio].agregar(datos)
        try:
            with (self.salida / 'muestras.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'dominio': dominio, **datos}, ensure_ascii=False) + '\n')
        except OSError:
            logging.exception('No se pudo guardar la proyección PCA')

    def error(self, dominio, mensaje):
        logging.error('PCA %s: %s', dominio, mensaje)
        self.ventana.graficos[dominio].estado.setText('PCA no disponible: ' + mensaje)
