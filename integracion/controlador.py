"""Estado único para acciones manuales y aceptadas por voz."""
import logging
import time
from PySide6.QtCore import QObject, QTimer, Signal, Qt
from integracion.proceso import Proceso


class Controlador(QObject):
    cambio = Signal()
    muestra = Signal(object)
    resultado = Signal(object)
    onda = Signal(object)
    comando_audio = Signal(object)
    dispositivos = Signal(object)
    cerrado = Signal()

    def __init__(self, raiz, python_audio=None, python_huellas=None, parent=None,
                 fabrica=Proceso, demora_ms=3000):
        super().__init__(parent)
        self.audio = fabrica('audio', raiz, python_audio, self)
        self.huellas = fabrica('huellas', raiz, python_huellas, self)
        self.audio.evento.connect(self.evento_audio)
        self.huellas.evento.connect(self.evento_huella)
        for proceso in (self.audio, self.huellas):
            proceso.terminado.connect(self.comprobar_cierre)
        self.actual = None
        self.ocupado = False
        self.cerrando = False
        self.estado = 'SIN HUELLA'
        self.detalle = 'Primero debés generar una huella.'
        self.fase = 'inactivo'
        self.ultimo_comando = None
        self.operacion_audio = None
        self.inicio_procesamiento = None
        self.inicio_grabacion = None
        self.respuesta_pendiente = None
        self.demora_ms = demora_ms
        self.presentacion = QTimer(self)
        self.presentacion.setSingleShot(True)
        self.presentacion.setTimerType(Qt.PreciseTimer)
        self.presentacion.timeout.connect(self.entregar_audio)
        self.limite_cierre = QTimer(self)
        self.limite_cierre.setSingleShot(True)
        self.limite_cierre.timeout.connect(self.forzar_cierre)

    def notificar(self, estado, detalle='', fase='inactivo'):
        self.estado, self.detalle, self.fase = estado, detalle, fase
        self.cambio.emit()

    def generar(self, identidad, anguli=None):
        if self.ocupado or self.cerrando:
            return False
        # Invalidar inmediatamente: un error nuevo nunca consulta la muestra vieja.
        self.actual = None
        self.ultimo_comando = None
        self.muestra.emit(None)
        self.resultado.emit({})
        self.ocupado = True
        self.notificar('PROCESANDO', 'Generando con Anguli y procesando la huella…', 'generando')
        self.huellas.solicitar('generar', identidad=identidad, anguli=anguli)
        return True

    def evento_huella(self, tipo, datos):
        if self.cerrando:
            return
        if tipo == 'error':
            self.fallar(datos['mensaje'])
        elif tipo == 'resultado':
            self.actual = datos
            self.ocupado = False
            self.muestra.emit(datos)
            self.notificar('LISTO', 'Huella generada · Lista para recibir un comando')

    def grabar(self, dispositivo=None):
        if self.ocupado or self.cerrando:
            return False
        if self.actual is None:
            self.accion('reconocer')
            return False
        self.ocupado = True
        self.operacion_audio = 'grabar'
        self.inicio_procesamiento = None
        self.inicio_grabacion = None
        self.onda.emit([])
        self.notificar('PREPARANDO', 'Validando modelo y abriendo el micrófono…', 'preparando')
        self.audio.solicitar('grabar', dispositivo=dispositivo)
        return True

    def listar_dispositivos(self):
        if self.ocupado or self.cerrando:
            return
        self.ocupado = True
        self.operacion_audio = 'dispositivos'
        self.notificar('PROCESANDO', 'Consultando micrófonos…', 'dispositivos')
        self.audio.solicitar('dispositivos')

    def evento_audio(self, tipo, datos):
        if self.cerrando:
            return
        if tipo == 'grabando':
            self.inicio_grabacion = time.monotonic()
            self.notificar('GRABANDO', 'Micrófono activo · Captura de 2 segundos', 'grabando')
        elif tipo == 'capturado':
            self.inicio_procesamiento = time.monotonic()
            self.onda.emit({'real': datos['onda']})
            self.notificar('PROCESANDO COMANDO...', 'Analizando la captura real…', 'procesando_audio')
        elif tipo == 'normalizado':
            self.onda.emit({'normalizada': datos['onda']})
        elif tipo == 'error':
            prefijo = 'Error de micrófono o audio. Revisá el dispositivo seleccionado. '
            self.fallar(prefijo + datos['mensaje'])
        elif tipo == 'resultado':
            if self.operacion_audio == 'dispositivos':
                self.ocupado = False
                self.dispositivos.emit(datos)
                self.notificar('LISTO' if self.actual else 'SIN HUELLA', 'Seleccioná el micrófono por nombre e índice.')
            else:
                self.respuesta_pendiente = datos
                transcurrido = (time.monotonic() - (self.inicio_procesamiento or time.monotonic())) * 1000
                self.presentacion.start(max(0, int(self.demora_ms - transcurrido + 1)))

    def entregar_audio(self):
        if self.cerrando or self.respuesta_pendiente is None:
            return
        datos, self.respuesta_pendiente = self.respuesta_pendiente, None
        self.ocupado = False
        logging.info('Decisión audio: %s', datos)
        self.comando_audio.emit(datos)
        comando = datos.get('comando', 'no_reconocido')
        if comando == 'no_reconocido':
            texto = ('No se detectó un comando válido. Volvé a grabar.'
                     if datos.get('diagnostico', {}).get('propuesta') is None
                     else 'Comando no reconocido. Intentá nuevamente.')
            self.notificar('COMANDO NO RECONOCIDO', texto)
        else:
            self.accion(comando)

    def accion(self, comando):
        if self.cerrando:
            return False
        if comando == 'salir':
            self.cerrar()
            return True
        if self.ocupado:
            return False
        if comando not in ('reconocer', 'identificar'):
            self.notificar('COMANDO NO RECONOCIDO', 'Comando no reconocido. Intentá nuevamente.')
            return False
        self.ultimo_comando = comando
        if self.actual is None:
            self.notificar('SIN HUELLA', 'Primero debés generar una huella.')
            return False
        r = self.actual
        aceptada = r['autorizada'] == 1
        identidad = r.get('identidad') if aceptada and comando == 'identificar' else None
        pred = r['reconocimiento']
        self.resultado.emit({'comando': comando, 'autorizada': aceptada,
                             'identidad': identidad, 'distancia': pred['distancia'],
                             'limite': pred['limite_distancia'],
                             'archivo_resultado': r['archivo_resultado']})
        self.notificar('AUTORIZADO' if aceptada else 'NO AUTORIZADO', 'Resultado disponible para la huella actual.')
        return True

    def fallar(self, mensaje):
        logging.error(mensaje)
        self.ocupado = False
        self.notificar('ERROR', mensaje)

    def cerrar(self):
        if self.cerrando:
            return
        self.cerrando = True
        self.presentacion.stop()
        self.respuesta_pendiente = None
        self.notificar('CERRANDO', 'Esperando la finalización de las operaciones y sus procesos hijos…', 'cerrando')
        for proceso in (self.audio, self.huellas):
            proceso.cerrar()
        self.limite_cierre.start(180000)
        self.comprobar_cierre()

    def forzar_cierre(self):
        logging.error('Cierre: vencieron 180 segundos; terminando árboles de procesos')
        self.detalle = 'Terminando procesos que excedieron el tiempo de cierre…'
        self.cambio.emit()
        for proceso in (self.audio, self.huellas):
            proceso.forzar_arbol()

    def comprobar_cierre(self):
        if self.cerrando and not self.audio.activo and not self.huellas.activo:
            self.limite_cierre.stop()
            self.cerrado.emit()
