"""Transporte asíncrono Qt; nunca espera bloqueando el hilo de la GUI."""
import json
import logging
import os
from pathlib import Path
import signal
import sys
import uuid
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


class Proceso(QObject):
    evento = Signal(str, object)
    terminado = Signal()

    def __init__(self, dominio, raiz, interprete=None, parent=None):
        super().__init__(parent)
        self.dominio, self.raiz = dominio, Path(raiz).resolve()
        self.interprete = interprete or sys.executable
        self.p = QProcess(self)
        self.buffer = b''
        self.sid = None
        self.pendiente = None
        self.cerrando = False
        self.p.readyReadStandardOutput.connect(self.leer)
        self.p.readyReadStandardError.connect(self.errores)
        self.p.started.connect(self.enviar_pendiente)
        self.p.finished.connect(self.finalizado)
        self.p.errorOccurred.connect(self.error_proceso)
        self.asesino = None

    @property
    def activo(self):
        return self.p.state() != QProcess.NotRunning

    def solicitar(self, operacion, **argumentos):
        if self.sid is not None or self.cerrando:
            raise RuntimeError('Ya hay una operación en curso o el proceso está cerrando')
        self.sid = uuid.uuid4().hex
        self.pendiente = {'id': self.sid, 'operacion': operacion, 'argumentos': argumentos}
        if not self.activo:
            self.buffer = b''
            entorno = QProcessEnvironment.systemEnvironment()
            entorno.insert('PYTHONIOENCODING', 'utf-8')
            entorno.insert('PYTHONDONTWRITEBYTECODE', '1')
            self.p.setProcessEnvironment(entorno)
            self.p.setWorkingDirectory(str(self.raiz))
            worker = Path(__file__).with_name('trabajador.py')
            self.p.start(self.interprete, ['-u', str(worker), '--dominio', self.dominio,
                                            '--raiz', str(self.raiz)])
        else:
            self.enviar_pendiente()

    def escribir(self, req):
        self.p.write((json.dumps(req, ensure_ascii=False) + '\n').encode('utf-8'))

    def enviar_pendiente(self):
        if self.pendiente:
            self.escribir(self.pendiente)
            self.pendiente = None

    def leer(self):
        self.buffer += bytes(self.p.readAllStandardOutput())
        while b'\n' in self.buffer:
            linea, self.buffer = self.buffer.split(b'\n', 1)
            texto = linea.decode('utf-8', errors='replace').strip()
            if texto.startswith('@@HV@@'):
                try:
                    r = json.loads(texto[6:])
                    if r['id'] != self.sid:
                        continue
                    if r['tipo'] in ('resultado', 'error'):
                        self.sid = None
                    self.evento.emit(r['tipo'], r['datos'])
                except (KeyError, ValueError) as exc:
                    logging.exception('Protocolo inválido: %s', exc)
            elif texto.startswith('GRABANDO') and self.sid:
                self.evento.emit('grabando', {'mensaje': texto})
            elif texto:
                logging.info('%s: %s', self.dominio, texto)

    def errores(self):
        texto = bytes(self.p.readAllStandardError()).decode('utf-8', errors='replace')
        if texto:
            logging.warning('%s: %s', self.dominio, texto.rstrip())

    def error_proceso(self, error):
        if error == QProcess.FailedToStart:
            self.sid = self.pendiente = None
            self.evento.emit('error', {'mensaje': f'No se pudo iniciar {self.interprete}: {self.p.errorString()}'})
            self.terminado.emit()

    def finalizado(self, codigo, estado):
        self.leer()
        self.errores()
        if self.sid is not None:
            self.sid = self.pendiente = None
            self.evento.emit('error', {'mensaje': f'El proceso {self.dominio} terminó sin respuesta (código {codigo})'})
        self.terminado.emit()

    def cerrar(self):
        self.cerrando = True
        if self.activo:
            # Se encola después de la operación. La captura/generación termina
            # y recoge su propio hijo antes de que el servidor salga.
            self.escribir({'id': 'cierre', 'operacion': 'cerrar'})

    def forzar_arbol(self):
        if not self.activo:
            return
        pid = int(self.p.processId())
        if os.name == 'nt':
            self.asesino = QProcess(self)
            self.asesino.start('taskkill', ['/PID', str(pid), '/T', '/F'])
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
