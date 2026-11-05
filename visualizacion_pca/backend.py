"""Procesos de proyección separados de la GUI y de los clasificadores."""
from collections import deque
from pathlib import Path
import sys
import uuid
from PySide6.QtCore import QObject, QProcessEnvironment, QTimer, Signal
from integracion.proceso import Proceso


class ProcesoPCA(Proceso):
    def solicitar(self, operacion, **argumentos):
        if self.sid is not None or self.cerrando:
            raise RuntimeError('PCA ocupado o cerrando')
        self.sid = uuid.uuid4().hex
        self.pendiente = {'id': self.sid, 'operacion': operacion, 'argumentos': argumentos}
        if not self.activo:
            self.buffer = b''
            env = QProcessEnvironment.systemEnvironment()
            env.insert('PYTHONIOENCODING', 'utf-8')
            env.insert('PYTHONDONTWRITEBYTECODE', '1')
            self.p.setProcessEnvironment(env)
            self.p.setWorkingDirectory(str(self.raiz))
            self.p.start(self.interprete, ['-u', str(Path(__file__).with_name('trabajador.py')),
                                         '--raiz', str(self.raiz), '--dominio', self.dominio])
        else:
            self.enviar_pendiente()


class Backend(QObject):
    base = Signal(str, object)
    punto = Signal(str, object)
    error = Signal(str, str)
    cerrado = Signal()

    def __init__(self, raiz, python_audio=None, python_huellas=None, parent=None):
        super().__init__(parent)
        self.procesos = {d: ProcesoPCA(d, raiz, exe, self) for d, exe in
                         [('audio', python_audio), ('huellas', python_huellas)]}
        self.colas = {d: deque() for d in self.procesos}
        self.actual = {d: None for d in self.procesos}
        self.listos = {d: False for d in self.procesos}
        self.cerrando = False
        for d, proc in self.procesos.items():
            proc.evento.connect(lambda tipo, datos, dominio=d: self.recibir(dominio, tipo, datos))
            proc.terminado.connect(self.comprobar_cierre)
        self.limite = QTimer(self)
        self.limite.setSingleShot(True)
        self.limite.timeout.connect(self.forzar)

    def iniciar(self, modelos=None, datos=None):
        for d in self.procesos:
            self.encolar(d, 'base', modelo=(modelos or {}).get(d), datos=(datos or {}).get(d))

    def encolar(self, dominio, operacion, **args):
        if not self.cerrando:
            self.colas[dominio].append((operacion, args))
            self.despachar(dominio)

    def despachar(self, dominio):
        if self.cerrando or self.actual[dominio] is not None or not self.colas[dominio]:
            return
        operacion, args = self.colas[dominio].popleft()
        if operacion != 'base' and not self.listos[dominio]:
            self.error.emit(dominio, 'PCA no disponible; la clasificación continúa normalmente.')
            return
        self.actual[dominio] = operacion
        self.procesos[dominio].solicitar(operacion, **args)

    def recibir(self, dominio, tipo, datos):
        if tipo not in ('resultado', 'error') or self.cerrando:
            return
        operacion, self.actual[dominio] = self.actual[dominio], None
        if tipo == 'error':
            self.error.emit(dominio, datos['mensaje'])
            if operacion == 'base':
                self.colas[dominio].clear()
        elif operacion == 'base':
            self.listos[dominio] = True
            self.base.emit(dominio, datos)
        else:
            self.punto.emit(dominio, datos)
        self.despachar(dominio)

    def cerrar(self):
        if self.cerrando:
            return
        self.cerrando = True
        for d, p in self.procesos.items():
            self.colas[d].clear()
            p.cerrar()
        self.limite.start(180000)
        self.comprobar_cierre()

    def forzar(self):
        for p in self.procesos.values():
            p.forzar_arbol()

    def comprobar_cierre(self):
        if self.cerrando and not any(p.activo for p in self.procesos.values()):
            self.limite.stop()
            self.cerrado.emit()
