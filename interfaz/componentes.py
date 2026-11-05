import math
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QSize
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget, QSizePolicy


class Imagen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.texto = 'Generá una huella para comenzar'
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def cargar(self, ruta=None):
        self.pixmap = QPixmap(str(ruta)) if ruta else QPixmap()
        self.texto = 'Imagen no disponible' if ruta else 'Generá una huella para comenzar'
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor('#294e6a'), 1))
        p.setBrush(QColor('#030e17'))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        if self.pixmap.isNull():
            p.setPen(QColor('#829db8'))
            p.drawText(self.rect(), Qt.AlignCenter, self.texto)
        else:
            area = self.rect().adjusted(16, 16, -16, -16)
            imagen = self.pixmap.scaled(area.size().boundedTo(QSize(290, 330)), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap(area.center().x() - imagen.width() // 2,
                         area.center().y() - imagen.height() // 2, imagen)


class Onda(QWidget):
    def __init__(self):
        super().__init__()
        self.datos = []
        self.tipo = 'real'
        self.animada = False
        self.paso = 0
        self.setMinimumHeight(60)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.avanzar)

    def avanzar(self):
        self.paso += 1
        self.update()

    def animar(self, activar):
        self.animada = activar
        if activar:
            self.timer.start(70)
        else:
            self.timer.stop()
        self.update()

    def cargar(self, datos, tipo='real'):
        self.tipo = tipo
        self.datos = datos
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor('#194561'), 1))
        p.setBrush(QColor('#020f18'))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        area = self.rect().adjusted(15, 20, -15, -28)
        centro = area.center().y()
        p.setPen(QPen(QColor('#00d8f6'), 1.5))
        if self.animada:
            for i in range(70):
                x = area.left() + i * area.width() / 69
                h = (6 + 16 * abs(math.sin(i * .4 + self.paso * .2)))
                p.drawLine(QPointF(x, centro - h), QPointF(x, centro + h))
            texto = 'Indicador animado · no representa una medición'
        elif self.datos:
            pico = max(max(abs(a), abs(b)) for a, b in self.datos) or 1
            escala = area.height() * .46 / pico
            for i, (a, b) in enumerate(self.datos):
                x = area.left() + i * area.width() / max(1, len(self.datos) - 1)
                p.drawLine(QPointF(x, centro - b * escala), QPointF(x, centro - a * escala))
            texto = ('Audio normalizado' if self.tipo == 'normalizada' else 'Captura real') + ' · amplitud autoajustada'
        else:
            p.drawLine(area.left(), centro, area.right(), centro)
            texto = 'La onda real aparecerá al terminar la captura'
        p.setPen(QColor('#8ba8c3'))
        p.drawText(self.rect().adjusted(8, 0, -8, -7), Qt.AlignBottom | Qt.AlignHCenter, texto)


class Emblema(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(42, 48)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor('#22b7ff'), 2.2, Qt.SolidLine, Qt.RoundCap))
        for i in range(5):
            p.drawArc(QRectF(3 + i*3.5, 4 + i*3.5, 35 - i*7, 39 - i*7), 5*16, 275*16)
