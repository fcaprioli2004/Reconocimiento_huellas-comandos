#!/usr/bin/env python3
"""Inicio único. Ejecutar desde cualquier directorio; recursos relativos a este archivo."""
import argparse
import logging
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description='Huellas + Voz · Integración de escritorio')
    p.add_argument('--python-audio', help='Python del entorno funcional de audio')
    p.add_argument('--python-huellas', help='Python del entorno funcional de huellas')
    p.add_argument('--anguli', help='Ruta opcional de Anguli.exe; no cambia la configuración original')
    args = p.parse_args()
    raiz = Path(__file__).resolve().parent
    faltantes = [x for x in ('audio', 'huellas') if not (raiz / x).is_dir()]
    if faltantes:
        p.error('Copiá ARCHIVOS_NUEVOS dentro de IA_Trabajo_final, junto a audio/ y huellas/. Falta: ' + ', '.join(faltantes))
    logs = raiz / 'salidas_integracion/logs'
    logs.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[RotatingFileHandler(logs / 'interfaz.log', maxBytes=2_000_000, backupCount=3, encoding='utf-8'),
                  logging.StreamHandler()])
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        p.error('Falta PySide6 en este intérprete. Instalá requirements_integracion.txt con este mismo Python.')
    from integracion.controlador import Controlador
    from interfaz.ventana_principal import VentanaPrincipal
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Huellas + Voz')
    c = Controlador(raiz, args.python_audio, args.python_huellas)
    v = VentanaPrincipal(c, args.anguli)
    v.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
