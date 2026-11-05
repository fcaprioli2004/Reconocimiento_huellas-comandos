#!/usr/bin/env python3
"""Lanzador opcional. Abre la interfaz existente y una segunda ventana PCA 3D."""
import argparse
import logging
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python-audio')
    p.add_argument('--python-huellas')
    p.add_argument('--anguli')
    p.add_argument('--pca-audio', type=Path)
    p.add_argument('--pca-huellas', type=Path)
    p.add_argument('--datos-audio', type=Path)
    p.add_argument('--datos-huellas', type=Path)
    args = p.parse_args()
    raiz = Path(__file__).resolve().parent
    if not all((raiz / d).is_dir() for d in ('audio', 'huellas', 'interfaz', 'integracion')):
        p.error('Copiá ARCHIVOS_NUEVOS dentro de IA_Trabajo_final, junto a la integración que ya funciona.')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from integracion.controlador import Controlador
    from interfaz.ventana_principal import VentanaPrincipal
    from visualizacion_pca.sesion import SesionPCA
    logs = raiz / 'salidas_integracion/logs'
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(logs / 'pca.log', encoding='utf-8'), logging.StreamHandler()])
    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    c = Controlador(raiz, args.python_audio, args.python_huellas)
    v = VentanaPrincipal(c, args.anguli)
    sesion = SesionPCA(c, raiz, args.python_audio, args.python_huellas,
                      modelos={'audio': args.pca_audio, 'huellas': args.pca_huellas},
                      datos={'audio': args.datos_audio, 'huellas': args.datos_huellas})
    def terminar_pca():
        sesion.ventana.close()
        sesion.backend.cerrar()
    c.cerrado.connect(terminar_pca)
    sesion.backend.cerrado.connect(app.quit)
    v.show()
    sesion.ventana.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
