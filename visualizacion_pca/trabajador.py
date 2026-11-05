import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raiz', type=Path, required=True)
    p.add_argument('--dominio', choices=['audio', 'huellas'], required=True)
    a = p.parse_args()
    if os.name != 'nt':
        os.setsid()
    sys.path.insert(0, str(a.raiz / a.dominio))
    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    from visualizacion_pca.proyeccion import Proyector
    salida = sys.stdout
    proyector = None
    for linea in sys.stdin:
        req = json.loads(linea)
        if req['operacion'] == 'cerrar':
            break
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if req['operacion'] == 'base':
                    proyector = Proyector(a.dominio, a.raiz, **req['argumentos'])
                    datos = proyector.base()
                elif req['operacion'] == 'proyectar' and proyector is not None:
                    datos = proyector.proyectar(**req['argumentos'])
                else:
                    raise ValueError('El PCA de base todavía no está disponible')
            tipo = 'resultado'
        except (Exception, SystemExit) as exc:
            traceback.print_exc(file=sys.stderr)
            tipo, datos = 'error', {'mensaje': f'{type(exc).__name__}: {exc}'}
        print('@@HV@@' + json.dumps({'tipo': tipo, 'datos': datos, 'id': req['id']},
                                    default=str, ensure_ascii=False), file=salida, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
