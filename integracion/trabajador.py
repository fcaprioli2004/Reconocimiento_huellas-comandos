"""Servidor JSON por proceso. No importar los dos dominios en un intérprete."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

MARCA = '@@HV@@'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dominio', choices=['audio', 'huellas'], required=True)
    p.add_argument('--raiz', type=Path, required=True)
    a = p.parse_args()
    if os.name != 'nt':
        os.setsid()  # Permite terminar también los descendientes al vencer el cierre.
    raiz = a.raiz.resolve()
    sys.path.insert(0, str(raiz / a.dominio))
    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    salida = sys.stdout

    def emitir(tipo, datos, solicitud):
        print(MARCA + json.dumps({'tipo': tipo, 'datos': datos, 'id': solicitud},
                                ensure_ascii=False, default=str), file=salida, flush=True)

    adaptador = None
    for linea in sys.stdin:
        req = json.loads(linea)
        sid = req['id']
        op = req['operacion']
        if op == 'cerrar':
            return 0
        try:
            # Los mensajes de bibliotecas van a stderr. El subproceso de captura
            # conserva su stdout nativo: su aviso GRABANDO se recibe por separado.
            with contextlib.redirect_stdout(sys.stderr):
                if adaptador is None:
                    if a.dominio == 'audio':
                        from integracion.adaptador_audio import AdaptadorAudio
                        adaptador = AdaptadorAudio(raiz)
                    else:
                        from integracion.adaptador_huellas import AdaptadorHuellas
                        adaptador = AdaptadorHuellas(raiz)
                args = req.get('argumentos', {})
                evento = lambda tipo, datos: emitir(tipo, datos, sid)
                if a.dominio == 'huellas' and op == 'generar':
                    resultado = adaptador.generar(**args)
                elif a.dominio == 'audio' and op in ('grabar', 'archivo'):
                    resultado = getattr(adaptador, op)(emitir=evento, **args)
                elif a.dominio == 'audio' and op in ('dispositivos', 'diagnostico'):
                    resultado = getattr(adaptador, op)()
                else:
                    raise ValueError('Operación no admitida')
            emitir('resultado', resultado, sid)
        except (Exception, SystemExit) as exc:
            traceback.print_exc(file=sys.stderr)
            emitir('error', {'mensaje': f'{type(exc).__name__}: {exc}'}, sid)
    return 0


if __name__ == '__main__':
    sys.exit(main())
