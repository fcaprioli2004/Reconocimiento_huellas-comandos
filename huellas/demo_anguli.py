#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anguli -> 2 o 3 transformaciones en cadena -> normalización -> reconocimiento.

    py demo_anguli.py --listar
    py demo_anguli.py --identidad 1 --anguli "C:/ruta/Anguli.exe" --modo identificar
    py demo_anguli.py --identidad intruso --anguli "C:/ruta/Anguli.exe" --modo autorizar
    py demo_anguli.py --identidad 1 --anguli "C:/ruta/Anguli.exe" --solo-generar
    py demo_anguli.py --identidad 1 --anguli "C:/ruta/Anguli.exe" --preparar
    py demo_anguli.py --identidad 1 --cantidad-transformaciones 3 --modo identificar
    py demo_anguli.py --identidad 1 --transformaciones rotacion gaussiano --semilla-transformaciones 42
    py demo_anguli.py --identidad 1 --sin-transformaciones

Copiar junto a transformaciones_en_linea.py y los cinco archivos del reconocimiento
CORREGIDO. --solo-generar también aplica las transformaciones, sin reconocer.
--listar y --preparar no requieren OpenCV. --preparar no ejecuta Anguli.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from generador_anguli import GeneradorAnguli, write_json

NOMBRES_TRANSFORMACIONES = ('rotacion', 'traslacion', 'escalado', 'elastica', 'presion',
                           'recorte', 'contraste', 'gaussiano', 'sal_pimienta')


def _validar_transformaciones(transformar, cantidad, semilla, nombres):
    if not transformar:
        if cantidad is not None or semilla is not None or nombres is not None:
            raise ValueError('--sin-transformaciones no se combina con parámetros de transformaciones')
        return
    if cantidad is not None and cantidad not in (2, 3):
        raise ValueError('La cantidad debe ser 2 o 3')
    if semilla is not None and (isinstance(semilla, bool) or not isinstance(semilla, int)
                                or not 0 <= semilla < 2**64):
        raise ValueError('La semilla debe ser un entero entre 0 y 2**64-1')
    if nombres is not None:
        if len(nombres) not in (2, 3) or len(set(nombres)) != len(nombres):
            raise ValueError('Elegí exactamente 2 o 3 transformaciones distintas')
        if set(nombres)-set(NOMBRES_TRANSFORMACIONES):
            raise ValueError('Nombre de transformación desconocido')
        if cantidad is not None and cantidad != len(nombres):
            raise ValueError('La cantidad no coincide con las transformaciones indicadas')


def _preparar_imagen_reconocimiento(generation, transformar=True, cantidad=None,
                                   semilla=None, nombres=None):
    run = Path(generation['carpeta_ejecucion'])
    original = Path(generation['imagen_generada'])
    try:
        if transformar:
            from transformaciones_en_linea import transformar_archivo
            target = run/'transformaciones'/'huella_transformada.png'
            meta = transformar_archivo(original, target, cantidad, semilla, nombres)
            generation['imagen_a_procesar'] = str(target)
            generation['transformaciones'] = {'habilitadas': True, **meta}
        else:
            generation['imagen_a_procesar'] = str(original)
            generation['transformaciones'] = {'habilitadas': False, 'cantidad': 0, 'aplicadas': []}
        generation['estado_transformaciones'] = 'ok'
        return generation
    except Exception as exc:
        generation['estado_transformaciones'] = 'error'
        generation['error_transformaciones'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(run/'generacion.json', generation)


def generar_y_reconocer(identidad, config, anguli=None, modo='autorizar',
                        modelo=None, umbrales=None, salida=None, *, transformar=True,
                        cantidad_transformaciones=None, semilla_transformaciones=None,
                        nombres_transformaciones=None):
    """Punto de entrada para la futura GUI; ejecutar en un trabajador.

    La identidad elegida sólo llega al generador. El reconocedor recibe
    exclusivamente los píxeles y el modelo: nunca se fuerza 0 para intrusos
    ni 1 para una identidad autorizada seleccionada.
    """
    if modo not in {'autorizar', 'identificar'}:
        raise ValueError('Modo debe ser autorizar o identificar')
    _validar_transformaciones(transformar, cantidad_transformaciones,
                             semilla_transformaciones, nombres_transformaciones)
    # Validar modelo/umbrales ANTES de iniciar la generación costosa.
    import reconocer_huella as rec
    base = Path(config).resolve().parent
    model_path = Path(modelo).resolve() if modelo else base/'modelo_kmeans.joblib'
    threshold_path = Path(umbrales).resolve() if umbrales else base/'umbrales_rechazo.json'
    model, settings = rec.load_experiment(model_path, threshold_path)
    generator = GeneradorAnguli(config, ejecutable=anguli, salida=salida)
    generation = generator.generar(identidad)
    run = Path(generation['carpeta_ejecucion'])
    try:
        _preparar_imagen_reconocimiento(generation, transformar, cantidad_transformaciones,
                                       semilla_transformaciones, nombres_transformaciones)
        # Única entrada al normalizador: la imagen final de la cadena.
        vector, images, info, n_minutiae = rec.extract_vector(Path(generation['imagen_a_procesar']), settings)
        prediction = rec.predict_vector(vector, model, settings)
        if (prediction['decision'] == 'Autorizada') != (prediction['distancia'] <= prediction['limite_distancia']):
            raise RuntimeError('La decisión no coincide con el límite aplicado. '
                               'Reemplazá reconocer_huella.py por la versión corregida.')
        accepted = int(prediction['decision'] == 'Autorizada')
        result = {'estado': 'ok', 'modo': modo, 'autorizada': accepted,
                  'identidad': prediction['prediccion'] if accepted and modo == 'identificar' else None,
                  'seleccion_generador': str(identidad),
                  'imagen_generada': generation['imagen_generada'],
                  'imagen_a_procesar': generation['imagen_a_procesar'],
                  'transformaciones': generation['transformaciones'],
                  'minucias': n_minutiae, 'avisos': info.get('avisos', ''),
                  'reconocimiento': prediction}
        for folder, image in zip(['grises', 'binarias', 'esqueletos', 'mascaras'], images):
            rec.norm.guardar(run/'procesamiento'/(folder+'.png'), image)
    except Exception as exc:
        # Un fallo no se etiqueta como intruso: no hay decisión biométrica.
        result = {'estado': 'error', 'autorizada': None, 'identidad': None,
                  'seleccion_generador': str(identidad),
                  'imagen_generada': generation['imagen_generada'],
                  'imagen_a_procesar': generation.get('imagen_a_procesar'),
                  'transformaciones': generation.get('transformaciones'),
                  'error': f'{type(exc).__name__}: {exc}'}
        write_json(run/'resultado.json', result)
        raise
    write_json(run/'resultado.json', result)
    result['archivo_resultado'] = str(run/'resultado.json')
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=here/'config_anguli.json')
    parser.add_argument('--anguli', help='Ruta al ejecutable original, con sus recursos y bibliotecas')
    parser.add_argument('--identidad', choices=['1', '2', '3', 'intruso'])
    parser.add_argument('--modo', choices=['autorizar', 'identificar'], default='autorizar')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--listar', action='store_true')
    action.add_argument('--preparar', action='store_true')
    action.add_argument('--solo-generar', action='store_true')
    parser.add_argument('--modelo', type=Path)
    parser.add_argument('--umbrales', type=Path)
    parser.add_argument('--salida', type=Path)
    parser.add_argument('--cantidad-transformaciones', type=int, choices=[2, 3],
                        help='Cantidad fija; por defecto sortea 2 o 3 en cada llamada')
    parser.add_argument('--semilla-transformaciones', type=int,
                        help='Reproducir los aumentos de Python sobre la misma entrada')
    parser.add_argument('--transformaciones', nargs='+', choices=NOMBRES_TRANSFORMACIONES,
                        help='Elegir exactamente 2 o 3 métodos; por defecto se sortean sin repetición')
    parser.add_argument('--sin-transformaciones', action='store_true',
                        help='Control: procesar directamente la impresión original de Anguli')
    args = parser.parse_args()
    if not args.listar and args.identidad is None:
        parser.error('Indicá --identidad 1, 2, 3 o intruso')
    try:
        _validar_transformaciones(not args.sin_transformaciones, args.cantidad_transformaciones,
                                 args.semilla_transformaciones, args.transformaciones)
        generator = GeneradorAnguli(args.config, args.anguli, args.salida)
        if args.listar:
            print(json.dumps(generator.disponibles(), ensure_ascii=False, indent=2))
        elif args.preparar:
            result = generator.preparar(args.identidad, verificar_ejecutable=False)
            result['transformaciones_solicitadas'] = {
                'habilitadas': not args.sin_transformaciones,
                'cantidad': (0 if args.sin_transformaciones else len(args.transformaciones)
                             if args.transformaciones else args.cantidad_transformaciones
                             or 'aleatoria: 2 o 3'),
                'semilla': args.semilla_transformaciones,
                'nombres': ([] if args.sin_transformaciones else args.transformaciones
                            or 'selección aleatoria sin repetición')}
            write_json(Path(result['carpeta_ejecucion'])/'generacion.json', result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print('Plan guardado. Anguli NO fue ejecutado.')
        elif args.solo_generar:
            generation = generator.generar(args.identidad)
            _preparar_imagen_reconocimiento(generation, not args.sin_transformaciones,
                                           args.cantidad_transformaciones,
                                           args.semilla_transformaciones, args.transformaciones)
            print(json.dumps(generation, ensure_ascii=False, indent=2))
        else:
            result = generar_y_reconocer(args.identidad, args.config, args.anguli,
                                        args.modo, args.modelo, args.umbrales, args.salida,
                                        transformar=not args.sin_transformaciones,
                                        cantidad_transformaciones=args.cantidad_transformaciones,
                                        semilla_transformaciones=args.semilla_transformaciones,
                                        nombres_transformaciones=args.transformaciones)
            print(f"Autorizada: {result['autorizada']}")
            if args.modo == 'identificar':
                print('Identidad:', result['identidad'] or 'No autorizada')
            print('Imagen original de Anguli:', result['imagen_generada'])
            print('Imagen que se procesó:', result['imagen_a_procesar'])
            print('Transformaciones:', ', '.join(t['nombre'] for t in result['transformaciones']['aplicadas']) or 'ninguna')
            print('Resultado:', result['archivo_resultado'])
        return 0
    except Exception as exc:
        # El registro detallado queda en la carpeta de ejecución si Anguli ya inició.
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
