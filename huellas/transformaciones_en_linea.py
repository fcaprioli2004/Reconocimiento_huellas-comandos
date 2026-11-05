#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aplica 2 o 3 aumentos distintos EN CADENA a una impresión de Anguli.

Basado en generar_transformaciones(2).py proporcionado por el usuario.
Conserva los parámetros efectivos de ejecutar(), no los defaults de las
funciones sin usar: elástica 5px, sal/pimienta 5%, recorte 8%, contraste 0.5.
El script original de la base de datos no se modifica ni se ejecuta.

Selección sin repetición. Orden de aplicación fijo entre las seleccionadas:
rotación, traslación, escala, elástica, presión, recorte, contraste,
ruido gaussiano, sal/pimienta. Recorte precede al ruido añadido para no
estimar su región sobre esos nuevos píxeles de ruido.

    py transformaciones_en_linea.py --imagen "entrada.png" --salida "salida.png"
    py transformaciones_en_linea.py --imagen "entrada.png" --salida "otra.png" --cantidad 3 --semilla 42
    py transformaciones_en_linea.py --imagen "entrada.png" --salida "otra.png" --transformaciones rotacion gaussiano

Guarda imagen final, JSON con orden/parámetros/semilla y pasos intermedios.
Una semilla fija reproduce los aumentos de Python sobre la MISMA entrada;
no fija el generador aleatorio interno de Anguli.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys

import cv2
import numpy as np

ORDEN = ('rotacion', 'traslacion', 'escalado', 'elastica', 'presion',
         'recorte', 'contraste', 'gaussiano', 'sal_pimienta')


def validar_solicitud(cantidad=None, nombres=None, semilla=None):
    if cantidad is not None and cantidad not in (2, 3):
        raise ValueError('La cantidad de transformaciones debe ser 2 o 3')
    if nombres is not None:
        if not 2 <= len(nombres) <= 3 or len(set(nombres)) != len(nombres):
            raise ValueError('Elegí 2 o 3 transformaciones distintas')
        if set(nombres)-set(ORDEN):
            raise ValueError('Transformaciones desconocidas: '+', '.join(sorted(set(nombres)-set(ORDEN))))
        if cantidad is not None and len(nombres) != cantidad:
            raise ValueError('Cantidad incompatible con la lista de transformaciones')
    if semilla is not None and (isinstance(semilla, bool) or not isinstance(semilla, int)
                                or not 0 <= semilla < 2**64):
        raise ValueError('La semilla debe ser un entero entre 0 y 2**64-1')


def transformar_afin(imagen, angulo=0.0, escala=1.0, tx=0, ty=0):
    alto, ancho = imagen.shape
    matriz = cv2.getRotationMatrix2D((ancho/2, alto/2), angulo, escala)
    matriz[:, 2] += [tx, ty]
    return cv2.warpAffine(imagen, matriz, (ancho, alto), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def deformacion_elastica(imagen, rng, amplitud=5):
    alto, ancho = imagen.shape
    ny, nx = max(3, alto//50), max(3, ancho//50)
    def campo():
        base = rng.normal(0, 1, (ny, nx)).astype(np.float32)
        suave = cv2.resize(base, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        suave = cv2.GaussianBlur(suave, (0, 0), sigmaX=12)
        maximo = max(float(np.max(np.abs(suave))), 1e-6)
        return amplitud*suave/maximo
    dx, dy = campo(), campo()
    x, y = np.meshgrid(np.arange(ancho, dtype=np.float32), np.arange(alto, dtype=np.float32))
    return cv2.remap(imagen, x+dx, y+dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def presion_simulada(imagen):
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(imagen, nucleo, iterations=1)


def ruido_gaussiano(imagen, rng, sigma=5):
    ruido = rng.normal(0, sigma, imagen.shape)
    return np.clip(imagen.astype(np.float32)+ruido, 0, 255).astype(np.uint8)


def ruido_sal_pimienta(imagen, rng, proporcion=.05):
    salida = imagen.copy()
    cantidad = min(imagen.size, max(2, round(imagen.size*proporcion)))
    indices = rng.choice(imagen.size, size=cantidad, replace=False)
    plano = salida.reshape(-1)
    mitad = cantidad//2
    plano[indices[:mitad]] = 0
    plano[indices[mitad:]] = 255
    return salida


def recorte_parcial(imagen, lado='derecha', fraccion=.03):
    salida = imagen.copy()
    yy, xx = np.where(imagen < 220)
    if xx.size == 0:
        return salida
    izquierda, derecha = int(xx.min()), int(xx.max())+1
    arriba, abajo = int(yy.min()), int(yy.max())+1
    if lado == 'derecha':
        desde = derecha-max(1, round((derecha-izquierda)*fraccion))
        salida[:, desde:derecha] = 255
    elif lado == 'izquierda':
        hasta = izquierda+max(1, round((derecha-izquierda)*fraccion))
        salida[:, izquierda:hasta] = 255
    elif lado == 'arriba':
        hasta = arriba+max(1, round((abajo-arriba)*fraccion))
        salida[arriba:hasta, :] = 255
    elif lado == 'abajo':
        desde = abajo-max(1, round((abajo-arriba)*fraccion))
        salida[desde:abajo, :] = 255
    else:
        raise ValueError('Lado de recorte inválido')
    return salida


def contraste(imagen, factor=.8):
    datos = factor*(imagen.astype(np.float32)-127.5)+127.5
    return np.clip(datos, 0, 255).astype(np.uint8)


def aplicar_transformaciones(imagen, cantidad=None, semilla=None, nombres=None):
    validar_solicitud(cantidad, nombres, semilla)
    if imagen.ndim != 2 or imagen.dtype != np.uint8 or min(imagen.shape) < 3:
        raise ValueError('Se espera imagen gris uint8 de al menos 3x3 píxeles')
    seed = secrets.randbits(64) if semilla is None else semilla
    # Separar sorteos permite repetir con la cantidad o los nombres registrados
    # sin desplazar la secuencia de ruido/parámetros de las transformaciones.
    streams = np.random.SeedSequence(seed).spawn(3)
    rng_count, rng_select, rng = [np.random.default_rng(stream) for stream in streams]
    if nombres is None:
        count = int(rng_count.integers(2, 4)) if cantidad is None else cantidad
        names = set(rng_select.choice(ORDEN, size=count, replace=False).tolist())
    else:
        names = set(nombres)
    ordered = [name for name in ORDEN if name in names]
    salida = imagen.copy()
    pasos, registro = [], []
    for name in ordered:
        antes = salida
        if name == 'rotacion':
            p = {'angulo_grados': int(rng.choice([-10, 10]))}
            salida = transformar_afin(antes, angulo=p['angulo_grados'])
        elif name == 'traslacion':
            p = {'tx': int(rng.integers(-8, 8)), 'ty': int(rng.integers(-8, 8))}
            salida = transformar_afin(antes, tx=p['tx'], ty=p['ty'])
        elif name == 'escalado':
            p = {'factor': float(rng.choice([.85, 1.15]))}
            salida = transformar_afin(antes, escala=p['factor'])
        elif name == 'elastica':
            p = {'amplitud_px': 3}
            salida = deformacion_elastica(antes, rng, p['amplitud_px'])
        elif name == 'presion':
            p = {'metodo': 'erosion_3x3', 'iteraciones': 1}
            salida = presion_simulada(antes)
        elif name == 'recorte':
            p = {'lado': str(rng.choice(['izquierda', 'derecha', 'arriba', 'abajo'])), 'fraccion_roi': .05}
            salida = recorte_parcial(antes, p['lado'], p['fraccion_roi'])
        elif name == 'contraste':
            p = {'factor': .8}
            salida = contraste(antes, p['factor'])
        elif name == 'gaussiano':
            p = {'sigma': 5}
            salida = ruido_gaussiano(antes, rng, p['sigma'])
        elif name == 'sal_pimienta':
            p = {'proporcion': .01}
            salida = ruido_sal_pimienta(antes, rng, p['proporcion'])
        registro.append({'orden': len(registro)+1, 'nombre': name, 'parametros': p,
                         'pixeles_modificados_en_paso': int(np.count_nonzero(salida != antes))})
        pasos.append(salida.copy())
    meta = {'version': 1, 'semilla_python': seed, 'cantidad': len(ordered),
            'esquema_aleatorio': 'SeedSequence.spawn(3): cantidad, seleccion, parametros',
            'aplicadas': registro, 'alto': imagen.shape[0], 'ancho': imagen.shape[1],
            'pixeles_modificados_total': int(np.count_nonzero(salida != imagen)),
            'versiones': {'numpy': np.__version__, 'opencv': cv2.__version__}}
    return salida, meta, pasos


def guardar_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError('No se pudo codificar la imagen PNG')
    encoded.tofile(str(path))


def transformar_archivo(origen, destino, cantidad=None, semilla=None, nombres=None):
    source, target = Path(origen).resolve(), Path(destino).resolve()
    if source == target:
        raise ValueError('La salida debe ser distinta de la imagen original')
    if target.suffix.lower() != '.png':
        raise ValueError('La salida transformada debe tener extensión .png')
    record = target.with_suffix('.json')
    steps_dir = target.parent/(target.stem+'_pasos')
    if target.exists() or record.exists() or steps_dir.exists():
        raise FileExistsError(f'Ya existen resultados para {target.name}; elegí otra salida')
    raw = source.read_bytes()
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f'No se pudo leer la imagen de Anguli: {source}')
    image, meta, steps = aplicar_transformaciones(gray, cantidad, semilla, nombres)
    meta.update(imagen_origen=str(source), imagen_transformada=str(target),
                sha256_archivo_origen=hashlib.sha256(raw).hexdigest())
    for step, item in zip(steps, meta['aplicadas']):
        path = steps_dir/f"{item['orden']:02d}_{item['nombre']}.png"
        guardar_png(path, step)
        item['archivo_paso'] = str(path)
    guardar_png(target, image)
    meta['sha256_archivo_transformado'] = hashlib.sha256(target.read_bytes()).hexdigest()
    meta['archivo_registro'] = str(record)
    record.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--imagen', type=Path, required=True)
    parser.add_argument('--salida', type=Path, required=True)
    parser.add_argument('--cantidad', type=int, choices=[2, 3])
    parser.add_argument('--semilla', type=int)
    parser.add_argument('--transformaciones', choices=ORDEN, nargs='+')
    args = parser.parse_args()
    try:
        meta = transformar_archivo(args.imagen, args.salida, args.cantidad, args.semilla, args.transformaciones)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
