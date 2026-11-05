#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconocimiento EXPERIMENTAL con K-Means guardado y umbral de rechazo.

Instalar:
  py -m pip install numpy scipy scikit-learn opencv-python-headless scikit-image joblib
El modelo se generó con scikit-learn 1.9.1; conservar versiones compatibles.
Usar junto a normalizar_huellas.py, extraer_caracteristicas.py,
modelo_kmeans.joblib y umbrales_rechazo.json incluidos en el paquete.

Una imagen:
  py reconocer_huella.py --imagen "ruta/huella.png"
Una carpeta (recursivo):
  py reconocer_huella.py --carpeta "Huellas_pruebas"
Guardar además las imágenes procesadas:
  py reconocer_huella.py --carpeta "Huellas_intruso" --guardar-imagenes

No entrena ni recalibra. Aplica los mismos parámetros de normalización y
extracción documentados en umbrales_rechazo.json. Convierte los vectores al
tipo numérico de los centroides para compatibilidad float32/float64.
Distancia <= límite del clúster: devuelve la clase asociada.
Distancia > límite: devuelve 'No autorizada'. Distancia no es probabilidad.
La decisión usa EXCLUSIVAMENTE limites_por_cluster de umbrales_rechazo.json.
umbral_normalizado conserva la referencia de la calibración original, pero
no reemplaza límites por clúster que se hayan modificado. En cada resultado,
limite_puntaje_normalizado = limite_distancia / escala_del_cluster muestra
el umbral normalizado efectivo que corresponde al límite aplicado.
Un fallo de lectura/procesamiento se registra como 'No procesable', no como
un rechazo biométrico exitoso. En carpetas omite duplicados de píxeles dentro
de cada subcarpeta y registra qué archivo conserva.

El umbral incluido es experimental: en la prueba reservada aceptó 2 de 10
imágenes intrusas. No resuelve el problema de rechazo de desconocidos.
Para cambiar modelo o descriptores se necesita una nueva calibración.
Los hashes impiden mezclar accidentalmente un umbral con otro modelo/código.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

try:
    import numpy as np
    import cv2
    import joblib
    from threadpoolctl import threadpool_limits
    import normalizar_huellas as norm
    import extraer_caracteristicas as feat
except ImportError as exc:
    raise SystemExit('Falta una dependencia o uno de los dos scripts auxiliares. '
                     'Ver LEEME.md del paquete.') from exc

EXTENSIONS = {'.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff', '.pgm'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_experiment(model_path, config_path):
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if config.get('version') != 'rechazo_huellas_v1':
        raise ValueError('Formato de umbrales incompatible')
    checks = [(model_path, 'modelo_sha256'),
              (Path(norm.__file__), 'normalizador_sha256'),
              (Path(feat.__file__), 'extractor_sha256')]
    for path, field in checks:
        if sha256(path) != config[field]:
            raise ValueError(f'{path.name} no corresponde al experimento de umbrales; '
                             'usá los archivos incluidos en el mismo paquete')
    # Cargar sólo el modelo propio incluido; joblib no es un formato seguro
    # para modelos de procedencia desconocida.
    model = joblib.load(model_path)
    if model.get('tipo') != 'kmeans_huellas_v1':
        raise ValueError('Modelo incompatible')
    if not np.array_equal(model['nombres_caracteristicas'], feat.names_features()):
        raise ValueError('Características u orden incompatibles')
    limits = np.asarray(config['limites_por_cluster'], np.float64)
    if limits.shape != (3,) or not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError('Límites de rechazo inválidos')
    scales = np.asarray(config['escalas_por_cluster'], np.float64)
    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Escalas por clúster inválidas')
    if set(model['cluster_a_clase']) != {0, 1, 2}:
        raise ValueError('El modelo no tiene las tres clases asociadas')
    return model, config


def extract_vector(path, config):
    normal_args = SimpleNamespace(**config['normalizacion'])
    feature_args = SimpleNamespace(**config['extraccion'])
    images, normal_info = norm.procesar(path, normal_args)
    gray, binary_image, skeleton_image, mask_image = images[:4]
    binary, skeleton, mask = binary_image == 0, skeleton_image == 0, mask_image > 0
    if (skeleton & ~binary).any() or (skeleton & ~mask).any():
        raise ValueError('Esqueleto incompatible con binaria/máscara')
    feat.bbox(mask)
    if not 0.02 < binary[mask].mean() < .9 or skeleton.sum() < 20:
        raise ValueError('Binaria/esqueleto insuficiente o saturado')
    theta, quality = feat.orientation(gray)
    points, clean = feat.minutiae(skeleton, mask, theta, quality, feature_args)
    vector, blocks = feat.descriptors(gray, binary, mask, theta, quality, points)
    if not np.isfinite(vector).all():
        raise ValueError('Vector no finito')
    return vector, images[:4], normal_info, len(points)


def predict_vector(vector, model, config):
    X = np.asarray(vector).reshape(1, -1)
    if X.shape[1] != len(model['nombres_caracteristicas']) or not np.isfinite(X).all():
        raise ValueError('Vector incompatible o no finito')
    scaled = np.ascontiguousarray(model['scaler'].transform(X),
                                 dtype=model['kmeans'].cluster_centers_.dtype)
    with threadpool_limits(limits=1):
        distances = model['kmeans'].transform(scaled)[0]
    if not np.isfinite(distances).all():
        raise ValueError('Distancias no finitas')
    cluster = int(np.argmin(distances))
    distance = float(distances[cluster])
    scale = float(config['escalas_por_cluster'][cluster])
    limit = float(config['limites_por_cluster'][cluster])
    if not np.isfinite(limit) or limit <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Límite o escala de rechazo inválidos')
    score = distance/scale
    # Única fuente de decisión: el mismo límite que se informa al usuario.
    accepted = distance <= limit
    nearest = model['cluster_a_clase'][cluster]
    return {'cluster': cluster, 'clase_cercana': nearest,
            'distancia': distance, 'limite_distancia': limit,
            'puntaje_normalizado': score, 'limite_puntaje_normalizado': limit/scale,
            'decision': 'Autorizada' if accepted else 'No autorizada',
            'prediccion': nearest if accepted else 'No autorizada',
            **{f'distancia_C{k}': float(distances[k]) for k in range(3)}}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--imagen', type=Path)
    source.add_argument('--carpeta', type=Path)
    parser.add_argument('--modelo', type=Path, default=here/'modelo_kmeans.joblib')
    parser.add_argument('--umbrales', type=Path, default=here/'umbrales_rechazo.json')
    parser.add_argument('--salida', type=Path, default=here/'Resultados_reconocimiento')
    parser.add_argument('--guardar-imagenes', action='store_true')
    args = parser.parse_args()
    try:
        model, config = load_experiment(args.modelo, args.umbrales)
        cv2.setNumThreads(1)
        if args.imagen:
            if not args.imagen.is_file():
                raise ValueError('No existe la imagen')
            root = args.imagen.resolve().parent
            files = [args.imagen.resolve()]
        else:
            if not args.carpeta.is_dir():
                raise ValueError('No existe la carpeta')
            root = args.carpeta.resolve()
            output_base = args.salida.resolve()
            files = sorted(p for p in root.rglob('*') if p.is_file()
                           and p.suffix.lower() in EXTENSIONS
                           and not p.resolve().is_relative_to(output_base))
            files.sort(key=lambda p: (not p.name.startswith('imp_'), p.as_posix()))
        if not files:
            raise ValueError('No se encontraron imágenes compatibles')
        run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out = args.salida/run
        out.mkdir(parents=True, exist_ok=False)
        applied = {'regla_decision': 'distancia <= limite_distancia',
                   'limites_por_cluster_aplicados': config['limites_por_cluster'],
                   'escalas_por_cluster': config['escalas_por_cluster'],
                   'limites_normalizados_efectivos': [float(l)/float(s) for l, s in zip(
                       config['limites_por_cluster'], config['escalas_por_cluster'])],
                   'modelo_sha256': config['modelo_sha256'],
                   'umbrales_archivo_sha256': sha256(args.umbrales)}
        (out/'parametros_aplicados.json').write_text(json.dumps(applied, indent=2), encoding='utf-8')
        rows, duplicates, seen, vectors, vector_ids = [], [], {}, [], []
        for i, path in enumerate(files, 1):
            key = path.relative_to(root).as_posix()
            try:
                pixels = norm.leer(path)
                pixel_hash = hashlib.sha256(pixels.tobytes()+str(pixels.shape).encode()).hexdigest()
                duplicate_key = (path.parent, pixel_hash)
                if duplicate_key in seen:
                    duplicates.append({'omitido': key, 'conservado': seen[duplicate_key]})
                    continue
                seen[duplicate_key] = key
                vector, images, info, n_minutiae = extract_vector(path, config)
                row = {'id': key, 'estado': 'ok', **predict_vector(vector, model, config),
                       'minucias': n_minutiae, 'avisos': info.get('avisos', '')}
                if args.guardar_imagenes:
                    for folder, image in zip(['grises', 'binarias', 'esqueletos', 'mascaras'], images):
                        rel = Path(key).with_name(Path(key).name+'.png')
                        norm.guardar(out/'imagenes'/folder/rel, image)
                vectors.append(vector); vector_ids.append(key)
                print(f"[{i}/{len(files)}] {key}: {row['prediccion']} (distancia {row['distancia']:.3f})", flush=True)
            except Exception as exc:
                row = {'id': key, 'estado': 'error', 'decision': 'No procesable',
                       'prediccion': 'No procesable', 'avisos': f'{type(exc).__name__}: {exc}'}
                print(f'[{i}/{len(files)}] ERROR {key}: {exc}', flush=True)
            rows.append(row)
        fields = ['id', 'estado', 'decision', 'prediccion', 'cluster', 'clase_cercana',
                  'distancia', 'limite_distancia', 'puntaje_normalizado',
                  'limite_puntaje_normalizado', 'distancia_C0',
                  'distancia_C1', 'distancia_C2', 'minucias', 'avisos']
        with (out/'predicciones.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fields, delimiter=';'); writer.writeheader(); writer.writerows(rows)
        (out/'predicciones.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
        (out/'duplicados.json').write_text(json.dumps(duplicates, indent=2, ensure_ascii=False), encoding='utf-8')
        if vectors:
            np.savez_compressed(out/'vectores.npz', X=np.stack(vectors), ids=np.asarray(vector_ids),
                                nombres_caracteristicas=feat.names_features())
        print(f'Resultados: {out}')
        print('Experimento de rechazo: hubo falsas aceptaciones en la evaluación incluida; ver informe.')
        return 1 if any(r['estado'] == 'error' for r in rows) else 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
