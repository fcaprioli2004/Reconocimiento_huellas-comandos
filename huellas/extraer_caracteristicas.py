#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extractor BASE de características de huellas (Python >= 3.10).

INSTALAR:
    py -m pip install numpy opencv-python-headless scipy scikit-image
EJECUTAR, al lado de data_sheet, después de normalizar:
    py extraer_caracteristicas.py --raiz "data_sheet" --marcar

Es independiente de normalizar_huellas.py: usa sus imágenes en grises,
binarias, esqueletos y máscaras; comprueba dimensiones/polaridad y recalcula
coherencia sobre los grises. No usa el PNG 'calidad' del normalizador.
Acepta también --raiz data_sheet/Huellas_1/Huellas_normalizadas.

Por imagen: vector fijo de 91 valores + lista variable de minucias.
  Orientación axial global: 16; orientación espacial 2x2x8: 32;
  LBP uniforme: 10; textura GLCM: 6; momentos de Hu: 7;
  densidad espacial de minucias 2x3x3: 18; densidad global por tipo: 2.
X contiene características en sus unidades originales. NO ajusta un scaler
ni normaliza L2 toda la fila: StandardScaler se ajusta SOLO con entrenamiento.
Guarda los bloques por separado para comparar alternativas.
Es una referencia clásica para K-Means k=3, no un embedding aprendido;
no garantiza invariancia a deformación elástica ni reconocimiento perfecto.

MINUCIAS:
Crossing Number sobre esqueleto: CN=1 terminación, CN=3 bifurcación.
Filtra componentes pequeños, proximidad al borde, coherencia insuficiente
y candidatos próximos (heurística). La orientación es la TANGENTE local
de la cresta, NO la dirección de cada rama de una bifurcación; se representa
módulo pi (sin sentido). La coherencia no es una probabilidad de autenticidad.
No deshace defectos de binarización ni valida todas las falsas minucias.
Conserva minucias en JSON/CSV para futuros comparadores geométricos.

SALIDA: data_sheet/Caracteristicas_huellas/<ejecucion_UTC>/
  vectores.npz: X, orientacion_global, orientacion_espacial, lbp, glcm, hu,
               densidad_espacial, densidad_global,
               ids, grupos, grupos_aumento, etiquetas, nombres_caracteristicas.
  vectores.csv: id + etiqueta + vector X (una fila por imagen).
  muestras.csv: índice de filas, rutas, calidad y cantidades de minucias.
  minucias.csv: id, tipo, posición, tangente, coherencia, distancia al borde.
  etiquetas_para_completar.csv: plantilla para definir identidad real.
  errores.csv: archivos incompletos/ilegibles; continúa con los restantes.
  configuracion.json: parámetros, definición de bloques y versiones.
  por_imagen/<grupo>/<ruta>.npz: vector individual.
  por_imagen/<grupo>/<ruta>.json: minucias y metadatos individuales.
  marcadas/<grupo>/<ruta>.png: opcional con --marcar.

ETIQUETAS:
No supone que Huellas_1 represente un dedo/persona. Por defecto etiqueta="".
El grupo es sólo la carpeta contenedora, NO una identidad confirmada.
Completá la columna 'etiqueta' de etiquetas_para_completar.csv y ejecutá:
    py extraer_caracteristicas.py --raiz data_sheet --etiquetas "ruta/etiquetas.csv"
Alternativa, SOLO si cada carpeta representa un único dedo:
    py extraer_caracteristicas.py --raiz data_sheet --identidad-carpeta
Variantes como imp_01_original e imp_01_rotacion comparten grupo_aumento.
No se agrupan/median automáticamente sus vectores. Al evaluar, separar por
identidad y mantener variantes de una misma impresión en el mismo conjunto.
Para verificar dedos conocidos, diseñar una partición por impresiones/sesiones
independientes; no repartir aumentos de una impresión entre train y test.

CARGAR PARA ML:
    import numpy as np
    datos = np.load('.../vectores.npz', allow_pickle=False)
    X, y = datos['X'], datos['etiquetas']
    # X[i] corresponde a ids[i] y a la fila i de muestras.csv.
ENTRENAMIENTO K-MEANS (requiere scikit-learn; usar SOLO datos de entrenamiento):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    modelo = make_pipeline(StandardScaler(),
                           KMeans(n_clusters=3, n_init=20, random_state=42))
    modelo.fit(X_train)
    clusters_test = modelo.predict(X_test)
Centroides: modelo.named_steps['kmeans'].cluster_centers_ (espacio escalado).
Guardar scaler y KMeans juntos para predecir nuevas huellas del mismo modo.
Los índices 0/1/2 son arbitrarios; asignarlos a clases autorizadas requiere
etiquetas de entrenamiento o validación. k=3 no rechaza desconocidos:
K-Means siempre asigna el centro más cercano. Si hace falta rechazo,
calibrar un umbral de distancia con validación independiente.

REFERENCIAS:
https://scikit-image.org/docs/stable/api/skimage.feature.html
https://doi.org/10.3390/s24020664
"""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

try:
    import cv2
    import numpy as np
    import scipy
    from scipy import ndimage as ndi
    import skimage
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    from skimage.morphology import skeletonize
except ImportError as exc:
    raise SystemExit('Instalá dependencias: py -m pip install numpy '
                     'opencv-python-headless scipy scikit-image') from exc

VERSION = '1.0'
BLOCKS = [('orientacion_global', 16, 'Histograma axial, ponderado por coherencia'),
          ('orientacion_espacial', 32, 'Histograma axial en grilla 2x2x8'),
          ('lbp', 10, 'LBP uniforme P=8 R=1, frecuencias relativas'),
          ('glcm', 6, 'GLCM en ROI; promedio de direcciones y distancias'),
          ('hu', 7, 'Momentos de Hu de crestas binarias, log firmado'),
          ('densidad_espacial', 18, 'Terminaciones/bifurcaciones 3x3, por 10000 px de ROI'),
          ('densidad_global', 2, 'Terminaciones/bifurcaciones por 10000 px de ROI')]
FOLDERS = {'binarias', 'esqueletos', 'mascaras', 'calidad'}
EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm'}


def read(path):
    a = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f'Imagen ilegible: {path.name}')
    if a.ndim != 2 or a.dtype != np.uint8:
        raise ValueError(f'Se espera imagen en grises de 8 bits: {path.name}')
    return a


def write_png(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', a)
    if not ok:
        raise OSError(f'No se pudo codificar {path.name}')
    encoded.tofile(str(path))


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


def orientation(gray, sigma=5):
    a = gray.astype(np.float32)/255
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    smooth = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    xx, yy, xy = smooth(gx*gx), smooth(gy*gy), smooth(gx*gy)
    q = np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-8)
    tangent = (0.5*np.arctan2(2*xy, xx-yy)+np.pi/2) % np.pi
    return tangent, np.clip(q, 0, 1)


def crossing_number(skeleton):
    s = np.pad(skeleton.astype(np.int8), 1)
    h, w = skeleton.shape
    offsets = [(-1,0), (-1,1), (0,1), (1,1),
               (1,0), (1,-1), (0,-1), (-1,-1)]
    neighbors = [s[1+dy:1+dy+h, 1+dx:1+dx+w] for dy, dx in offsets]
    cn = np.zeros((h, w), np.int16)
    for a, b in zip(neighbors, neighbors[1:]+neighbors[:1]):
        cn += np.abs(a-b)
    return (cn//2)*skeleton


def clean_skeleton(skeleton, min_size):
    labels, n = ndi.label(skeleton, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False
    return skeletonize(keep[labels], method='zhang')


def minutiae(skeleton, mask, theta, quality, args):
    skeleton = clean_skeleton(skeleton, args.componente_minimo)
    # Padding falso: aun con ROI tocando lienzo, el borde se considera inválido.
    distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    cn = crossing_number(skeleton)
    ys, xs = np.nonzero(((cn == 1) | (cn == 3)) & mask &
                         (distance >= args.distancia_borde) &
                         (quality >= args.coherencia_minima))
    # Preferencia por coherencia; no mezclar ni promediar ubicaciones.
    candidates = sorted(zip(ys.tolist(), xs.tolist()),
                        key=lambda p: (-float(quality[p]), p[0], p[1]))
    selected = []
    grid = {}
    cell = max(1, args.separacion_minucias)
    for y, x in candidates:
        gy, gx = int(y//cell), int(x//cell)
        nearby = [p for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  for p in grid.get((gy+dy, gx+dx), [])]
        if any((x-p['x'])**2+(y-p['y'])**2 < cell**2 for p in nearby):
            continue
        p = {'tipo': 'terminacion' if cn[y, x] == 1 else 'bifurcacion',
             'x': x, 'y': y, 'tangente_rad': float(theta[y, x]),
             'coherencia': float(quality[y, x]), 'distancia_borde_px': float(distance[y, x])}
        selected.append(p)
        grid.setdefault((gy, gx), []).append(p)
    return sorted(selected, key=lambda p: (p['y'], p['x'])), skeleton


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise ValueError('Máscara vacía o ROI insuficiente')
    return int(ys.min()), int(ys.max()+1), int(xs.min()), int(xs.max()+1)


def texture_glcm(gray, mask):
    """GLCM sin pares de fondo: nivel 0 reservado; crestas cuantizadas 1..16."""
    quant = (gray.astype(np.uint16)*16//256+1).astype(np.uint8)
    quant[~mask] = 0
    matrices = graycomatrix(quant, distances=[1, 2],
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=17, symmetric=True, normed=False)
    # Excluir todo par con al menos un píxel fuera de ROI.
    matrices = matrices[1:, 1:, :, :].astype(np.float64)
    totals = matrices.sum(axis=(0, 1), keepdims=True)
    if (totals == 0).any():
        raise ValueError('Pares de textura insuficientes dentro de ROI')
    matrices /= totals
    names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    return np.asarray([graycoprops(matrices, name).mean() for name in names], np.float32)


def descriptors(gray, binary, mask, theta, quality, points):
    y0, y1, x0, x1 = bbox(mask)
    inside = ndi.binary_erosion(mask, iterations=2)
    if inside.sum() < 50:
        raise ValueError('ROI demasiado pequeña para textura')
    lbp_map = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp = np.bincount(lbp_map[inside].astype(int), minlength=10).astype(np.float32)
    lbp /= max(lbp.sum(), 1)
    global_hist = np.histogram(theta[inside], bins=16, range=(0, np.pi),
                               weights=quality[inside])[0].astype(np.float32)
    global_hist /= max(float(global_hist.sum()), 1e-8)
    orient = np.zeros((2, 2, 8), np.float32)
    ys, xs = np.nonzero(inside)
    gy = np.clip(((ys-y0)*2/(y1-y0)).astype(int), 0, 1)
    gx = np.clip(((xs-x0)*2/(x1-x0)).astype(int), 0, 1)
    direction = np.minimum((theta[inside]*8/np.pi).astype(int), 7)
    np.add.at(orient, (gy, gx, direction), quality[inside])
    # Cada celda es una distribución, no un conteo dependiente del tamaño.
    orient /= np.maximum(orient.sum(axis=2, keepdims=True), 1e-8)
    glcm = texture_glcm(gray, inside)
    moments = cv2.HuMoments(cv2.moments((binary & mask).astype(np.uint8),
                                      binaryImage=True)).ravel()
    # Valores casi cero se saturan; evita infinitos y ruido numérico extremo.
    hu = (-np.sign(moments)*np.log10(np.maximum(np.abs(moments), 1e-20))).astype(np.float32)
    cells = np.zeros((3, 3), np.float32)
    cy = np.clip(((ys-y0)*3/(y1-y0)).astype(int), 0, 2)
    cx = np.clip(((xs-x0)*3/(x1-x0)).astype(int), 0, 2)
    np.add.at(cells, (cy, cx), 1)
    density = np.zeros((2, 3, 3), np.float32)
    counts = np.zeros(2, np.float32)
    for p in points:
        p['x_roi'] = float((p['x']-x0)/max(x1-x0-1, 1))
        p['y_roi'] = float((p['y']-y0)/max(y1-y0-1, 1))
        i = 0 if p['tipo'] == 'terminacion' else 1
        y, x = min(2, int(p['y_roi']*3)), min(2, int(p['x_roi']*3))
        density[i, y, x] += 1
        counts[i] += 1
    # Área de cada celda medida en ROI erosionada usada para la descripción.
    density *= 10000/np.maximum(cells[None, :, :], 1)
    global_density = counts*(10000/max(int(inside.sum()), 1))
    raw = {'orientacion_global': global_hist, 'orientacion_espacial': orient.ravel(),
           'lbp': lbp, 'glcm': glcm, 'hu': hu,
           'densidad_espacial': density.ravel(), 'densidad_global': global_density}
    vector = np.concatenate([raw[name] for name, _, _ in BLOCKS]).astype(np.float32)
    if not np.isfinite(vector).all() or vector.size != 91:
        raise ValueError('Vector inválido')
    return vector, raw


def input_sample(source, path, args):
    rel = path.relative_to(source)
    gray = read(path)
    required = {name: source/name/rel for name in ('binarias', 'esqueletos', 'mascaras')}
    for p in required.values():
        if not p.is_file():
            raise ValueError(f'Falta archivo complementario: {p}')
    arrays = {name: read(p) for name, p in required.items()}
    if any(a.shape != gray.shape for a in arrays.values()):
        raise ValueError('Grises, binaria, esqueleto y máscara tienen dimensiones diferentes')
    for name, a in arrays.items():
        if not np.isin(a, [0, 255]).all():
            raise ValueError(f'{name}: se esperan valores exactos 0/255')
    mask = arrays['mascaras'] > 0
    binary = arrays['binarias'] == 0
    skeleton = arrays['esqueletos'] == 0
    bbox(mask)
    if (skeleton & ~binary).any() or (skeleton & ~mask).any():
        raise ValueError('Esqueleto incompatible con binaria/máscara; verificar misma ejecución')
    if not 0.02 < binary[mask].mean() < 0.9 or skeleton.sum() < 20:
        raise ValueError('Binaria/esqueleto vacío, saturado o de polaridad incorrecta')
    theta, quality = orientation(gray)
    points, clean = minutiae(skeleton, mask, theta, quality, args)
    vector, raw = descriptors(gray, binary, mask, theta, quality, points)
    warnings = []
    if len(points) < 5:
        warnings.append('Pocas minucias confiables; densidades pueden ser débiles/nulas')
    if len(points) > 150:
        warnings.append('Muchas minucias: posible ruido/roturas/puentes')
    if quality[mask].mean() < 0.35:
        warnings.append('Baja coherencia media')
    info = {'ancho': gray.shape[1], 'alto': gray.shape[0],
            'terminaciones': sum(p['tipo'] == 'terminacion' for p in points),
            'bifurcaciones': sum(p['tipo'] == 'bifurcacion' for p in points),
            'coherencia_media': float(quality[mask].mean()),
            'fraccion_crestas': float(binary[mask].mean()),
            'pixeles_esqueleto': int(clean.sum()), 'avisos': ' | '.join(warnings)}
    return vector, raw, points, info, gray


def load_labels(path):
    if path is None:
        return {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        if not {'id', 'etiqueta'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV de etiquetas requiere columnas id;etiqueta y separador ;')
        result = {}
        for row in reader:
            key, label = row['id'].strip(), row['etiqueta'].strip()
            if key in result:
                raise ValueError(f'ID duplicado en etiquetas: {key}')
            result[key] = label
        return result


def find_sources(root):
    if root.name.casefold() == 'huellas_normalizadas':
        return [root]
    return sorted(p for p in root.rglob('*') if p.is_dir() and
                  p.name.casefold() == 'huellas_normalizadas')


def names_features():
    names = [f'orientacion_global_bin_{i:02d}' for i in range(16)]
    names += [f'orientacion_celda_y{y}_x{x}_bin_{b}'
              for y in range(2) for x in range(2) for b in range(8)]
    names += [f'lbp_uniforme_{i:02d}' for i in range(10)]
    names += ['glcm_'+p for p in ['contrast', 'dissimilarity', 'homogeneity',
                                  'energy', 'correlation', 'ASM']]
    names += [f'hu_log_{i+1}' for i in range(7)]
    names += [f'densidad_{kind}_y{y}_x{x}' for kind in ['terminacion', 'bifurcacion']
              for y in range(3) for x in range(3)]
    names += ['densidad_global_terminacion', 'densidad_global_bifurcacion']
    return np.asarray(names)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raiz', type=Path, default=Path('data_sheet'))
    p.add_argument('--salida', type=Path, help='Directorio base; crea subcarpeta por ejecución')
    labels = p.add_mutually_exclusive_group()
    labels.add_argument('--etiquetas', type=Path)
    labels.add_argument('--identidad-carpeta', action='store_true')
    p.add_argument('--marcar', action='store_true', help='PNG con terminaciones rojas y bifurcaciones azules')
    p.add_argument('--distancia-borde', type=float, default=16, help='Píxeles en lienzo normalizado')
    p.add_argument('--coherencia-minima', type=float, default=0.35)
    p.add_argument('--separacion-minucias', type=float, default=7)
    p.add_argument('--componente-minimo', type=int, default=16)
    a = p.parse_args()
    if not a.raiz.is_dir():
        p.error(f'No existe la carpeta {a.raiz}')
    if a.etiquetas is not None and not a.etiquetas.is_file():
        p.error('No existe el CSV de etiquetas')
    if not 0 <= a.coherencia_minima <= 1:
        p.error('Coherencia mínima fuera de [0,1]')
    if a.distancia_borde < 1 or a.separacion_minucias < 1 or a.componente_minimo < 1:
        p.error('Distancias y componente mínimo deben ser >= 1')
    return a


def main():
    args = parse_args()
    root = args.raiz.resolve()
    sources = find_sources(root)
    if not sources:
        print('No se encontraron carpetas Huellas_normalizadas.', file=sys.stderr)
        return 2
    try:
        labels = load_labels(args.etiquetas)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    base = args.salida or ((root.parent if root.name.casefold() == 'huellas_normalizadas'
                           else root)/'Caracteristicas_huellas')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = base.resolve()/run
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    metadata, errors, point_rows, vectors, raw_rows = [], [], [], [], []
    known_ids = set()
    # Configuración escrita antes de procesar para conservar parámetros aun si se interrumpe.
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    start = 0
    spans = []
    for name, size, description in BLOCKS:
        spans.append({'nombre': name, 'inicio_inclusivo': start,
                      'fin_exclusivo': start+size, 'descripcion': description})
        start += size
    config.update(version_script=VERSION, ejecucion_utc=run, dimension_vector=start,
                  bloques=spans, ejes='x derecha; y abajo; tangente módulo pi desde +x hacia +y',
                  etiquetas='CSV explícito' if args.etiquetas else
                  ('carpeta asumida como identidad por opción explícita' if args.identidad_carpeta else 'sin identidad asignada'),
                  versiones={'numpy': np.__version__, 'opencv': cv2.__version__,
                             'scipy': scipy.__version__, 'skimage': skimage.__version__})
    (out/'configuracion.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    for source in sources:
        parent = source.parent
        # Ruta relativa evita colisiones si distintos padres comparten el mismo nombre.
        group = parent.relative_to(root).as_posix() if root != source and parent != root else parent.name
        files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXT
                       and not any(part.casefold() in FOLDERS for part in p.relative_to(source).parts[:-1]))
        print(f'\n{group}: {len(files)} imágenes', flush=True)
        for i, path in enumerate(files, 1):
            rel = path.relative_to(source)
            key = group+'/'+rel.as_posix()
            try:
                if key in known_ids:
                    raise ValueError(f'ID duplicado: {key}')
                known_ids.add(key)
                vector, raw, points, info, gray = input_sample(source, path, args)
                label = labels.get(key, '') if args.etiquetas else (group if args.identidad_carpeta else '')
                impression = re.match(r'(?i)^(imp[_-]?\d+)(?:[_\-.]|$)', rel.name)
                augmentation = group+'/'+rel.parent.as_posix()+'/'+(impression.group(1).lower() if impression else rel.name)
                row = {'fila': len(vectors), 'id': key, 'grupo_carpeta': group,
                       'grupo_aumento': augmentation, 'etiqueta': label,
                       'archivo_normalizado': str(path),
                       'sha256_grises': hashlib.sha256(path.read_bytes()).hexdigest(), **info}
                if args.etiquetas and key not in labels:
                    row['avisos'] += ' | Sin entrada en CSV de etiquetas'
                per = out/'por_imagen'/group/rel
                per.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(per)+'.npz', vector=vector,
                                    nombres_caracteristicas=names_features(), **raw)
                record = {'version_script': VERSION, 'muestra': row, 'minucias': points,
                          'orientacion': 'Tangente de cresta módulo pi, no ángulos individuales de ramas'}
                Path(str(per)+'.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                if args.marcar:
                    marked = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    for pt in points:
                        color = (0, 0, 255) if pt['tipo'] == 'terminacion' else (255, 0, 0)
                        cv2.circle(marked, (pt['x'], pt['y']), 4, color, 1)
                        dx = int(round(10*np.cos(pt['tangente_rad'])))
                        dy = int(round(10*np.sin(pt['tangente_rad'])))
                        cv2.line(marked, (pt['x']-dx, pt['y']-dy), (pt['x']+dx, pt['y']+dy), color, 1)
                    write_png(Path(str(out/'marcadas'/group/rel)+'.png'), marked)
                vectors.append(vector)
                raw_rows.append(raw)
                metadata.append(row)
                point_rows.extend({'id': key, 'indice_minucia': n, **pt} for n, pt in enumerate(points))
                print(f'[{i}/{len(files)}] OK {rel}: {len(points)} minucias', flush=True)
            except Exception as exc:
                errors.append({'id': key, 'error': f'{type(exc).__name__}: {exc}'})
                print(f'[{i}/{len(files)}] ERROR {rel}: {exc}', flush=True)
    write_csv(out/'errores.csv', errors, ['id', 'error'])
    if not vectors:
        print(f'No se extrajeron vectores. Revisar {out}/errores.csv', file=sys.stderr)
        return 2
    X = np.stack(vectors).astype(np.float32)
    matrices = {name: np.stack([r[name] for r in raw_rows]).astype(np.float32) for name, _, _ in BLOCKS}
    np.savez_compressed(out/'vectores.npz', X=X,
                        ids=np.asarray([r['id'] for r in metadata]),
                        grupos=np.asarray([r['grupo_carpeta'] for r in metadata]),
                        grupos_aumento=np.asarray([r['grupo_aumento'] for r in metadata]),
                        etiquetas=np.asarray([r['etiqueta'] for r in metadata]),
                        nombres_caracteristicas=names_features(), **matrices)
    write_csv(out/'muestras.csv', metadata, list(metadata[0]))
    write_csv(out/'minucias.csv', point_rows, ['id', 'indice_minucia', 'tipo', 'x', 'y',
              'tangente_rad', 'coherencia', 'distancia_borde_px', 'x_roi', 'y_roi'])
    write_csv(out/'etiquetas_para_completar.csv',
              [{k: r[k] for k in ('id', 'grupo_carpeta', 'grupo_aumento', 'etiqueta')} for r in metadata],
              ['id', 'grupo_carpeta', 'grupo_aumento', 'etiqueta'])
    with (out/'vectores.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['id', 'etiqueta']+names_features().tolist())
        for row, vector in zip(metadata, X):
            writer.writerow([row['id'], row['etiqueta']]+[format(float(v), '.8g') for v in vector])
    print(f'\nResultado: {len(X)} vectores de {X.shape[1]} valores; {len(errors)} errores.')
    print(f'Salida: {out}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
