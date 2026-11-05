#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalizador BASE de huellas dactilares (Python >= 3.10).

INSTALACION (Windows, terminal):
    py -m pip install numpy opencv-python-headless scipy scikit-image
USO, desde la carpeta que contiene data_sheet:
    py normalizar_huellas.py --raiz "data_sheet"
También acepta la ruta a un grupo Huellas_1 o a su carpeta Huellas.

ESTRUCTURA (no modifica originales):
    data_sheet/Huellas_1/Huellas/imp_01_original.png
    data_sheet/Huellas_1/Huellas_normalizadas/imp_01_original.png.png
    data_sheet/Huellas_1/Huellas_normalizadas/binarias/imp_01_original.png.png
    data_sheet/Huellas_1/Huellas_normalizadas/esqueletos/imp_01_original.png.png
    data_sheet/Huellas_1/Huellas_normalizadas/mascaras/imp_01_original.png.png
    data_sheet/Huellas_1/Huellas_normalizadas/informe_<ejecucion>.csv
    data_sheet/Huellas_1/Huellas_normalizadas/config_<ejecucion>.json

Se conserva la extensión original en el nombre para evitar colisiones entre,
por ejemplo, a.jpg y a.bmp. Se preservan subcarpetas bajo 'Huellas'.
Grises/binarias/esqueletos: crestas NEGRAS sobre fondo BLANCO.
Máscara: blanco = ROI válida. La carpeta principal contiene los grises finales
para textura/orientación/momentos, sin Gabor. Binarias y esqueletos usan Gabor
como paso intermedio. Estas cuatro salidas alimentan extraer_caracteristicas.py.
Con --guardar-diagnosticos se agregan calidad/coherencia y calidad/realce,
que son sólo diagnósticos y no se usan como muestras de entrenamiento.

METODOS:
Mediana selectiva de impulsos -> segmentación por textura/coherencia ->
erosión configurable del halo de la máscara -> PCA del contorno -> una
transformación afín para zoom proporcional, rotación y centrado -> contraste
CLAHE o local (GRISES FINALES) -> Gabor para binarización -> umbral adaptativo
-> eliminación conservadora de componentes pequeños -> esqueleto Zhang.

LIMITACIONES:
* PCA alinea el eje del CONTORNO, no un punto anatómico; ambigüedad de 180°.
  Recortes/huellas parciales pueden producir una orientación incorrecta.
* Por defecto --encuadre ajustar amplía/reduce la ROI hasta el margen del
  lienzo, conservando proporciones. No añade detalle ni garantiza escala física;
  una huella parcial puede quedar más ampliada que una impresión completa.
  Para preservar escala física usar --encuadre conservar --escala ppi
  --ppi-entrada 500 --ppi-salida 500. Para escala por separación de crestas,
  usar --encuadre conservar --escala crestas.
* No se rectifica deformación elástica ni se recuperan crestas fusionadas por
  presión. No se usa otra impresión/original ni se aprende de otros archivos.
* No es un extractor/depurador completo de minucias: no borra espuelas a ciegas.
  Las terminaciones próximas al borde deben descartarse en el extractor.
* Los parámetros son de partida, NO están calibrados con tu base real.
* Binarización/esqueleto son para minucias clásicas. Para una CNN comparar
  también las imágenes en grises y el preprocesamiento mínimo.

AJUSTES:
    py normalizar_huellas.py --raiz data_sheet --sin-rotacion
    py normalizar_huellas.py --raiz data_sheet --contraste local --sin-gabor
    py normalizar_huellas.py --raiz data_sheet --escala ninguna --tamano 768
    py normalizar_huellas.py --raiz data_sheet --margen 8 --erosion-roi 4 --sobrescribir
    py normalizar_huellas.py --raiz data_sheet --sigma-ruido 10 --guardar-diagnosticos
    py normalizar_huellas.py --raiz data_sheet --sobrescribir
    py normalizar_huellas.py --help

Fuentes de los métodos y APIs:
https://www.peterkovesi.com/matlabfns/FingerPrints/Docs/index.html
https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
https://scikit-image.org/docs/stable/api/skimage.morphology.html
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

try:
    import cv2
    import numpy as np
    import scipy
    from scipy import ndimage as ndi
    import skimage
    from skimage.morphology import skeletonize
except ImportError as exc:
    raise SystemExit("Falta una dependencia. Ejecutá: python -m pip install "
                     "numpy opencv-python-headless scipy scikit-image") from exc

EXTENSIONES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm'}
SALIDA = 'Huellas_normalizadas'


def impar(n):
    return max(3, int(round(n)) | 1)


def leer(path):
    # imdecode/fromfile permite rutas con tildes en Windows.
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError('Archivo ilegible o formato no soportado')
    if im.dtype != np.uint8:
        raise ValueError('Se esperan imágenes de 8 bits; convertir explícitamente las de 16 bits')
    if im.ndim == 3:
        if im.shape[2] == 4:
            alpha = im[:, :, 3:4].astype(np.float32) / 255
            im = np.rint(im[:, :, :3] * alpha + 255 * (1-alpha)).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if min(im.shape) < 48:
        raise ValueError('Imagen demasiado pequeña: mínimo 48 píxeles por lado')
    return im


def guardar(path, im):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', im)
    if not ok:
        raise OSError(f'No se pudo codificar {path}')
    temporal = path.with_name(path.name + '.tmp')
    encoded.tofile(str(temporal))
    temporal.replace(path)


def tensor(im, sigma=5):
    f = im.astype(np.float32) / 255
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    smooth = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    xx, yy, xy = smooth(gx*gx), smooth(gy*gy), smooth(gx*gy)
    coherence = np.sqrt((xx-yy)**2 + 4*xy**2) / (xx+yy+1e-8)
    # Dirección NORMAL a la cresta, tal como la espera getGaborKernel.
    theta = (0.5*np.arctan2(2*xy, xx-yy)) % np.pi
    return theta, np.clip(coherence, 0, 1)


def quitar_impulsos(im, umbral):
    if umbral <= 0:
        return im.copy()
    med = cv2.medianBlur(im, 3)
    # Sólo extremos aislados; no aplicar mediana a todas las crestas finas.
    extrema = (im < 20) | (im > 235)
    replace = extrema & (np.abs(im.astype(np.float32)-med) > umbral)
    return np.where(replace, med, im).astype(np.uint8)


def segmentar(im, args):
    f = im.astype(np.float32)
    w = impar(args.ventana_roi)
    mean = cv2.blur(f, (w, w))
    std = np.sqrt(np.maximum(cv2.blur(f*f, (w, w))-mean*mean, 0))
    _, coherence = tensor(im, max(2, w/4))
    seed = (std >= args.std_roi) & (coherence >= args.coherencia_roi)
    # Estas operaciones se aplican a la máscara, NO a las crestas.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))
    mask = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        raise ValueError('No se detectó ROI; revisar std-roi/coherencia-roi')
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = ndi.binary_fill_holes(labels == idx)
    # La ventana de textura ensancha la ROI hacia el fondo. Recorte conservador
    # del halo; se puede desactivar con --erosion-roi 0 para conservar bordes.
    if args.erosion_roi:
        mask = ndi.binary_erosion(mask, iterations=args.erosion_roi, border_value=0)
    if mask.sum() < max(400, im.size*0.01):
        raise ValueError('ROI demasiado pequeña o sin textura suficiente')
    return mask


def periodos(im, mask, minimo=4, maximo=24, bloque=64):
    """Picos espectrales locales; rechaza parches sin periodicidad/coherencia."""
    _, coherence = tensor(im)
    h, w = im.shape
    b = min(bloque, h, w)
    step = max(16, b//2)
    freq = np.fft.fftfreq(b)
    fy, fx = np.meshgrid(freq, freq, indexing='ij')
    radius = np.hypot(fx, fy)
    band = (radius >= 1/maximo) & (radius <= 1/minimo)
    window = np.outer(np.hanning(b), np.hanning(b))
    observations = []
    for y in range(0, h-b+1, step):
        for x in range(0, w-b+1, step):
            sl = np.s_[y:y+b, x:x+b]
            if mask[sl].mean() < 0.9 or coherence[sl].mean() < 0.25:
                continue
            patch = im[sl].astype(np.float32)
            patch = (patch-patch.mean())*window
            power = np.abs(np.fft.fft2(patch))**2
            selected = np.where(band, power, 0)
            peak = np.unravel_index(np.argmax(selected), selected.shape)
            if selected[peak] < 8*(power[band].mean()+1e-9):
                continue
            observations.append((y+b//2, x+b//2, float(1/radius[peak])))
    return observations


def transformar(im, mask, args, avisos):
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    ratio = float(values[-1]/max(values[0], 1e-9))
    angle = 0.0
    if not args.sin_rotacion:
        if ratio >= args.ratio_pca:
            vx, vy = vectors[:, -1]
            angle = float(np.degrees(np.arctan2(vy, vx)) % 180 - 90)
            avisos.append('Rotación PCA aproximada; no resuelve ambigüedad de 180 grados')
        else:
            avisos.append('Rotación omitida: contorno sin eje principal confiable')
    observations = periodos(im, mask, args.periodo_min, args.periodo_max)
    period = float(np.median([p[2] for p in observations])) if len(observations)>=3 else None
    scale = 1.0
    if args.escala == 'ppi':
        scale = args.ppi_salida/args.ppi_entrada
    elif args.escala == 'crestas':
        if period is None:
            avisos.append('Escala sin corregir: periodicidad insuficiente')
        else:
            scale = args.periodo_objetivo/period
            limited = float(np.clip(scale, 0.5, 2.0))
            if limited != scale:
                avisos.append('Factor de escala limitado al intervalo 0.5..2')
            scale = limited
    matrix = cv2.getRotationMatrix2D(tuple(points.mean(axis=0)), angle, scale)
    corners = points @ matrix[:, :2].T + matrix[:, 2]
    low, high = corners.min(axis=0), corners.max(axis=0)
    available = args.tamano - 2*args.margen
    factor = available/max(high-low+1)
    fit = factor if args.encuadre == 'ajustar' else min(1.0, factor)
    if fit < 1 and args.encuadre == 'conservar':
        avisos.append('ROI reducida para caber en lienzo; escala objetivo no alcanzada')
    if abs(fit-1) > 1e-8:
        matrix[:, :2] *= fit
        matrix[:, 2] *= fit
        low, high = low*fit, high*fit
    matrix[:, 2] += (args.tamano-1)/2 - (low+high)/2
    # Antialias antes de reducir; una única interpolación afín.
    net_scale = scale*fit
    if net_scale < 1:
        sigma = max(0.01, (1/net_scale-1)/2)
        im = cv2.GaussianBlur(im, (0, 0), sigma)
    size = (args.tamano, args.tamano)
    interpolation = cv2.INTER_CUBIC if net_scale > 1 else cv2.INTER_LINEAR
    aligned = cv2.warpAffine(im, matrix, size, flags=interpolation, borderValue=255)
    roi = cv2.warpAffine(mask.astype(np.uint8), matrix, size,
                         flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
    return aligned, roi, {'angulo_grados': angle, 'ratio_pca': ratio,
                         'periodo_entrada_px': period, 'escala_aplicada': net_scale,
                         'matriz_afin': matrix.tolist()}


def contraste(im, mask, modo, clip):
    f = im.astype(np.float32)
    lo, hi = np.percentile(f[mask], [1, 99])
    if hi-lo < 2:
        raise ValueError('Contraste insuficiente en ROI')
    # Fondo al tono medio antes del contraste: reduce influencia de márgenes.
    f[~mask] = np.median(f[mask])
    f = np.clip((f-lo)*255/(hi-lo), 0, 255).astype(np.uint8)
    if modo == 'clahe':
        f = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(f)
    elif modo == 'local':
        a = f.astype(np.float32)
        weights = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 12)
        mean = cv2.GaussianBlur(a*mask, (0, 0), 12)/np.maximum(weights, 1e-5)
        sq = cv2.GaussianBlur(a*a*mask, (0, 0), 12)/np.maximum(weights, 1e-5)
        sd = np.sqrt(np.maximum(sq-mean*mean, 0))
        f = np.clip(128+45*(a-mean)/np.maximum(sd, 10), 0, 255).astype(np.uint8)
    f[~mask] = 255
    return f


def realzar(im, mask, args, avisos):
    theta, coherence = tensor(im, 5)
    observations = periodos(im, mask, 4, 24)
    if args.sin_gabor:
        return im, coherence
    if len(observations) < 3:
        avisos.append('Gabor omitido: no hay frecuencia local confiable')
        return im, coherence
    # Mapa de periodos: observaciones interpoladas por el vecino más próximo
    # y suavizadas. Tres bandas y 16 orientaciones limitan el coste del banco.
    locations = np.zeros(im.shape, bool)
    periods = np.zeros(im.shape, np.float32)
    for y, x, p in observations:
        locations[y, x] = True
        periods[y, x] = p
    indices = ndi.distance_transform_edt(~locations, return_distances=False, return_indices=True)
    local = cv2.GaussianBlur(periods[tuple(indices)], (0, 0), 8)
    bands = np.unique(np.round(np.percentile([p[2] for p in observations], [20, 50, 80]), 1))
    pidx = np.argmin(np.abs(local[:, :, None]-bands), axis=2)
    tidx = (np.rint(theta*16/np.pi).astype(int)) % 16
    valid = mask & (coherence >= args.coherencia_gabor)
    response = np.zeros(im.shape, np.float32)
    signal = (127.5-im.astype(np.float32))/127.5
    signal[~mask] = 0
    for j, period in enumerate(bands):
        for i in range(16):
            select = valid & (pidx == j) & (tidx == i)
            if not select.any():
                continue
            k = impar(4*period)
            kernel = cv2.getGaborKernel((k, k), 0.55*period, i*np.pi/16,
                                       float(period), 0.7, 0, ktype=cv2.CV_32F)
            kernel -= kernel.mean()
            kernel /= np.sum(np.abs(kernel))+1e-8
            filtered = cv2.filter2D(signal, cv2.CV_32F, kernel)
            response[select] = filtered[select]
    out = im.copy()
    if valid.sum() > 100:
        amplitude = np.percentile(np.abs(response[valid]), 95)
        if amplitude > 1e-6:
            enhanced = np.clip(127.5-110*response/amplitude, 0, 255)
            # Mezcla gradual para no crear saltos en el umbral de coherencia.
            weight = np.clip((coherence-args.coherencia_gabor)/0.3, 0, 0.85)
            out[valid] = np.clip((1-weight[valid])*im[valid] +
                                 weight[valid]*enhanced[valid], 0, 255).astype(np.uint8)
    out[~mask] = 255
    return out, coherence


def componentes(binary, minimo):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= minimo
    keep[0] = False
    return keep[labels]


def procesar(path, args):
    im = leer(path)
    avisos = []
    if args.invertir:
        im = 255-im
    clean = quitar_impulsos(im, args.umbral_impulsos)
    mask = segmentar(clean, args)
    band = max(1, args.erosion_roi+1)
    edge = np.r_[mask[:band, :].ravel(), mask[-band:, :].ravel(),
                 mask[:, :band].ravel(), mask[:, -band:].ravel()]
    if edge.any():
        avisos.append('ROI toca borde de entrada: posible recorte/huella parcial')
    aligned, roi, info = transformar(clean, mask, args, avisos)
    if args.sigma_ruido > 0:
        aligned = cv2.bilateralFilter(aligned, 5, args.sigma_ruido, 2)
    normalized = contraste(aligned, roi, args.contraste, args.clip_clahe)
    enhanced, quality = realzar(normalized, roi, args, avisos)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, args.ventana_binaria,
                                   args.constante_binaria) > 0
    safe = cv2.erode(roi.astype(np.uint8), np.ones((5, 5), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    binary = componentes(binary & safe, args.area_minima)
    skeleton = skeletonize(binary, method='zhang')
    if skeleton.sum() < 20:
        raise ValueError('Esqueleto vacío o insuficiente; revisar máscara/umbrales')
    fraction = float(binary.sum()/max(safe.sum(), 1))
    if not 0.2 <= fraction <= 0.65:
        avisos.append('Proporción de crestas atípica; revisar contraste/binarización/presión')
    if float(quality[roi].mean()) < 0.35:
        avisos.append('Baja coherencia media: revisar visualmente')
    info.update(roi_fraccion=float(mask.mean()), fraccion_crestas=fraction,
                coherencia_media=float(quality[roi].mean()),
                pixeles_esqueleto=int(skeleton.sum()), avisos=' | '.join(avisos))
    # Guardar grises contrastados ANTES de Gabor: conserva mejor la textura.
    images = [normalized, np.where(binary, 0, 255).astype(np.uint8),
              np.where(skeleton, 0, 255).astype(np.uint8), roi.astype(np.uint8)*255]
    if args.guardar_diagnosticos:
        images += [(quality*roi*255).astype(np.uint8), enhanced]
    return images, info


def fuentes(root):
    if root.name.casefold() == 'huellas':
        return [root]
    # No procesa sus propios resultados en ejecuciones posteriores.
    return sorted(p for p in root.rglob('*') if p.is_dir() and
                  p.name.casefold() == 'huellas' and
                  SALIDA.casefold() not in [v.casefold() for v in p.relative_to(root).parts])


def argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raiz', type=Path, default=Path('data_sheet'))
    p.add_argument('--tamano', type=int, default=512, help='Lienzo cuadrado de salida')
    p.add_argument('--margen', type=int, default=12, help='Margen mínimo del encuadre en píxeles de salida')
    p.add_argument('--encuadre', choices=['ajustar', 'conservar'], default='ajustar',
                   help='ajustar: zoom de ROI al lienzo; conservar: respeta escala elegida')
    p.add_argument('--escala', choices=['crestas', 'ppi', 'ninguna'], default='ninguna')
    p.add_argument('--ppi-entrada', type=float)
    p.add_argument('--ppi-salida', type=float, default=500)
    p.add_argument('--periodo-objetivo', type=float, default=10, help='Píxeles por ciclo cresta-valle')
    p.add_argument('--periodo-min', type=float, default=4)
    p.add_argument('--periodo-max', type=float, default=24)
    p.add_argument('--sin-rotacion', action='store_true')
    p.add_argument('--ratio-pca', type=float, default=1.3)
    p.add_argument('--invertir', action='store_true', help='Para crestas claras sobre fondo oscuro')
    p.add_argument('--ventana-roi', type=int, default=17)
    p.add_argument('--erosion-roi', type=int, default=3,
                   help='Recorta halo de máscara en píxeles de entrada; 0 conserva todo el borde')
    p.add_argument('--std-roi', type=float, default=6)
    p.add_argument('--coherencia-roi', type=float, default=0.18)
    p.add_argument('--umbral-impulsos', type=float, default=80, help='0 desactiva mediana selectiva')
    p.add_argument('--sigma-ruido', type=float, default=0,
                   help='Bilateral opcional, por ejemplo 10; 0 conserva más detalle')
    p.add_argument('--contraste', choices=['clahe', 'local', 'percentiles'], default='clahe')
    p.add_argument('--clip-clahe', type=float, default=2)
    p.add_argument('--sin-gabor', action='store_true')
    p.add_argument('--coherencia-gabor', type=float, default=0.25)
    p.add_argument('--ventana-binaria', type=int, default=31)
    p.add_argument('--constante-binaria', type=float, default=3)
    p.add_argument('--area-minima', type=int, default=8, help='Área mínima de componente; 1 conserva todos')
    p.add_argument('--sobrescribir', action='store_true')
    p.add_argument('--guardar-diagnosticos', action='store_true',
                   help='Guarda mapas de coherencia y realce Gabor dentro de calidad')
    p.add_argument('--debug', action='store_true')
    a = p.parse_args()
    if not a.raiz.is_dir():
        p.error(f'No existe la carpeta: {a.raiz}')
    if a.tamano < 96 or a.margen < 0 or 2*a.margen >= a.tamano-32:
        p.error('Tamaño/margen incompatibles')
    if a.ventana_binaria < 3 or a.ventana_binaria % 2 == 0:
        p.error('--ventana-binaria debe ser impar y >= 3')
    if a.escala == 'ppi' and (a.ppi_entrada is None or a.ppi_entrada <= 0):
        p.error('--escala ppi requiere --ppi-entrada positivo')
    if a.encuadre == 'ajustar' and a.escala != 'ninguna':
        p.error('Para --escala ppi/crestas usar --encuadre conservar; el zoom no conserva esa escala')
    if a.erosion_roi < 0 or a.sigma_ruido < 0 or a.umbral_impulsos < 0:
        p.error('Erosión, sigma-ruido y umbral-impulsos deben ser >= 0')
    if not (2 <= a.periodo_min < a.periodo_max <= 32) or a.periodo_objetivo <= 0:
        p.error('Periodos inválidos; se requiere 2 <= mínimo < máximo <= 32')
    if any(not 0 <= v <= 1 for v in [a.coherencia_roi, a.coherencia_gabor]):
        p.error('Coherencias deben estar en [0,1]')
    if min(a.ppi_salida, a.std_roi, a.clip_clahe, a.area_minima) <= 0 or a.ratio_pca <= 1:
        p.error('PPI, std-roi, clip-clahe, área deben ser positivos y ratio-pca > 1')
    return a


def main():
    args = argumentos()
    groups = fuentes(args.raiz.resolve())
    if not groups:
        print('No se encontraron carpetas llamadas Huellas.', file=sys.stderr)
        return 2
    cv2.setNumThreads(1)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    counts = {'ok': 0, 'error': 0, 'omitida': 0}
    fields = ['archivo', 'estado', 'avisos', 'angulo_grados', 'ratio_pca',
              'periodo_entrada_px', 'escala_aplicada', 'matriz_afin', 'roi_fraccion',
              'fraccion_crestas', 'coherencia_media', 'pixeles_esqueleto']
    for source in groups:
        out = source.parent/SALIDA
        files = sorted(f for f in source.rglob('*') if f.is_file() and f.suffix.lower() in EXTENSIONES)
        out.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(version_script='2.0', fuente=str(source), ejecucion_utc=run,
                      grises_finales='Contraste antes de Gabor; Gabor sólo para binarización',
                      versiones={'opencv': cv2.__version__, 'numpy': np.__version__,
                                 'scipy': scipy.__version__, 'skimage': skimage.__version__})
        (out/f'config_{run}.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'\n{source}: {len(files)} imágenes -> {out}')
        with (out/f'informe_{run}.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fields, delimiter=';')
            writer.writeheader()
            for i, path in enumerate(files, 1):
                rel = path.relative_to(source)
                rel = rel.with_name(rel.name+'.png')
                destinations = [out/rel] + [out/folder/rel for folder in
                                             ['binarias', 'esqueletos', 'mascaras']]
                if args.guardar_diagnosticos:
                    destinations += [out/'calidad'/'coherencia'/rel, out/'calidad'/'realce'/rel]
                row = {'archivo': str(path.relative_to(source))}
                if any(d.exists() for d in destinations) and not args.sobrescribir:
                    row.update(estado='omitida', avisos='Hay resultados previos; usar --sobrescribir para regenerar todos')
                else:
                    try:
                        images, info = procesar(path, args)
                        for destination, image in zip(destinations, images):
                            guardar(destination, image)
                        row.update(info, estado='ok')
                    except Exception as exc:
                        row.update(estado='error', avisos=f'{type(exc).__name__}: {exc}')
                        if args.debug:
                            traceback.print_exc()
                counts[row['estado']] += 1
                writer.writerow(row)
                handle.flush()
                print(f"[{i}/{len(files)}] {row['estado']}: {row['archivo']}", flush=True)
    print('\nResumen:', counts)
    if sum(counts.values()) == 0:
        print('No se encontraron imágenes compatibles.', file=sys.stderr)
        return 2
    return 1 if counts['error'] else 0


if __name__ == '__main__':
    sys.exit(main())
