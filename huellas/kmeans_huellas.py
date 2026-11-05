#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-Means k=3 y centroides de vectores de huellas. Python >= 3.10.

INSTALAR:
    py -m pip install numpy scipy scikit-learn matplotlib joblib
ENTRENAR, después de extraer características:
    py kmeans_huellas.py --raiz "data_sheet"
O elegir exactamente un archivo:
    py kmeans_huellas.py --datos "ruta/a/vectores.npz"
PREDECIR nuevos vectores SIN reajustar el modelo:
    py kmeans_huellas.py --datos "nuevas/vectores.npz" --predecir "ruta/modelo_kmeans.joblib"

Se usan las características originales X del extractor, NO las coordenadas
PCA 2D. Ajusta StandardScaler y KMeans SOLO con filas de entrenamiento.
K=3 fijo; inicialización k-means++, n_init=50, semilla=42 por defecto.
Cada centroide es la media de las muestras de entrenamiento asignadas al
clúster, en el espacio estandarizado usado para medir distancias.
También exporta centroides en las unidades originales de las características.
La PCA 2D se ajusta a entrenamiento sólo para DIBUJAR, no para agrupar.

PARTICIONES OPCIONALES:
Sin --particiones, TODAS las filas son entrenamiento. Esto determina centros
exploratorios, pero no mide generalización. No hace una separación aleatoria
de variantes de la misma impresión.
Con --particiones "particiones.csv": CSV con separador ;, columnas id;conjunto.
conjunto puede ser train, validacion o test. Todas las filas deben figurar.
Las variantes que comparten grupo_aumento no pueden cruzar particiones.
Genera particiones_para_completar.csv para preparar este archivo.

IDENTIDADES:
Los clústeres 0, 1 y 2 son números arbitrarios, no identidades automáticas.
Si el NPZ contiene etiquetas, las usa como referencia, sin influir en KMeans.
Opcional --etiquetas "etiquetas.csv": columnas id;etiqueta, separador ;.
SOLO si cada carpeta corresponde a un dedo autorizado:
    py kmeans_huellas.py --raiz data_sheet --identidad-carpeta
Con exactamente 3 clases etiquetadas en entrenamiento, asocia clúster/clase
por asignación uno a uno que maximiza coincidencias en entrenamiento.
Guarda la tabla de conteos para revisar si esa asociación es convincente.
No usa test ni validación para definir esa asociación. No fuerza que los
clústeres coincidan con las identidades ni corrige la agrupación con etiquetas.

SALIDA: junto a vectores.npz, KMeans_huellas/<ejecucion_UTC>/
  centroides_estandarizados.csv: 3 centros usados para las distancias.
  centroides_originales.csv: 3 centros en las unidades originales.
  centroides.npz: arrays numéricos y nombres de características.
  representantes.csv: muestra de entrenamiento más cercana a cada centro.
  asignaciones.csv: clúster, distancias a los 3 centros y clase asociada.
  correspondencia_clusters.csv: asociación con clases, si hay etiquetas.
  conteos_cluster_clase_train.csv: evidencia de esa asociación.
  conteos_cluster_grupo.csv: distribución de carpetas entre los clústeres.
  modelo_kmeans.joblib: scaler + KMeans + PCA visual + nombres + asociación.
  clusters_pca.png: grupos predichos y centroides proyectados.
  clases_pca.png: clases/grupos de carpeta como comparación, si hay >=2.
  resumen.json: métricas descriptivas y advertencias.
Los archivos se conservan por ejecución, sin sobrescribir resultados previos.

IMPORTANTE:
K-Means SIEMPRE asigna uno de los tres centros; no rechaza huellas desconocidas.
La distancia no es una probabilidad. Un umbral de rechazo necesitaría
validación con muestras autorizadas y no autorizadas independientes.
Para nuevas muestras, usar el mismo normalizador/extractor y orden de
características. Cargar únicamente archivos joblib propios/de confianza.
Las métricas de entrenamiento no son exactitud de reconocimiento futura.

Referencia:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

try:
    import numpy as np
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from threadpoolctl import threadpool_limits
except ImportError as exc:
    raise SystemExit('Instalá: py -m pip install numpy scipy scikit-learn matplotlib joblib') from exc

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(columns)
        writer.writerows(rows)


def find_data(root, explicit):
    if explicit:
        if not explicit.is_file():
            raise ValueError(f'No existe {explicit}')
        return explicit.resolve()
    if not root.is_dir():
        raise ValueError(f'No existe {root}')
    if (root/'vectores.npz').is_file():
        return (root/'vectores.npz').resolve()
    base = root if root.name == 'Caracteristicas_huellas' else root/'Caracteristicas_huellas'
    candidates = sorted(base.glob('*/vectores.npz'), key=lambda p: p.parent.name)
    if not candidates:
        raise ValueError('No se encontró vectores.npz; ejecutar extractor o usar --datos')
    return candidates[-1].resolve()


def load_data(path):
    with np.load(path, allow_pickle=False) as d:
        X = np.asarray(d['X'], np.float64)
        if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
            raise ValueError('X debe ser una matriz no vacía, sin NaN/inf')
        n, p = X.shape
        def texts(key, default):
            result = np.asarray(d[key]).astype(str) if key in d else np.asarray(default)
            if result.shape != (n,):
                raise ValueError(f'{key} incompatible con X')
            return np.char.strip(result)
        ids = texts('ids', [str(i) for i in range(n)])
        groups = texts('grupos', [key.split('/')[0] if '/' in key else 'Sin_grupo' for key in ids])
        labels = texts('etiquetas', ['']*n)
        augmentation = texts('grupos_aumento', ['']*n)
        names = np.asarray(d['nombres_caracteristicas']).astype(str) if 'nombres_caracteristicas' in d else np.asarray([f'f{i:03d}' for i in range(p)])
        if names.shape != (p,) or len(set(names)) != p:
            raise ValueError('Nombres de características inválidos')
    if len(set(ids)) != n or (ids == '').any():
        raise ValueError('IDs duplicados o vacíos')
    return X, ids, groups, labels, augmentation, names


def read_mapping(path, value_column):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        if not {'id', value_column}.issubset(reader.fieldnames or []):
            raise ValueError(f'{path.name} requiere columnas id;{value_column}')
        result = {}
        for row in reader:
            key = row['id'].strip()
            if key in result:
                raise ValueError(f'ID duplicado en {path.name}: {key}')
            result[key] = row[value_column].strip()
        return result


def partitions(path, ids, augmentation):
    if not path:
        return np.repeat('train', len(ids))
    mapping = read_mapping(path, 'conjunto')
    missing = set(ids)-set(mapping)
    if missing:
        raise ValueError(f'Faltan {len(missing)} IDs en particiones')
    result = np.asarray([mapping[key].lower() for key in ids])
    if not set(result).issubset({'train', 'validacion', 'test'}):
        raise ValueError('conjunto debe ser train, validacion o test')
    seen = {}
    for group, part in zip(augmentation, result):
        if group:
            if group in seen and seen[group] != part:
                raise ValueError(f'Variantes de {group} cruzan particiones: mantenerlas juntas')
            seen[group] = part
    return result


def class_mapping(predicted, labels, train):
    known = train & (labels != '')
    classes = sorted(set(labels[known]))
    counts = np.zeros((3, len(classes)), dtype=int)
    for c, name in enumerate(classes):
        for cluster in range(3):
            counts[cluster, c] = np.sum(known & (predicted == cluster) & (labels == name))
    result = {}
    if len(classes) == 3 and np.all(counts.sum(axis=1) > 0):
        rows, columns = linear_sum_assignment(-counts)
        if np.all(counts[rows, columns] > 0):
            result = {int(r): str(classes[c]) for r, c in zip(rows, columns)}
    return result, classes, counts


def project2(pca, Xs):
    Z = pca.transform(Xs)
    return np.column_stack((Z[:, 0], Z[:, 1] if Z.shape[1] >= 2 else np.zeros(len(Z))))


def plot_clusters(Z, centers, color_labels, parts, ratio, path, title):
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    names = sorted(set(color_labels))
    for i, name in enumerate(names):
        color = COLORS[i % len(COLORS)]
        for part, marker in [('train', 'o'), ('validacion', '^'), ('test', 's'), ('prediccion', 's')]:
            selected = (color_labels == name) & (parts == part)
            if selected.any():
                ax.scatter(Z[selected, 0], Z[selected, 1], s=30, color=color,
                           marker=marker, alpha=.65, label=f'{name} · {part} (n={selected.sum()})')
    if centers is not None:
        for k, point in enumerate(centers):
            ax.scatter(*point, marker='X', s=220, color=COLORS[k], edgecolors='black', linewidths=1)
            ax.annotate(f'C{k}', point, xytext=(7, 7), textcoords='offset points', weight='bold')
    ax.set(xlabel=f'PC1 ({ratio[0]*100:.1f}%)',
           ylabel=f'PC2 ({ratio[1]*100:.1f}%)' if len(ratio)>1 else 'PC2 no disponible (0)',
           title=title+f'\nProyección 2D: {np.sum(ratio[:2])*100:.1f}% de varianza de entrenamiento')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='best')
    fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raiz', type=Path, default=Path('data_sheet'))
    parser.add_argument('--datos', type=Path)
    parser.add_argument('--salida', type=Path)
    parser.add_argument('--particiones', type=Path)
    parser.add_argument('--predecir', type=Path, help='Modelo propio previamente guardado; no entrena de nuevo')
    labels_group = parser.add_mutually_exclusive_group()
    labels_group.add_argument('--etiquetas', type=Path)
    labels_group.add_argument('--identidad-carpeta', action='store_true')
    parser.add_argument('--semilla', type=int, default=42)
    args = parser.parse_args()
    if args.predecir and args.particiones:
        parser.error('--predecir no admite --particiones: todas las filas son nuevas predicciones')
    try:
        path = find_data(args.raiz, args.datos)
        X, ids, groups, labels, augmentation, names = load_data(path)
        if args.etiquetas:
            mapping_labels = read_mapping(args.etiquetas, 'etiqueta')
            labels = np.asarray([mapping_labels.get(key, '') for key in ids])
        elif args.identidad_carpeta:
            labels = groups.copy()
        warnings = []
        mode = 'prediccion' if args.predecir else 'entrenamiento'
        if args.predecir:
            model = joblib.load(args.predecir)
            if model.get('tipo') != 'kmeans_huellas_v1':
                raise ValueError('Modelo incompatible; usar el generado por este script')
            if not np.array_equal(model['nombres_caracteristicas'], names):
                raise ValueError('Las características o su orden no coinciden con el modelo')
            scaler, km, visual = model['scaler'], model['kmeans'], model['pca_visual']
            assignment = model['cluster_a_clase']
            parts = np.repeat('prediccion', len(X))
            train = np.zeros(len(X), bool)
            Xs = scaler.transform(X)
            predicted = km.predict(Xs)
            classes, counts = [], np.zeros((3, 0), dtype=int)
        else:
            parts = partitions(args.particiones, ids, augmentation)
            train = parts == 'train'
            if train.sum() < 3 or np.unique(X[train], axis=0).shape[0] < 3:
                raise ValueError('Se necesitan al menos tres vectores de entrenamiento distintos')
            scaler = StandardScaler()
            scaler.fit(X[train])
            Xs = scaler.transform(X)
            if not np.isfinite(Xs).all():
                raise ValueError('Escalado inválido; revisar rango numérico de las características')
            if np.max(np.var(Xs[train], axis=0)) < 1e-12:
                raise ValueError('No hay variación útil en entrenamiento')
            km = KMeans(n_clusters=3, init='k-means++', n_init=50,
                        max_iter=500, tol=1e-6, random_state=args.semilla, algorithm='lloyd')
            # Evita exceso de hilos en matrices pequeñas y mejora reproducibilidad.
            with threadpool_limits(limits=1):
                km.fit(Xs[train])
                predicted = km.predict(Xs)
            if len(set(predicted[train])) != 3:
                raise ValueError('No se obtuvieron tres clústeres distintos; revisar vectores')
            visual = PCA(n_components=min(2, X.shape[1], int(train.sum())-1), svd_solver='full')
            visual.fit(Xs[train])
            assignment, classes, counts = class_mapping(predicted, labels, train)
            if km.n_iter_ >= km.max_iter:
                warnings.append('Se alcanzó max_iter: revisar convergencia antes de interpretar centros.')
            if not args.particiones:
                warnings.append('Todas las muestras son entrenamiento; no hay evaluación independiente.')
            if not assignment:
                warnings.append('Sin asociación a identidad: faltan tres clases etiquetadas en entrenamiento o un clúster no tiene etiquetas.')
            if args.particiones and (augmentation == '').any():
                warnings.append('Faltan grupos_aumento en algunas filas: no se pudo comprobar toda posible fuga entre variantes.')
            model = {'tipo': 'kmeans_huellas_v1', 'scaler': scaler, 'kmeans': km,
                     'pca_visual': visual, 'cluster_a_clase': assignment,
                     'nombres_caracteristicas': names, 'ids_entrenamiento': ids[train],
                     'fuente_entrenamiento': str(path), 'semilla': args.semilla}
        distances = km.transform(Xs)
        if not np.isfinite(distances).all():
            raise ValueError('Distancias inválidas')
        if not np.array_equal(predicted, distances.argmin(axis=1)):
            raise ValueError('Inconsistencia entre clústeres y distancias')
        originals = scaler.inverse_transform(km.cluster_centers_)
        Z = project2(visual, Xs)
        centers2d = project2(visual, km.cluster_centers_)
        run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out = (args.salida or path.parent/'KMeans_huellas')/run
        out.mkdir(parents=True, exist_ok=False)
        predicted_class = np.asarray([assignment.get(int(k), '') for k in predicted])
        metadata = ['id', 'grupo_carpeta', 'etiqueta_referencia', 'grupo_aumento', 'conjunto',
                    'cluster', 'clase_asociada', 'distancia_centroide', 'distancia_C0', 'distancia_C1', 'distancia_C2', 'PC1', 'PC2']
        write_csv(out/'asignaciones.csv', metadata,
                  ([ids[i], groups[i], labels[i], augmentation[i], parts[i], int(predicted[i]),
                    predicted_class[i], float(distances[i, predicted[i]]), *distances[i].tolist(),
                    *Z[i].tolist()] for i in range(len(X))))
        for filename, centers in [('centroides_estandarizados.csv', km.cluster_centers_),
                                  ('centroides_originales.csv', originals)]:
            write_csv(out/filename, ['cluster']+names.tolist(),
                      ([k, *center.tolist()] for k, center in enumerate(centers)))
        np.savez_compressed(out/'centroides.npz', estandarizados=km.cluster_centers_,
                            originales=originals, nombres_caracteristicas=names,
                            centroides_pca_2d=centers2d)
        write_csv(out/'correspondencia_clusters.csv', ['cluster', 'clase_asociada'],
                  ([k, assignment.get(k, '')] for k in range(3)))
        write_csv(out/'conteos_cluster_grupo.csv', ['conjunto', 'grupo_carpeta', 'C0', 'C1', 'C2'],
                  ([part, group, *[int(np.sum((parts == part) & (groups == group) & (predicted == k)))
                                  for k in range(3)]]
                   for part in sorted(set(parts)) for group in sorted(set(groups))
                   if np.any((parts == part) & (groups == group))))
        if not args.predecir:
            joblib.dump(model, out/'modelo_kmeans.joblib')
            write_csv(out/'conteos_cluster_clase_train.csv', ['cluster']+classes,
                      ([k, *counts[k].tolist()] for k in range(3)))
            representatives = []
            for k in range(3):
                candidates = np.flatnonzero(train & (predicted == k))
                nearest = candidates[np.argmin(distances[candidates, k])]
                representatives.append([k, ids[nearest], float(distances[nearest, k]),
                                        assignment.get(k, '')])
            write_csv(out/'representantes.csv', ['cluster', 'id_mas_cercano', 'distancia', 'clase_asociada'], representatives)
            write_csv(out/'particiones_para_completar.csv', ['id', 'grupo_aumento', 'conjunto'],
                      zip(ids.tolist(), augmentation.tolist(), parts.tolist()))
        plot_clusters(Z, centers2d, np.asarray([f'Clúster {k}' for k in predicted]),
                      parts, visual.explained_variance_ratio_, out/'clusters_pca.png',
                      'K-Means k=3 · X = centroide proyectado (se agrupa en todas las características)')
        references = np.asarray([label if label else group for label, group in zip(labels, groups)])
        if len(set(references)) >= 2:
            plot_clusters(Z, None, references, parts, visual.explained_variance_ratio_,
                          out/'clases_pca.png', 'Referencia por etiqueta disponible / carpeta sin etiquetar')
        metrics = {}
        for part in sorted(set(parts)):
            select = parts == part
            known = select & (labels != '')
            metrics[part] = {'n': int(select.sum()),
                             'conteos_clusters': {str(k): int(np.sum(select & (predicted == k))) for k in range(3)}}
            if known.sum() > 1 and len(set(labels[known])) > 1:
                metrics[part]['ARI_etiquetas_vs_clusters'] = float(adjusted_rand_score(labels[known], predicted[known]))
            if assignment and known.any():
                metrics[part]['acierto_asociacion_clase'] = float(np.mean(predicted_class[known] == labels[known]))
                metrics[part]['n_etiquetadas'] = int(known.sum())
        if not args.predecir:
            if 1 < len(set(predicted[train])) < train.sum():
                try:
                    with threadpool_limits(limits=1):
                        metrics['train']['silhouette'] = float(silhouette_score(
                            Xs[train], predicted[train], sample_size=min(2000, int(train.sum())), random_state=args.semilla))
                except ValueError:
                    warnings.append('Silhouette no calculable para la muestra seleccionada.')
            metrics['train'].update(inercia=float(km.inertia_), iteraciones=int(km.n_iter_))
        warnings.append('K-Means asigna siempre un clúster: no hay rechazo de huellas desconocidas.')
        summary = {'modo': mode, 'entrada': str(path), 'modelo_entrada': str(args.predecir) if args.predecir else None,
                   'k': 3, 'muestras': len(X), 'caracteristicas': X.shape[1],
                   'espacio_clustering': 'Todas las características, StandardScaler ajustado sólo con train',
                   'pca_solo_visualizacion': True, 'varianza_pca_2d_porcentaje': float(visual.explained_variance_ratio_.sum()*100),
                   'cluster_a_clase': assignment, 'metricas': metrics, 'avisos': warnings,
                   'versiones': {'numpy': np.__version__, 'sklearn': sklearn.__version__}}
        (out/'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Entrada: {path}\nModo: {mode} · {len(X)} muestras · {X.shape[1]} características')
        print('Centroides estandarizados:', km.cluster_centers_.shape)
        for part, values in metrics.items():
            print(part, values)
        print('Asociación clúster/clase:', assignment or 'Sin etiquetas suficientes')
        for warning in warnings:
            print('AVISO:', warning)
        print(f'Resultados: {out}')
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
