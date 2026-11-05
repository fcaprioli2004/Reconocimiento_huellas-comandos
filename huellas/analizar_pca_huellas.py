#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PCA exploratorio de vectores generados por extraer_caracteristicas.py.

Python >= 3.10. Instalación:
    py -m pip install numpy scikit-learn matplotlib plotly joblib
Uso desde la carpeta que contiene data_sheet:
    py analizar_pca_huellas.py --raiz "data_sheet"
Elige la última ejecución con vectores.npz dentro de Caracteristicas_huellas.
También acepta una carpeta de ejecución o un archivo explícito:
    py analizar_pca_huellas.py --datos "ruta/a/vectores.npz"
Filtrar sólo los tres grupos deseados:
    py analizar_pca_huellas.py --raiz data_sheet --grupos Huellas_1 Huellas_2 Huellas_3
Etiquetas reales opcionales: CSV del extractor, separador ; y columnas id;etiqueta:
    py analizar_pca_huellas.py --raiz data_sheet --etiquetas "ruta/etiquetas.csv"

PROCESAMIENTO:
Valida X e IDs -> selecciona muestras -> elimina columnas constantes ->
StandardScaler -> PCA completo con SVD -> proyecciones 2D/3D y varianza.
Usa X (características originales del extractor), no los píxeles ni máscaras.
No fuerza 3 componentes por haber 3 huellas. Calcula todas las componentes
disponibles y cuántas explican el 90%/95% de la varianza.
Colores: etiquetas si TODAS están completas; de lo contrario grupo de carpeta.
El origen de colores se registra explícitamente. --color permite elegirlo.
Las etiquetas sólo colorean; NO participan en el ajuste de PCA.

SALIDA: al lado de vectores.npz, carpeta PCA_huellas/<ejecucion_UTC>/
  informe_pca.html: interactivo, sin Internet; hover con ID y variante.
  pca_2d.png, pca_3d.png (si hay >=3 componentes), varianza.png,
  cargas_principales.png: figuras para informes.
  proyecciones.csv / proyecciones.npz: coordenadas de todas las componentes.
  varianza_explicada.csv: porcentaje individual y acumulado.
  cargas_pca.csv: coeficientes de PCA para cada característica activa.
  influencia_caracteristicas.csv: varianza aportada y magnitud de pesos PC1/2/3.
  centroides_grupos.csv: MEDIAS de grupos en PCA, NO centroides de K-Means.
  modelo_pca.joblib: scaler, PCA, selección y orden de características.
  resumen.json: entrada, parámetros, avisos, tamaños y varianza conservada.

INTERPRETACION:
Puntos del mismo color próximos sugieren similitud en la proyección.
El solapamiento en PC1/PC2 no demuestra que los vectores completos sean
inseparables. Ver la varianza de 2D/3D y revisar las componentes restantes.
PCA maximiza varianza, NO separación de identidades: cambios de presión,
ruido o normalización también pueden dominar una componente.
Las cargas grandes indican influencia, NO poder discriminante validado.
Si se colorea por carpetas, confirmar que cada carpeta sea una identidad.
Este es un análisis EXPLORATORIO: ajusta scaler y PCA con todas las filas
seleccionadas. Para evaluar un modelo posterior, ajustarlos de nuevo sólo
con entrenamiento; mantener variantes de la misma impresión juntas.
K-Means k=3 es una etapa distinta; no se asignan clases con PCA ni se
entrena aquí K-Means. No limitarlo automáticamente a las dos primeras PCs.

CARGAR NUEVOS VECTORES (mismas características y orden):
    import joblib
    modelo = joblib.load('.../modelo_pca.joblib')  # sólo archivos propios
    Xs = modelo['scaler'].transform(X_nuevo[:, modelo['columnas_activas']])
    Z = modelo['pca'].transform(Xs)

Referencias:
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
"""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import sys

try:
    import numpy as np
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit('Instalá dependencias: py -m pip install numpy '
                     'scikit-learn matplotlib plotly joblib') from exc

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']


def csv_write(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(fields)
        writer.writerows(rows)


def resolve_data(root, explicit):
    if explicit:
        if not explicit.is_file():
            raise ValueError(f'No existe {explicit}')
        return explicit.resolve()
    if not root.is_dir():
        raise ValueError(f'No existe la carpeta {root}')
    if (root/'vectores.npz').is_file():
        return (root/'vectores.npz').resolve()
    base = root if root.name == 'Caracteristicas_huellas' else root/'Caracteristicas_huellas'
    candidates = sorted(base.glob('*/vectores.npz'), key=lambda p: p.parent.name)
    if not candidates:
        raise ValueError('No se encontró vectores.npz. Ejecutá el extractor o indicá --datos.')
    return candidates[-1].resolve()


def strings(data, key, n, default):
    a = np.asarray(data[key]).astype(str) if key in data else np.repeat(default, n)
    if a.shape != (n,):
        raise ValueError(f'{key}: cantidad de elementos incompatible con X')
    return np.char.strip(a)


def load_data(path, labels_path):
    with np.load(path, allow_pickle=False) as d:
        if 'X' not in d or 'ids' not in d:
            raise ValueError('El archivo debe contener X e ids, como la salida del extractor.')
        X = np.asarray(d['X'], dtype=np.float64)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError('X debe ser una matriz no vacía de muestras x características')
        n, p = X.shape
        ids = strings(d, 'ids', n, '')
        groups = strings(d, 'grupos', n, '')
        groups = np.asarray([g if g else (key.split('/')[0] if '/' in key else 'Sin_grupo')
                             for g, key in zip(groups, ids)])
        labels = strings(d, 'etiquetas', n, '')
        names = strings(d, 'nombres_caracteristicas', p, '')
        if (names == '').any():
            names = np.asarray([f'caracteristica_{i:03d}' for i in range(p)])
        augmentation = strings(d, 'grupos_aumento', n, '')
    if (ids == '').any() or len(set(ids)) != n:
        raise ValueError('IDs vacíos o duplicados; revisar el archivo del extractor')
    if len(set(names)) != p:
        raise ValueError('Los nombres de características deben ser únicos')
    if not np.isfinite(X).all():
        bad = np.flatnonzero(~np.isfinite(X).all(axis=1))[:5]
        raise ValueError('Hay NaN/inf; corregir extracción. Ejemplos: '+', '.join(ids[bad]))
    if labels_path:
        lookup = {}
        with labels_path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            if not {'id', 'etiqueta'}.issubset(reader.fieldnames or []):
                raise ValueError('CSV de etiquetas: columnas id;etiqueta, separadas por ;')
            for row in reader:
                key = row['id'].strip()
                if key in lookup:
                    raise ValueError(f'ID repetido en etiquetas: {key}')
                lookup[key] = row['etiqueta'].strip()
        labels = np.asarray([lookup.get(key, '') for key in ids])
    return X, ids, groups, labels, names, augmentation


def variant(key):
    match = re.search(r'(?i)imp[_-]?\d+[_-](.*)', key.rsplit('/', 1)[-1])
    value = match.group(1) if match else key.rsplit('/', 1)[-1]
    return re.sub(r'(?i)(\.(png|jpg|jpeg|bmp|tif|tiff|pgm))+$', '', value)


def scatter_static(Z, groups, explained, path, dims):
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d' if dims == 3 else None)
    for i, group in enumerate(sorted(set(groups))):
        selected = groups == group
        coords = [Z[selected, j] for j in range(dims)]
        ax.scatter(*coords, color=COLORS[i % len(COLORS)], s=30, alpha=0.7,
                   label=f'{group} (n={selected.sum()})')
        center = Z[selected, :dims].mean(axis=0)
        ax.scatter(*[[v] for v in center], color=COLORS[i % len(COLORS)],
                   marker='X', s=160, edgecolors='black', linewidths=0.8)
    labels = [f'PC{i+1} ({explained[i]*100:.1f}%)' for i in range(dims)]
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    if dims == 3:
        ax.set_zlabel(labels[2])
    else:
        ax.axhline(0, color='#aaaaaa', lw=.5); ax.axvline(0, color='#aaaaaa', lw=.5)
        ax.grid(alpha=.2)
    ax.set_title(f'Distribución PCA {dims}D · varianza visible {explained[:dims].sum()*100:.1f}%\nX = media de cada grupo; no es un centro de K-Means')
    ax.legend(loc='best', fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_interactive(Z, colors, ids, groups, labels, variants, ratio, dims):
    fig = go.Figure()
    for i, name in enumerate(sorted(set(colors))):
        idx = np.flatnonzero(colors == name)
        custom = np.column_stack((ids[idx], groups[idx], labels[idx], variants[idx])).tolist()
        args = dict(x=Z[idx, 0].tolist(), y=Z[idx, 1].tolist(), mode='markers',
                    name=f'{name} (n={len(idx)})', customdata=custom,
                    marker=dict(color=COLORS[i % len(COLORS)], size=6, opacity=.75),
                    hovertemplate='ID: %{customdata[0]}<br>Grupo: %{customdata[1]}<br>'
                    'Etiqueta: %{customdata[2]}<br>Variante: %{customdata[3]}<br>'
                    'PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>')
        if dims == 3:
            args['z'] = Z[idx, 2].tolist()
            args['marker']['size'] = 4
            args['hovertemplate'] = args['hovertemplate'].replace('<extra>', '<br>PC3: %{z:.3f}<extra>')
            fig.add_trace(go.Scatter3d(**args))
        else:
            fig.add_trace(go.Scattergl(**args))
    axis = [f'PC{i+1} ({ratio[i]*100:.1f}%)' for i in range(dims)]
    fig.update_layout(template='plotly_white', height=580,
                      title=f'PCA {dims}D · {ratio[:dims].sum()*100:.1f}% de varianza',
                      legend_title_text='Grupo de color', margin=dict(l=40, r=30, t=70, b=40))
    if dims == 3:
        fig.update_layout(scene=dict(xaxis_title=axis[0], yaxis_title=axis[1], zaxis_title=axis[2]))
    else:
        fig.update_layout(xaxis_title=axis[0], yaxis_title=axis[1])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raiz', type=Path, default=Path('data_sheet'))
    parser.add_argument('--datos', type=Path)
    parser.add_argument('--etiquetas', type=Path)
    parser.add_argument('--grupos', nargs='+', help='Nombres exactos de carpetas/grupos a incluir')
    parser.add_argument('--color', choices=['auto', 'grupos', 'etiquetas'], default='auto')
    parser.add_argument('--salida', type=Path, help='Directorio base de resultados; crea subcarpeta por ejecución')
    args = parser.parse_args()
    try:
        data_path = resolve_data(args.raiz, args.datos)
        X, ids, groups, labels, names, augmentation = load_data(data_path, args.etiquetas)
        if args.grupos:
            absent = set(args.grupos)-set(groups)
            if absent:
                raise ValueError('Grupos inexistentes: '+', '.join(sorted(absent)))
            keep = np.isin(groups, args.grupos)
            X, ids, groups, labels, augmentation = [a[keep] for a in (X, ids, groups, labels, augmentation)]
        if len(X) < 3:
            raise ValueError('PCA 2D requiere al menos tres muestras seleccionadas')
        active = np.var(X, axis=0) > 1e-12
        if active.sum() < 2:
            raise ValueError('Hay menos de dos características con variación suficiente')
        warnings = []
        source_color = args.color
        if source_color == 'auto':
            source_color = 'etiquetas' if np.all(labels != '') else 'grupos'
            if source_color == 'grupos':
                warnings.append('Colores por carpeta: faltan etiquetas completas; confirmar identidad de cada grupo.')
        if source_color == 'etiquetas' and (labels == '').any():
            raise ValueError('Para --color etiquetas todas las muestras deben tener etiqueta')
        colors = labels if source_color == 'etiquetas' else groups
        if len(set(colors)) != 3:
            warnings.append(f'Se encontraron {len(set(colors))} grupos de color, no tres. Revisar etiquetas o usar --grupos.')
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X[:, active])
        n_components = min(len(X)-1, int(active.sum()))
        pca = PCA(n_components=n_components, svd_solver='full')
        Z = pca.fit_transform(Xs)
        ratio = pca.explained_variance_ratio_
        cumulative = np.cumsum(ratio)
        if not np.isfinite(Z).all() or not np.isfinite(ratio).all():
            raise ValueError('PCA produjo valores inválidos; revisar rango de las características')
        k90 = min(n_components, int(np.searchsorted(cumulative, .90)+1))
        k95 = min(n_components, int(np.searchsorted(cumulative, .95)+1))
        run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out = (args.salida or data_path.parent/'PCA_huellas')/run
        out.mkdir(parents=True, exist_ok=False)
        variants = np.asarray([variant(key) for key in ids])
        pc_names = [f'PC{i+1}' for i in range(n_components)]
        meta_names = ['id', 'grupo_carpeta', 'etiqueta', 'grupo_color', 'grupo_aumento', 'variante']
        meta = np.column_stack((ids, groups, labels, colors, augmentation, variants))
        csv_write(out/'proyecciones.csv', meta_names+pc_names,
                  ([*row.tolist(), *coords.tolist()] for row, coords in zip(meta, Z)))
        np.savez_compressed(out/'proyecciones.npz', Z=Z, ids=ids, grupos=groups,
                            etiquetas=labels, grupos_color=colors,
                            varianza_explicada=ratio, varianza_acumulada=cumulative)
        csv_write(out/'varianza_explicada.csv', ['componente', 'porcentaje', 'porcentaje_acumulado'],
                  ([pc_names[i], ratio[i]*100, cumulative[i]*100] for i in range(n_components)))
        csv_write(out/'cargas_pca.csv', ['caracteristica']+pc_names,
                  ([name, *weights.tolist()] for name, weights in zip(names[active], pca.components_.T)))
        influence = (pca.components_**2).T @ ratio
        order = np.argsort(-influence)
        csv_write(out/'influencia_caracteristicas.csv', ['caracteristica', 'influencia_varianza_total',
                  *[f'abs_carga_PC{i+1}' for i in range(min(3, n_components))]],
                  ([names[active][j], influence[j],
                    *np.abs(pca.components_[:3, j]).tolist()] for j in order))
        centers = [(name, int((colors == name).sum()), *Z[colors == name].mean(axis=0).tolist())
                   for name in sorted(set(colors))]
        csv_write(out/'centroides_grupos.csv', ['grupo', 'n']+pc_names, centers)
        joblib.dump({'scaler': scaler, 'pca': pca, 'columnas_activas': active,
                     'nombres_caracteristicas': names, 'origen_ajuste': str(data_path),
                     'ids_ajuste': ids, 'uso': 'PCA exploratorio, no validación predictiva'}, out/'modelo_pca.joblib')
        scatter_static(Z, colors, ratio, out/'pca_2d.png', 2)
        if n_components >= 3:
            scatter_static(Z, colors, ratio, out/'pca_3d.png', 3)
        else:
            warnings.append('No hay tres componentes disponibles: no se genera gráfico 3D.')
        plt.rcParams.update({'font.size': 10})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        shown = min(20, n_components)
        axes[0].bar(range(1, shown+1), ratio[:shown]*100, color=COLORS[0])
        axes[0].set(title=f'Varianza individual · primeras {shown} PCs', xlabel='Componente', ylabel='Varianza (%)')
        axes[1].plot(range(1, n_components+1), cumulative*100, color=COLORS[2], lw=2)
        for target in [90, 95]:
            axes[1].axhline(target, linestyle='--', color='#888888', lw=.8)
        axes[1].set(title=f'Acumulada · 95% con {k95} componentes', xlabel='Número de componentes', ylabel='Varianza acumulada (%)', ylim=(0, 102))
        fig.savefig(out/'varianza.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1, min(3, n_components), figsize=(15, 6), constrained_layout=True)
        for pc, ax in enumerate(np.atleast_1d(axes)):
            weights = pca.components_[pc]
            top = np.argsort(np.abs(weights))[-10:]
            ax.barh(np.arange(len(top)), weights[top], color=[COLORS[0] if v>=0 else COLORS[1] for v in weights[top]])
            ax.set_yticks(np.arange(len(top)), names[active][top], fontsize=7)
            ax.set(title=f'PC{pc+1} · mayores cargas', xlabel='Coeficiente en espacio estandarizado')
            ax.axvline(0, color='#666666', lw=.6)
        fig.savefig(out/'cargas_principales.png', dpi=180); plt.close(fig)
        summary = dict(archivo_entrada=str(data_path), ejecucion_utc=run, muestras=len(X),
                       caracteristicas_entrada=X.shape[1], caracteristicas_activas=int(active.sum()),
                       descartadas_por_varianza_baja=names[~active].tolist(), umbral_varianza=1e-12,
                       fuente_colores=source_color, conteos={str(name): int((colors==name).sum()) for name in sorted(set(colors))},
                       varianza_2d_porcentaje=float(ratio[:2].sum()*100),
                       varianza_3d_porcentaje=float(ratio[:3].sum()*100) if n_components>=3 else None,
                       componentes_90_por_ciento=k90, componentes_95_por_ciento=k95,
                       avisos=warnings, uso='Exploración; scaler y PCA ajustados a todas las filas seleccionadas',
                       versiones={'numpy': np.__version__, 'sklearn': sklearn.__version__,
                                  'matplotlib': matplotlib.__version__, 'plotly': plotly.__version__})
        (out/'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        parts = [scatter_interactive(Z, colors, ids, groups, labels, variants, ratio, 2).to_html(
                    full_html=False, include_plotlyjs=True, config={'responsive': True})]
        if n_components >= 3:
            parts.append(scatter_interactive(Z, colors, ids, groups, labels, variants, ratio, 3).to_html(
                         full_html=False, include_plotlyjs=False, config={'responsive': True}))
        variance_plot = go.Figure()
        variance_plot.add_bar(x=list(range(1, n_components+1)), y=(ratio*100).tolist(), name='Individual (%)')
        variance_plot.add_scatter(x=list(range(1, n_components+1)), y=(cumulative*100).tolist(), name='Acumulada (%)')
        variance_plot.update_layout(template='plotly_white', title='Varianza explicada', xaxis_title='Componente', yaxis_title='Porcentaje', height=420)
        parts.append(variance_plot.to_html(full_html=False, include_plotlyjs=False))
        counts_html = ''.join(f'<li>{escape(name)}: {count} imágenes</li>' for name, count in summary['conteos'].items())
        alerts = ''.join(f'<li>{escape(w)}</li>' for w in warnings)
        html = '''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PCA de huellas</title>
<style>body{font:16px/1.6 system-ui;background:#f2f5f9;color:#172b45;margin:0}
main{max-width:1200px;margin:32px auto;padding:28px;background:white;border-radius:14px}
h1{font-size:30px}h2{font-size:22px}.notice{background:#eef5fc;padding:16px;border-radius:8px}
.plots{overflow:hidden}small{color:#52667e}code{background:#eef2f6;padding:2px 5px}</style><main>'''
        html += f'<h1>Distribución de características · PCA</h1><p>{len(X)} imágenes · {int(active.sum())} características activas · colores por {escape(source_color)}.</p>'
        html += f'<div class="notice">PC1 + PC2 conservan <b>{summary["varianza_2d_porcentaje"]:.1f}%</b> de la varianza. Se necesitan <b>{k95}</b> componentes para conservar 95%.</div>'
        html += '<h2>Grupos representados</h2><ul>'+counts_html+'</ul>'
        if alerts:
            html += '<h2>Observaciones</h2><ul>'+alerts+'</ul>'
        html += '<p>Pasá el cursor por los puntos para identificar la imagen y su variante. En 3D, arrastrá para rotar. La leyenda permite ocultar grupos.</p>'
        html += '<div class="plots">'+''.join(parts)+'</div>'
        html += '<h2>Cómo leerlo</h2><p>PCA busca variación, no separación de clases. La superposición en 2D puede ocultar separación en otras componentes. Las deformaciones y el ruido también pueden explicar variación.</p>'
        html += '<p>Es un análisis exploratorio, no una medida de precisión. Las medias de grupos exportadas no son centros de K-Means. Para entrenar y evaluar modelos, ajustá scaler y PCA sólo con entrenamiento.</p>'
        html += f'<small>Entrada: {escape(str(data_path))}. Detalles, coeficientes y datos exportados junto a este informe.</small></main></html>'
        (out/'informe_pca.html').write_text(html, encoding='utf-8')
        print(f'Entrada: {data_path}\nMatriz: {X.shape}; activas: {active.sum()}')
        print(f'Colores por {source_color}: {summary["conteos"]}')
        print(f'Varianza 2D: {summary["varianza_2d_porcentaje"]:.2f}%; 95% con {k95} componentes.')
        for w in warnings:
            print('AVISO:', w)
        print(f'Resultados: {out}\nAbrí informe_pca.html en el navegador.')
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
