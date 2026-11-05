#!/usr/bin/env python3
"""PCA exploratorio de los comandos presentes en dataset.npz.

Guardar en IA_Trabajo_final/audio, junto a extraer_caracteristicas.py.
Instalar: python -m pip install numpy scikit-learn matplotlib
Ejecutar: python analizar_pca.py
Sin ventanas: python analizar_pca.py --sin-mostrar

Lee caracteristicas/Voz_1/dataset.npz. Guarda en analisis_pca/Voz_1.
Rutas predeterminadas relativas al script; las rutas explicitas, a la terminal.
No modifica los audios ni las caracteristicas. No sobrescribe una salida existente.
PCA se ajusta a toda la base SOLO para exploracion. Las etiquetas solo colorean
los puntos; no intervienen en el ajuste. Para entrenar/evaluar un clasificador,
volver a ajustar StandardScaler y PCA dentro de cada particion de entrenamiento.
No aplicar pesos manuales. No interpretar contribucion PCA como poder predictivo.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


COMANDOS = ("identificar", "reconocer", "salir")
COLORES = ("#0072B2", "#D55E00", "#009E73")
MARCADORES = ("o", "^", "s")


def leer_dataset(path):
    with np.load(path, allow_pickle=False) as d:
        faltantes = {"X", "y", "archivos", "nombres"} - set(d.files)
        if faltantes:
            raise ValueError(f"Faltan claves en dataset.npz: {sorted(faltantes)}")
        X = np.asarray(d["X"], dtype=float)
        y = np.asarray(d["y"], dtype=str)
        archivos = np.asarray(d["archivos"], dtype=str)
        nombres = np.asarray(d["nombres"], dtype=str)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("Se necesitan al menos 3 audios y 2 caracteristicas")
    if y.shape != (len(X),) or archivos.shape != (len(X),) or nombres.shape != (X.shape[1],):
        raise ValueError("Las dimensiones de X, y, archivos y nombres no coinciden")
    if not np.isfinite(X).all():
        raise ValueError("X contiene NaN o infinito. Revisar extraccion; no se imputan valores")
    if len(set(nombres)) != len(nombres):
        raise ValueError("Hay nombres de caracteristicas duplicados")
    if len(set(y)) < 2 or any(not c.strip() for c in y):
        raise ValueError("Se requieren al menos dos comandos con etiquetas no vacias")
    return X, y, archivos, nombres


def ajustar_pca(X):
    variables = np.var(X, axis=0) > 0
    if variables.sum() < 2:
        raise ValueError("Hay menos de 2 caracteristicas con variacion")
    scaler = StandardScaler()
    Z = scaler.fit_transform(X[:, variables])
    rango = int(np.linalg.matrix_rank(Z))
    if rango < 2:
        raise ValueError("El conjunto tiene rango menor a 2; no permite una distribucion PCA 2D")
    pca = PCA(n_components=rango, svd_solver="full", whiten=False)
    scores = pca.fit_transform(Z)
    return variables, scaler, pca, scores


def csv_guardar(path, encabezado, filas):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(encabezado)
        w.writerows(filas)


def graficar(scores, y, ratio, salida, plt):
    comandos = tuple(sorted(set(y)))
    colores = [plt.get_cmap("tab20")(i % 20) for i in range(len(comandos))]
    marcadores = [MARCADORES[i % len(MARCADORES)] for i in range(len(comandos))]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for comando, color, marcador in zip(comandos, colores, marcadores):
        indices = np.flatnonzero(y == comando)
        ax.scatter(scores[indices, 0], scores[indices, 1], s=65, color=color,
                   marker=marcador, edgecolors="white", linewidth=.6,
                   label=f"{comando} (n={len(indices)})")
        for i in indices:
            ax.annotate(str(i+1), (scores[i, 0], scores[i, 1]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel=f"PC1 ({ratio[0]*100:.1f}% de varianza)",
           ylabel=f"PC2 ({ratio[1]*100:.1f}% de varianza)",
           title=f"Comandos de voz · PCA 2D\nVarianza acumulada: {ratio[:2].sum()*100:.1f}%")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(salida / "pca_2d.png", dpi=180)
    if scores.shape[1] >= 3:
        fig3 = plt.figure(figsize=(9, 7), layout="constrained")
        ax3 = fig3.add_subplot(111, projection="3d")
        for comando, color, marcador in zip(comandos, colores, marcadores):
            indices = np.flatnonzero(y == comando)
            ax3.scatter(*scores[indices, :3].T, s=55, color=color, marker=marcador,
                        label=f"{comando} (n={len(indices)})", depthshade=False)
            for i in indices:
                ax3.text(*scores[i, :3], str(i+1), fontsize=8)
        ax3.set(xlabel=f"PC1 ({ratio[0]*100:.1f}%)", ylabel=f"PC2 ({ratio[1]*100:.1f}%)",
                zlabel=f"PC3 ({ratio[2]*100:.1f}%)",
                title=f"Comandos de voz · PCA 3D\nVarianza acumulada: {ratio[:3].sum()*100:.1f}%")
        ax3.legend()
        fig3.savefig(salida / "pca_3d.png", dpi=180)
    figv, (a, b) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    componentes = np.arange(1, len(ratio)+1)
    a.bar(componentes, ratio*100, color="#0072B2")
    a.set(xlabel="Componente", ylabel="Varianza explicada (%)", title="Varianza por componente")
    b.plot(componentes, np.cumsum(ratio)*100, "o-", color="#009E73")
    b.axhline(95, color="#777777", linestyle="--", label="95%")
    b.set(xlabel="Cantidad de componentes", ylabel="Varianza acumulada (%)",
          title="Varianza acumulada", ylim=(0, 103))
    b.legend()
    for axis in (a, b):
        axis.set_xticks(componentes)
        axis.grid(axis="y", alpha=.2)
    figv.savefig(salida / "varianza_explicada.png", dpi=180)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrada", type=Path, default=base/"caracteristicas/Voz_1/dataset.npz")
    parser.add_argument("--salida", type=Path, default=base/"analisis_pca/Voz_1")
    parser.add_argument("--sin-mostrar", action="store_true", help="Guardar graficos sin abrir ventanas")
    args = parser.parse_args()
    entrada, salida = args.entrada.resolve(), args.salida.resolve()
    if not entrada.is_file():
        parser.error(f"No existe {entrada}. Ejecuta primero extraer_caracteristicas.py")
    if salida.exists() and (not salida.is_dir() or any(f.is_file() or f.is_symlink() for f in salida.rglob("*"))):
        parser.error("La salida ya existe. Usa otra --salida para conservar los resultados")
    if entrada.parent == salida or entrada.parent in salida.parents or salida in entrada.parents:
        parser.error("La carpeta de salida debe estar separada de la entrada")
    try:
        X, y, archivos, nombres = leer_dataset(entrada)
        variables, scaler, pca, scores = ajustar_pca(X)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    import matplotlib
    if args.sin_mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    salida.mkdir(parents=True, exist_ok=True)
    ratio = pca.explained_variance_ratio_
    acumulada = np.cumsum(ratio)
    n95 = min(len(ratio), int(np.searchsorted(acumulada, .95) + 1))
    pcs = [f"PC{i+1}" for i in range(len(ratio))]
    csv_guardar(salida/"coordenadas_pca.csv", ["id", "archivo", "comando"] + pcs,
                [[i+1, archivos[i], y[i], *scores[i]] for i in range(len(X))])
    csv_guardar(salida/"varianza_explicada.csv", ["componente", "varianza", "proporcion", "acumulada"],
                [[pcs[i], pca.explained_variance_[i], ratio[i], acumulada[i]] for i in range(len(ratio))])
    # Direcciones/eigenvectores, no correlaciones feature-PC.
    # Porcentaje = coeficiente^2 / suma de coeficientes^2 del eje, que suma 100.
    coef = np.zeros((len(ratio), X.shape[1]))
    coef[:, variables] = pca.components_
    contrib = coef ** 2 * 100
    csv_guardar(salida/"coeficientes_pca.csv", ["caracteristica"]+pcs,
                [[nombre, *coef[:, j]] for j, nombre in enumerate(nombres)])
    csv_guardar(salida/"contribuciones_pca_porcentaje.csv", ["caracteristica"]+pcs,
                [[nombre, *contrib[:, j]] for j, nombre in enumerate(nombres)])
    # Parametros numericos para reproducir la proyeccion sin pickle.
    np.savez_compressed(salida/"transformacion_pca.npz", nombres=nombres,
                        columnas_variables=variables, scaler_media=scaler.mean_,
                        scaler_escala=scaler.scale_, pca_media=pca.mean_,
                        componentes=pca.components_, varianza_explicada=ratio)
    conteo = {c: int(np.sum(y == c)) for c in sorted(set(y))}
    resumen = [
        "PCA EXPLORATORIO DE COMANDOS DE VOZ", "",
        f"Audios: {len(X)} | Caracteristicas originales: {X.shape[1]}",
        f"Audios por comando: {conteo}",
        f"Caracteristicas constantes excluidas: {list(nombres[~variables])}",
        f"Rango numerico / componentes conservados: {len(ratio)}",
        f"Varianza PC1+PC2: {acumulada[1]*100:.2f}%",
        f"Componentes para alcanzar 95% de varianza: {n95}", "",
        "Cada punto representa un audio; su numero coincide con id en coordenadas_pca.csv.",
        "Las etiquetas se usan SOLO para colores y marcadores; PCA no usa las clases.",
        "Distancias: similitud en la proyeccion de las caracteristicas estandarizadas.",
        "La proyeccion pierde informacion; solapamiento 2D no implica clases inseparables.",
        "Separacion visual no demuestra precision ni generalizacion a nuevos audios/voces.",
        "Los porcentajes son varianza de las caracteristicas escaladas, NO acierto del modelo.",
        "Contribuciones: coeficientes al cuadrado, suman 100% por componente; NO relevancia predictiva.",
        "No se aplicaron pesos manuales: cada caracteristica variable tiene varianza unitaria.",
        "Los grupos con mas columnas pueden influir mas; PCA no iguala bloques de caracteristicas.",
        "Escalador y PCA ajustados a toda la base exclusivamente para exploracion.",
        "Para evaluar un clasificador, ajustar ambos SOLO con entrenamiento dentro de cada particion.",
        "No estandarizar de nuevo cada audio de forma independiente.", "",
        "Caracteristicas con mayor contribucion a los primeros ejes:",
    ]
    for k in range(min(3, len(ratio))):
        resumen.append(f"{pcs[k]} ({ratio[k]*100:.2f}% de varianza):")
        for j in np.argsort(-contrib[k])[:5]:
            resumen.append(f"  {nombres[j]}: {contrib[k,j]:.2f}%")
    (salida/"resumen.txt").write_text("\n".join(resumen)+"\n", encoding="utf-8")
    config = {"entrada": str(entrada), "escalado": "StandardScaler por columna, sin pesos",
              "pca": "SVD full, whiten=False, n_components=rango numerico", "uso": "exploratorio",
              "versiones": {"python": sys.version, "numpy": np.__version__,
                            "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}}
    (salida/"configuracion.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    graficar(scores, y, ratio, salida, plt)
    print("\n".join(resumen[:11]))
    print(f"\nResultados: {salida}")
    print("PCA muestra variacion; no mide precision de reconocimiento.")
    if args.sin_mostrar:
        plt.close("all")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
