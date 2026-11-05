#!/usr/bin/env python3
"""Reconocimiento de comandos con KNN puro (3 vecinos ponderados por distancia).

Sin entrenamiento, validacion cruzada ni calibracion: lee los vectores de
caracteristicas existentes en caracteristicas/Voz_1/dataset.npz. No crea ni
lee modelos_knn/ ni archivos .joblib.

Ejemplos desde la carpeta audio/:
    python reconocer_comandos.py dispositivos
    python reconocer_comandos.py escuchar
    python reconocer_comandos.py escuchar --archivo ejemplo.wav
    python reconocer_comandos.py escuchar --dispositivo 2
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

import normalizar_audios as normalizador
import extraer_caracteristicas as extractor

BASE = Path(__file__).resolve().parent
DATASET = BASE / "caracteristicas" / "Voz_1" / "dataset.npz"

# ================== CONFIGURACION MANUAL ==================
K = 3                       # SIEMPRE 3 vecinos para la votacion.
APOYO_MINIMO = 0.70        # De 0.0 a 1.0. No es una tasa real de aciertos.
UMBRALES_MANUALES = {      # Distancias maximas en espacio estandarizado.
    "identificar": 10.0,
    "reconocer": 10.0,
    "salir": 10.0,
}
# Se conserva la metrica original de rechazo: media de los 3 vecinos
# mas cercanos DE LA CLASE PROPUESTA (no confundir con los 3 votos KNN).
VECINOS_DISTANCIA = 3
# ==========================================================

ULTIMA_CAPTURA = None


class KNNPonderado:
    """KNN por fuerza bruta; carga referencias, nunca ajusta un modelo.

    El peso de cada uno de los 3 votos es 1/distancia. Si hay coincidencias
    de distancia cero, votan solo los vecinos de distancia cero, como en
    weights='distance' de scikit-learn.
    """

    def __init__(self, Z, y):
        self.Z = np.asarray(Z, dtype=float)
        self.y = np.asarray(y, dtype=str)
        self.classes_ = np.unique(self.y)
        self.n_features_in_ = int(self.Z.shape[1])

    def predict_proba(self, vectores):
        vectores = np.atleast_2d(np.asarray(vectores, dtype=float))
        if vectores.ndim != 2 or vectores.shape[1] != self.n_features_in_:
            raise ValueError("Cantidad incorrecta de caracteristicas")
        probabilidades = np.zeros((len(vectores), len(self.classes_)), dtype=float)
        for i, vector in enumerate(vectores):
            distancias = np.linalg.norm(self.Z - vector, axis=1)
            indices = np.argsort(distancias, kind="stable")[:K]
            d = distancias[indices]
            if np.any(d == 0):
                pesos = (d == 0).astype(float)
            else:
                pesos = 1.0 / d
            for clase, peso in zip(self.y[indices], pesos):
                j = int(np.searchsorted(self.classes_, clase))
                probabilidades[i, j] += float(peso)
            probabilidades[i] /= probabilidades[i].sum()
        return probabilidades


def _huella_archivo(modulo):
    return hashlib.sha256(Path(modulo.__file__).read_bytes()).hexdigest()


def cargar_modelo(path=DATASET):

    dataset = Path(path)
    if not dataset.is_file():
        raise FileNotFoundError(f"Falta el dataset de audio: {dataset}")
    with np.load(dataset, allow_pickle=False) as datos:
        X = np.asarray(datos["X"], dtype=float)
        y = np.asarray(datos["y"], dtype=str)
        nombres = np.asarray(datos["nombres"], dtype=str)

    if X.ndim != 2 or X.shape[1] != 43 or len(X) < K:
        raise ValueError("Dataset invalido: X debe contener al menos 3 vectores de 43 valores")
    if y.shape != (len(X),) or not np.isfinite(X).all():
        raise ValueError("Dataset invalido: etiquetas o numeros inconsistentes")
    if not np.array_equal(nombres, np.asarray(extractor.nombres_vector())):
        raise ValueError("El orden de caracteristicas no coincide con el extractor")
    clases = set(y.tolist())
    if clases != set(UMBRALES_MANUALES):
        raise ValueError(f"Las clases {sorted(clases)} no coinciden con los umbrales {sorted(UMBRALES_MANUALES)}")
    if any(np.sum(y == clase) < VECINOS_DISTANCIA for clase in clases):
        raise ValueError(f"Se requieren {VECINOS_DISTANCIA} referencias como minimo por comando")
    if not (0.0 <= APOYO_MINIMO <= 1.0):
        raise ValueError("APOYO_MINIMO debe estar entre 0 y 1")
    if any(not np.isfinite(u) or u <= 0 for u in UMBRALES_MANUALES.values()):
        raise ValueError("Todos los umbrales deben ser positivos y finitos")

    config_path = dataset.parent / "configuracion.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Falta configuracion.json junto a {dataset}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("version_pipeline") != 4:
        raise ValueError("El dataset no corresponde a la version 4 del procesamiento")
    if cfg.get("parametros") != asdict(extractor.Config()):
        raise ValueError("Los parametros de extraccion no coinciden con el dataset")
    if cfg.get("normalizador_sha256") != _huella_archivo(normalizador):
        raise ValueError("El normalizador fue modificado; hay que actualizar los datos de referencia")
    if cfg.get("extractor_sha256") != _huella_archivo(extractor):
        raise ValueError("El extractor fue modificado; hay que actualizar los datos de referencia")
    config_normalizacion = cfg["normalizacion"]
    normalizador.validar_config(normalizador.Config(**config_normalizacion))

    media = X.mean(axis=0)
    escala = X.std(axis=0, ddof=0)
    escala[escala == 0] = 1.0
    Z = (X - media) / escala
    return {
        "version_pipeline": 4,
        "k": K,
        "knn": KNNPonderado(Z, y),
        "Z": Z,
        "y": y,
        "media": media,
        "escala": escala,
        "umbrales": dict(UMBRALES_MANUALES),
        "apoyo_minimo": APOYO_MINIMO,
        "vecinos_distancia": VECINOS_DISTANCIA,
        "estado_umbral": "manual desde script",
        "normalizacion": config_normalizacion,
        "extraccion": asdict(extractor.Config()),
    }


def puntuar(modelo, vector):
    """Propone una clase por voto ponderado y calcula distancia de rechazo."""
    v = np.asarray(vector, dtype=float)
    if v.shape != (43,) or not np.isfinite(v).all():
        raise ValueError("Vector de caracteristicas invalido")
    z = (v - modelo["media"]) / modelo["escala"]
    apoyos = modelo["knn"].predict_proba(z.reshape(1, -1))[0]
    indice = int(np.argmax(apoyos))
    propuesta = str(modelo["knn"].classes_[indice])
    referencias = modelo["Z"][modelo["y"] == propuesta]
    distancias = np.sort(np.linalg.norm(referencias - z, axis=1))[:modelo["vecinos_distancia"]]
    return {
        "propuesta": propuesta,
        "apoyo": float(apoyos[indice]),
        "distancia": float(distancias.mean()),
    }


def decidir(modelo, resultado):
    """Acepta si apoyo ponderado y radio manual cumplen sus limites."""
    propuesta = resultado.get("propuesta")
    if propuesta is None:
        return "no_reconocido"
    return (propuesta if resultado["apoyo"] >= modelo["apoyo_minimo"]
            and resultado["distancia"] <= modelo["umbrales"][propuesta]
            else "no_reconocido")


def procesar(modelo, audio, sr):
    """Conserva exactamente la firma y la tupla usadas por la integracion."""
    normalizado, metricas = normalizador.normalizar(
        audio, sr, normalizador.Config(**modelo["normalizacion"]))
    datos = extractor.extraer(
        normalizado, modelo["extraccion"]["frecuencia"],
        extractor.Config(**modelo["extraccion"]))
    resultado = puntuar(modelo, datos["vector"])
    return normalizado, datos["vector"], resultado, metricas


def grabar(dispositivo, duracion=2.0):
    global ULTIMA_CAPTURA
    from captura_segura import grabar_seguro
    audio, sr, metadata = grabar_seguro(
        dispositivo, duracion=duracion, timeout=max(12.0, duracion + 9.0))
    ULTIMA_CAPTURA = metadata
    return audio, sr


def guardar_toma(carpeta, nombre, modelo, audio, sr):
    """Mantiene registros de sesiones de escucha; NO guarda modelos KNN."""
    carpeta.mkdir(parents=True, exist_ok=True)
    sf.write(str(carpeta / f"{nombre}_original.wav"), audio, sr, subtype="FLOAT")
    try:
        normalizado, vector, resultado, metricas = procesar(modelo, audio, sr)
        sf.write(str(carpeta / f"{nombre}_normalizado.wav"), normalizado,
                 modelo["extraccion"]["frecuencia"], subtype="FLOAT")
        np.save(carpeta / f"{nombre}_vector.npy", vector)
        if metricas.get("avisos"):
            print("Avisos:", metricas["avisos"])
    except ValueError as exc:
        resultado = {"propuesta": None, "apoyo": 0.0, "distancia": None, "error": str(exc)}
    resultado["salida"] = decidir(modelo, resultado)
    resultado["umbrales_usados"] = modelo["umbrales"].copy()
    resultado["apoyo_minimo_usado"] = modelo["apoyo_minimo"]
    resultado["estado_umbral"] = modelo["estado_umbral"]
    if ULTIMA_CAPTURA is not None:
        resultado["captura"] = ULTIMA_CAPTURA.copy()
    (carpeta / f"{nombre}_resultado.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    return resultado


def mostrar(modelo, resultado):
    print("RESULTADO:", resultado["salida"])
    if resultado["propuesta"] is None:
        print("No se pudo reconocer:", resultado.get("error", ""))
    else:
        p = resultado["propuesta"]
        print(f"Propuesta: {p} | apoyo ponderado: {resultado['apoyo']:.1%}")
        print(f"Distancia: {resultado['distancia']:.3f} | umbral: {modelo['umbrales'][p]:.3f}")


def escuchar(args):
    modelo = cargar_modelo(args.dataset)
    print(f"KNN ponderado, k={modelo['k']} | umbrales manuales")
    carpeta = args.registros / ("sesion_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    carpeta.mkdir(parents=True, exist_ok=False)
    if args.archivo:
        audio, sr = sf.read(str(args.archivo), dtype="float64", always_2d=False)
        mostrar(modelo, guardar_toma(carpeta, "001", modelo, audio, sr))
        return
    print(f"Enter para grabar {args.duracion:g} segundos; q y Enter para salir")
    i = 0
    while True:
        if input("\nEnter para grabar > ").strip().lower() in ("q", "salir"):
            break
        audio, sr = grabar(args.dispositivo, args.duracion)
        i += 1
        mostrar(modelo, guardar_toma(carpeta, f"{i:03d}", modelo, audio, sr))
    print("Registros:", carpeta)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="modo", required=True)
    p = sub.add_parser("escuchar", help="Reconocer una grabacion o desde el microfono")
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--dispositivo", type=int, default=None)
    p.add_argument("--duracion", type=float, default=2.0)
    p.add_argument("--archivo", type=Path, default=None)
    p.add_argument("--registros", type=Path, default=BASE / "registros_vivo" / "Voz_1")
    sub.add_parser("dispositivos", help="Listar los microfonos")
    args = parser.parse_args()
    if getattr(args, "duracion", 1) <= 0:
        parser.error("La duracion debe ser positiva")
    try:
        if args.modo == "escuchar":
            escuchar(args)
        else:
            import sounddevice as sd
            print(sd.query_devices())
    except (KeyboardInterrupt, EOFError):
        print("\nSesion finalizada")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
