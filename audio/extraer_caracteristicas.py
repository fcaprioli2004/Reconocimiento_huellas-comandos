#!/usr/bin/env python3
"""Extrae MFCC, delta MFCC, ZCR y centroide de los WAV normalizados.

Python >= 3.9. Dependencias: python -m pip install numpy scipy soundfile
Guardar en IA_Trabajo_final/audio y ejecutar: python extraer_caracteristicas.py
Las rutas por defecto se resuelven respecto de ESTE archivo, no de la terminal.

Vector de 43 valores: media MFCC (13), std MFCC (13), std delta (13),
media y std ZCR (2), media y std centroide en Hz (2). Std poblacional ddof=0.
Prioridad propuesta: MFCC principal, delta alta, ZCR y centroide complementaria.
No se inventan pesos numericos ni se ajusta un escalador con toda la base.
Para entrenar: StandardScaler dentro del Pipeline/particion de entrenamiento.
No hacer CMVN por archivo antes del resumen: anularia las medias/std elegidas.

Los NPZ guardan tambien las secuencias completas para conservar el orden temporal.
La mascara de energia NO es VAD: puede seleccionar ruido. No corta el audio.
La entrada proviene del normalizador actualizado; conserva sus parametros
para que el modelo reutilice exactamente la misma configuracion en vivo.
"""

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.fft import dct
from scipy.signal import get_window, savgol_filter


@dataclass(frozen=True)
class Config:
    frecuencia: int = 16000
    ventana_muestras: int = 400  # 25 ms
    salto_muestras: int = 160    # 10 ms
    n_fft: int = 512
    bandas_mel: int = 40
    n_mfcc: int = 13             # C0 a C12
    fmin_hz: float = 20.0
    fmax_hz: float = 8000.0
    delta_ventanas: int = 9
    umbral_absoluto_dbfs: float = -60.0
    umbral_relativo_db: float = 30.0


def nombres_vector():
    return ([f"mfcc_{i:02d}_media" for i in range(13)]
            + [f"mfcc_{i:02d}_std" for i in range(13)]
            + [f"delta_mfcc_{i:02d}_std" for i in range(13)]
            + ["zcr_media", "zcr_std", "centroide_media_hz", "centroide_std_hz"])


def banco_mel(cfg):
    """40 triangulos Mel HTK, altura 1, sin normalizacion de area.

    m(f)=2595*log10(1+f/700). Se evalua en las frecuencias reales de la FFT,
    sin redondear los vertices a bins. No depende de valores por defecto externos.
    """
    mel_min = 2595 * np.log10(1 + cfg.fmin_hz / 700)
    mel_max = 2595 * np.log10(1 + cfg.fmax_hz / 700)
    vertices = 700 * (10 ** (np.linspace(mel_min, mel_max, cfg.bandas_mel + 2) / 2595) - 1)
    frecuencias = np.fft.rfftfreq(cfg.n_fft, 1 / cfg.frecuencia)
    filtros = []
    for a, b, c in zip(vertices[:-2], vertices[1:-1], vertices[2:]):
        filtros.append(np.maximum(0, np.minimum((frecuencias-a)/(b-a),
                                                (c-frecuencias)/(c-b))))
    return np.asarray(filtros), frecuencias


def extraer(audio, sr, cfg=Config()):
    """Entrada mono flotante a 16 kHz. No vuelve a normalizar ni remuestrear.

    Tramas reales de 400 muestras, Hann periodica y ceros SOLO para FFT512.
    No rellena tramas temporales; descarta la cola incompleta (<10 ms).
    Delta: regresion lineal Savitzky-Golay de 9 tramas, unidades por trama,
    bordes extendidos con el valor mas cercano; calculada antes de enmascarar.
    """
    x = np.asarray(audio, dtype=np.float64)
    if sr != cfg.frecuencia:
        raise ValueError(f"Se esperan {cfg.frecuencia} Hz; recibido {sr}. Usar audio normalizado")
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Se espera audio mono, no vacio y sin NaN/infinito")
    minimo = cfg.ventana_muestras + (cfg.delta_ventanas - 1) * cfg.salto_muestras
    if len(x) < minimo:
        raise ValueError(f"Audio demasiado corto para delta: minimo {minimo/sr:.3f} s")
    tramas = np.lib.stride_tricks.sliding_window_view(x, cfg.ventana_muestras)[::cfg.salto_muestras]
    rms = np.sqrt(np.mean(tramas ** 2, axis=1))
    umbral = max(10 ** (cfg.umbral_absoluto_dbfs / 20),
                 float(rms.max()) * 10 ** (-cfg.umbral_relativo_db / 20))
    activa = rms >= umbral
    if np.count_nonzero(activa) < cfg.delta_ventanas:
        raise ValueError("Menos de 9 tramas energeticas: revisar silencio/nivel/duracion")
    ventana = get_window("hann", cfg.ventana_muestras, fftbins=True)
    magnitud = np.abs(np.fft.rfft(tramas * ventana, n=cfg.n_fft, axis=1)).T
    filtros, frecuencias = banco_mel(cfg)
    potencia_mel = filtros @ (magnitud ** 2)
    # Referencia fija 1, piso 1e-10, sin recorte relativo top_db.
    log_mel = 10 * np.log10(np.maximum(potencia_mel, 1e-10))
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:cfg.n_mfcc]
    delta = savgol_filter(mfcc, window_length=cfg.delta_ventanas, polyorder=1,
                          deriv=1, delta=1.0, axis=1, mode="nearest")
    # Cero pertenece al signo positivo; fraccion de los 399 pares adyacentes.
    zcr = np.mean(np.signbit(tramas[:, 1:]) != np.signbit(tramas[:, :-1]), axis=1)
    suma = magnitud.sum(axis=0)
    centroide = np.divide(frecuencias @ magnitud, suma,
                         out=np.zeros_like(suma), where=suma > 0)
    vector = np.concatenate((mfcc[:, activa].mean(axis=1),
                             mfcc[:, activa].std(axis=1, ddof=0),
                             delta[:, activa].std(axis=1, ddof=0),
                             [zcr[activa].mean(), zcr[activa].std(ddof=0),
                              centroide[activa].mean(), centroide[activa].std(ddof=0)]))
    if vector.shape != (43,) or not np.isfinite(vector).all():
        raise ValueError("Vector invalido")
    tiempos = (np.arange(len(tramas)) * cfg.salto_muestras + cfg.ventana_muestras / 2) / sr
    return {
        "vector": vector, "nombres": np.asarray(nombres_vector()),
        "mfcc": mfcc, "delta_mfcc": delta, "zcr": zcr,
        "centroide_hz": centroide, "tramas_activas": activa,
        "tiempos_centro_s": tiempos,
    }


def escribir_csv(path, campos, filas):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrada", type=Path, default=base / "data_sheet_normalizada/Voz_1")
    parser.add_argument("--salida", type=Path, default=base / "caracteristicas/Voz_1")
    args = parser.parse_args()
    entrada, salida = args.entrada.resolve(), args.salida.resolve()
    if not entrada.is_dir():
        parser.error(f"No existe la entrada: {entrada}. Guarda el script dentro de audio o usa --entrada")
    if entrada == salida or entrada in salida.parents or salida in entrada.parents:
        parser.error("Entrada y salida deben estar separadas sin contenerse")
    if salida.exists() and (not salida.is_dir() or any(f.is_file() or f.is_symlink() for f in salida.rglob("*"))):
        parser.error("La salida ya existe; usa otra --salida para evitar sobrescribir")
    archivos = sorted(p for p in entrada.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not archivos:
        parser.error("No hay archivos WAV en la entrada")
    destinos = set()
    for archivo in archivos:
        rel = archivo.relative_to(entrada)
        if archivo.is_symlink() or entrada not in archivo.resolve().parents:
            parser.error(f"Enlace no admitido: {rel}")
        if len(rel.parts) < 2:
            parser.error(f"WAV sin carpeta de comando: {rel}")
        clave = rel.with_suffix(".npz").as_posix().casefold()
        if clave in destinos:
            parser.error(f"Destinos duplicados: {rel}")
        destinos.add(clave)
    cfg = Config()
    import normalizar_audios as normalizador
    origen_cfg = entrada/"configuracion.json"
    if not origen_cfg.is_file():
        parser.error("Falta configuracion.json de normalizacion: ejecuta primero el normalizador actualizado")
    origen = json.loads(origen_cfg.read_text(encoding="utf-8"))
    if origen.get("version_normalizacion") != 4:
        parser.error("La entrada no fue generada con la normalizacion ROI actual; regenera los audios")
    if origen.get("normalizador_sha256") != hashlib.sha256(Path(normalizador.__file__).read_bytes()).hexdigest():
        parser.error("El normalizador cambio desde que se genero la entrada: vuelve a normalizar con la version actual")
    normalizador.validar_config(normalizador.Config(**origen["parametros"]))
    salida.mkdir(parents=True, exist_ok=True)
    config = {
        "version_pipeline": 4,
        "normalizacion": origen["parametros"],
        "normalizador_sha256": hashlib.sha256(Path(normalizador.__file__).read_bytes()).hexdigest(),
        "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parametros": asdict(cfg), "orden_vector": nombres_vector(),
        "prioridades_propuestas": {"MFCC": "principal", "delta_MFCC": "alta",
                                   "ZCR": "complementaria", "centroide": "complementaria"},
        "pesos_numericos_aplicados": False,
        "estadisticas": "media/std poblacional sobre tramas energeticas; delta solo std",
        "mel": "HTK, triangulos altura 1, sin normalizar area",
        "log_mel": "10*log10(max(potencia_mel,1e-10)), referencia 1, sin top_db",
        "dct": "II ortonormal; C0..C12", "delta": "regresion lineal, 9 tramas, borde nearest",
        "zcr": "cambios de signo / 399; cero positivo", "centroide": "magnitud, Hz",
        "mascara": "energia; NO detector de voz; no se concatenan tramas",
        "escalado": "No aplicado. Ajustar StandardScaler solo con entrenamiento",
        "versiones": {"python": sys.version, "numpy": np.__version__,
                      "scipy": scipy.__version__, "soundfile": sf.__version__},
    }
    (salida / "configuracion.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    dataset, reporte = [], []
    vectores, etiquetas, rutas = [], [], []
    campos = ["archivo", "comando"] + nombres_vector()
    for archivo in archivos:
        rel = archivo.relative_to(entrada)
        comando = rel.parts[0]
        fila = {"archivo": rel.as_posix(), "comando": comando, "estado": "ERROR", "detalle": ""}
        try:
            x, sr = sf.read(str(archivo), dtype="float64", always_2d=False)
            datos = extraer(x, sr, cfg)
            destino = salida / rel.with_suffix(".npz")
            destino.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destino, **datos, comando=np.asarray(comando), archivo=np.asarray(rel.as_posix()))
            valores = dict(zip(nombres_vector(), datos["vector"].tolist()))
            registro = {"archivo": rel.as_posix(), "comando": comando, **valores}
            dataset.append(registro)
            escribir_csv(destino.with_suffix(".csv"), campos, [registro])
            vectores.append(datos["vector"])
            etiquetas.append(comando)
            rutas.append(rel.as_posix())
            fila.update(estado="OK", tramas_total=len(datos["zcr"]),
                        tramas_activas=int(datos["tramas_activas"].sum()))
            print(f"OK    {rel} -> 43 caracteristicas")
        except (ValueError, RuntimeError, OSError) as exc:
            fila["detalle"] = str(exc)
            print(f"ERROR {rel}: {exc}")
        reporte.append(fila)
    escribir_csv(salida / "dataset_caracteristicas.csv", campos, dataset)
    escribir_csv(salida / "reporte.csv", ["archivo", "comando", "estado", "detalle", "tramas_total", "tramas_activas"], reporte)
    X = np.vstack(vectores) if vectores else np.empty((0, 43))
    np.savez_compressed(salida / "dataset.npz", X=X, y=np.asarray(etiquetas, dtype=str),
                        archivos=np.asarray(rutas, dtype=str), nombres=np.asarray(nombres_vector()))
    errores = len(archivos) - len(dataset)
    print(f"\nDetectados: {len(archivos)} | OK: {len(dataset)} | Errores: {errores}")
    print(f"X: {X.shape} | Salida: {salida}")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
