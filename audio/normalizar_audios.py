#!/usr/bin/env python3
"""Normalizacion consistente de ROI: bordes cortos, filtrado suave y RMS activo.

Guardar en audio. python normalizar_audios.py
Lee data_sheet/Voz_1 y guarda en data_sheet_normalizada/Voz_1.
Todas las subcarpetas se conservan, incluidos comandos nuevos.
No borra originales ni sobrescribe salidas. No cambia tono ni velocidad.
La deteccion por energia NO es un VAD: no identifica si se pronuncio una palabra.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import resample_poly, butter, sosfiltfilt, welch
from scipy.ndimage import uniform_filter1d
import soundfile as sf

VERSION_NORMALIZACION = 4


@dataclass(frozen=True)
class Config:
    frecuencia: int = 16000
    rms_objetivo_dbfs: float = -25.0
    pico_maximo_dbfs: float = -1.0
    ganancia_maxima_db: float = 20.0
    silencio_dbfs: float = -65.0
    umbral_activo_absoluto_dbfs: float = -65.0
    umbral_activo_relativo_db: float = 36.0
    umbral_fuerte_relativo_db: float = 18.0
    ventana_ms: float = 10.0
    referencia_superior_ms: float = 100.0
    percentil_fondo: float = 20.0
    margen_sobre_fondo_db: float = 6.0
    actividad_minima_ms: float = 80.0
    segmento_minimo_ms: float = 35.0
    hueco_maximo_ms: float = 40.0
    margen_recorte_ms: float = 10.0
    contexto_fijo_ms: float = 20.0
    paso_alto_hz: float = 70.0  # 0 desactiva el filtro para una comparacion controlada.
    orden_filtro: int = 2
    contexto_filtro_ms: float = 150.0
    fundido_ms: float = 3.0
    rechazar_ruido_estacionario: bool = True


def db(valor):
    return float(20*np.log10(max(float(valor), 1e-15)))


def validar_config(cfg):
    if cfg.frecuencia != 16000:
        raise ValueError("Esta integracion requiere 16000 Hz")
    numericos = [v for v in asdict(cfg).values() if isinstance(v, (int, float))]
    if not all(np.isfinite(v) for v in numericos):
        raise ValueError("Configuracion con valores no finitos")
    if not cfg.rms_objetivo_dbfs < cfg.pico_maximo_dbfs < 0:
        raise ValueError("El RMS objetivo debe estar por debajo del pico, ambos negativos")
    if cfg.ganancia_maxima_db < 0 or cfg.ventana_ms <= 0 or cfg.referencia_superior_ms <= 0:
        raise ValueError("Ganancia o ventanas invalidas")
    if not 0 < cfg.umbral_fuerte_relativo_db < cfg.umbral_activo_relativo_db:
        raise ValueError("Los umbrales fuerte/debil relativos son invalidos")
    if not 0 <= cfg.percentil_fondo <= 100 or not 0 <= cfg.paso_alto_hz < 1000:
        raise ValueError("Fondo o frecuencia del pasaaltos invalidos")
    if not isinstance(cfg.orden_filtro, int) or not 1 <= cfg.orden_filtro <= 6:
        raise ValueError("Orden de filtro invalido")
    if min(cfg.actividad_minima_ms, cfg.segmento_minimo_ms) <= 0:
        raise ValueError("Duracion minima invalida")
    if min(cfg.hueco_maximo_ms, cfg.margen_recorte_ms, cfg.contexto_fijo_ms, cfg.fundido_ms) < 0:
        raise ValueError("Margenes o fundido negativos")
    if cfg.contexto_filtro_ms < 50:
        raise ValueError("El contexto del filtro debe ser al menos 50 ms")


def _intervalos(mascara):
    bordes = np.diff(np.r_[False, mascara, False].astype(np.int8))
    return list(zip(np.flatnonzero(bordes == 1), np.flatnonzero(bordes == -1)))


def detectar_actividad(x, sr, cfg):
    # Envolvente centrada de RMS, con resolucion de una muestra. Los ceros
    # externos son contexto de analisis, no se estiman como ruido real.
    w = max(3, int(round(sr*cfg.ventana_ms/1000)))
    if len(x) < w:
        raise ValueError("Audio demasiado corto para detectar la ROI")
    envolvente = np.sqrt(np.maximum(0, uniform_filter1d(x*x, size=w, mode="constant")))
    n_top = min(len(x), max(1, round(sr*cfg.referencia_superior_ms/1000)))
    superiores = np.partition(envolvente, len(envolvente)-n_top)[-n_top:]
    referencia = float(np.median(superiores))
    if db(referencia) < cfg.silencio_dbfs:
        raise ValueError("Sin actividad suficiente o solo un impulso demasiado breve")
    fondo = float(np.percentile(envolvente, cfg.percentil_fondo))
    contraste = db(referencia)-db(fondo)
    bajo = max(10**(cfg.umbral_activo_absoluto_dbfs/20),
               referencia*10**(-cfg.umbral_activo_relativo_db/20))
    # Estimar ruido SOLO si hay contexto externo a la actividad preliminar.
    # El percentil de una palabra sin pausas no es una muestra de ruido.
    preliminar = np.flatnonzero(envolvente >= bajo)
    externos = np.r_[envolvente[:preliminar[0]], envolvente[preliminar[-1]+1:]] if len(preliminar) else np.array([])
    fondo_confiable = len(externos) >= round(.1*sr)
    if fondo_confiable:
        fondo = float(np.percentile(externos,cfg.percentil_fondo))
        fondo_confiable = fondo < referencia*10**(-20/20)
    contraste = db(referencia)-db(fondo)
    if fondo_confiable:
        bajo = max(bajo, fondo*10**(cfg.margen_sobre_fondo_db/20))
    alto = max(bajo, referencia*10**(-cfg.umbral_fuerte_relativo_db/20))
    debil = envolvente >= bajo
    fuerte = envolvente >= alto
    hueco = round(sr*cfg.hueco_maximo_ms/1000)
    for ini, fin in _intervalos(~debil):
        if ini > 0 and fin < len(x) and fin-ini <= hueco:
            debil[ini:fin] = True
    limpia = np.zeros(len(x), dtype=bool)
    minimo = round(sr*cfg.segmento_minimo_ms/1000)
    for ini, fin in _intervalos(debil):
        if fin-ini >= minimo and fuerte[ini:fin].any():
            limpia[ini:fin] = True
    actividad_ms = np.count_nonzero(limpia)/sr*1000
    if actividad_ms < cfg.actividad_minima_ms:
        raise ValueError(f"Actividad demasiado breve: {actividad_ms:.0f} ms")
    # Una envolvente plana + espectro muy plano es compatible con ruido continuo.
    # Un tono/vocal sostenida no se rechaza solo por no tener silencios.
    contraste_envolvente = db(np.percentile(envolvente,95))-db(np.percentile(envolvente,5))
    planitud = None
    if cfg.rechazar_ruido_estacionario and contraste_envolvente < 3:
        _, potencia = welch(x, fs=sr, nperseg=min(512,len(x)))
        potencia = potencia[1:]
        planitud = float(np.exp(np.mean(np.log(np.maximum(potencia,1e-30))))/max(np.mean(potencia),1e-30))
        if planitud > .3:
            raise ValueError("Ruido estacionario sin actividad diferenciada; no se normaliza como comando")
    return limpia, {"actividad_ms": float(actividad_ms), "referencia_dbfs": db(referencia),
                    "fondo_estimado_dbfs": db(fondo), "fondo_utilizado": bool(fondo_confiable),
                    "contraste_db": contraste, "umbral_actividad_dbfs": db(bajo),
                    "umbral_fuerte_dbfs": db(alto), "planitud_ruido": planitud}


def normalizar(audio, sr, cfg=Config()):
    """API compatible: audio original -> (mono float32 normalizado, metricas).

    No reconstruye fonemas que no se hayan grabado. La deteccion es energetica,
    no un VAD ni un reconocimiento de palabras. No estira ni cambia el tono.
    """
    validar_config(cfg)
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError("Frecuencia invalida")
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim not in (1,2) or not x.size or not np.isfinite(x).all():
        raise ValueError("Audio vacio, dimensiones invalidas o NaN/infinito")
    canales = 1 if x.ndim == 1 else x.shape[1]
    rms_multicanal = float(np.sqrt(np.mean(x*x)))
    cerca_escala = float(np.mean(abs(x) >= .999))
    x = x if x.ndim == 1 else x.mean(axis=1)
    if canales > 1 and rms_multicanal > 1e-12 and np.sqrt(np.mean(x*x)) < .1*rms_multicanal:
        raise ValueError("Posible cancelacion entre canales al convertir a mono")
    duracion_original = len(x)/sr
    # Registrar mediana; el pasaaltos elimina DC sin sustraer una mediana
    # dependiente de cuanta pausa contenga el archivo. Media final sobre la ROI.
    dc = float(np.median(x))
    if not cfg.paso_alto_hz:
        x = x-dc
    if sr != cfg.frecuencia:
        divisor = math.gcd(int(sr),cfg.frecuencia)
        x = resample_poly(x,cfg.frecuencia//divisor,int(sr)//divisor,window=("kaiser",5.0),padtype="constant")
    sr_out = cfg.frecuencia
    if cfg.paso_alto_hz:
        sos = butter(cfg.orden_filtro,cfg.paso_alto_hz,btype="highpass",fs=sr_out,output="sos")
        p = max(round(sr_out*cfg.contexto_filtro_ms/1000),32)
        xp = np.pad(x,(p,p),mode="constant")
        x = sosfiltfilt(sos,xp,padtype=None)[p:-p]
    mascara, actividad = detectar_actividad(x,sr_out,cfg)
    puntos = np.flatnonzero(mascara)
    roi_ini, roi_fin = int(puntos[0]), int(puntos[-1]+1)
    margen = round(sr_out*cfg.margen_recorte_ms/1000)
    ini = max(0,roi_ini-margen)
    fin = min(len(x),roi_fin+margen)
    faltante_ini = max(0,margen-roi_ini)
    faltante_fin = max(0,roi_fin+margen-len(x))
    recortado = np.pad(x[ini:fin].copy(),(faltante_ini,faltante_fin))
    activa = np.pad(mascara[ini:fin],(faltante_ini,faltante_fin))
    recortado -= recortado.mean()
    fade = min(round(sr_out*cfg.fundido_ms/1000),len(recortado)//2)
    if fade > 1:
        rampa = np.sin(np.linspace(0,np.pi/2,fade))**2
        recortado[:fade] *= rampa
        recortado[-fade:] *= rampa[::-1]
    # Restar la media tras el fundido mantiene la componente continua nula.
    recortado -= recortado.mean()
    rms = float(np.sqrt(np.mean(recortado[activa]**2)))
    pico = float(np.max(abs(recortado)))
    if min(rms,pico) <= 1e-15:
        raise ValueError("Sin energia util en la ROI")
    deseada = 10**(cfg.rms_objetivo_dbfs/20)/rms
    max_g = 10**(cfg.ganancia_maxima_db/20)
    max_p = 10**(cfg.pico_maximo_dbfs/20)/pico
    ganancia = min(deseada,max_g,max_p)
    limite = "objetivo RMS" if ganancia >= deseada*(1-1e-9) else ("ganancia maxima" if max_g <= max_p else "pico maximo")
    contexto = round(sr_out*cfg.contexto_fijo_ms/1000)
    salida_roi = (recortado*ganancia).astype(np.float32)
    rms_real_salida = float(np.sqrt(np.mean(salida_roi[activa].astype(np.float64)**2)))
    y = np.pad(salida_roi,(contexto,contexto))
    avisos = []
    if cerca_escala:
        avisos.append("muestras originales cercanas a escala completa; revisar saturacion")
    if limite != "objetivo RMS":
        avisos.append("RMS objetivo no alcanzado por " + limite)
    if roi_ini == 0 or roi_fin == len(x):
        avisos.append("actividad toca un extremo: revisar si la palabra original esta completa")
    metricas = {"canales_originales":canales,"frecuencia_original":int(sr),
                "duracion_original_s":duracion_original,"duracion_salida_s":len(y)/sr_out,
                "roi_inicio_s":roi_ini/sr_out,"roi_fin_s":roi_fin/sr_out,
                "recorte_inicio_s":ini/sr_out,"recorte_final_s":(len(x)-fin)/sr_out,
                "contexto_fijo_ms":cfg.contexto_fijo_ms,"margen_recorte_ms":cfg.margen_recorte_ms,
                "relleno_extremo_inicio_ms":faltante_ini/sr_out*1000,
                "relleno_extremo_final_ms":faltante_fin/sr_out*1000,
                "dc_original":dc,"dc_salida":float(np.mean(y,dtype=np.float64)),
                "rms_activo_entrada_dbfs":db(rms),"rms_activo_salida_dbfs":db(rms_real_salida),
                "pico_salida_dbfs":db(np.max(abs(y))),"ganancia_db":db(ganancia),
                "limite_aplicado":limite,"fraccion_original_cerca_escala_completa":cerca_escala,
                **actividad,"avisos":"; ".join(avisos)}
    return y, metricas


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entrada", type=Path, default=base/"data_sheet/Voz_1")
    p.add_argument("--salida", type=Path, default=base/"data_sheet_normalizada/Voz_1")
    p.add_argument("--config", type=Path, help="JSON con los parametros o un configuracion.json de esta version")
    a = p.parse_args()
    cfg = Config()
    if a.config:
        contenido = json.loads(a.config.read_text(encoding="utf-8"))
        cfg = Config(**contenido.get("parametros", contenido))
    validar_config(cfg)
    entrada, salida = a.entrada.resolve(), a.salida.resolve()
    if not entrada.is_dir():
        p.error(f"No existe {entrada}")
    if salida.exists() and (not salida.is_dir() or any(f.is_file() or f.is_symlink() for f in salida.rglob("*"))):
        p.error("La salida ya existe: usa otra --salida, sin mezclar versiones")
    if entrada == salida or entrada in salida.parents or salida in entrada.parents:
        p.error("Entrada y salida deben estar separadas sin contenerse")
    archivos = sorted(f for f in entrada.rglob("*") if f.is_file() and f.suffix.lower() in {".wav", ".flac", ".ogg", ".aif", ".aiff"})
    if not archivos:
        p.error("No hay audios compatibles. MP4 requiere conversion real a WAV")
    claves = set()
    for f in archivos:
        if f.is_symlink() or entrada not in f.resolve().parents:
            p.error(f"Enlace no admitido: {f}")
        rel = f.relative_to(entrada)
        if len(rel.parts) < 2:
            p.error(f"Audio fuera de una carpeta de comando: {rel}")
        clave = rel.with_suffix(".wav").as_posix().casefold()
        if clave in claves:
            p.error(f"Destinos duplicados: {rel}")
        claves.add(clave)
    salida.mkdir(parents=True, exist_ok=True)
    config = {"version_normalizacion": VERSION_NORMALIZACION, "parametros": asdict(cfg),
              "normalizador_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "entrada": str(entrada), "formato": "WAV FLOAT32 mono",
              "nota": "ROI por histeresis de energia, contexto corto fijo y filtro pasaaltos. No es VAD ni estima SNR real",
              "versiones": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__}}
    (salida/"configuracion.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    filas = []
    for archivo in archivos:
        rel = archivo.relative_to(entrada)
        fila = {"archivo": rel.as_posix(), "estado": "ERROR", "detalle": ""}
        try:
            x, sr = sf.read(str(archivo), dtype="float64", always_2d=True)
            y, met = normalizar(x, sr, cfg)
            destino = salida/rel.with_suffix(".wav")
            destino.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(destino), y, cfg.frecuencia, subtype="FLOAT")
            fila.update(met, estado="OK")
            print(f"OK {rel}: {met['duracion_original_s']:.2f}s -> {met['duracion_salida_s']:.2f}s")
        except (ValueError, RuntimeError, OSError) as exc:
            fila["detalle"] = str(exc)
            print(f"ERROR {rel}: {exc}")
        filas.append(fila)
    campos = ["archivo", "estado", "detalle"]
    for fila in filas:
        campos.extend(k for k in fila if k not in campos)
    with (salida/"reporte.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(filas)
    errores = sum(f["estado"] != "OK" for f in filas)
    print(f"\nTotal: {len(filas)} | OK: {len(filas)-errores} | Revisar: {errores}\nSalida: {salida}")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
