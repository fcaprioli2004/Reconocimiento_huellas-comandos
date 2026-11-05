#!/usr/bin/env python3
r"""Reconoce todos los WAV originales de una carpeta con el modelo actual.

python reconocer_carpeta.py --entrada ".\pruebas\Voz_1\lote_01"
Usa las funciones de reconocer_comandos.py: no entrena, calibra ni cambia k.
La normalizacion y extraccion son automaticas. No usar WAV ya normalizados.
Si hay carpetas identificar/reconocer/salir/desconocido, calcula metricas.
Sin etiquetas, muestra decisiones sin atribuirles exactitud.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import reconocer_comandos as rc


def escribir_csv(path, campos, filas):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def resumen_metricas(filas, clases=None):
    clases = tuple(clases or rc.CLASES)
    validas = [r for r in filas if r["estado"] != "error_lectura"]
    conocidas = [r for r in validas if r["real"] in clases]
    negativas = [r for r in validas if r["real"] == "desconocido"]
    def tasa(rows, predicado):
        return float(np.mean([predicado(r) for r in rows])) if rows else None
    return {
        "total_archivos": len(filas), "errores_lectura": sum(r["estado"] == "error_lectura" for r in filas),
        "sin_senal_valida": sum(r["estado"] == "sin_senal_valida" for r in filas),
        "sin_etiqueta": sum(not r["real"] for r in filas),
        "distribucion_salidas": dict(Counter(r["resultado"] for r in validas)),
        "conocidos_evaluados": len(conocidas), "desconocidos_evaluados": len(negativas),
        "aciertos_conocidos": sum(r["resultado"] == r["real"] for r in conocidas),
        "rechazos_conocidos": sum(r["resultado"] == "no_reconocido" for r in conocidas),
        "confusiones_aceptadas": sum(r["resultado"] not in (r["real"], "no_reconocido") for r in conocidas),
        "exactitud_conocidos": tasa(conocidas, lambda r: r["resultado"] == r["real"]),
        "tasa_rechazo_conocidos": tasa(conocidas, lambda r: r["resultado"] == "no_reconocido"),
        "tasa_falsa_aceptacion_desconocidos": tasa(negativas, lambda r: r["resultado"] != "no_reconocido"),
        "por_comando": {c: {"cantidad": sum(r["real"] == c for r in conocidas),
                           "aceptacion_correcta": tasa([r for r in conocidas if r["real"] == c], lambda r: r["resultado"] == c)} for c in clases},
        "nota": "Las etiquetas proceden de carpetas, no de la prediccion. Errores de lectura excluidos y contados aparte. Silencio sin caracteristicas se considera rechazo, no acierto de un comando conocido. No usar este lote para ajustar el modelo si se desea conservarlo como prueba independiente.",
    }


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entrada", type=Path, required=True)
    p.add_argument("--modelo", type=Path, default=base/"modelos_knn/Voz_1/modelo.joblib")
    p.add_argument("--salida", type=Path, default=base/"resultados_pruebas/Voz_1"/datetime.now().strftime("lote_%Y%m%d_%H%M%S_%f"))
    a = p.parse_args()
    entrada, salida = a.entrada.resolve(), a.salida.resolve()
    if not entrada.is_dir():
        p.error(f"No existe la carpeta: {entrada}")
    if salida.exists():
        p.error("La salida ya existe; elegi otra --salida")
    if entrada == salida or entrada in salida.parents or salida in entrada.parents:
        p.error("Entrada y salida deben ser carpetas separadas sin contenerse")
    archivos = sorted(f for f in entrada.rglob("*") if f.is_file() and f.suffix.lower() == ".wav")
    if not archivos:
        p.error("No hay archivos WAV. Converti MP4 a WAV antes de continuar")
    destinos = set()
    for f in archivos:
        if f.is_symlink() or entrada not in f.resolve().parents:
            p.error(f"No se admiten enlaces a archivos: {f}")
        if f.stem.lower().endswith("_normalizado"):
            p.error("La entrada incluye archivos *_normalizado.wav. Selecciona una carpeta con originales para evitar procesarlos dos veces")
        clave = f.relative_to(entrada).as_posix().casefold()
        if clave in destinos:
            p.error(f"Nombres duplicados al ignorar mayusculas: {f.name}")
        destinos.add(clave)
    modelo = rc.cargar_modelo(a.modelo)
    clases = rc.clases_modelo(modelo)
    hash_antes = hashlib.sha256(a.modelo.read_bytes()).hexdigest()
    salida.mkdir(parents=True)
    json_config = {"entrada": str(entrada), "modelo": str(a.modelo.resolve()), "modelo_sha256": hash_antes,
                   "k": modelo["k"], "umbrales": modelo["umbrales"],
                   "apoyo_minimo": modelo["apoyo_minimo"], "estado_umbral": modelo["estado_umbral"],
                   "normalizacion": modelo["normalizacion"], "extraccion": modelo["extraccion"]}
    (salida/"configuracion.json").write_text(json.dumps(json_config, indent=2, ensure_ascii=False), encoding="utf-8")
    campos = ["archivo", "real", "candidata", "resultado", "apoyo", "distancia", "umbral",
              "correcto", "estado", "motivo", "avisos"]
    filas = []
    for archivo in archivos:
        rel = archivo.relative_to(entrada)
        real = rel.parts[0] if len(rel.parts) > 1 and rel.parts[0] in (*clases, "desconocido") else ""
        r = dict.fromkeys(campos, "")
        r.update(archivo=rel.as_posix(), real=real, estado="OK")
        try:
            x, sr = sf.read(str(archivo), dtype="float64", always_2d=False)
        except (OSError, RuntimeError, ValueError) as exc:
            r.update(estado="error_lectura", resultado="error", motivo=str(exc))
        else:
            try:
                yn, vector, rr, metricas = rc.procesar(modelo, x, sr)
            except ValueError as exc:
                r.update(estado="sin_senal_valida", resultado="no_reconocido", motivo=str(exc))
            else:
                limite = modelo["umbrales"][rr["propuesta"]]
                razones = []
                if rr["apoyo"] < modelo["apoyo_minimo"]:
                    razones.append("apoyo insuficiente")
                if rr["distancia"] > limite:
                    razones.append("distancia superior al umbral")
                r.update(candidata=rr["propuesta"], resultado=rc.decidir(modelo, rr),
                         apoyo=rr["apoyo"], distancia=rr["distancia"], umbral=limite,
                         motivo="; ".join(razones) if razones else "aceptado", avisos=metricas["avisos"])
                destino = salida/"normalizados"/rel
                destino.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(destino), yn, modelo["extraccion"]["frecuencia"], subtype="FLOAT")
                vec_path = salida/"vectores"/rel.with_suffix(".npy")
                vec_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(vec_path, vector)
        if real and r["estado"] != "error_lectura":
            esperado = "no_reconocido" if real == "desconocido" else real
            r["correcto"] = r["resultado"] == esperado
        filas.append(r)
        print(f"{rel} -> {r['resultado']}" + (f" | real: {real}" if real else ""))
        # Guardado incremental: las filas completas sobreviven a una interrupcion.
        escribir_csv(salida/"resultados.csv", campos, filas)
    resumen = resumen_metricas(filas, clases)
    (salida/"resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    columnas = [*clases, "no_reconocido"]
    matriz = []
    for real in (*clases, "desconocido"):
        elegibles = [r for r in filas if r["real"] == real and r["estado"] != "error_lectura"]
        matriz.append(dict(real=real, **{c: sum(r["resultado"] == c for r in elegibles) for c in columnas}))
    escribir_csv(salida/"confusion.csv", ["real"]+columnas, matriz)
    hash_despues = hashlib.sha256(a.modelo.read_bytes()).hexdigest()
    if hash_antes != hash_despues:
        raise RuntimeError("El archivo del modelo cambio durante la prueba; revisar si otro proceso lo estaba actualizando")
    print(f"\nArchivos: {len(filas)} | Resultados: {salida}")
    if resumen["conocidos_evaluados"]:
        print(f"Comandos conocidos: {resumen['aciertos_conocidos']}/{resumen['conocidos_evaluados']} correctos ({resumen['exactitud_conocidos']:.1%})")
        print(f"Rechazos: {resumen['rechazos_conocidos']} | Confusiones aceptadas: {resumen['confusiones_aceptadas']}")
    if resumen["desconocidos_evaluados"]:
        print(f"Falsa aceptacion de desconocidos: {resumen['tasa_falsa_aceptacion_desconocidos']:.1%}")
    if resumen["sin_etiqueta"]:
        print("Los archivos sin carpeta de etiqueta tienen prediccion, pero no se evalua su exactitud.")
    return 1 if resumen["errores_lectura"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("Prueba interrumpida. Las filas completadas estan guardadas en resultados.csv")
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}")
