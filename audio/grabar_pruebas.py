#!/usr/bin/env python3
r"""Graba WAV originales de prueba sin entrenar ni calibrar modelos.

Guardar junto a reconocer_comandos.py y los dos modulos de preprocesamiento.
python grabar_pruebas.py --dispositivo 2 --salida ".\pruebas\Voz_1\lote_01"
Cada Enter graba 2 s (configurable) usando la misma funcion del reconocimiento.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import soundfile as sf
import reconocer_comandos as rc


def guardar_manifest(carpeta, manifest):
    (carpeta/"grabaciones.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dispositivo", type=int, required=True, help="Indice del microfono; JBL era 2 en tu lista")
    p.add_argument("--por-clase", type=int, default=5)
    p.add_argument("--comandos", nargs="+", help="Comandos a grabar; por defecto, los del modelo v3 o las carpetas de data_sheet")
    p.add_argument("--duracion", type=float, default=2.0)
    p.add_argument("--modelo", type=Path, default=base/"modelos_knn/Voz_1/modelo.joblib")
    p.add_argument("--desconocidos", type=int, default=5, help="Otras palabras, ruido y silencio; 0 para omitir")
    p.add_argument("--salida", type=Path, default=base/"pruebas/Voz_1"/datetime.now().strftime("lote_%Y%m%d_%H%M%S_%f"))
    a = p.parse_args()
    if a.por_clase < 1 or a.desconocidos < 0:
        p.error("--por-clase debe ser positivo y --desconocidos no negativo")
    if a.duracion <= 0:
        p.error("Duracion debe ser positiva")
    if a.comandos:
        clases = tuple(dict.fromkeys(a.comandos))
    elif a.modelo.is_file():
        clases = rc.clases_modelo(rc.cargar_modelo(a.modelo))
    else:
        carpetas = sorted(f.name for f in (base/"data_sheet/Voz_1").iterdir() if f.is_dir()) if (base/"data_sheet/Voz_1").is_dir() else []
        clases = tuple(c for c in carpetas if c not in ("desconocido", "no_reconocido")) or rc.CLASES
    if any(c in ("desconocido", "no_reconocido", ".", "..") or any(ch in c for ch in '/\\:') for c in clases):
        p.error("Nombres de comandos invalidos o reservados")
    if a.salida.exists():
        p.error("La salida ya existe. Elegi otro lote, por ejemplo lote_02")
    import sounddevice as sd
    info = sd.query_devices(a.dispositivo, "input")
    sr = int(round(info["default_samplerate"]))
    sd.check_input_settings(device=a.dispositivo, channels=1, samplerate=sr, dtype="float32")
    salida = a.salida.resolve()
    salida.mkdir(parents=True)
    manifest = {"uso": "prueba independiente; no se agrega al entrenamiento",
                "dispositivo": a.dispositivo, "microfono": info["name"],
                "frecuencia": sr, "duracion_s": a.duracion, "comandos": clases, "formato": "WAV FLOAT32 original",
                "grabaciones": []}
    guardar_manifest(salida, manifest)
    print(f"Microfono: {info['name']}\nSalida: {salida}")
    print("Una palabra por toma. Enter inicia; q y Enter termina y conserva las tomas.")
    try:
        for clase in (*clases, "desconocido"):
            total = a.desconocidos if clase == "desconocido" else a.por_clase
            if clase == "desconocido" and total:
                print("\nDeci otras palabras (hola, ventana, musica...), y en alguna toma silencio o ruido.")
            for i in range(1, total+1):
                prompt = f"\n{clase} · {i}/{total} · Enter para grabar, q para terminar > "
                if input(prompt).strip().lower() == "q":
                    return 0
                x, frecuencia = rc.grabar(a.dispositivo, a.duracion)
                relativo = Path(clase)/f"{clase}_{i:03d}.wav"
                destino = salida/relativo
                destino.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(destino), x, frecuencia, subtype="FLOAT")
                manifest["grabaciones"].append({"archivo": relativo.as_posix(), "real": clase,
                                                "frecuencia": frecuencia, "muestras": len(x)})
                guardar_manifest(salida, manifest)
                print("Guardado:", relativo)
    except (KeyboardInterrupt, EOFError):
        print("\nGrabacion interrumpida; las tomas terminadas quedan guardadas.")
    finally:
        print(f"\nTomas guardadas: {len(manifest['grabaciones'])}\nCarpeta: {salida}")
        print(f'Para reconocerlas: python reconocer_carpeta.py --entrada "{salida}"')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}")
