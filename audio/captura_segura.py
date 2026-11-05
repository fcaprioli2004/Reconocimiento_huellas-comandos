#!/usr/bin/env python3
r"""Captura WAV con conteo de muestras y un limite externo de tiempo.

El proceso padre no abre PortAudio. Un hijo abre el microfono; si el controlador
se cuelga al abrir, grabar o cerrar, el padre termina el hijo tras el timeout.
No cambia el microfono automaticamente, ni la normalizacion, ni el clasificador.

Prueba sin modelo:
python captura_segura.py --dispositivo 17 --salida prueba_microfono.wav
"""
import argparse
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


def _comando(dispositivo, duracion, wav, meta):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--duracion", str(duracion), "--salida", str(wav), "--metadata", str(meta)]
    if dispositivo is not None:
        cmd += ["--dispositivo", str(dispositivo)]
    return cmd


def grabar_seguro(dispositivo=None, duracion=3.0, timeout=12.0):
    """Devuelve (audio mono float32, frecuencia, metadatos).

    El timeout incluye arranque, apertura, captura y cierre del dispositivo.
    Nunca devuelve un buffer incompleto ni espera indefinidamente sd.wait().
    """
    if duracion <= 0 or timeout < duracion + 2:
        raise ValueError("Duracion invalida o timeout demasiado corto")
    with tempfile.TemporaryDirectory(prefix="captura_voz_") as tmp:
        wav = Path(tmp)/"audio.wav"
        meta = Path(tmp)/"captura.json"
        try:
            proceso = subprocess.run(_comando(dispositivo, duracion, wav, meta), timeout=timeout,
                                     check=False)  # stdout visible, sin buffers de consola.
        except subprocess.TimeoutExpired as exc:
            # subprocess.run mata y espera al hijo antes de propagar TimeoutExpired.
            raise RuntimeError(
                f"El microfono no termino dentro de {timeout:g} segundos. "
                "La captura se cancelo y no se usara audio parcial. Cierra otras aplicaciones "
                "que usen el microfono y comprueba el dispositivo. Para tus JBL puedes probar "
                "--dispositivo 17 (WASAPI), verificando antes su nombre en la lista."
            ) from exc
        if proceso.returncode != 0:
            detalle = "La captura fallo; revisa el mensaje anterior y el dispositivo seleccionado"
            if meta.exists():
                detalle = json.loads(meta.read_text(encoding="utf-8")).get("error", detalle)
            raise RuntimeError(detalle)
        if not wav.exists() or not meta.exists():
            raise RuntimeError("La captura no genero el WAV y sus metadatos")
        import numpy as np
        import soundfile as sf
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        datos = json.loads(meta.read_text(encoding="utf-8"))
        esperado = int(round(duracion * sr))
        if audio.ndim != 1 or len(audio) != esperado or not np.isfinite(audio).all():
            raise RuntimeError("Grabacion incompleta o invalida; no se procesara")
        return audio, sr, datos


def _worker(args):
    """Todo acceso nativo al microfono queda dentro de este proceso."""
    import numpy as np
    import soundfile as sf
    import sounddevice as sd
    info = sd.query_devices(args.dispositivo, "input")
    sr = int(round(info["default_samplerate"]))
    sd.check_input_settings(device=args.dispositivo, channels=1, samplerate=sr, dtype="float32")
    total = int(round(args.duracion * sr))
    audio = np.empty(total, dtype=np.float32)
    estado = {"posicion": 0, "error": None}
    terminado = threading.Event()

    def callback(indata, frames, time_info, status):
        try:
            if status:
                estado["error"] = f"Perdida o irregularidad de muestras: {status}"
                raise sd.CallbackAbort()
            inicio = estado["posicion"]
            cantidad = min(frames, total - inicio)
            audio[inicio:inicio+cantidad] = indata[:cantidad, 0]
            estado["posicion"] += cantidad
            if estado["posicion"] == total:
                raise sd.CallbackStop()
        except (sd.CallbackStop, sd.CallbackAbort):
            raise
        except Exception as exc:
            estado["error"] = f"Error en callback de captura: {exc}"
            terminado.set()
            raise sd.CallbackAbort()

    stream = sd.InputStream(samplerate=sr, device=args.dispositivo, channels=1,
                            dtype="float32", callback=callback, finished_callback=terminado.set)
    print(f"GRABANDO {args.duracion:g} segundos · {info['name']} · habla ahora", flush=True)
    try:
        stream.start()
        if not terminado.wait(args.duracion + 3.0):
            raise RuntimeError(f"El dispositivo no completo la captura: {estado['posicion']}/{total} muestras")
        if estado["error"]:
            raise RuntimeError(estado["error"])
        if estado["posicion"] != total:
            raise RuntimeError(f"Captura interrumpida: {estado['posicion']}/{total} muestras")
    finally:
        # Si una llamada nativa se bloquea aqui, el padre tambien impone su limite.
        try:
            stream.abort(ignore_errors=True)
        finally:
            stream.close(ignore_errors=True)
    sf.write(str(args.salida), audio, sr, subtype="FLOAT")
    metadata = {"dispositivo_solicitado": args.dispositivo, "nombre": info["name"],
                "hostapi": int(info["hostapi"]), "frecuencia": sr, "canales": 1,
                "muestras": total, "duracion_s": args.duracion, "captura": "proceso aislado"}
    args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Grabacion terminada.", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--metadata", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--dispositivo", type=int, default=None)
    p.add_argument("--duracion", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=12.0)
    p.add_argument("--salida", type=Path, required=True)
    a = p.parse_args()
    if a.duracion <= 0:
        p.error("Duracion debe ser positiva")
    if a.worker:
        try:
            return _worker(a)
        except Exception as exc:
            print(f"ERROR DE CAPTURA: {exc}", file=sys.stderr, flush=True)
            if a.metadata:
                a.metadata.write_text(json.dumps({"error": str(exc)}, ensure_ascii=False), encoding="utf-8")
            return 1
    if a.salida.exists() or a.salida.with_suffix(".json").exists():
        p.error("La salida ya existe; elegi otro nombre")
    try:
        x, sr, datos = grabar_seguro(a.dispositivo, a.duracion, a.timeout)
        import soundfile as sf
        a.salida.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(a.salida), x, sr, subtype="FLOAT")
        a.salida.with_suffix(".json").write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Guardado:", a.salida.resolve())
    except KeyboardInterrupt:
        print("\nCaptura cancelada.")
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
