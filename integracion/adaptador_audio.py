"""Se importa únicamente dentro del proceso dedicado a audio."""
from pathlib import Path


class AdaptadorAudio:
    def __init__(self, raiz):
        import reconocer_comandos as api
        self.api = api
        self.raiz = Path(raiz)
        self.modelo = None

    def cargar(self):
        if self.modelo is None:
            self.modelo = self.api.cargar_modelo(
                self.raiz / "audio/caracteristicas/Voz_1/dataset.npz"
            )
        return self.modelo

    def diagnostico(self):
        m = self.cargar()
        return {
            "k": m["k"], "apoyo_minimo": m["apoyo_minimo"],
            "umbrales": m["umbrales"], "caracteristicas": m["knn"].n_features_in_,
        }

    def dispositivos(self):
        import sounddevice as sd
        return [
            {"id": i, "nombre": d["name"], "hostapi": d["hostapi"]}
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0
        ]

    @staticmethod
    def onda(audio):
        import numpy as np
        x = np.asarray(audio)
        if x.ndim > 1:
            x = x.mean(axis=1)
        # Envolvente min/max real, conserva picos; solo visualizacion.
        return [
            [float(b.min()), float(b.max())]
            for b in np.array_split(x, min(360, len(x))) if len(b)
        ]

    def evaluar(self, audio, sr, emitir=None):
        m = self.cargar()
        vector = None
        try:
            normalizado, vector, info, metricas = self.api.procesar(m, audio, sr)
            if emitir is not None:
                emitir("normalizado", {"onda": self.onda(normalizado)})
        except ValueError as exc:
            info = {"propuesta": None, "apoyo": 0.0, "distancia": None, "error": str(exc)}
            metricas = {}
        return {
            "comando": self.api.decidir(m, info), "diagnostico": info,
            "metricas": metricas, "umbrales": m["umbrales"],
            "apoyo_minimo": m["apoyo_minimo"],
            "vector_caracteristicas": None if vector is None else [float(v) for v in vector],
        }

    def grabar(self, dispositivo, emitir):
        self.cargar()  # Validar antes de abrir el dispositivo.
        emitir("preparando", {})
        audio, sr = self.api.grabar(dispositivo, duracion=2.0)
        emitir("capturado", {
            "onda": self.onda(audio), "frecuencia": sr,
            "captura": self.api.ULTIMA_CAPTURA,
        })
        return self.evaluar(audio, sr, emitir=emitir)

    def archivo(self, ruta, emitir):
        import soundfile as sf
        self.cargar()
        audio, sr = sf.read(str(ruta), dtype="float64", always_2d=False)
        emitir("capturado", {"onda": self.onda(audio), "frecuencia": sr})
        return self.evaluar(audio, sr, emitir=emitir)
