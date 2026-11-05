#!/usr/bin/env python3
"""Reconocimiento de comandos descubiertos desde las carpetas del dataset.

Guardar junto a normalizar_audios.py y extraer_caracteristicas.py, dentro de audio.
Instalar: python -m pip install numpy scipy soundfile scikit-learn sounddevice joblib
1) python reconocer_comandos.py entrenar
2) python reconocer_comandos.py escuchar
3) Opcional: python reconocer_comandos.py calibrar
Listar microfonos: python reconocer_comandos.py dispositivos
Seleccionar uno: python reconocer_comandos.py escuchar --dispositivo 2
Probar WAV original: python reconocer_comandos.py escuchar --archivo ejemplo.wav

Predeterminado: caracteristicas/Voz_1/dataset.npz, grabacion de 2 segundos.
No usa PCA. Compara k=3,5,7 con CV estratificada; scaler ajustado por particion.
La puntuacion CV selecciona k, NO constituye una prueba independiente final.
Los umbrales iniciales son geometricos y provisionales, no garantizan rechazo.
Calibrar usa audios NUEVOS y negativos; su resultado tampoco es un test final.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import normalizar_audios as normalizador
    import extraer_caracteristicas as extractor
except ImportError as exc:
    raise SystemExit("Guarda este script junto a normalizar_audios.py y extraer_caracteristicas.py. " + str(exc))

BASE = Path(__file__).resolve().parent
CLASES = ("identificar", "reconocer", "salir")
APOYO_MINIMO = 0.70  # Heuristica inicial, NO probabilidad calibrada.
UMBRALES_MANUALES = { 
   "identificar": 8.0,
    "reconocer": 8.0,
    "salir": 8.5,
    # "nuevo_comando": 8.0,
}
VECINOS_DISTANCIA = 5
ULTIMA_CAPTURA = None


def clases_modelo(modelo):
    return tuple(str(c) for c in modelo["knn"].classes_)


def evaluar_registros(modelo, registros):
    """Evaluacion DESCRIPTIVA con etiquetas; no reemplaza un test independiente."""
    clases = clases_modelo(modelo)
    conocidos = [r for r in registros if r["real"] in clases]
    negativos = [r for r in registros if r["real"] == "desconocido"]
    def tasa(filas, funcion):
        return float(np.mean([funcion(r) for r in filas])) if filas else None
    return {
        "cantidad_conocidos": len(conocidos), "cantidad_desconocidos": len(negativos),
        "acierto_knn_sin_rechazo": tasa(conocidos, lambda r: r.get("propuesta") == r["real"]),
        "aceptacion_correcta": tasa(conocidos, lambda r: decidir(modelo, r) == r["real"]),
        "rechazo_conocidos": tasa(conocidos, lambda r: decidir(modelo, r) == "no_reconocido"),
        "comando_incorrecto_aceptado": tasa(conocidos, lambda r: decidir(modelo, r) not in (r["real"], "no_reconocido")),
        "falsa_aceptacion_desconocidos": tasa(negativos, lambda r: decidir(modelo, r) != "no_reconocido"),
        "aceptacion_correcta_por_clase": {
            c: tasa([r for r in conocidos if r["real"] == c], lambda r: decidir(modelo, r) == r["real"])
            for c in clases},
    }


def comprobar_calibracion(metricas):
    """Condiciones minimas sobre ajuste; no son garantia de generalizacion."""
    razones = []
    for clase, recall in metricas["aceptacion_correcta_por_clase"].items():
        if recall is None or recall < .80:
            razones.append(f"{clase}: aceptacion correcta {0 if recall is None else recall:.0%}; minimo 80%")
    far = metricas["falsa_aceptacion_desconocidos"]
    if far is None or far > .10 + 1e-12:
        razones.append(f"Falsa aceptacion de desconocidos {0 if far is None else far:.0%}; maximo 10%")
    return razones


def huellas():
    return {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (normalizador, extractor)}


def csv_guardar(path, campos, filas):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def json_guardar(path, contenido):
    path.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")


def entrenar(args):
    with np.load(args.dataset, allow_pickle=False) as d:
        X = np.asarray(d["X"], dtype=float)
        y = np.asarray(d["y"], dtype=str)
        nombres = np.asarray(d["nombres"], dtype=str)
    if X.ndim != 2 or X.shape[1] != 43 or y.shape != (len(X),):
        raise ValueError("Dataset invalido: se esperan X=(N,43) e y=(N,)")
    if not np.array_equal(nombres, np.asarray(extractor.nombres_vector())):
        raise ValueError("El orden de caracteristicas no coincide con el extractor")
    clases = tuple(sorted(set(y.tolist())))
    if not np.isfinite(X).all() or len(clases) < 2 or any(not c.strip() for c in clases):
        raise ValueError("Revisar valores y etiquetas: se requieren al menos dos comandos")
    if set(clases) & {"desconocido", "no_reconocido"}:
        raise ValueError("desconocido y no_reconocido son etiquetas reservadas para pruebas, no comandos de entrenamiento")
    cuentas = {c: int(np.sum(y == c)) for c in clases}
    if min(cuentas.values()) < 5:
        raise ValueError("Se requieren al menos 5 grabaciones por comando")
    # Comprueba parametros cuando existe el JSON generado por nuestro extractor.
    cfg_path = args.dataset.parent / "configuracion.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("parametros") != asdict(extractor.Config()):
            raise ValueError("La configuracion del dataset no coincide con el extractor actual")
        if cfg.get("version_pipeline") != 4:
            raise ValueError("Regenera el dataset con la normalizacion ROI actualizada")
        actuales = huellas()
        if cfg.get("normalizador_sha256") != actuales["normalizar_audios"] or cfg.get("extractor_sha256") != actuales["extraer_caracteristicas"]:
            raise ValueError("Los scripts no coinciden con los usados para producir el dataset; regenera sus etapas")
        cfg_normalizacion = cfg["normalizacion"]
        normalizador.validar_config(normalizador.Config(**cfg_normalizacion))
    else:
        raise ValueError("Falta configuracion.json de extraccion; regenera el dataset actualizado")
    if args.salida.exists() and (not args.salida.is_dir() or any(f.is_file() or f.is_symlink() for f in args.salida.rglob("*"))):
        raise ValueError("La salida ya existe. Usa --salida con una carpeta nueva")
    if np.any(np.ptp(X, axis=0) == 0):
        print("AVISO: hay columnas constantes; StandardScaler las deja sin variacion.")
    if not np.any(np.ptp(X, axis=0) > 0):
        raise ValueError("No hay variacion en las caracteristicas")
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
    comparacion, predicciones = [], {}
    for k in (3, 5, 7):
        if k > min(len(train) for train, _ in cv):
            continue
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=k, weights="distance", metric="euclidean", algorithm="brute"))
        pred = cross_val_predict(pipe, X, y, cv=cv)
        score = float(balanced_accuracy_score(y, pred))
        comparacion.append({"k": k, "exactitud_balanceada_cv": score})
        predicciones[k] = pred
        print(f"k={k}: exactitud balanceada CV = {score:.1%}")
    elegido = max(comparacion, key=lambda r: (r["exactitud_balanceada_cv"], -r["k"]))
    k = elegido["k"]
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean", algorithm="brute").fit(Z, y)
    umbrales = {}
    for c in clases:
        puntos = Z[y == c]
        distancias = np.linalg.norm(puntos[:, None, :] - puntos[None, :, :], axis=2)
        np.fill_diagonal(distancias, np.inf)  # Nunca usar el propio punto como vecino.
        locales = np.sort(distancias, axis=1)[:, :VECINOS_DISTANCIA].mean(axis=1)
        umbrales[c] = max(1e-6, float(np.quantile(locales, 0.95)) * 1.15)
    modelo = {"version": 2, "version_pipeline": 4, "clases": clases, "scaler": scaler, "knn": knn, "Z": Z, "y": y,
              "nombres": nombres, "k": k, "umbrales": umbrales,
              "apoyo_minimo": APOYO_MINIMO, "vecinos_distancia": VECINOS_DISTANCIA,
              "estado_umbral": "provisional", "huellas": huellas(),
              "normalizacion": cfg_normalizacion, "extraccion": asdict(extractor.Config())}
    args.salida.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, args.salida / "modelo.joblib")
    csv_guardar(args.salida/"comparacion_k.csv", ["k", "exactitud_balanceada_cv"], comparacion)
    matriz = confusion_matrix(y, predicciones[k], labels=clases)
    csv_guardar(args.salida/"confusion_cv.csv", ["real"]+list(clases),
                [dict(real=c, **dict(zip(clases, matriz[i].tolist()))) for i, c in enumerate(clases)])
    reporte = {"audios_por_clase": cuentas, "k_elegido": k, "comparacion_cv": comparacion,
               "umbrales_provisionales": umbrales, "apoyo_minimo": APOYO_MINIMO,
               "criterio_distancia": "media de 3 vecinos de la clase propuesta en espacio estandarizado",
               "criterio_umbral_inicial": "percentil95 de distancias intra-clase sin propio punto, x1.15",
               "advertencia": "Radio geometrico calculado con scaler final de entrenamiento; no es evaluacion de rechazo. CV selecciona k, no prueba final independiente.",
               "normalizacion": modelo["normalizacion"], "extraccion": modelo["extraccion"]}
    json_guardar(args.salida/"entrenamiento.json", reporte)
    print(f"\nk elegido: {k}. Modelo: {args.salida/'modelo.joblib'}")
    print("Umbrales PROVISIONALES:", umbrales)
    print("Calibra con grabaciones nuevas y palabras desconocidas antes de evaluar rechazo.")


def cargar_modelo(path):
    modelo = joblib.load(path)
    if modelo.get("version") != 2 or modelo.get("version_pipeline") != 4:
        raise ValueError("Modelo de otra normalizacion: vuelve a normalizar, extraer y entrenar")
    if modelo["huellas"] != huellas():
        raise ValueError("Los scripts de normalizacion/extraccion cambiaron. Regenera la base y reentrena")
    for comando, umbral in UMBRALES_MANUALES.items():
        if comando not in modelo["umbrales"]:
            raise ValueError(f"El modelo no conoce el comando: {comando}")

        if umbral <= 0:
            raise ValueError(f"Umbral inválido para {comando}: {umbral}")

        modelo["umbrales"][comando] = float(umbral)

    if UMBRALES_MANUALES:
        modelo["estado_umbral"] = "manual desde script"

    return modelo


def puntuar(modelo, vector):
    v = np.asarray(vector, dtype=float)
    if v.shape != (43,) or not np.isfinite(v).all():
        raise ValueError("Vector invalido")
    z = modelo["scaler"].transform(v.reshape(1, -1))
    apoyos = modelo["knn"].predict_proba(z)[0]
    indice = int(np.argmax(apoyos))
    propuesta = str(modelo["knn"].classes_[indice])
    puntos = modelo["Z"][modelo["y"] == propuesta]
    distancias = np.sort(np.linalg.norm(puntos-z[0], axis=1))[:modelo["vecinos_distancia"]]
    return {"propuesta": propuesta, "apoyo": float(apoyos[indice]),
            "distancia": float(distancias.mean())}


def decidir(modelo, resultado):
    if resultado.get("propuesta") is None:
        return "no_reconocido"
    c = resultado["propuesta"]
    ok = (resultado["apoyo"] >= modelo["apoyo_minimo"]
          and resultado["distancia"] <= modelo["umbrales"][c])
    return c if ok else "no_reconocido"


def procesar(modelo, audio, sr):
    normalizado, metricas = normalizador.normalizar(audio, sr, normalizador.Config(**modelo["normalizacion"]))
    datos = extractor.extraer(normalizado, modelo["extraccion"]["frecuencia"],
                             extractor.Config(**modelo["extraccion"]))
    resultado = puntuar(modelo, datos["vector"])
    return normalizado, datos["vector"], resultado, metricas


def grabar(dispositivo, duracion=2.0):
    global ULTIMA_CAPTURA
    from captura_segura import grabar_seguro
    audio, sr, metadata = grabar_seguro(dispositivo, duracion=duracion, timeout=max(12.0, duracion+9.0))
    ULTIMA_CAPTURA = metadata
    return audio, sr


def sesion(path, prefijo):
    carpeta = path / (prefijo + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    carpeta.mkdir(parents=True, exist_ok=False)
    return carpeta


def guardar_toma(carpeta, nombre, modelo, audio, sr):
    carpeta.mkdir(parents=True, exist_ok=True)
    sf.write(str(carpeta/f"{nombre}_original.wav"), audio, sr, subtype="FLOAT")
    try:
        yn, vector, resultado, metricas = procesar(modelo, audio, sr)
        sf.write(str(carpeta/f"{nombre}_normalizado.wav"), yn,
                 modelo["extraccion"]["frecuencia"], subtype="FLOAT")
        np.save(carpeta/f"{nombre}_vector.npy", vector)
        if metricas["avisos"]:
            print("Aviso de normalizacion:", metricas["avisos"])
    except ValueError as exc:
        resultado = {"propuesta": None, "apoyo": 0.0, "distancia": None, "error": str(exc)}
    resultado["salida"] = decidir(modelo, resultado)
    resultado["umbrales_usados"] = modelo["umbrales"].copy()
    resultado["apoyo_minimo_usado"] = modelo["apoyo_minimo"]
    resultado["estado_umbral"] = modelo["estado_umbral"]
    if ULTIMA_CAPTURA is not None:
        resultado["captura"] = ULTIMA_CAPTURA.copy()
    json_guardar(carpeta/f"{nombre}_resultado.json", resultado)
    return resultado


def escuchar(args):
    modelo = cargar_modelo(args.modelo)
    print(f"KNN k={modelo['k']} | umbral: {modelo['estado_umbral']}")
    carpeta = sesion(args.registros, "sesion_")
    if args.archivo:
        x, sr = sf.read(str(args.archivo), dtype="float64", always_2d=False)
        mostrar(modelo, guardar_toma(carpeta, "001", modelo, x, sr))
        return
    print(f"Enter: grabar {args.duracion:g} segundos. Escribi q y Enter para salir.")
    i = 0
    while True:
        if input("\nEnter para grabar > ").strip().lower() in ("q", "salir"):
            break
        x, sr = grabar(args.dispositivo, args.duracion)
        i += 1
        mostrar(modelo, guardar_toma(carpeta, f"{i:03d}", modelo, x, sr))
    print("Registros:", carpeta)


def mostrar(modelo, resultado):
    print("RESULTADO:", resultado["salida"])
    if resultado["propuesta"] is not None:
        print(f"Candidata: {resultado['propuesta']} | apoyo ponderado: {resultado['apoyo']:.1%}")
        print(f"Distancia: {resultado['distancia']:.3f} | limite: {modelo['umbrales'][resultado['propuesta']]:.3f}")
    else:
        print("No se pudo extraer una senal valida:", resultado.get("error", ""))


def optimizar_umbrales(modelo, registros):
    """Maximiza media de aceptacion correcta y rechazo de no-clase, por clase.

    Mantiene el umbral de apoyo fijo. En empate elige menor radio.
    Los errores de clasificacion cuentan como fallos de aceptacion del comando real.
    """
    resultado = {}
    clases = clases_modelo(modelo)
    for clase in clases:
        positivos = np.array([r["real"] == clase for r in registros])
        if not positivos.any() or positivos.all():
            raise ValueError("La calibracion necesita positivos y negativos por clase")
        elegible = np.array([r.get("propuesta") == clase and r["apoyo"] >= modelo["apoyo_minimo"] for r in registros])
        d = np.array([r["distancia"] if r.get("distancia") is not None else np.inf for r in registros])
        candidatos = np.unique(np.r_[0.0, d[elegible & np.isfinite(d)]])
        mejor = None
        for limite in candidatos:
            acepta = elegible & (d <= limite)
            sensibilidad = float(acepta[positivos].mean())
            rechazo = float((~acepta[~positivos]).mean())
            score = (sensibilidad + rechazo) / 2
            opcion = (score, -float(limite), sensibilidad, rechazo)
            if mejor is None or opcion[:2] > mejor[:2]:
                mejor = opcion
        resultado[clase] = {"umbral": -mejor[1], "aceptacion_correcta_calibracion": mejor[2],
                            "rechazo_no_clase_calibracion": mejor[3]}
    return resultado


def calibrar(args):
    modelo = cargar_modelo(args.modelo)
    clases = clases_modelo(modelo)
    if args.por_clase < 5 or args.negativos < 10:
        raise ValueError("Graba al menos 5 ejemplos nuevos por comando y 10 negativos")
    carpeta = sesion(args.modelo.parent/"calibraciones", "calibracion_")
    print("No repetir archivos de entrenamiento. Habla con variaciones naturales.")
    print("En desconocido, prioriza palabras ajenas (hola, abrir, musica, ventana, perro...).")
    print("Inclui tambien un poco de ruido y silencio, pero no uses solo silencio.")
    print("Se conserva el modelo anterior y se actualizan radios solo al completar la sesion.")
    registros = []
    for clase in (*clases, "desconocido"):
        cantidad = args.negativos if clase == "desconocido" else args.por_clase
        for i in range(cantidad):
            input(f"\n{clase} · toma {i+1}/{cantidad}: prepara el ejemplo y presiona Enter > ")
            x, sr = grabar(args.dispositivo, args.duracion)
            r = guardar_toma(carpeta/clase, f"{i+1:03d}", modelo, x, sr)
            r["real"] = clase
            registros.append(r)
    for clase in clases:
        validos = sum(r["real"] == clase and r.get("propuesta") is not None for r in registros)
        if validos < 3:
            raise ValueError(f"Calibracion incompleta: solo {validos} tomas validas de {clase}. Modelo sin cambios; revisa las grabaciones")
    negativos_validos = sum(r["real"] == "desconocido" and r.get("propuesta") is not None for r in registros)
    if negativos_validos < 3:
        raise ValueError("Se necesitan al menos 3 negativos con caracteristicas validas (otras palabras). Modelo sin cambios; no calibres solo con silencio")
    tabla = optimizar_umbrales(modelo, registros)
    candidato = dict(modelo)
    candidato["umbrales"] = {c: tabla[c]["umbral"] for c in clases}
    metricas_antes = evaluar_registros(modelo, registros)
    metricas_despues = evaluar_registros(candidato, registros)
    razones = comprobar_calibracion(metricas_despues)
    # Evita reemplazar una version que acepta mas comandos correctos.
    if metricas_despues["aceptacion_correcta"] + 1e-12 < metricas_antes["aceptacion_correcta"]:
        razones.append("La aceptacion correcta disminuye respecto del modelo actual")
    auditoria = {"estado": "rechazada" if razones else "aprobada en calibracion",
                 "razones": razones, "antes": metricas_antes, "candidato": metricas_despues,
                 "radios_actuales": modelo["umbrales"], "radios_propuestos": candidato["umbrales"],
                 "nota": "Metricas sobre ajuste, no sobre prueba independiente"}
    json_guardar(carpeta/"auditoria.json", auditoria)
    # Incluye decisiones explicitas antes y despues, no solo las del modelo viejo.
    for r in registros:
        r["salida_antes"] = decidir(modelo, r)
        r["salida_candidato"] = decidir(candidato, r)
    json_guardar(carpeta/"resumen.json", {"ajuste": tabla, "registros": registros, "auditoria": auditoria})
    if razones:
        print("\nCALIBRACION RECHAZADA. El modelo actual NO se modifico.")
        for razon in razones:
            print("-", razon)
        print("Revisar captura y diferencias con la base antes de ampliar umbrales.")
        print("Informe y audios:", carpeta)
        return
    # No contar la misma calibracion como evaluacion final.
    previo = carpeta/"modelo_anterior.joblib"
    joblib.dump(modelo, previo)
    modelo["umbrales"] = {c: tabla[c]["umbral"] for c in clases}
    modelo["estado_umbral"] = "calibrado (requiere prueba independiente)"
    temporal = args.modelo.with_suffix(".temporal.joblib")
    joblib.dump(modelo, temporal)
    os.replace(temporal, args.modelo)
    print("\nUmbrales actualizados:", modelo["umbrales"])
    for clase in clases:
        tasa = tabla[clase]["aceptacion_correcta_calibracion"]
        print(f"{clase}: aceptacion correcta en calibracion = {tasa:.0%}")
        if tasa < 0.8:
            print("  AVISO: aceptacion baja; revisar microfono, grabaciones y caracteristicas antes de ampliar radios.")
    print("Prueba ahora con NUEVAS grabaciones conocidas y desconocidas.")
    print("Copia del modelo anterior y registros:", carpeta)


def diagnosticar(args):
    """Audita una calibracion completa guardada sin modificar el modelo."""
    modelo = cargar_modelo(args.modelo)
    carpeta = args.calibracion
    if carpeta is None:
        opciones = sorted(args.modelo.parent.glob("calibraciones/*/resumen.json"))
        if not opciones:
            raise ValueError("No hay calibraciones completas guardadas")
        carpeta = opciones[-1].parent
    resumen = json.loads((carpeta/"resumen.json").read_text(encoding="utf-8"))
    registros = resumen["registros"]
    resultado = evaluar_registros(modelo, registros)
    print("Calibracion:", carpeta)
    print("k:", modelo["k"], "| umbrales actuales:", modelo["umbrales"])
    print("Estas metricas corresponden a las tomas guardadas, no a audios nuevos.")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print("\nComprobacion del ajuste:")
    for razon in comprobar_calibracion(resultado):
        print("-", razon)
    print("Si KNN ya propone otra clase antes del rechazo, ampliar el radio no lo corrige.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="modo", required=True)
    p = sub.add_parser("entrenar")
    p.add_argument("--dataset", type=Path, default=BASE/"caracteristicas/Voz_1/dataset.npz")
    p.add_argument("--salida", type=Path, default=BASE/"modelos_knn/Voz_1")
    for modo in ("escuchar", "calibrar"):
        p = sub.add_parser(modo)
        p.add_argument("--modelo", type=Path, default=BASE/"modelos_knn/Voz_1/modelo.joblib")
        p.add_argument("--dispositivo", type=int, default=None)
        p.add_argument("--duracion", type=float, default=2.0, help="Segundos por toma; valor inicial 2")
        if modo == "escuchar":
            p.add_argument("--archivo", type=Path, help="Probar WAV original en vez de grabar")
            p.add_argument("--registros", type=Path, default=BASE/"registros_vivo/Voz_1")
        else:
            p.add_argument("--por-clase", type=int, default=5)
            p.add_argument("--negativos", type=int, default=10)
    sub.add_parser("dispositivos")
    p = sub.add_parser("diagnosticar")
    p.add_argument("--modelo", type=Path, default=BASE/"modelos_knn/Voz_1/modelo.joblib")
    p.add_argument("--calibracion", type=Path, default=None)
    args = parser.parse_args()
    if hasattr(args, "duracion") and args.duracion <= 0:
        parser.error("La duracion debe ser positiva")
    try:
        if args.modo == "entrenar":
            entrenar(args)
        elif args.modo == "escuchar":
            escuchar(args)
        elif args.modo == "calibrar":
            calibrar(args)
        elif args.modo == "diagnosticar":
            diagnosticar(args)
        else:
            import sounddevice as sd
            print(sd.query_devices())
    except (KeyboardInterrupt, EOFError):
        print("\nSesion terminada. Los registros ya guardados se conservan.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
