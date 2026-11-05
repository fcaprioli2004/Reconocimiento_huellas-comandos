# Revisión del proyecto e instrucciones de integración

Revisión del ZIP `huellas(1).zip`, recibido el 29 de octubre de 2026. Estructura indicada por el usuario: `IA_Trabajo_final/huellas` y `IA_Trabajo_final/audio`. Solo se revisó huellas; audio no fue adjuntado en este ZIP ni analizado. Este documento describe los archivos recibidos, no modifica el proyecto y no implementa todavía la interfaz.

## 1. Resultado de la revisión

El flujo de huellas está preparado para conectarlo a una interfaz: Anguli → transformaciones → normalización → vector de características → escalador y K-Means guardados → rechazo por distancia → respuesta.

Comprobaciones realizadas:

- Los ocho scripts Python de la raíz tienen sintaxis válida.
- Los hashes del modelo, normalizador y extractor coinciden con los declarados en `umbrales_rechazo.json`.
- Se cargó el modelo recibido: tres centroides y 91 características por vector.
- Las tres imágenes de `maestras` son copias exactas de las maestras originales de cada identidad en `data_sheet`.
- Se reprocesaron las 12 imágenes transformadas con resultados guardados: coincidieron exactamente las distancias y decisiones registradas, usando los límites actuales.
- La versión corregida de `reconocer_huella.py` decide exclusivamente con `distancia <= limite_distancia`. El puntaje normalizado es diagnóstico.

No se ejecutó el Anguli de Windows en esta revisión. Sus registros incluidos muestran ejecuciones anteriores. La reproducción se hizo sobre las imágenes ya generadas. No se verificó el funcionamiento del audio ni de una interfaz, que aún no existe.

## 2. Organización actual

La raíz general tiene dos carpetas hermanas:

- `IA_Trabajo_final/huellas`: sistema de huellas y materiales de entrenamiento.
- `IA_Trabajo_final/audio`: sistema de voz; fuera del alcance de esta revisión.

Dentro de `huellas`, el ZIP contiene:

| Archivo o carpeta | Uso | Necesario para ejecutar desde la interfaz |
|---|---|---|
| `demo_anguli.py` | Coordinador de generación y reconocimiento | Sí |
| `generador_anguli.py` | Adaptador del ejecutable Anguli | Sí |
| `transformaciones_en_linea.py` | Aumentos encadenados de la nueva impresión | Sí |
| `reconocer_huella.py` | Carga del experimento, extracción y predicción | Sí |
| `normalizar_huellas.py` | Normalización de cada imagen | Sí |
| `extraer_caracteristicas.py` | Vector de 91 características | Sí |
| `modelo_kmeans.joblib` | Escalador, centroides y asociación de clases | Sí |
| `umbrales_rechazo.json` | Límites y configuración del procesamiento | Sí |
| `config_anguli.json` | Rutas y parámetros del generador | Sí |
| `Anguli_base.conf` | Plantilla para generar intrusos | Sí |
| `maestras/identidad_1.png`, `identidad_2.png`, `identidad_3.png` | Maestras para generar nuevas impresiones de cada identidad | Sí |
| `referencia/meta_identidad_1.txt`, `meta_identidad_2.txt`, `meta_identidad_3.txt` | Metadatos de referencia; no reconstruyen por sí solos una huella | No para la llamada actual; conservar como documentación |
| `Generaciones_anguli/` | Historial de ejecuciones | Se crea automáticamente; no requiere historial previo |
| `data_sheet/` | Base, normalizados, vectores, PCA y entrenamiento | No durante inferencia; conservar para reproducir el trabajo |
| `Huellas_pruebas/` y `Huellas_intruso/` | Evaluación con imágenes guardadas | No durante generación; conservar para pruebas |
| `analizar_pca_huellas.py` | Análisis y visualización PCA | No durante inferencia |
| `kmeans_huellas.py` | Entrenamiento y análisis de clústeres | No durante inferencia |
| `particiones_huellas.csv` | Separación de entrenamiento, validación y prueba | No durante inferencia; conservar |
| `.venv/` | Entorno Python instalado en Windows | Necesario si se usa ese intérprete; recreable |
| `__pycache__/` | Caché de Python | No; recreable |

En `data_sheet` existen `Huellas_1`, `Huellas_2`, `Huellas_3` y `Caracteristicas_huellas`. Cada identidad tiene `Huellas`, `Huellas_normalizadas` y los originales de Anguli en `info` o `Info`. Esa diferencia de mayúsculas debe respetarse si se programa algo para Linux. La inferencia usa `maestras`, no esas rutas originales.

El experimento guardado está en `data_sheet/Caracteristicas_huellas/20261029T134624_945103Z/`. Contiene vectores, archivos por imagen, imágenes marcadas y las carpetas PCA y K-Means. El entrenamiento utilizado está bajo `KMeans_huellas/20261029T144702_486297Z/`. Para la interfaz, la copia vigente del modelo es la ubicada en la raíz de `huellas`.

No es necesario reorganizar estas carpetas para integrar. Mantener inicialmente las rutas existentes reduce cambios.

## 3. Ajustes concretos encontrados

### Semillas autorizadas incompletas

Los `Anguli.conf` originales de la base indican semillas 1, 20 y 42 para las identidades 1, 2 y 3. Sin embargo, `config_anguli.json` contiene solamente `[1]` en `semillas_autorizadas_conocidas`.

Actualizar ese campo a:

```json
"semillas_autorizadas_conocidas": [1, 20, 42]
```

El generador ya consulta esta lista para excluir semillas al crear intrusos. La exclusión mejora la consistencia de la generación; una semilla diferente no garantiza que el clasificador rechace la imagen.

### Recorte: el valor que realmente se aplica es 8 %

En `transformaciones_en_linea.py`, `recorte_parcial` tiene un valor por defecto `fraccion=.03`, pero `aplicar_transformaciones` le pasa explícitamente `fraccion_roi=.08`. Por eso el flujo aplica 8 %, no 3 %.

Si se desea 3 %, cambiar también el valor explícito del registro de parámetros en la rama `recorte`. Si se desea conservar 8 %, dejar la llamada y alinear el valor por defecto para evitar confusiones. No se eligió una opción por el usuario en esta revisión.

### Contraste: factor 0.2 reduce más que 0.5

El código recibido aplica `factor=.2` mediante `factor * (imagen - 127.5) + 127.5`. Esto aproxima los píxeles al gris y disminuye más el contraste que el factor .5 de la entrega anterior. Es un cambio válido si es intencional; no interpretarlo como una perturbación más suave.

Otros parámetros efectivos recibidos: rotación ±10 grados; traslación x/y entre -8 y 7 píxeles; escala .85 o 1.15; elasticidad amplitud 3; presión por erosión 3×3; gaussiano sigma 5; sal y pimienta proporción .01. Conservar estos valores salvo que el usuario solicite cambiarlos.

### Texto histórico en los umbrales

`umbrales_rechazo.json` conserva en `regla` una descripción de la comparación normalizada original. La decisión real corregida usa los límites por clúster. Actualizar solo la descripción a `Aceptar si distancia al centro mas cercano <= limite_del_cluster` evitará confusiones.

Los campos de selección y calibración describen el experimento original; no prueban que los límites modificados manualmente sigan cumpliendo esas métricas. No sobrescribir los valores actuales durante la integración.

## 4. Contrato de la interfaz y de la voz

La interfaz permitirá seleccionar qué generar: `"1"`, `"2"`, `"3"` o `"intruso"`. Esta selección se transmite únicamente a Anguli. El clasificador recibe los píxeles procesados y debe determinar la respuesta por sí mismo.

| Acción visible o comando de voz | Acción interna | Respuesta |
|---|---|---|
| Reconocer | `modo="autorizar"` | `autorizada`: entero 1 o 0 |
| Identificar | `modo="identificar"` | Identidad estimada si se acepta; “No autorizada” si se rechaza |
| Salir | Cerrar aplicación y recursos de voz | No genera ni clasifica una huella |

En el flujo actual, reconocer e identificar generan y procesan una nueva imagen por llamada. Si se quiere identificar una imagen ya reconocida, habrá que reutilizar su resultado en el controlador: no llamar otra vez al generador sin intención.

Botones y comandos de voz deben invocar el mismo controlador de acciones. La voz solo debe entregar una intención; no debe duplicar la lógica del reconocimiento ni establecer autorización. No se especifica aquí cómo reconocer audio.

## 5. Punto de entrada real

Función en `demo_anguli.py`:

```python
generar_y_reconocer(
    identidad,
    config,
    anguli=None,
    modo="autorizar",
    modelo=None,
    umbrales=None,
    salida=None,
    *,
    transformar=True,
    cantidad_transformaciones=None,
    semilla_transformaciones=None,
    nombres_transformaciones=None,
)
```

Por defecto se sortean dos o tres transformaciones distintas. Se aplican en el orden definido por el módulo, no necesariamente en el orden escrito por el usuario. La semilla de transformaciones permite reproducir los aumentos sobre la misma imagen de entrada; no fija la generación de Anguli.

Los scripts actuales usan importaciones locales sin prefijo de paquete. La futura IA debe resolverlo explícitamente. Ejemplo de adaptación mínima desde un lanzador ubicado en `IA_Trabajo_final`, sin modificar los scripts del experimento:

```python
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
HUELLAS = BASE / "huellas"
sys.path.insert(0, str(HUELLAS))

from demo_anguli import generar_y_reconocer

def procesar_solicitud(seleccion, accion):
    # Ejecutar esta función en un trabajador, fuera del hilo de la ventana.
    modo = {"reconocer": "autorizar", "identificar": "identificar"}[accion]
    return generar_y_reconocer(
        identidad=seleccion,
        config=HUELLAS / "config_anguli.json",
        modo=modo,
    )
```

Este ejemplo no implementa ventana ni voz. Evitar nombres de módulos que colisionen entre audio y huellas. Una conversión futura a paquete exige ajustar importaciones y revisar los hashes de los archivos protegidos; no basta con añadir `__init__.py`.

El ejecutable está configurado como `C:/Anguli-MSVC/Anguli.exe`. Deben permanecer disponibles sus recursos y bibliotecas. Los parámetros opcionales `modelo` y `umbrales` deben ser rutas absolutas si se pasan desde otra ubicación; por defecto se buscan junto a `config_anguli.json`. Las rutas relativas de maestras y salida se resuelven respecto de esa configuración.

## 6. Resultado y presentación

En una ejecución correcta, el diccionario devuelto incluye:

- `estado`: `"ok"`.
- `autorizada`: 1 o 0.
- `identidad`: `"Huellas_1"`, `"Huellas_2"` o `"Huellas_3"` si se acepta en modo identificar; `None` en los demás casos.
- `imagen_generada`: original de Anguli.
- `imagen_a_procesar`: imagen final transformada, entrada real al normalizador.
- `transformaciones`: métodos, parámetros, semilla y rutas de pasos.
- `minucias`, `avisos`: información auxiliar.
- `reconocimiento`: clúster, clase cercana, distancias, límite aplicado y decisión.
- `archivo_resultado`: ubicación del JSON guardado, añadida al diccionario devuelto después de escribirlo; no asumir que aparece dentro del propio JSON.

Mostrar principalmente la imagen transformada y el resultado. Se pueden ofrecer vistas opcionales del original, normalizada, binaria y esqueleto. No mostrar la clase cercana como identidad confirmada cuando la huella se rechaza. `identidad=None` en modo autorizar no significa por sí solo rechazo: consultar `autorizada`.

Errores de generación, dependencias, lectura o procesamiento lanzan excepciones. Tras iniciar el procesamiento, algunos fallos también dejan un `resultado.json` con `estado="error"` y `autorizada=None`. Otros fallos previos pueden no generar ese archivo. El controlador debe capturar excepciones, mostrar “Error de procesamiento” y no convertirlas en un rechazo exitoso.

## 7. Archivos de cada ejecución

La salida predeterminada es `huellas/Generaciones_anguli/<fecha_UTC_y_codigo>/`:

- `salida_anguli/Impression_1/...`: impresión original; no asumir un nombre fijo del archivo, usar la ruta devuelta.
- `transformaciones/huella_transformada.png`: imagen final antes de normalizar.
- `transformaciones/huella_transformada.json`: parámetros, semilla y hashes.
- `transformaciones/huella_transformada_pasos/`: imagen después de cada paso.
- `procesamiento/grises.png`, `binarias.png`, `esqueletos.png`, `mascaras.png`: productos del normalizador.
- `generacion.json`, `anguli.log`, `resultado.json`: registros.

Los JSON históricos contienen rutas absolutas de la computadora original. Al mover el proyecto, usar las rutas de las ejecuciones nuevas o reconstruir las históricas respecto de la carpeta correspondiente; no tratarlas como configuración vigente.

## 8. Modelo y límites vigentes

Se estandariza el vector con el escalador guardado y se calculan distancias a los tres centroides. No entrenar, recalibrar ni ajustar el escalador con cada solicitud.

| Clúster | Clase guardada | Límite recibido |
|---|---|---|
| 0 | `Huellas_3` | 9.316211714119087 |
| 1 | `Huellas_2` | 9.933641332080787 |
| 2 | `Huellas_1` | 9.971949080236985 |

Leer asociación y límites desde los archivos, sin fijarlos en la interfaz. La regla usa la distancia al centro más cercano y el límite de ese clúster. No buscar otro clúster para forzar aceptación. La distancia no es una probabilidad.

Las 91 características incluyen orientación global y espacial, LBP, GLCM, momentos de Hu y densidades de minucias. PCA se usa para análisis y visualización, no hace falta ejecutarlo desde la interfaz. La normalización no garantiza reconstruir información perdida por presión, recorte ni deformaciones elásticas.

En las 12 ejecuciones con transformaciones guardadas había 10 selecciones autorizadas: 8 aceptadas con la identidad correspondiente y 2 rechazadas; los 2 intrusos fueron rechazados. Son registros de demostración, no una evaluación independiente ni una tasa de rendimiento general. Un registro anterior también muestra un intruso aceptado con límites anteriores más amplios.

La integración es viable, pero la evaluación de las combinaciones actuales y de los límites actuales sigue pendiente. Cambiar límites exige reevaluar aceptación y rechazo; no garantiza mejorar ambos a la vez.

## 9. Entorno y ejecución

El ZIP incluye un entorno Windows de aproximadamente 526 MiB. Versiones observadas en sus metadatos:

```text
numpy==2.5.4
scipy==1.18.1
scikit-learn==1.9.1
scikit-image==0.26.0
opencv-python-headless==5.0.0.93
joblib==1.6.0
threadpoolctl==3.7.0
```

PCA y los informes también usan bibliotecas gráficas; se observaron matplotlib 3.11.2 y plotly 7.1.0. Estas son versiones leídas del entorno recibido, no una búsqueda de versiones recomendadas. No copiar el entorno Windows a Linux ni actualizar dependencias del modelo sin verificar compatibilidad. Coordinar dependencias con audio cuando esa parte esté lista; aquí no se evaluaron.

Desde `IA_Trabajo_final`, se puede probar el flujo sin interfaz usando el intérprete del entorno existente:

```powershell
.\huellas\.venv\Scripts\python.exe .\huellas\demo_anguli.py --listar
.\huellas\.venv\Scripts\python.exe .\huellas\demo_anguli.py --identidad 1 --modo identificar
.\huellas\.venv\Scripts\python.exe .\huellas\demo_anguli.py --identidad intruso --modo autorizar
```

`--listar` verifica archivos de generación disponibles; no sustituye una prueba completa de Anguli y reconocimiento.

## 10. Limpieza aconsejada

No se eliminó ningún archivo. Recomendaciones específicas:

- Se puede eliminar `huellas/__pycache__/` y otras cachés equivalentes. Python las recrea.
- Se pueden eliminar `data_sheet/Huellas_1/Huellas_normalizadas/calidad`, y las carpetas equivalentes de identidades 2 y 3, si no se necesitan sus diagnósticos. La inferencia y extracción actual no las usan.
- Se pueden archivar o borrar ejecuciones antiguas dentro de `Generaciones_anguli` si ya no se necesitan sus imágenes ni registros. Conservar ejemplos para la presentación y la validación.
- Se pueden archivar `marcadas`, gráficos PCA e informes redundantes; no son necesarios para inferencia. Conservar al menos el experimento usado y los materiales de la entrega académica.
- `.venv` ocupa la mayor parte del ZIP. Excluirla de futuros ZIP y del repositorio. No borrar el entorno activo hasta tener dependencias registradas y otro entorno funcional.
- Conservar `data_sheet`, pruebas, script PCA, entrenamiento y particiones como materiales académicos, aunque la interfaz no los invoque.
- No borrar `binarias`, `esqueletos` ni `mascaras` de la base si se desea volver a extraer sus características. No borrar maestras, configuraciones, modelo ni umbrales de la raíz.

## 11. Requisitos para la futura IA integradora

1. Mantener inicialmente los archivos de huellas y sus rutas. Añadir el lanzador gráfico en la raíz del proyecto si conviene.
2. Aplicar o resolver los detalles de configuración del apartado 3 con el usuario, preservando sus parámetros intencionales.
3. Resolver las importaciones de los scripts actuales sin modificar accidentalmente los archivos con hash.
4. Crear un controlador compartido por botones y voz; evitar solicitudes duplicadas mientras hay una en curso.
5. Ejecutar la generación y el procesamiento fuera del hilo gráfico; actualizar los componentes gráficos desde el mecanismo de eventos del framework.
6. Manejar estados listo, procesando, resultado y error. Restablecer controles también al fallar.
7. Al salir, detener la escucha y gestionar el trabajador en curso. La función actual es bloqueante y no ofrece cancelación ni callbacks de progreso: no prometer esas capacidades sin implementarlas. Anguli tiene timeout configurado de 120 segundos.
8. Probar las tres selecciones autorizadas y el intruso, ambos modos, varios comandos sucesivos, un error de Anguli y un error de procesamiento.
9. No modificar la clasificación para que coincida con la selección del generador. No usar el nombre del archivo ni la carpeta como característica.
10. Conservar modelo, escalador y umbrales durante la integración. Separar cambios de interfaz de futuros experimentos de mejora del reconocimiento.

La parte de voz debe aportar sus comandos y su mecanismo para detener la escucha cuando esté terminada. No se presupone ninguna librería, función ni modelo de audio a partir de este ZIP.
