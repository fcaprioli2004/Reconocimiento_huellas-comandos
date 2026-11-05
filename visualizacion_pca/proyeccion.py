"""Carga de PCA existentes y proyección de nuevas muestras, sin fit().
Este módulo se ejecuta en un proceso por dominio, con sus imports aislados.
"""
import hashlib
import csv
from pathlib import Path
import numpy as np


class Proyector:
    def __init__(self, dominio, raiz, modelo=None, datos=None):
        self.dominio, self.raiz = dominio, Path(raiz).resolve()
        self.rec = None
        if dominio == 'audio':
            import analizar_pca as original
            self.archivo = Path(modelo) if modelo else self.raiz / 'audio/analisis_pca/Voz_1/transformacion_pca.npz'
            self.datos = Path(datos) if datos else self.raiz / 'audio/caracteristicas/Voz_1/dataset.npz'
            self.X, self.grupos, self.ids, nombres = original.leer_dataset(self.datos)
            with np.load(self.archivo, allow_pickle=False) as d:
                self.modelo = {k: d[k].copy() for k in d.files}
            self.nombres = self.modelo['nombres'].astype(str)
            if not np.array_equal(nombres, self.nombres):
                raise ValueError('El PCA de audio y el dataset tienen distintas características u orden')
            import extraer_caracteristicas as extractor
            if not np.array_equal(self.nombres, extractor.nombres_vector()):
                raise ValueError('El PCA de audio no corresponde al extractor vigente')
            self.ratio = self.modelo['varianza_explicada']
            self.Z = self.transformar(self.X)
            archivo_csv = self.archivo.with_name('coordenadas_pca.csv')
            if archivo_csv.is_file():
                with archivo_csv.open(encoding='utf-8-sig', newline='') as f:
                    filas = list(csv.DictReader(f))
                anteriores = np.array([[float(row[f'PC{i}']) for i in (1, 2, 3)] for row in filas])
                if anteriores.shape != self.Z[:, :3].shape or not np.allclose(anteriores, self.Z[:, :3], rtol=1e-8, atol=1e-8):
                    raise ValueError('El dataset de audio no reproduce las coordenadas PCA guardadas')
        else:
            import analizar_pca_huellas as original
            import joblib
            if modelo:
                self.archivo = Path(modelo)
            else:
                candidatos = list((self.raiz / 'huellas/data_sheet').glob('Caracteristicas_huellas/*/PCA_huellas/*/modelo_pca.joblib'))
                if len(candidatos) != 1:
                    raise ValueError('Se esperaba un único PCA de huellas. Indicá --pca-huellas con la ruta exacta.')
                self.archivo = candidatos[0]
            self.modelo = joblib.load(self.archivo)
            self.datos = Path(datos) if datos else self.archivo.parents[2] / 'vectores.npz'
            X, ids, grupos, etiquetas, nombres, _ = original.load_data(self.datos, None)
            self.nombres = np.asarray(self.modelo['nombres_caracteristicas']).astype(str)
            if not np.array_equal(nombres, self.nombres):
                raise ValueError('PCA de huellas y dataset incompatibles')
            indices = {str(key): i for i, key in enumerate(ids)}
            elegidos = [indices[str(key)] for key in self.modelo['ids_ajuste']]
            self.X, self.ids = X[elegidos], ids[elegidos]
            self.grupos = etiquetas[elegidos] if np.all(etiquetas[elegidos] != '') else grupos[elegidos]
            self.ratio = self.modelo['pca'].explained_variance_ratio_
            self.Z = self.transformar(self.X)
            # Reproducir las coordenadas del gráfico original, incluidos signo y orden.
            archivo_z = self.archivo.with_name('proyecciones.npz')
            with np.load(archivo_z, allow_pickle=False) as d:
                if not np.array_equal(d['ids'].astype(str), self.ids.astype(str)) or not np.allclose(d['Z'], self.Z, rtol=1e-8, atol=1e-8):
                    raise ValueError('Las proyecciones guardadas no coinciden con el PCA y el dataset elegidos')
        if len(self.ratio) < 3 or not np.isfinite(self.Z).all():
            raise ValueError('El PCA no permite una proyección 3D válida')

    def transformar(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != len(self.nombres) or not np.isfinite(X).all():
            raise ValueError('Vector ausente o incompatible con las características del PCA')
        m = self.modelo
        if self.dominio == 'audio':
            z = (X[:, m['columnas_variables']] - m['scaler_media']) / m['scaler_escala']
            return (z - m['pca_media']) @ m['componentes'].T
        return m['pca'].transform(m['scaler'].transform(X[:, m['columnas_activas']]))

    def base(self):
        return {'coordenadas': self.Z[:, :3].tolist(), 'grupos': self.grupos.tolist(),
                'ids': self.ids.tolist(), 'varianza': self.ratio[:3].tolist(),
                'modelo': str(self.archivo), 'dataset': str(self.datos),
                'modelo_sha256': hashlib.sha256(self.archivo.read_bytes()).hexdigest(),
                'dataset_sha256': hashlib.sha256(self.datos.read_bytes()).hexdigest()}

    def proyectar(self, vector=None, imagen=None, **metadata):
        if self.dominio == 'huellas':
            if self.rec is None:
                import reconocer_huella as rec
                _, self.cfg = rec.load_experiment(self.raiz / 'huellas/modelo_kmeans.joblib',
                                                  self.raiz / 'huellas/umbrales_rechazo.json')
                if not np.array_equal(self.nombres, rec.feat.names_features()):
                    raise ValueError('El PCA de huellas no corresponde al extractor vigente')
                self.rec = rec
            vector, _, _, _ = self.rec.extract_vector(Path(imagen), self.cfg)
        if vector is None:
            return {'sin_vector': True, 'motivo': 'No hay características válidas para proyectar.', **metadata}
        return {'coordenadas': self.transformar(vector)[0, :3].tolist(),
                'imagen': imagen, **metadata}
