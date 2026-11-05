"""Reutiliza la API pública; clasifica una sola vez por muestra."""
from pathlib import Path


class AdaptadorHuellas:
    def __init__(self, raiz):
        import demo_anguli as api
        self.api = api
        self.raiz = Path(raiz)

    def generar(self, identidad, anguli=None):
        if identidad not in ('1', '2', '3', 'intruso'):
            raise ValueError('Selección de generador inválida')
        r = self.api.generar_y_reconocer(
            identidad, self.raiz / 'huellas/config_anguli.json',
            anguli=anguli or None, modo='identificar',
            salida=self.raiz / 'salidas_integracion/huellas')
        return self.preparar(r)

    @staticmethod
    def preparar(r):
        if r.get('estado') != 'ok' or r.get('autorizada') not in (0, 1):
            raise ValueError('La ejecución no devolvió una clasificación válida')
        r = dict(r)
        carpeta = Path(r['archivo_resultado']).parent
        r['vistas'] = {'Transformada': r['imagen_a_procesar']}
        for nombre, archivo in [('Normalizada', 'grises.png'), ('Binaria', 'binarias.png'),
                                ('Esqueleto', 'esqueletos.png')]:
            ruta = carpeta / 'procesamiento' / archivo
            if ruta.is_file():
                r['vistas'][nombre] = str(ruta)
        return r
