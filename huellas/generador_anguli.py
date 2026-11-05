#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptador de Anguli por consola, reutilizable desde una futura GUI.

GeneradorAnguli(...).generar('1') crea una impresión desde su maestra.
GeneradorAnguli(...).generar('intruso') crea otra maestra con nueva semilla
y solicita una impresión. No determina autorización ni identidad predicha.

Usa subprocess con argumentos separados (sin shell), cwd de Anguli para sus
recursos y carpeta nueva por ejecución. No modifica el Anguli.conf original.
La disponibilidad real de la consola depende de la distribución de Anguli.
Referencia: https://dsl.cds.iisc.ac.in/projects/Anguli/userguide.html
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import time
from uuid import uuid4
import xml.etree.ElementTree as ET

EXTENSIONS = {'.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff', '.pgm'}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def number(value, name, minimum=0, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f'{name}: valor numérico inválido')
    result = float(value)
    if not result >= minimum or (maximum is not None and result > maximum):
        raise ValueError(f'{name}: valor fuera de rango')
    return result


class GeneradorAnguli:
    def __init__(self, configuracion, ejecutable=None, salida=None):
        self.config_path = Path(configuracion).resolve()
        self.home = self.config_path.parent
        self.config = json.loads(self.config_path.read_text(encoding='utf-8-sig'))
        if self.config.get('version') != 1:
            raise ValueError('Formato de config_anguli.json incompatible')
        self.executable_setting = ejecutable or os.environ.get('ANGULI_EXE') or self.config.get('ejecutable')
        self.output = self.path(salida or self.config.get('salida', 'Generaciones_anguli'))
        params = self.config.get('impresiones', {})
        self.noise = self.interval(params.get('ruido', [0, 2]), 'ruido', 8)
        self.scratches = self.interval(params.get('rasgunos', [0, 2]), 'rasgunos', 100)
        self.translation = number(params.get('traslacion', 10), 'traslacion', maximum=100)
        self.rotation = number(params.get('rotacion', 10), 'rotacion', maximum=180)
        self.timeout = number(self.config.get('timeout_segundos', 120), 'timeout', minimum=1, maximum=600)

    def path(self, value):
        p = Path(os.path.expandvars(str(value))).expanduser()
        return p.resolve() if p.is_absolute() else (self.home/p).resolve()

    @staticmethod
    def interval(value, name, maximum):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f'{name}: se espera [mínimo, máximo]')
        lo, hi = [number(v, name, maximum=maximum) for v in value]
        if lo > hi or int(lo) != lo or int(hi) != hi:
            raise ValueError(f'{name}: se esperan enteros con mínimo <= máximo')
        return [int(lo), int(hi)]

    def executable(self, must_exist=True):
        value = self.executable_setting
        if not value:
            value = shutil.which('Anguli.exe') or shutil.which('Anguli')
        if not value:
            raise ValueError('Indicá --anguli "C:/ruta/Anguli.exe" o completá ejecutable en config_anguli.json')
        found = shutil.which(str(value))
        path = Path(found).resolve() if found else self.path(value)
        if must_exist and not path.is_file():
            raise FileNotFoundError(f'No existe el ejecutable de Anguli: {path}')
        return path

    def disponibles(self):
        result = {}
        for key in ['1', '2', '3']:
            entry = self.config.get('identidades', {}).get(key, {})
            master = self.path(entry['maestra']) if entry.get('maestra') else None
            result[key] = {'nombre': entry.get('nombre', f'Identidad {key}'),
                           'disponible': bool(master and master.is_file()),
                           'maestra': str(master) if master else None}
        template = self.path(self.config.get('configuracion_intruso', 'Anguli_base.conf'))
        result['intruso'] = {'nombre': 'Intruso', 'disponible': template.is_file(),
                            'configuracion': str(template)}
        return result

    def _new_intruder_config(self, destination, output):
        template = self.path(self.config.get('configuracion_intruso', 'Anguli_base.conf'))
        tree = ET.parse(template)
        fingerprint = tree.getroot().find('FingerprintGen')
        if fingerprint is None:
            raise ValueError('La configuración no contiene FingerprintGen')
        def set_values(tag, **values):
            node = fingerprint.find(tag)
            if node is None:
                node = ET.SubElement(fingerprint, tag)
            for key, value in values.items():
                node.set(key, str(value))
        original = fingerprint.find('customSeed')
        original_seed = int(original.get('seedValue', '1')) if original is not None else 1
        forbidden = set(self.config.get('semillas_autorizadas_conocidas', [1])) | {original_seed}
        seed = secrets.randbelow(2**31-2)+1
        while seed in forbidden:
            seed = secrets.randbelow(2**31-2)+1
        set_values('customSeed', value=1, seedValue=seed)
        set_values('NumFingerprints', value=1)
        set_values('ImprPerFinger', value=1)
        set_values('NumThreads', value=1)
        set_values('StartFinger', value=1)
        set_values('SaveMetaInfo', value=1)
        set_values('OutDir', value=output.as_posix())
        set_values('Translation', value=f'{self.translation:g}')
        set_values('Rotation', value=f'{self.rotation:g}')
        set_values('NoiseLevel', low=self.noise[0], high=self.noise[1])
        set_values('NumScratches', low=self.scratches[0], high=self.scratches[1])
        tree.write(destination, encoding='utf-8', xml_declaration=True)
        return seed

    def preparar(self, identidad, verificar_ejecutable=True):
        key = str(identidad).lower()
        if key not in {'1', '2', '3', 'intruso'}:
            raise ValueError('Identidad debe ser 1, 2, 3 o intruso')
        availability = self.disponibles()[key]
        if not availability['disponible']:
            raise FileNotFoundError(f'Falta la maestra de Identidad {key}: '
                                    f'{availability.get("maestra", availability.get("configuracion"))}')
        executable = self.executable(verificar_ejecutable)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid4().hex[:8]
        run = self.output/stamp
        run.mkdir(parents=True, exist_ok=False)
        output = run/'salida_anguli'
        output.mkdir()
        details = {'estado': 'preparado', 'seleccion_generador': key,
                   'ejecutable': str(executable), 'directorio_trabajo': str(executable.parent),
                   'carpeta_ejecucion': str(run), 'carpeta_salida_anguli': str(output),
                   'timeout_segundos': self.timeout}
        if key != 'intruso':
            master = Path(availability['maestra'])
            input_folder = run/'entrada_maestra'
            input_folder.mkdir()
            # Una sola maestra por llamada: Anguli no generará todas las identidades.
            staged = input_folder/('1'+master.suffix.lower())
            shutil.copy2(master, staged)
            command = [str(executable), '-impr', '-indir', str(input_folder),
                       '-ni', '1', '-outdir', str(output), '-numT', '1',
                       '-noise', f'{self.noise[0]} {self.noise[1]}',
                       '-scratch', f'{self.scratches[0]} {self.scratches[1]}',
                       '-trans', f'{self.translation:g}', '-rot', f'{self.rotation:g}']
            details.update(modo_anguli='impresion_de_maestra', maestra=str(master),
                           maestra_sha256=hashlib.sha256(master.read_bytes()).hexdigest())
        else:
            generated_config = run/'Anguli_intruso.conf'
            seed = self._new_intruder_config(generated_config, output)
            command = [str(executable), '-cfg', str(generated_config), '-num', '1',
                       '-ni', '1', '-numT', '1', '-outdir', str(output)]
            details.update(modo_anguli='nueva_huella_e_impresion', semilla_intruso=seed,
                           configuracion_anguli=str(generated_config))
        details['comando'] = command
        write_json(run/'generacion.json', details)
        return details

    def ejecutar_preparado(self, details):
        run = Path(details['carpeta_ejecucion'])
        logfile = run/'anguli.log'
        started = time.monotonic()
        try:
            with logfile.open('wb') as log:
                completed = subprocess.run(details['comando'], cwd=details['directorio_trabajo'],
                                           stdout=log, stderr=subprocess.STDOUT,
                                           timeout=details['timeout_segundos'], shell=False,
                                           check=False)
            details['codigo_salida'] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f'Anguli terminó con código {completed.returncode}. Revisá {logfile}')
            output = Path(details['carpeta_salida_anguli'])
            images = sorted(p for p in output.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            impressions = [p for p in images if any(part.lower().startswith('impression_') for part in p.relative_to(output).parts[:-1])]
            if len(impressions) != 1:
                raise RuntimeError(f'Se esperaba una impresión nueva y se encontraron {len(impressions)}. '
                                   f'Revisá {logfile} y {output}. No se usará la maestra como sustituto.')
            image = impressions[0]
            if image.stat().st_size == 0:
                raise RuntimeError('Anguli produjo un archivo vacío')
            details.update(estado='generado', imagen_generada=str(image),
                           imagen_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
            return details
        except Exception as exc:
            details.update(estado='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            details['duracion_segundos'] = time.monotonic()-started
            details['registro_anguli'] = str(logfile)
            write_json(run/'generacion.json', details)

    def generar(self, identidad):
        return self.ejecutar_preparado(self.preparar(identidad))
