"""
Telco Customer Churn - Descarga y Procesamiento
================================================
Descarga el dataset desde Kaggle usando API token y lo procesa en memoria.
"""

import os
import json
import subprocess
import sys
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------
#  CONFIGURACION
#  Datos de configuracion del dataset a descargar utilizando plataforma de kaggle
# --------------------------------------------------------------
#DATASET_REF=Referencia del dataset donde blastchar es el nombre del autor y telco-customer-churn es el nombre del repositorio
#DATASET_URL = es la URL donde esta guardado el archivo
#DATASET_FILE = Debemos de indicarle el nombre del archivo vamos importar con su extension

DATASET_REF = "blastchar/telco-customer-churn"
DATASET_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
DATASET_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# RAW_DIR = Nombre de la carpeta que almacena los datos crudos o sin procesar
RAW_DIR = "raw"


# --------------------------------------------------------------
#  CLASE IMPORTADORA de Kaggle.
#Recordar los siguientes pasos antes de ejecutar dicha clase.
#1- Instalar Kaggle desde el CMD del equipo pip install kaggle o en la terminal de pychart
#2- Crear una carpeta en la raíz del usuario llamada: .kaggle (C:\Users\"Nombre del usuario"\.kaggle
#3- Crear un archivo json llamado: kaggle.json
#4- El archivo desde de llevar los siguiente datos: {"username":"tu_usuario","key":"tu_clave_aqui"}
#5- El username se obtiene en setting-your profile y API key se obtiene setting--your api token, new token
# --------------------------------------------------------------

class KaggleDatasetImporter:
    """
    Importa datasets desde Kaggle usando API token.
    Compatible con kaggle < 1.6 (KaggleApiExtended) y >= 1.6 (nuevo CLI).
    """

    def __init__(self, username: str = None, api_key: str = None, download_dir: str = RAW_DIR):
        self.download_dir = Path(download_dir).resolve()
        self.download_dir.mkdir(parents=True, exist_ok=True)

        if username and api_key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = api_key

        try:
            import kaggle
        except ImportError:
            raise ImportError(
                "La librería 'kaggle' no está instalada.\n"
                "Instálala con: python -m pip install kaggle"
            )

        try:
            from kaggle.api.kaggle_api_extended import KaggleApiExtended
            self._api = KaggleApiExtended()
            self._api.authenticate()
            self._legacy = True
        except Exception:
            self._api = None
            self._legacy = False

    @classmethod
    def from_json(cls, json_path: str = "~/.kaggle/kaggle.json", **kwargs):
        """Crea instancia leyendo credenciales desde kaggle.json."""
        path = Path(json_path).expanduser()

        if not path.exists():
            win_path = Path.home() / ".kaggle" / "kaggle.json"
            if win_path.exists():
                path = win_path
            else:
                raise FileNotFoundError(
                    f"No se encontró el archivo de credenciales en {path} ni en {win_path}\n"
                    "Crea tu token en https://www.kaggle.com/settings"
                )
        with open(path) as f:
            creds = json.load(f)
        print(f"Credenciales cargadas desde: {path}")
        return cls(username=creds["username"], api_key=creds["key"], **kwargs)

    def download(self, dataset_ref: str, force: bool = False) -> Path:
        """Descarga y descomprime el dataset en download_dir."""
        if not force and any(self.download_dir.iterdir()):
            print(f"La carpeta destino ya contiene archivos. Omitiendo descarga.")
            return self.download_dir

        print(f"Descargando: {dataset_ref} ...")

        if self._legacy:
            self._api.dataset_download_files(
                dataset=dataset_ref,
                path=str(self.download_dir),
                unzip=True,
                quiet=False,
            )
        else:
            result = subprocess.run(
                [
                    sys.executable, "-m", "kaggle",
                    "datasets", "download",
                    "-d", dataset_ref,
                    "-p", str(self.download_dir),
                    "--unzip",
                ],
                capture_output=True,
                text=True,
                check=True
            )

        print(f"Descarga completada en: {self.download_dir}/")
        return self.download_dir

    def load(self, dataset_ref: str, file_name: str = None, **read_kwargs) -> pd.DataFrame:
        """Descarga el dataset (si no existe) y lo carga como DataFrame."""
        data_path = self.download(dataset_ref)
        return self._read_files(data_path, file_name, **read_kwargs)

    def _read_files(self, data_path: Path, file_name: str = None, **kwargs) -> pd.DataFrame:
        if file_name:
            target = data_path / file_name
            if not target.exists():
                raise FileNotFoundError(f"'{file_name}' no encontrado en {data_path}")
            return self._read_single(target, **kwargs)

        candidates = list(data_path.glob("*.csv")) + list(data_path.glob("*.xlsx"))
        if not candidates:
            raise FileNotFoundError(f"No se encontraron archivos válidos en {data_path}")

        if len(candidates) == 1:
            return self._read_single(candidates[0], **kwargs)

        print(f"\nMúltiples archivos encontrados. Cargando el primero por defecto: {candidates[0].name}")
        return self._read_single(candidates[0], **kwargs)

    @staticmethod
    def _read_single(file_path: Path, **kwargs) -> pd.DataFrame:
        ext = file_path.suffix.lower()
        print(f"Cargando: {file_path.name}")
        if ext == ".csv":
            return pd.read_csv(file_path, **kwargs)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Formato no soportado: {ext}")

# --------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------
def main():
    print("=" * 60)
    print("DESCARGA DE DATASET: Telco Customer Churn")
    print(f"Fuente : {DATASET_URL}")
    print("=" * 60)

    try:
        importer = KaggleDatasetImporter.from_json()
    except FileNotFoundError:
        print("Aviso: kaggle.json no encontrado, usando configuración por defecto de Kaggle.")
        importer = KaggleDatasetImporter()

    df_raw = importer.load(
        dataset_ref=DATASET_REF,
        file_name=DATASET_FILE,
    )

    print(f"\nDataset raw: {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas")



    # SE REMOVIÓ LA EXPORTACIÓN DEL CSV
    print("\nEl procesamiento se completó correctamente en memoria.")

    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()