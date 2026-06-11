"""
Telco Customer Churn - Descarga y Procesamiento
================================================
Descarga el dataset desde Kaggle usando API token y lo procesa.
"""

import os
import io
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

#RAW_DIR=Nombre de la carpeta que almacena los datos crudos o sin procesar
#PROCESSED_DIR=Nombre de los archivos que almacena los datos procesados
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


# --------------------------------------------------------------
#  CLASE IMPORTADORA de Kaggle.
#Recordar los siguientes pasos antes de ejecutar dicha clase.
#1- Instalar Kaggle desde el CMD del equipo pip install kaggle o en la terminal de pychart
#2- Crear una carpeta en la raíz del usuario llamada: .kaggle (C:\Users\"Nombre del usuario"\.kaggle
#3- Crear un archivo json llamado: kaggle.json
#4- El archivo desde de llevar los siguiente: {"username":"tu_usuario","key":"tu_clave_aqui"}
#5- El username se obtiene en setting-your profile y API key se obtiene setting--your api token, new token
# --------------------------------------------------------------



class KaggleDatasetImporter:
    """
    Importa datasets desde Kaggle usando API token.
    Compatible con kaggle < 1.6 (KaggleApiExtended) y >= 1.6 (nuevo CLI).
    """
    #____________________________________________________________
    #Parametros del constructor
    #username:Nombre del usuario en la cuenta de kaggle
    #api_key: Api key de kaggle
    #download_dir: Nombre de la carpeta a crear
    #_____________________________________________________________

    def __init__(self, username: str = None, api_key: str = None, download_dir: str = RAW_DIR):
        self.download_dir = Path(download_dir).resolve()  # Ruta absoluta para evitar fallos con subprocess
        self.download_dir.mkdir(parents=True, exist_ok=True) #Crea la carpeta donde se va a guardar los archivos descargados

         #Si se pasan credenciales manuales, se setean en el entorno
        if username and api_key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = api_key


        #try que valida que kaggle este importado anteriormente o que emite mensaje que debe de instalarlo
        try:
            import kaggle
        except ImportError:
            raise ImportError(
                "La librería 'kaggle' no está instalada.\n"
                "Instálala con: python -m pip install kaggle"
            )

        # Detecta versión del paquete kaggle
        try:
            from kaggle.api.kaggle_api_extended import KaggleApiExtended
            self._api = KaggleApiExtended()
            self._api.authenticate()
            self._legacy = True  # kaggle < 1.6
        except Exception:
            # Capturamos Exception general por si la autenticación falla por entorno vacío
            self._api = None
            self._legacy = False  # kaggle >= 1.6

    #Metodo from_json= permite leer credenciales de kaggle en el archivo kaggle.json que habia creado anteriormente
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

    #Metodo download=permite descargar el dataset
    def download(self, dataset_ref: str, force: bool = False) -> Path:
        """Descarga y descomprime el dataset en download_dir."""

        # Evitar descargar si ya hay archivos y no se fuerza
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
            # check=True lanzará un CalledProcessError automáticamente si falla
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
#  PROCESAMIENTO DEL DATASET
# --------------------------------------------------------------
def process_telco_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y prepara el dataset Telco Customer Churn."""
    print("\n" + "=" * 60)
    print("PROCESANDO DATASET: Telco Customer Churn")
    print("=" * 60)

    df = df.copy()

    # 1. TotalCharges string -> float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nulls = df["TotalCharges"].isna().sum()
    if nulls > 0:
        print(f"  {nulls} filas con TotalCharges vacío -> eliminadas")
        df.dropna(subset=["TotalCharges"], inplace=True)

    # 2. SeniorCitizen: 0/1 -> 'No'/'Yes'
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 3. Churn: 'Yes'/'No' -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 4. customerID como índice (Validando que exista)
    if "customerID" in df.columns:
        df.set_index("customerID", inplace=True)

    print(f"Dataset limpio: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # Distribución con manejo de excepciones por si el mapeo falló
    try:
        churn_counts = df["Churn"].value_counts()
        total = len(df)
        print(f"\nDistribución de Churn:")
        print(f"  No churned : {churn_counts.get(0, 0):,} ({churn_counts.get(0, 0) / total * 100:.1f}%)")
        print(f"  Churned    : {churn_counts.get(1, 0):,} ({churn_counts.get(1, 0) / total * 100:.1f}%)")
    except Exception as e:
        print(f"No se pudo desplegar la distribución de Churn: {e}")

    return df


def save_processed(df: pd.DataFrame, output_dir: str = PROCESSED_DIR) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "telco_churn_clean.csv"
    df.to_csv(output_path)
    print(f"Dataset guardado en: {output_path}")
    return str(output_path)


# --------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------
def main():
    print("=" * 60)
    print("DESCARGA DE DATASET: Telco Customer Churn")
    print(f"Fuente : {DATASET_URL}")
    print("=" * 60)

    # Inicializar importador (busca json o variables de entorno automáticamente)
    try:
        importer = KaggleDatasetImporter.from_json()
    except FileNotFoundError:
        # Si no encuentra el JSON, intenta usar directo (requiere variables de entorno)
        print("Aviso: kaggle.json no encontrado, usando configuración por defecto de Kaggle.")
        importer = KaggleDatasetImporter()

    # Carga inteligente (Tu clase ya sabe si descargar o no gracias al refactor)
    df_raw = importer.load(
        dataset_ref=DATASET_REF,
        file_name=DATASET_FILE,
    )

    print(f"\nDataset raw: {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas")

    # Procesamiento y Guardado
    df_clean = process_telco_churn(df_raw)
    output_path = save_processed(df_clean)

    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)
    print(f"  Procesado: {output_path}")


if __name__ == "__main__":
    main()