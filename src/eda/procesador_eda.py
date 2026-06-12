import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os  # Librería para manejar rutas de archivos.
from sklearn.preprocessing import OneHotEncoder


class ProcesadorEDA:  # Creamos la clase ProcesadorEDA la cual nos ayudará a realizar un análisis EDA.
    def __init__(self, DF_data=pd.DataFrame()):  # Realizamos el constructor.
        self.__DF_data = DF_data  # Atributo privado que almacena el DataFrame.
        self.__num_filas = DF_data.shape[0]  # Atributos privados que almacenan el número de filas y columnas.
        self.__num_columnas = DF_data.shape[1]

    # Creamos los propertys (getters) para acceder a los atributos privados.
    @property
    def DF_data(self):
        return self.__DF_data

    @property
    def num_filas(self):
        return self.__num_filas

    @property
    def num_columnas(self):
        return self.__num_columnas

    # Creamos los setters para que podamos modificar los atributos privados si es necesario.
    @DF_data.setter
    def DF_data(self, DF_data):
        self.__DF_data = DF_data
        self.__num_filas = DF_data.shape[0]
        self.__num_columnas = DF_data.shape[1]

    @num_filas.setter
    def num_filas(self, num_filas):
        self.__num_filas = num_filas

    @num_columnas.setter
    def num_columnas(self, num_columnas):
        self.__num_columnas = num_columnas

    # -------------------------------------------------------------------------------------------------------------------#

    # 1. Método en el cual obtendremos información general del Dataset proporcionado.
    def informacion_data(self):
        print("Información general del dataset")
        print(f"Descripcion del dataset \n{self.__DF_data.info()}")
        print(f"Primeros 5 registros del data set: \n{self.__DF_data.head(5)}")
        print(f"Estadística básica del dataset:\n{self.__DF_data.describe()}")

    # -------------------------------------------------------------------------------------------------------------------#

    # 2. Método con el que podremos limpiar textos ya que tenemos varias variables STR
    def limpiar_texto(self):
    # Añadimos 'str' y str de forma explícita para evitar el Pandas4Warning
        columnas_texto = self.__DF_data.select_dtypes(
            include=['object', 'category', 'str', str]
        ).columns  # Selecciona las columnas de tipo texto.

        print("Iniciando la limpieza de textos...")
        for columna in columnas_texto:
                # Mostramos en consola qué variable se está limpiando justo ahora
            print(f" -> Limpiando texto en la variable: '{columna}'")

            self.__DF_data[columna] = (self.__DF_data[columna].astype(str).apply(
                lambda x: x.encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
            )  # Asegura que los datos sean de tipo string limpios.

        print("¡Todas las columnas categóricas han sido limpiadas con éxito!\n")

    # -------------------------------------------------------------------------------------------------------------------#

    # 3. Método para verificar, reportar y limpiar datos nulos
    def gestionar_datos_nulos(self):
        print("--- Verificación de Integridad de Datos (Nulos) ---")

        # 1. Evaluamos a nivel global si el dataset tiene CUALQUIER dato nulo
        if self.__DF_data.isnull().values.any():
            print("⚠️ Alerta: Se detectaron datos faltantes en el dataset.\n")

            # 2. Reportamos el conteo detallado por columna
            print("Conteo de nulos por variable:")
            print(self.__DF_data.isnull().sum())

            # 3. Ejecutamos la desinfección eliminando las filas con nulos
            print("\nProcediendo con la desinfección...")
            self.__DF_data.dropna(inplace=True)
            print("Los datos nulos han sido eliminados correctamente.")
        else:
            # Si no hay nulos, saltamos la limpieza e informamos al usuario
            print("¡Todo está bien! El dataset no contiene registros nulos. No se requiere limpieza.")

    # -------------------------------------------------------------------------------------------------------------------#

    #4 Gestionar datos duplicados
    def gestionar_datos_duplicados(self, eliminar=False):
        duplicados = self.__DF_data.duplicated().sum()
        print('Este dataset tiene los siguientes datos duplicados:')
        print(duplicados)

        if eliminar:
            self.__DF_data.drop_duplicates(inplace=True)
            print('Los datos duplicados han sido eliminados correctamente')
        else:
            print('No se eliminaron los duplicados')

    # -------------------------------------------------------------------------------------------------------------------#
    #5 Generar OneHotEncoder ya que existen varias variables que se deben de pasar de STR a int

    # 7. Método para aplicar One-Hot Encoding a variables categóricas
    def aplicar_one_hot_encoding(self, columnas_especificas=None):

        # Aplica One-Hot Encoding a las columnas categóricas indicadas o
        # detecta automáticamente todas las variables de texto si no se pasa ninguna.

        print("--- Aplicando One-Hot Encoding ---")

        # 1. Detectar columnas categóricas si no se especifican
        if columnas_especificas is None:
            columnas_especificas = self.__DF_data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()

        # 2. Evitar usar customerID
        if 'customerID' in columnas_especificas:
            columnas_especificas.remove('customerID')

        # 3. Validar que haya columnas
        if not columnas_especificas:
            print("No se encontraron variables categóricas para codificar.")
            return

        print(f"Variables seleccionadas para codificación: {columnas_especificas}")

        # 4. Inicializamos el codificador
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # 5. Transformar datos
        encoded_results = encoder.fit_transform(
            self.__DF_data[columnas_especificas]
        )

        # 6. Obtener nombres de nuevas columnas
        nuevos_nombres = encoder.get_feature_names_out(columnas_especificas)

        # 7. Crear DataFrame con resultados codificados
        df_encoded = pd.DataFrame(
            encoded_results,
            columns=nuevos_nombres,
            index=self.__DF_data.index
        )

        # 8. Unir y eliminar columnas originales
        self.__DF_data = self.__DF_data.join(df_encoded)
        self.__DF_data.drop(columns=columnas_especificas, inplace=True)

        # 9. Actualizar dimensiones
        self.__num_filas = self.__DF_data.shape[0]
        self.__num_columnas = self.__DF_data.shape[1]

        print("✅ One-Hot Encoding completado con éxito.")
        print(f"Nuevo tamaño del dataset: {self.__num_filas} filas x {self.__num_columnas} columnas.\n")


    # -------------------------------------------------------------------------------------------------------------------#

    # 8. Método para poder guardar nuestro csv limpio y guardarlo en la carpeta processed.
    def csv_limpio(self, ruta_guardar_csv='data/raw/data,processed/telco_churn_clean.csv'):
        carpeta = os.path.dirname(ruta_guardar_csv)  # Obtenemos la carpeta del path proporcionado.
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)  # Creamos la carpeta si no existe.
        self.__DF_data.to_csv(ruta_guardar_csv, index=False)  # Guardamos el DataFrame como un archivo CSV.
        print(f'El Dataset limpio se ha guardado en la ruta: {ruta_guardar_csv}')

    # -------------------------------------------------------------------------------------------------------------------#

    # 9 Matriz de correlación
    def eda_matriz_correlacion(self):
        # Seleccionar únicamente columnas numéricas
        df_numerico = self.__DF_data.select_dtypes(include=['number'])

        if df_numerico.empty:
            print("No hay columnas numéricas para generar la matriz de correlación.")
            return None

        matriz_correlacion = df_numerico.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matriz de Correlación')
        plt.show()

        return matriz_correlacion

    # -------------------------------------------------------------------------------------------------------------------#

    # 10 Histograma para cada columna numérica
    def eda_histogramas(self):
        # Genera un histograma para la variable "Number"
        columna = "Number"
        if columna in self.__DF_data.select_dtypes(include=['number']).columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.__DF_data[columna], kde=True, bins=10)
            plt.title("Histograma variable Número de camisa")
            plt.xlabel("Número de camisa")
            plt.ylabel('Cantidad de jugadores por número de camisa')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"La columna '{columna}' no existe o no es numérica.")

    # -------------------------------------------------------------------------------------------------------------------#

    # 11 Boxplot
    def generar_boxplots(self):
        # Genera un boxplot para la columna numérica age.
        columna = "Age"
        if columna in self.__DF_data.select_dtypes(include=['number']).columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=self.__DF_data[columna])
            plt.title("Box plot de edad de jugadores")
            plt.xlabel("Edad de jugadores")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"La columna '{columna}' no existe o no es numérica.")

    # -------------------------------------------------------------------------------------------------------------------#

    # 12. Método para realizar la limpieza general del dataset
    def ejecutar_eda(self):
        print("=" * 120)
        print("=" * 120)
        print("\n")
        print("Ejecución procesamiento EDA \n")
        print("\n")
        print("=" * 60)
        print("\n")
        print("#1 Informacion del dataset ""WA_Fn-UseC_-Telco-Customer-Churn.csv"", datos estadistico")
        print("\n")
        self.informacion_data()
        print("\n")
        print("=" * 60)
        print("#2 Limpieza texto de las variables STR")
        self.limpiar_texto()
        print("\n")
        print("=" * 60)
        print("#3 Verificacion datos nulos")
        self.gestionar_datos_nulos()
        print("\n")
        print("=" * 60)
        print("#4 Validacion datos duplicados")
        self.gestionar_datos_duplicados()
        print("\n")
        print("=" * 60)
        print("#5 Aplicar onehotencoding al dataset")
        self.aplicar_one_hot_encoding()
        print("=" * 60)
        print("\n")
        print("Matriz correlación, histograma y boxplot")
        print("\n")
        print("=" * 60)
        self.eda_matriz_correlacion()
        self.eda_histogramas()
        self.generar_boxplots()
        print("#8 Generar dataset limpio")
        self.csv_limpio()
        print("\n")


# =============================================================================
# INSTANCIACIÓN CON TU ARCHIVO REAL
# =============================================================================

if __name__ == "__main__":

    # 1. Obtenemos la ruta absoluta de la carpeta donde está este script (src/eda)
    directorio_actual = os.path.dirname(os.path.abspath(__file__))


    ruta_real_archivo = os.path.normpath(
        os.path.join(
            directorio_actual,
            "..", "..",
            "data", "raw",
            "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    )

    print(f"Buscando archivo en la ruta calculada:\n--> {ruta_real_archivo}\n")

    # 3. Validamos si el archivo existe e iniciamos el EDA
    if os.path.exists(ruta_real_archivo):
        print("¡Archivo encontrado con éxito! Cargando datos...")
        df_clientes = pd.read_csv(ruta_real_archivo)

        # Instanciamos la clase y corremos el proceso
        analisis_churn = ProcesadorEDA(DF_data=df_clientes)
        analisis_churn.ejecutar_eda()
    else:
        print("❌ ERROR: El archivo no se encuentra en la ruta calculada.")
        print(f"Por favor, verifica que el archivo exista en: {ruta_real_archivo}")