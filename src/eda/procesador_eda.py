import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os  # Librería para manejar rutas de archivos.


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

    # 3. Método en el cual obtendremos aquellos datos nulos.
    def datos_nulos(self):
        print("Este dataset tiene datos nulos en:")
        print(self.__DF_data.isnull().sum())

    # 4.Eliminar los datos nulos.
    def eliminar_datos_nulos(self):
        self.__DF_data.dropna(inplace=True)
        print('Los datos nulos han sido eliminados')

        # 5. Imputar los datos nulos (utilizar la media para los numéricos y la moda para las categóricas).
    def imputar_datos_nulos(self):
        # 1. Calculamos los nulos por variable
        df_var = self.__DF_data.isnull().sum()
        porcentaje_eliminacion = 0.1  # 10%

        # 2. Filtramos variables con menos del 10% de nulos
        df_var = df_var[df_var < porcentaje_eliminacion * len(self.__DF_data)]
        lista_variables_OK = df_var.index

        print(f"Variables seleccionadas para imputación (menos del 10% nulos): {list(lista_variables_OK)}")

        # 3. Hacemos la imputación real sobre esas variables en el DataFrame original
        for columna in lista_variables_OK:
            # Si la columna tiene nulos, procedemos a llenarlos
            if self.__DF_data[columna].isnull().any():
                if self.__DF_data[columna].dtype in ['float64', 'int64']:
                    # Imputamos con la media para numéricos
                    media = self.__DF_data[columna].mean()
                    self.__DF_data[columna].fillna(media, inplace=True)
                    print(f" -> Variable numérica '{columna}' imputada con la media: {media:.2f}")
                else:
                    # Imputamos con la moda para categóricos (texto)
                    moda = self.__DF_data[columna].mode()
                    if not moda.empty:
                        self.__DF_data[columna].fillna(moda[0], inplace=True)
                        print(f" -> Variable categórica '{columna}' imputada con la moda: '{moda[0]}'")


            print("\nPrimeros 5 registros del DataFrame después de filtrar e imputar:")
            print(self.__DF_data[lista_variables_OK].head(5))

    # -------------------------------------------------------------------------------------------------------------------#

    # 6. Método en el cual podremos obtener los valores duplicados.
    def datos_duplicados(self):
        print('Este dataset tiene los siguientes datos duplicados: ')
        print(self.__DF_data.duplicated().sum())

    # 7.Eliminar los datos duplicados.
    def eliminar_datos_duplicados(self):
        self.__DF_data.drop_duplicates(inplace=True)
        print('Los datos duplicados del dataset han sido eliminados correctamente')

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
        print("Ejecución procesamiento EDA \n")
        print("#1 Informacion del dataset raw, datos estadistico")
        self.informacion_data()
        print("\n")
        print("#2 Limpieza texto de las variables STR")
        self.limpiar_texto()
        print("\n")
        print("#3 Verificacion datos nulos")
        self.datos_nulos()
        print("\n")
        print("#4 Eliminar datos nulos ")
        self.eliminar_datos_nulos()
        print("\n")
        print("#5 Imputar datos nulos")
        self.imputar_datos_nulos()
        print("\n")
        print("#6 Datos duplicados")
        self.datos_duplicados()
        print("\n")
        print("#7 Eliminar datos duplicados")
        self.datos_duplicados()
        print("\n")
        print("#8 Generar dataset limpio")
        self.csv_limpio()
        print("\n")
        print("---------------------------------------------------------------------------------------------------\n")
        print("Matriz correlación, histograma y boxplot")
        print("\n")
        self.eda_matriz_correlacion()
        self.eda_histogramas()
        self.generar_boxplots()


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
            "data", "raw", "data", "raw",
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
        print("Por favor, verifica el bloque de código de la ruta.")