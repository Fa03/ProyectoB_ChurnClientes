from pathlib import Path
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
        print(f"Descripcion del dataset \n{self.__DF_data.info()}") #Resumen de las variables del dataset
        print(f"Primeros 5 registros del data set: \n{self.__DF_data.head(5)}") #Obtencion de las primeras 5 lineas del dataset
        print(f"Estadística básica del dataset:\n{self.__DF_data.describe()}") #Estadistica basica del dataset

    # -------------------------------------------------------------------------------------------------------------------#

    # 2. Método con el que podremos limpiar textos ya que tenemos varias variables STR
    #Dicho metodo nos permite evaluar las variables que contengan texto y les realiza una limpieza de formato
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
    # Este metodo revisa las variables y verifica si alguna fila esta vacia, el metodo nos genera alerta si se encuentra algun registro faltante
    def gestionar_datos_nulos(self):
        print("--- Verificación de Integridad de Datos (Nulos) ---")

        #Evaluamos a nivel global si el dataset tiene CUALQUIER dato nulo
        if self.__DF_data.isnull().values.any():
            print("Alerta: Se detectaron datos faltantes en el dataset.\n")

        #Reportamos el conteo detallado por columna
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
    #Este metodo nos permite determinar si existen valores duplicados.
    #Realiza 2 acciones:
    #1- Si encuentra valores nulos los elimina
    #2- Sino encuentra valores nulos indicara que no se requiere eliminar duplicados
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

    #5 Reemplazar Yes/No de la variable objetivo Churn por 1-0
    #Dicho metodo nos permite identificar la variable objetivo "Churn"y aplicacion un ciclo si encuentra que sea "Si" lo cambio por "1", pero si encuentra que sea "no" lo cambio por "0"
    # En dicho metodo se utiliza pandas.map() para realizar el mapeo de Si/No
    def transformar_target_churn(self):
        print("--- Transformando Variable Objetivo (Churn) ---")

        #Antes de realizar el cambio de valores de la variable verifica si existe en el DF
        if 'Churn' not in self.__DF_data.columns:
            print("La columna 'Churn' no se encuentra en el dataset o ya fue transformada.")
            return

        #Aplicamos el mapeo de 'Yes'/'No' a 1/0
        # Usamos .map() que es la forma más rápida y estándar en Pandas
        self.__DF_data['Churn'] = self.__DF_data['Churn'].map({'Yes': 1, 'No': 0})

        print("✅ Transformación de 'Churn' completada con éxito (Yes -> 1, No -> 0).")

    # -------------------------------------------------------------------------------------------------------------------#
    #6 Generar OneHotEncoder ya que existen varias variables que se deben de pasar de STR a int
    #La variables que se requieren pasar a OneHotEncoder son: gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    #Se implementa OneHotEncoder debido a que las variables mencionadas anteriormente tiene datos como "YES" o "NO"
    def aplicar_one_hot_encoding(self, columnas_especificas=None, columnas_a_excluir=None):
        print("--- Aplicando One-Hot Encoding Optimizado ---")

        # Antes de aplicar el OneHotEncoder se debe definir columnas a excluir por defecto que no se les aplicara la codificacion.
        if columnas_a_excluir is None:
            columnas_a_excluir = ['customerID','MonthlyCharges', 'TotalCharges', 'Churn','tenure']
        else:
            columnas_a_excluir = list(set(columnas_a_excluir + ['customerID', 'TotalCharges']))

        # Seleccionar todas las categóricas
        if columnas_especificas is None:
            columnas_especificas = self.__DF_data.select_dtypes(
                include=['object', 'category', 'string']
            ).columns.tolist()

        # 4. Excluir la variable objetivo y las columnas problemáticas
        columnas_especificas = [col for col in columnas_especificas if col not in columnas_a_excluir]

        # 5. Filtrar columnas con alta cardinalidad
        columnas_especificas = [
            col for col in columnas_especificas
            if self.__DF_data[col].nunique() < 20
        ]

        if not columnas_especificas:
            print("No se encontraron variables categóricas para codificar.")
            return

        print(f"Variables seleccionadas para codificación: {columnas_especificas}")

        # Proceso de One-Hot Encoding optimizado para no duplicar variables binarias
        # Usamos drop='if_binary' para mantener 1 sola columna en variables de Sí/No
        encoder = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
        encoded_results = encoder.fit_transform(self.__DF_data[columnas_especificas])
        nuevos_nombres = encoder.get_feature_names_out(columnas_especificas)

        df_encoded = pd.DataFrame(
            encoded_results,
            columns=nuevos_nombres,
            index=self.__DF_data.index
        )

        #Unir resultados y eliminar las columnas originales codificadas
        self.__DF_data = self.__DF_data.join(df_encoded)
        self.__DF_data.drop(columns=columnas_especificas, inplace=True)

        # Actualizar dimensiones internas
        self.__num_filas = self.__DF_data.shape[0]
        self.__num_columnas = self.__DF_data.shape[1]

        print("One-Hot Encoding completado con éxito.")
        print(
            f"Nuevo tamaño del dataset: {self.__num_filas} filas x {self.__num_columnas} columnas (Estructura compacta).\n")
        print("\n" + "=" * 100 + "\n")
        print("One-Hot Encoding")
        print("\n" + "=" * 100 + "\n")

        # 8. Configuración de Pandas y muestra
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print("--- Primeros 5 registros del Dataset Actualizado One-Hot Encoding ---")
        print(self.__DF_data.head(5))

        return self.__DF_data


    # -------------------------------------------------------------------------------------------------------------------#
    #7 Calculo de correlaciones
    #El metodo obtener_correlaciones_fuertes me permite hacer correlaciones con respecto la variable Churn me ordena de mayor a menor.

    def obtener_correlaciones_fuertes(self, umbral=0.1):
        print(f"--- Calculando Correlaciones Fuertes con Churn (Ordenadas de Mayor a Menor) ---")

        #$Asegurar que Churn exista y sea numérica
        if 'Churn' not in self.__DF_data.columns:
            print("❌ Error: La columna 'Churn' no existe. Asegúrate de haber mapeado 'Yes'/'No' a 1/0 primero.")
            return

        # En caso de que Churn siga detectada como objeto, forzar numérico temporalmente
        if self.__DF_data['Churn'].dtype == 'object':
            self.__DF_data['Churn'] = pd.to_numeric(self.__DF_data['Churn'], errors='coerce')

        # Selecciona únicamente las columnas numéricas (Correlación de Pearson requiere números)
        df_numerico = self.__DF_data.select_dtypes(include=['number'])

        # Calcular la matriz de correlación con respecto a Churn
        # .corr() calcula la relación entre todas; luego extraemos solo la serie de 'Churn'
        correlaciones = df_numerico.corr()['Churn'].drop('Churn')  # Eliminamos la correlación de Churn consigo misma

        # Filtrar por el umbral absoluto (mantiene las que superan el umbral en magnitud)
        correlaciones_filtradas = correlaciones[correlaciones.abs() >= umbral]

        # Ordenar estrictamente de mayor a menor (Valores reales de Pearson)
        # Al quitar el .abs() y poner ascending=False, las positivas altas van arriba y las negativas abajo
        correlaciones_ordenadas = correlaciones_filtradas.sort_values(ascending=False)

        if correlaciones_ordenadas.empty:
            print(f"No se encontraron variables con correlación mayor o igual al umbral {umbral}.")
            return

        # Mostrar los resultados formateados en pantalla
        print(f"\nVariables más correlacionadas con el Churn (Abandono):")
        print(f"{'Variable':<45} | {'Coeficiente de Correlación':<25}")
        print("-" * 75)
        for col, val in correlaciones_ordenadas.items():
            print(f"{col:<45} | {val:>24.4f}")

        print("-" * 75)
        print(f"Total de características con fuerte relación: {len(correlaciones_ordenadas)}\n")

        return correlaciones_ordenadas

    # -------------------------------------------------------------------------------------------------------------------#
    #8 Visualizar correlaciones
    #El metodo nos muestra las correlaciones fuertes con respecto a la variable "Churn"
    #Devuelve correlacion fuerte positiva="Roja" y correlacion fuerte negativa="Azul"
    def grafico_correlaciones_churn(self, target='Churn', top_n=10):

        print(f"--- Graficando Top {top_n} correlaciones con {target} ---")

        # 1. Validar que la variable objetivo exista en el DataFrame actual
        if target not in self.__DF_data.columns:
            print(f"La columna '{target}' no existe en el dataset.")
            print("Asegúrate de haber ejecutado 'transformar_target_churn()' primero.")
            return

        # Calcular la correlación de Pearson (solo numéricas)
        corr = self.__DF_data.corr(numeric_only=True)[target]

        if target in corr.index:
            corr_ordenado = corr.abs().sort_values(ascending=False).drop(target)
        else:
            corr_ordenado = corr.abs().sort_values(ascending=False)

        top_indices = corr_ordenado.head(top_n).index

        #Recuperar los valores con su signo real (+ o -) y ordenarlos para el gráfico de barras
        top_signed = corr[top_indices].sort_values(ascending=True)

        #Configuración estética del gráfico
        plt.figure(figsize=(10, 6))

        # Asignar color: Rojo para correlación positiva (fomenta el Churn) y Azul para negativa (retención)
        colores = np.where(top_signed.values >= 0, '#e74c3c', '#3498db')

        # Dibujar gráfico de barras horizontales
        barras = plt.barh(top_signed.index, top_signed.values, color=colores, edgecolor='black', alpha=0.8)

        # Añadir los valores numéricos al final de cada barra para facilitar la lectura
        for barra in barras:
            ancho = barra.get_width()
            posicion_x = ancho + 0.01 if ancho >= 0 else ancho - 0.05
            ha_alineacion = 'left' if ancho >= 0 else 'right'
            plt.text(posicion_x, barra.get_y() + barra.get_height() / 2,
                     f'{ancho:.2f}',
                     va='center', ha=ha_alineacion, fontsize=10, fontweight='bold')

        # Línea vertical en el cero para dar equilibrio visual
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

        # Títulos y etiquetas estilizadas
        plt.title(f'Top {top_n} Características con Mayor Impacto en el Churn', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Coeficiente de Correlación (Pearson)', fontsize=11, labelpad=10)
        plt.ylabel('Variables del Dataset', fontsize=11)

        # Ajustar márgenes para que no se corten las etiquetas de texto larguísimas del One-Hot
        plt.tight_layout()
        plt.show()


    # -------------------------------------------------------------------------------------------------------------------#
    #8 Generar dataset reducido con la variables fuertemente correlacionadas
    #El siguiente metodo nos permite generar 1 dataset limpio con las variables fuertemente correlacionadas y variables predefinadas para el modelo IA

    def generar_dataset_reducido(self, target='Churn', umbral=0.25):
        from pathlib import Path  # Nos aseguramos de tener Path disponible
        import pandas as pd

        print(f"--- Generando dataset reducido basado en correlación (Umbral: {umbral}) ---")

        # Validar que la variable objetivo exista con el nombre correcto
        if target not in self.__DF_data.columns:
            print(f"❌ La columna '{target}' no existe.")
            print("Asegúrate de haber ejecutado 'transformar_target_churn()' primero.")
            return None

        # Calcular correlaciones (solo numéricas)
        corr = self.__DF_data.corr(numeric_only=True)[target]

        # Seleccionar variables con correlación fuerte (usando el umbral absoluto de 0.25)
        variables_fuertes = corr[
            (corr >= umbral) | (corr <= -umbral)
            ].index.tolist()

        # Forzar la inclusión del target y de todas las columnas requeridas
        columnas_obligatorias = [
            target, 'gender_Male', 'TotalCharges', 'MonthlyCharges',
            'InternetService_DSL', 'InternetService_No'
        ]

        for col in columnas_obligatorias:
            if col in self.__DF_data.columns and col not in variables_fuertes:
                variables_fuertes.append(col)

        #NUEVA REGLA: Excluir columnas no deseadas explícitamente 🌟
        columnas_a_quitar = ['PaymentMethod_Electronic check']
        variables_fuertes = [col for col in variables_fuertes if col not in columnas_a_quitar]

        #ORDENAR COLUMNAS A TU GUSTO
        orden_deseado_inicio = [
            'Churn', 'gender_Male', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'InternetService_No', 'InternetService_DSL'
        ]

        # Filtramos solo las que realmente existan en 'variables_fuertes' por seguridad
        columnas_inicio = [col for col in orden_deseado_inicio if col in variables_fuertes]

        # El resto de las variables seleccionadas van al final (conservando el orden que tenían)
        columnas_resto = [col for col in variables_fuertes if col not in columnas_inicio]

        # Combinamos ambas listas para el orden final definitivo
        columnas_ordenadas_final = columnas_inicio + columnas_resto

        print(f"✅ Variables seleccionadas finales ordenadas ({len(columnas_ordenadas_final)}):")
        print(columnas_ordenadas_final)

        # Crear el nuevo dataset reducido con el nuevo orden de columnas
        self.__DF_reducido = self.__DF_data[columnas_ordenadas_final].copy()

        print("\n✅ Dataset reducido creado correctamente")
        print(f"Nuevo tamaño: {self.__DF_reducido.shape[0]} filas x {self.__DF_reducido.shape[1]} columnas")

        # Definir ruta y crear directorios si no existen
        ruta = Path('src/eda/processed/telco_churn_reducido.csv')
        ruta.parent.mkdir(parents=True, exist_ok=True)

        # Guardar el DataFrame a formato CSV
        self.__DF_reducido.to_csv(ruta, index=False)

        print(f'✅ El dataset reducido se ha guardado en: {ruta.resolve()}')
        print("\n--- Primeros 5 registros del Dataset Reducido (Columnas Ordenadas) ---")

        # Configuración de visualización en consola para el print
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(self.__DF_reducido.head(5))
        print("=" * 60 + "\n")

        return self.__DF_reducido
    # -------------------------------------------------------------------------------------------------------------------#

    # -------------------------------------------------------------------------------------------------------------------#

    # 12. Método para realizar la limpieza general del dataset
    #El metodo ejecutar_eda nos permite ejecutar todos los metodos que se crearon anteriormente con cierto formato para interpretarlo.
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
        print("5 Trasnformar target Churn")
        self.transformar_target_churn()
        print("\n")
        print("=" * 60)
        print("#6 Aplicar Onehotencoding al dataset")
        self.aplicar_one_hot_encoding()
        print("\n")
        print("=" * 60)
        print("#7 Calcular correlaciones")
        self.obtener_correlaciones_fuertes()
        print("\n")
        print("=" * 60)
        print("#8 Generar Grafico de correlaciones")
        self.grafico_correlaciones_churn()
        print("\n")
        print("=" * 60)
        print("#9 Generar dataset con las principales correlaciones")
        self.generar_dataset_reducido()
        print("\n")


# =============================================================================
# Ejecucion de la clase por medio del main
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
        print("ERROR: El archivo no se encuentra en la ruta calculada.")
        print(f"Por favor, verifica que el archivo exista en: {ruta_real_archivo}")