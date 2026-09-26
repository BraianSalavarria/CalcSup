# CalcSup — Visualizador Avanzado de Superficies Paramétricas

Un visualizador interactivo 3D para el cálculo multivariable desarrollado en Python con **CustomTkinter**, **Matplotlib** y **SymPy**. Diseñado con una interfaz moderna y fluida estilo GeoGebra para facilitar el análisis gráfico y matemático de superficies paramétricas, vectores tangentes, vectores normales, planos tangentes y cálculo de área mediante integrales dobles.

---

## 🌟 Características Principales

* **Renderizado 3D Interactivo Estilo GeoGebra:**
  * Ejes vectoriales graduados ($X$, $Y$, $Z$) con números y puntos de graduación (*ticks*).
  * Eje $Z$ negativo en línea punteada.
  * Cuadrícula base en el plano $Z = 0$.
  * Zoom dinámico mediante rueda del ratón y atajos de teclado (`Ctrl +` / `Ctrl -`).
* **Soporte Matemático Avanzado:**
  * Compatible con funciones trigonométricas (`sen`, `cos`, `tan`, `arcsen`, etc.), exponenciales (`e^u`, `exp`), logaritmos y constantes matemáticas (`pi`, `e`).
  * Autocompletado inteligente de funciones en tiempo real.
* **Múltiples Superficies en Escena:**
  * Añade y gestiona múltiples superficies paramétricas simultáneamente.
  * Menú desplegable para editar, actualizar o eliminar superficies específicas.
  * Asignación de mapas de color (*colormaps*): Viridis, Plasma, Inferno, Coolwarm, etc.
* **Geometría Diferencial (Vectores y Plano Tangente):**
  * Evaluación puntual en parámetros $(u_0, v_0)$.
  * Cálculo simbólico y numérico de derivadas parciales ($\mathbf{r}_u$, $\mathbf{r}_v$) y del vector normal ($\mathbf{N} = \mathbf{r}_u \times \mathbf{r}_v$).
  * Representación en formato **Cartesiano** `(x; y; z)` o **Vectorial** `xi + yj + zk`.
  * Visualización 3D del plano tangente en el punto evaluado.
* **Cálculo de Área de Superficie:**
  * Integración doble numérica sobre la magnitud del producto vectorial $\Vert{} \mathbf{r}_u \times \mathbf{r}_v \Vert{}$ empleando SciPy.
* **Log de Estado y Errores Dedicado:**
  * Panel de retroalimentación en tiempo real para capturar errores de sintaxis o cálculo sin colapsar la aplicación.
* **Gestión de Archivos y Proyectos:**
  * Exportación de la gráfica 3D en alta resolución (`PNG`, `JPG`, `PDF`).
  * Guardado y carga completa de proyectos en formato `.json` (mantiene estados de conmutadores, ecuaciones, resolución y configuraciones).

---

## 🚀 Requisitos e Instalación

### Prerrequisitos

Asegúrate de tener **Python 3.8** o superior instalado en tu sistema.

### Instalación de Dependencias

Clona este repositorio e instala las librerías requeridas ejecutando:

```bash
git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
cd TU_REPOSITORIO
pip install -r requirements.txt

Si prefieres instalar las dependencias manualmente:

```bash
pip install customtkinter matplotlib numpy sympy scipy
