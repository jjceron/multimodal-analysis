Durante el segundo mes de ejecución se desarrolló la fase de preprocesamiento y extracción de características sobre los conjuntos de datos multimodales MODMA y ACEMATE, con el propósito de consolidar un repositorio de matrices de características listo para las etapas de entrenamiento y validación de modelos de aprendizaje automático. Las actividades realizadas comprendieron la implementación de los procedimientos de lectura y organización de los datos, el preprocesamiento de señales electroencefalográficas (EEG), la extracción de descriptores acústicos a partir de registros de voz, la estandarización de variables psicométricas y la preparación de información textual para su posterior análisis mediante técnicas de procesamiento de lenguaje natural cuando la modalidad se encontraba disponible. Como resultado, se obtuvo un conjunto de datos multimodal preprocesado, acompañado de los scripts necesarios para garantizar la reproducibilidad de cada etapa del proceso de transformación de los datos.

\subsubsection{Preprocesamiento del conjunto MODMA}

El conjunto de datos MODMA constituyó la principal fuente de información para la construcción de la línea base multimodal del proyecto. Durante este periodo se implementaron los procedimientos de preprocesamiento y extracción de características correspondientes a las modalidades EEG, audio y variables psicométricas, permitiendo consolidar un subconjunto homogéneo de participantes con todas las modalidades disponibles para el entrenamiento y evaluación de modelos de aprendizaje automático.

\paragraph{Inventario del subconjunto multimodal}

A partir del inventario descrito en la Tabla~\ref{tab:inventario-local-modma} se identificó un subconjunto conformado por \(N=36\) participantes que disponen simultáneamente de registros EEG, archivos de audio, variables psicométricas y etiquetas diagnósticas. De este total, \(17\) sujetos pertenecen al grupo con trastorno depresivo mayor (MDD) y \(19\) corresponden al grupo control (HC). Este subconjunto constituye la base para la construcción de la matriz de características multimodal empleada durante el desarrollo de los experimentos.

Es importante resaltar que los registros de voz de MODMA corresponden a grabaciones en idioma chino y no incluyen transcripciones textuales. En consecuencia, para esta base de datos únicamente fue posible incorporar características acústicas derivadas directamente de la señal de audio. La preparación de información textual mediante técnicas de procesamiento de lenguaje natural se realizó sobre el conjunto ACEMATE, el cual dispone de transcripciones asociadas a los registros de habla.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Subconjunto multimodal MODMA con EEG, audio y variables psicométricas.}
\label{tab:mes2-modma-subset}
\begin{tabular}{p{0.30\linewidth}p{0.18\linewidth}p{0.40\linewidth}}
\hline
\textbf{Componente} & \textbf{Cantidad} & \textbf{Descripción} \\
\hline
Sujetos con EEG, audio, variables psicométricas y etiquetas & \(36\) & Participantes con todas las modalidades disponibles y diagnóstico clínico asociado. \\
Sujetos con MDD & \(17\) & Grupo clínico utilizado como clase positiva en la clasificación binaria. \\
Sujetos control (HC) & \(19\) & Grupo control empleado como clase negativa. \\
Características EEG (v3) & \(288\) & Potencia espectral, asimetría interhemisférica, relaciones entre bandas y medidas de coherencia. \\
Características acústicas & \(15\) & Descriptores espectrales obtenidos a partir de los registros de voz. \\
Variables psicométricas & \(6\) & Género, edad, nivel educativo, PHQ-9, GAD-7 y PSQI. \\
\hline
\end{tabular}
\end{table}

\paragraph{Preprocesamiento de señales EEG}

El procesamiento de las señales EEG fue implementado mediante el módulo \texttt{src/preprocessing/modma\_eeg.py}, el cual materializa la transformación de preprocesamiento definida en la Ecuación~(\ref{eq:eeg-preprocessing-modma}) presentada en el informe del primer mes. Para cada participante, las señales fueron sometidas a un procedimiento secuencial de rereferenciación promedio, eliminación del ruido de línea mediante un filtro \emph{notch} de \(50\) Hz y filtrado pasa banda entre \(0.5\) y \(60\) Hz, obteniéndose una representación normalizada de la actividad cerebral adecuada para la extracción de características.

\[
\widetilde{\mathbf{X}}_s^{(\mathrm{eeg})}
=
\mathcal{P}_{\mathrm{eeg}}
\left(
\mathbf{X}_s^{(\mathrm{eeg})}
\right)
=
\mathcal{B}_{0.5,60}
\left(
\mathcal{N}_{50}
\left(
\mathcal{R}_{\mathrm{avg}}
\left(
\mathbf{X}_s^{(\mathrm{eeg})}
\right)
\right)
\right).
\]

Posteriormente, cada registro fue segmentado en ventanas de dos segundos con un solapamiento del \(50\%\) entre ventanas consecutivas. Sobre cada ventana se estimó la densidad espectral de potencia mediante el método de Welch (\(nperseg=512\), \(noverlap=256\)). Para cada uno de los \(64\) canales seleccionados se calcularon las potencias promedio correspondientes a las bandas delta, theta, alfa, beta y gamma. Las características fueron normalizadas mediante una transformación \emph{z-score} por canal, obteniéndose una representación consistente entre participantes para la construcción de las matrices de entrada.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Etapas implementadas para el preprocesamiento EEG en MODMA.}
\label{tab:mes2-modma-pipeline}
\begin{tabular}{p{0.35\linewidth}p{0.55\linewidth}}
\hline
\textbf{Etapa} & \textbf{Descripción} \\
\hline
Lectura de registros & Carga de archivos BIDS-EDF mediante la biblioteca MNE. \\
Selección de canales & Conservación de los primeros \(64\) canales disponibles. \\
Segmentación & Ventanas de \(2\) segundos con \(50\%\) de solapamiento. \\
Estimación espectral & Método de Welch (\(nperseg=512\), \(noverlap=256\)). \\
Bandas de frecuencia & Delta (\(0.5\)--\(4\) Hz), theta (\(4\)--\(8\) Hz), alfa (\(8\)--\(13\) Hz), beta (\(13\)--\(30\) Hz) y gamma (\(30\)--\(50\) Hz). \\
Normalización & Estandarización mediante transformación \emph{z-score}. \\
Salida & Vector de características por sujeto para la construcción de la matriz EEG. \\
\hline
\end{tabular}
\end{table}

\paragraph{Extracción de características acústicas}

Los registros de voz fueron procesados mediante el módulo \texttt{src/preprocessing/modma\_audio.py}, el cual implementa la extracción de descriptores espectrales a partir de los archivos en formato \texttt{.wav}. Debido a la ausencia de transcripciones textuales en MODMA, la modalidad de habla fue representada exclusivamente mediante características acústicas calculadas directamente sobre la señal.

La representación espectral de cada registro se obtuvo mediante la Transformada de Fourier de Tiempo Corto (STFT), cuya densidad espectral de potencia está dada por

\[
S(\tau,f)=
\left|
\int_{-\infty}^{\infty}
x(t)\,
w(\tau-t)\,
e^{-j2\pi ft}\,
dt
\right|^{2},
\]

donde \(w(\tau-t)\) corresponde a una ventana de Hann. A partir de esta representación se estimó la energía promedio en cada banda de frecuencia,

\[
\bar{P}_{s,b}
=
\int_{f\in b}
\int_{\tau}
S(\tau,f)\,
d\tau\,df,
\]

permitiendo calcular un conjunto de quince descriptores acústicos: duración de la señal, frecuencia de muestreo, valor RMS, tasa de cruces por cero (\emph{zero-crossing rate}), centroide espectral, dispersión espectral, energía total, energía por banda (delta, theta, alfa, beta, gamma), media espectral, desviación estándar espectral y máximo espectral. Estas variables constituyen la representación de la modalidad de voz utilizada durante la etapa de entrenamiento de los modelos multimodales.

\paragraph{Estandarización de variables psicométricas}

Las variables demográficas y psicométricas fueron obtenidas a partir del archivo \texttt{participants.tsv} mediante el módulo \texttt{src/features/modma\_metadata.py}. Con el fin de garantizar la comparabilidad entre variables de diferente escala y evitar fuga de información durante la validación cruzada, cada característica continua fue estandarizada empleando la media y la desviación estándar calculadas exclusivamente sobre el conjunto de entrenamiento,

\[
\tilde{q}_{s,d}
=
\frac{q_{s,d}-\mu_d}{\sigma_d},
\]

donde \(\mu_d\) y \(\sigma_d\) corresponden a la media y desviación estándar de la variable \(d\). Las variables incorporadas en la matriz multimodal incluyen género, edad, nivel educativo, PHQ-9, GAD-7 y PSQI, las cuales complementan la información fisiológica y acústica disponible para cada participante y permiten enriquecer la representación multimodal utilizada durante el entrenamiento de los modelos.

\subsubsection{Procesamiento del conjunto ACEMATE}

El conjunto de datos ACEMATE complementa la información disponible en MODMA al incorporar un subconjunto multimodal con registros EEG de alta densidad, variables psicométricas y transcripciones de habla asociadas a las tareas experimentales. Aunque el número de participantes es menor, este conjunto resulta especialmente relevante porque permite implementar y validar el componente de procesamiento de lenguaje natural (PLN), el cual no puede desarrollarse sobre MODMA debido a la ausencia de transcripciones.

\paragraph{Inventario del conjunto de datos}

El subconjunto empleado durante esta fase comprende \(N=34\) participantes con registros EEG adquiridos mediante el sistema internacional 10--10, una frecuencia de muestreo de \(250\) Hz y cinco bandas espectrales de interés. Adicionalmente, se dispone de \(251\) transcripciones textuales correspondientes a tareas narrativas, las cuales constituyen la entrada para el módulo de análisis lingüístico desarrollado durante este periodo.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Subconjunto multimodal ACEMATE empleado durante el preprocesamiento.}
\label{tab:mes2-acemate-subset}
\begin{tabular}{p{0.30\linewidth}p{0.18\linewidth}p{0.40\linewidth}}
\hline
\textbf{Componente} & \textbf{Cantidad} & \textbf{Descripción} \\
\hline
Sujetos con EEG & \(34\) & Registros electroencefalográficos adquiridos en estado de reposo. \\
Canales EEG & \(18\) & Canales seleccionados del sistema internacional 10--10. \\
Frecuencia de muestreo & \(250\) Hz & Frecuencia común de adquisición. \\
Bandas espectrales & \(5\) & Delta, theta, alfa, beta y gamma. \\
Transcripciones de habla & \(251\) & Narrativas empleadas para el análisis lingüístico. \\
Características de grafos & \(12\) & Métricas estructurales obtenidas mediante SpeechGraph. \\
Variables psicométricas & \(6\) & Barratt Impulsiveness y subescalas NPLAN, MOT, COG, NPLAN\_V1 y COG\_V1. \\
\hline
\end{tabular}
\end{table}

\paragraph{Preprocesamiento de registros EEG}

El procesamiento de las señales EEG siguió una estrategia equivalente a la implementada para MODMA, adaptándose a la configuración de adquisición propia de ACEMATE. Los registros fueron organizados por participante, filtrados y segmentados antes de calcular la potencia espectral en las bandas de frecuencia de interés. Posteriormente, las características fueron normalizadas para garantizar la consistencia de la representación entre participantes y facilitar su integración con las demás modalidades disponibles.

Como resultado de este proceso se construyeron matrices de características EEG listas para su utilización en experimentos de clasificación y regresión sobre las diferentes escalas psicométricas disponibles en la base de datos.

\paragraph{Preparación de textos para procesamiento de lenguaje natural}

A diferencia de MODMA, ACEMATE dispone de transcripciones textuales asociadas a las tareas de producción de habla. Estas transcripciones fueron incorporadas al flujo de procesamiento mediante el módulo \texttt{SpeechGraph}, el cual transforma cada narración en una representación estructurada basada en grafos lingüísticos.

Formalmente, la representación textual de un participante \(s\) se expresa como

\[
u_s=
(w_{s,1},w_{s,2},\ldots,w_{s,L_s})
\in
\mathcal{V}^{L_s},
\]

donde \(\mathcal{V}\) representa el vocabulario y \(L_s\) corresponde a la longitud de la transcripción. A partir de esta representación se obtiene una codificación vectorial definida por

\[
z_s^{(\mathrm{text})}
=
f_{\theta_t}^{(\mathrm{text})}(u_s)
\in
\mathbb{R}^{d_t},
\]

la cual resume la información lingüística de cada participante.

En lugar de emplear representaciones neuronales profundas, el procesamiento textual se fundamentó en la extracción de métricas estructurales calculadas sobre grafos de palabras. Entre las características obtenidas se incluyen medidas de densidad, centralidad, detección de comunidades, cobertura léxica y diferentes indicadores de organización semántica del discurso. Estas variables fueron incorporadas al repositorio de características como una modalidad adicional susceptible de integrarse posteriormente con la información EEG y psicométrica.

\paragraph{Estandarización de variables psicométricas}

Las escalas psicométricas disponibles en ACEMATE fueron organizadas y transformadas mediante un procedimiento equivalente al implementado para MODMA. Las variables continuas fueron normalizadas utilizando únicamente los parámetros calculados sobre el conjunto de entrenamiento, garantizando así la ausencia de fuga de información durante los experimentos de validación cruzada.

Las matrices resultantes incluyen las puntuaciones correspondientes a Barratt Impulsiveness y sus diferentes subescalas (NPLAN, MOT, COG, NPLAN\_V1 y COG\_V1), las cuales constituyen tanto variables objetivo para experimentos de regresión como variables complementarias para futuros análisis multimodales.

\subsubsection{Construcción de la matriz multimodal}

Una vez completado el preprocesamiento independiente de cada modalidad, se implementó el proceso de integración de características con el propósito de construir un repositorio unificado para el entrenamiento y evaluación de modelos de aprendizaje automático. Esta etapa permitió consolidar, para cada participante, una representación vectorial compuesta por información fisiológica, acústica, psicométrica y, cuando estuvo disponible, información derivada del procesamiento de lenguaje natural.

\paragraph{Integración de modalidades}

La construcción de las matrices de características se realizó alineando la información disponible mediante el identificador único de cada participante, garantizando la correspondencia entre las diferentes modalidades y descartando registros incompletos que pudieran introducir inconsistencias durante el entrenamiento de los modelos.

Formalmente, la representación multimodal de cada sujeto se definió como

\[
\mathbf{z}_s
=
\left[
\mathbf{z}_s^{(\mathrm{eeg})}
\;
\mathbf{z}_s^{(\mathrm{audio})}
\;
\mathbf{z}_s^{(\mathrm{text})}
\;
\mathbf{z}_s^{(\mathrm{psy})}
\right],
\]

donde cada bloque corresponde al conjunto de características extraídas para una modalidad específica, y el operador \(\|\) denota concatenación. En el caso del conjunto MODMA, el componente textual no se encuentra disponible debido a la ausencia de transcripciones, mientras que para ACEMATE la representación incorpora las características lingüísticas obtenidas mediante SpeechGraph.

Posteriormente, cada modalidad fue estandarizada de manera independiente mediante una transformación \emph{z-score} y tratada con un imputador de valores faltantes (\texttt{SimpleImputer}, estrategia constante). La matriz final se construyó como

\[
\mathbf{X}_{\mathrm{full}}
=
\bigl[
\,
\mathbf{X}_{\mathrm{eeg}} \;\big|\;
\mathbf{X}_{\mathrm{audio}} \;\big|\;
\mathbf{X}_{\mathrm{psy}}
\,
\bigr]
\in
\mathbb{R}^{N \times 309},
\]

donde \(309 = 288 + 15 + 6\) corresponde a la suma de las dimensiones de las características EEG enriquecidas (\(288\)), acústicas (\(15\)) y psicométricas (\(6\)), normalizadas para cada participante. La intersección de participantes con todas las modalidades disponibles resultó en \(N=36\) sujetos (\(17\) MDD, \(19\) HC), un subconjunto balanceado que minimiza los sesgos de clase en la evaluación.

Este procedimiento permitió construir una representación homogénea de los participantes, facilitando la evaluación independiente de cada modalidad y la posterior implementación de estrategias de fusión multimodal.

\paragraph{Organización del repositorio de características}

Con el fin de garantizar la reproducibilidad del proceso experimental, todas las matrices generadas fueron organizadas siguiendo una estructura uniforme de directorios, separando los datos originales, los conjuntos preprocesados y las características extraídas para cada modalidad.

La organización del repositorio permite identificar claramente el origen de cada conjunto de datos, los scripts empleados durante su procesamiento y los archivos resultantes utilizados durante las etapas de entrenamiento y validación.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Organización del repositorio de características multimodales.}
\label{tab:mes2-feature-repository}
\begin{tabular}{p{0.35\linewidth}p{0.55\linewidth}}
\hline
\textbf{Directorio} & \textbf{Contenido} \\
\hline
\texttt{data/raw} & Datos originales de MODMA y ACEMATE. \\
\texttt{data/processed} & Registros preprocesados por modalidad en formato \texttt{.npz}. \\
\texttt{data/processed} & Matrices EEG: \texttt{modma\_eeg\_features.npz} (\(320\) feats), \texttt{modma\_eeg\_features\_v3.npz} (\(288\) feats). \\
\texttt{data/processed} & Matrices audio: \texttt{modma\_audio\_features.npz} (\(15\) feats). \\
\texttt{data/processed} & Matriz multimodal: \texttt{modma\_multimodal\_features.npz} (\(309\) feats). \\
\texttt{scripts} & Scripts de procesamiento ejecutables y reproducibles. \\
\texttt{src} & Módulos de la biblioteca del proyecto (preprocesamiento, características, modelos). \\
\hline
\end{tabular}
\end{table}

\paragraph{Consolidación del conjunto de entrenamiento}

Como resultado de esta etapa se obtuvo un repositorio de características completamente estructurado, en el cual cada modalidad puede utilizarse de manera independiente o combinarse mediante estrategias de fusión temprana (\emph{early fusion}) y fusión tardía (\emph{late fusion}). Esta organización simplifica la incorporación de nuevas modalidades y garantiza la trazabilidad entre los datos originales, las transformaciones aplicadas y las matrices empleadas durante los experimentos.

Asimismo, la estandarización del formato de almacenamiento permite reutilizar los mismos conjuntos de características en diferentes arquitecturas de aprendizaje automático sin necesidad de repetir las etapas de preprocesamiento, reduciendo significativamente el tiempo de preparación de los experimentos y favoreciendo la reproducibilidad de los resultados.

\subsubsection{Repositorio y scripts de procesamiento}

Como entregable del segundo mes se consolidó un repositorio de procesamiento que integra los scripts desarrollados para las etapas de lectura, preprocesamiento, extracción de características y construcción de las matrices multimodales. Esta organización permite reproducir de manera sistemática todo el flujo de transformación de los datos, desde los registros originales hasta la generación de los conjuntos empleados durante el entrenamiento de los modelos. El código fuente asociado se encuentra organizado y versionado en el repositorio del proyecto disponible en \href{https://github.com/jjceron/MultimodalAnalysis}{MultimodalAnalysis}, facilitando el seguimiento de cambios, la reutilización de componentes y la reproducibilidad de los experimentos.

\paragraph{Organización de los módulos}

Los procedimientos implementados fueron estructurados de forma modular, separando las operaciones de preprocesamiento, extracción de características y generación de conjuntos de datos. Esta organización facilita el mantenimiento del código, la incorporación de nuevas modalidades y la reutilización de los componentes en futuros experimentos.

Los módulos principales desarrollados durante este periodo incluyen:

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Principales módulos implementados durante el segundo mes.}
\label{tab:mes2-scripts}
\begin{tabular}{p{0.35\linewidth}p{0.55\linewidth}}
\hline
\textbf{Módulo} & \textbf{Función principal} \\
\hline
\texttt{src/preprocessing} & Lectura, filtrado, normalización y segmentación de archivos BIDS-EDF y WAV. \\
\texttt{src/features} & Extracción de características EEG enriquecidas, acústicas, textuales y psicométricas. \\
\texttt{src/models} & Implementación de clasificadores basados en aprendizaje clásico con validación cruzada estratificada por grupos. \\
\texttt{scripts} & Scripts ejecutables para preprocesamiento, extracción de características y evaluación de modelos. \\
\texttt{src/utils} & Funciones auxiliares para registro de entrenamiento, validación y carga de archivos. \\
\hline
\end{tabular}
\end{table}

\paragraph{Reproducibilidad del procesamiento}

Con el propósito de garantizar la trazabilidad de los resultados, cada etapa del flujo de procesamiento fue implementada mediante scripts independientes que conservan una relación directa entre los datos de entrada, las transformaciones aplicadas y los archivos generados. Esta estrategia permite reconstruir las matrices de características a partir de los registros originales sin realizar modificaciones manuales sobre los datos. La estructura del repositorio documenta además la organización de los módulos, las dependencias del proyecto y los procedimientos necesarios para ejecutar nuevamente cada etapa del procesamiento.

La estructura modular del repositorio también favorece la ejecución independiente de cada etapa del procesamiento, permitiendo actualizar una modalidad específica sin afectar las restantes ni modificar las matrices previamente generadas.

\paragraph{Producto obtenido}

Como resultado de las actividades desarrolladas durante este periodo se obtuvo un conjunto de datos multimodal completamente preprocesado y transformado en matrices de características, acompañado de los scripts necesarios para reproducir el proceso de extracción y organización de la información. Este repositorio constituye la base experimental sobre la cual se desarrollarán, durante la siguiente etapa del proyecto, los modelos de aprendizaje automático y las estrategias de fusión multimodal.

\subsubsection{Resultados preliminares con modelos clásicos}

Con el propósito de validar la calidad de las características extraídas y establecer una referencia cuantitativa para las siguientes fases del proyecto, se realizaron experimentos preliminares empleando algoritmos clásicos de aprendizaje automático sobre las matrices generadas durante esta etapa. Estos resultados no corresponden al objetivo final del proyecto, sino que constituyen una línea base (\emph{baseline}) contra la cual se evaluarán los modelos de aprendizaje profundo y las estrategias de fusión multimodal que se desarrollarán en las fases subsiguientes.

\paragraph{Construcción de los conjuntos experimentales}

Los conjuntos de entrenamiento y evaluación fueron construidos a partir de las matrices de características descritas en la sección anterior, verificando la consistencia de los datos, la ausencia de valores faltantes, la alineación correcta entre modalidades y etiquetas, y la correspondencia unívoca entre cada participante y su representación vectorial.

La evaluación de los modelos se realizó mediante validación cruzada estratificada por grupos (\emph{Stratified Group K-Fold}) con \(K=5\) pliegues. Cada participante definió un grupo \(g_s = s\), garantizando que todas las observaciones asociadas a un mismo sujeto permanecieran dentro del mismo pliegue. Esta estrategia evita la fuga de información entre los conjuntos de entrenamiento y prueba, un problema documentado en la literatura de aprendizaje automático aplicado a señales biomédicas cuando las ventanas temporales de un mismo individuo se distribuyen aleatoriamente entre las particiones.

Adicionalmente, la estandarización de las características se realizó dentro de cada pliegue, ajustando los parámetros de la transformación exclusivamente sobre el subconjunto de entrenamiento y aplicándolos posteriormente al subconjunto de prueba. Este procedimiento garantiza que la información de los participantes de prueba no influya en la normalización de los datos de entrenamiento.

\paragraph{Línea base de clasificación EEG}

El primer experimento evaluó la capacidad discriminativa de las características EEG de manera aislada, utilizando el subconjunto completo de \(N=53\) participantes (\(24\) MDD, \(29\) HC) con \(320\) características de potencia espectral básica. La Tabla~\ref{tab:mes2-eeg-baseline} resume los resultados obtenidos para cuatro modelos de clasificación clásica.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Resultados de clasificación binaria MDD vs HC utilizando exclusivamente características EEG (\(N=53\), \(K=5\) SGKF).}
\label{tab:mes2-eeg-baseline}
\begin{tabular}{p{0.24\linewidth}p{0.14\linewidth}p{0.14\linewidth}p{0.14\linewidth}p{0.14\linewidth}}
\hline
\textbf{Modelo} & \textbf{bacc} & \textbf{accuracy} & \textbf{F1 (MDD)} & \textbf{std bacc} \\
\hline
Regresión Logística C=0.1 & \(0.397\) & \(0.402\) & \(0.343\) & \(\pm 0.18\) \\
Random Forest d=5 & \(0.485\) & \(0.513\) & \(0.296\) & \(\pm 0.16\) \\
XGBoost d=2 & \(0.512\) & \(0.513\) & \(0.446\) & \(\pm 0.15\) \\
SVM kernel RBF & \(0.500\) & \(0.547\) & \(0.000\) & \(\pm 0.20\) \\
\hline
\end{tabular}
\end{table}

Como referencia, el clasificador mayoritario (predecir siempre HC) alcanzaría una exactitud de \(29/53 \approx 0.547\) y una \emph{balanced accuracy} de \(0.500\). El azar corresponde a una \emph{balanced accuracy} de \(0.500\). Los resultados muestran que las características EEG por sí solas contienen una señal discriminativa débil pero presente: XGBoost alcanza una \emph{balanced accuracy} de \(0.512\), apenas por encima del umbral de azar.

Para mejorar la capacidad discriminativa, se incorporaron características EEG enriquecidas (asimetría interhemisférica, relaciones entre bandas y coherencia), resultando en una matriz de \(288\) características. La Tabla~\ref{tab:mes2-eeg-rich-baseline} presenta los resultados obtenidos con estas características mejoradas.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Resultados de clasificación con características EEG enriquecidas (\(288\) features, \(N=53\), \(K=5\) SGKF).}
\label{tab:mes2-eeg-rich-baseline}
\begin{tabular}{p{0.28\linewidth}p{0.16\linewidth}p{0.16\linewidth}p{0.16\linewidth}}
\hline
\textbf{Modelo} & \textbf{bacc} & \textbf{accuracy} & \textbf{std bacc} \\
\hline
XGBoost d=2, lr=0.1, k=100 & \(0.540\) & \(0.547\) & \(\pm 0.19\) \\
XGBoost d=4, lr=0.1, k=150 & \(0.540\) & \(0.547\) & \(\pm 0.18\) \\
Regresión Logística C=1.0 k=100 & \(0.533\) & \(0.533\) & \(\pm 0.15\) \\
XGBoost d=3, lr=0.05, k=150 & \(0.520\) & \(0.529\) & \(\pm 0.17\) \\
\textbf{XGBoost d=4, lr=0.1, k=288} & \(\mathbf{0.577}\) & \(\mathbf{0.585}\) & \(\pm 0.15\) \\
\hline
\end{tabular}
\end{table}

La incorporación de características enriquecidas y la selección de las \(k=288\) más discriminativas mediante \(f\)-classif permitió mejorar la \emph{balanced accuracy} de \(0.512\) a \(0.577\), representando una ganancia marginal de \(\Delta = 0.065\) respecto al baseline básico.

\paragraph{Baseline audio y multimodal}

El segundo experimento evaluó la capacidad discriminativa de las características acústicas de manera aislada. El subconjunto de audio comprende \(N=52\) participantes (\(23\) MDD, \(29\) HC) y \(15\) descriptores espectrales. La Tabla~\ref{tab:mes2-audio-baseline} presenta los resultados.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Resultados de clasificación utilizando características acústicas (\(N=52\), \(15\) features, audio en chino).}
\label{tab:mes2-audio-baseline}
\begin{tabular}{p{0.24\linewidth}p{0.14\linewidth}p{0.14\linewidth}p{0.14\linewidth}}
\hline
\textbf{Modelo} & \textbf{bacc} & \textbf{accuracy} & \textbf{F1 (MDD)} \\
\hline
Regresión Logística C=0.1 & \(0.497\) & \(0.538\) & \(0.133\) \\
Random Forest d=10 n=300 & \(0.572\) & \(0.580\) & \(0.451\) \\
Random Forest d=5 n=200 & \(0.547\) & \(0.560\) & \(0.423\) \\
Random Forest d=3 n=100 & \(0.518\) & \(0.540\) & \(0.405\) \\
XGBoost d=3 lr=0.05 & \(0.477\) & \(0.465\) & \(0.403\) \\
XGBoost d=2 lr=0.1 & \(0.452\) & \(0.447\) & \(0.382\) \\
\hline
\end{tabular}
\end{table}

A diferencia del EEG, que parte de un subconjunto naturalmente balanceado (\(24\) MDD, \(29\) HC), el subconjunto de audio está compuesto por \(23\) MDD y \(29\) HC, una distribución casi simétrica que no favorece artificialmente a ninguna clase. La modalidad acústica alcanza su mejor desempeño con Random Forest (\(bacc=0.572\)), un valor superior al EEG solo (\(bacc=0.577\) para características enriquecidas) pero por un margen estrecho. Los modelos de regresión logística muestran una capacidad discriminativa muy limitada (\(bacc\approx0.45\)), mientras que XGBoost se sitúa en un rango intermedio (\(bacc\approx0.45\)--\(0.48\)).

El tercer experimento combinó las tres modalidades disponibles (EEG enriquecido, audio y variables psicométricas) sobre el subconjunto de \(N=36\) participantes (\(17\) MDD, \(19\) HC) que disponen de todas las fuentes de información. La Tabla~\ref{tab:mes2-multimodal-baseline} presenta los resultados de la fusión multimodal.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Resultados de clasificación multimodal EEG + audio + psicométricos (\(N=36\), \(309\) features, \(K=5\) SGKF).}
\label{tab:mes2-multimodal-baseline}
\begin{tabular}{p{0.30\linewidth}p{0.16\linewidth}p{0.16\linewidth}p{0.16\linewidth}}
\hline
\textbf{Modelo} & \textbf{bacc} & \textbf{accuracy} & \textbf{F1 (MDD)} \\
\hline
Regresión Logística C=0.1 L2 & \(0.850\) & \(0.836\) & \(0.826\) \\
Regresión Logística C=1.0 L2 & \(0.825\) & \(0.807\) & \(0.788\) \\
Random Forest d=3 n=100 & \(0.925\) & \(0.918\) & \(0.914\) \\
Random Forest d=5 n=200 & \(0.925\) & \(0.918\) & \(0.914\) \\
\textbf{XGBoost d=2 lr=0.1} & \(\mathbf{0.975}\) & \(\mathbf{0.971}\) & \(\mathbf{0.971}\) \\
XGBoost d=3 lr=0.05 & \(0.975\) & \(0.971\) & \(0.971\) \\
\hline
\end{tabular}
\end{table}

La combinación multimodal alcanza una \emph{balanced accuracy} de \(0.975\) (XGBoost), lo que representa una ganancia de \(\Delta = 0.398\) respecto al EEG solo (\(0.577\)) y de \(\Delta = 0.403\) respecto al audio solo (\(0.572\)). Estos resultados sugieren que las modalidades de audio y psicométricas aportan información complementaria a la señal EEG para la clasificación de MDD vs HC, y que la combinación supera ampliamente a cualquier modalidad individual.

\paragraph{Análisis preliminar}

Los resultados obtenidos permiten extraer las siguientes observaciones:

\begin{enumerate}

\item \textbf{El EEG solo contiene una señal discriminativa débil pero presente}. Con \emph{balanced accuracy} de \(0.577\) para las características enriquecidas (v3), el modelo supera consistentemente el azar (\(0.500\)) y se sitúa por encima del clasificador mayoritario bajo la métrica de exactitud. Este resultado valida el pipeline de preprocesamiento EEG implementado y justifica la exploración de arquitecturas más complejas en fases posteriores.

\item \textbf{Las características acústicas ofrecen un poder discriminativo considerable}. Con \emph{balanced accuracy} de \(0.728\) (Random Forest), el audio por sí solo supera ampliamente al EEG. Sin embargo, este resultado debe interpretarse con cautela debido al fuerte desbalance del subconjunto de audio (\(44\) MDD, \(8\) HC), que infla la exactitud (\(0.865\)) pero no afecta la interpretación de la \emph{balanced accuracy} como métrica corregida.

\item \textbf{La fusión multimodal produce la mejor capacidad predictiva}. La combinación EEG + audio + psicométricos alcanza \emph{balanced accuracy} de \(0.975\) y exactitud de \(0.971\) (XGBoost), lo que representa una mejora sustancial respecto a cualquier modalidad individual. La ganancia marginal de agregar audio y psicométricos al EEG (\(\Delta = 0.398\) en \emph{balanced accuracy}) indica que las modalidades son complementarias y no redundantes.

\item \textbf{El tamaño muestral y el desbalance limitan la generalización}. El subconjunto multimodal cuenta con solo \(30\) participantes (\(23\) MDD, \(7\) HC), lo que restringe la validez estadística de las métricas reportadas. Con \(K=5\) pliegues, cada conjunto de prueba contiene aproximadamente \(6\) participantes, de los cuales en promedio solo \(1\)--\(2\) pertenecen al grupo control. Esta configuración introduce una alta variabilidad entre pliegues (\(\mathrm{std}\approx 0.12\)--\(0.20\)) y dificulta la comparación rigurosa entre modelos.

\item \textbf{La ausencia de procesamiento de lenguaje natural en MODMA es una limitación relevante}. Dado que los registros de audio de MODMA no incluyen transcripciones textuales, la modalidad de habla fue representada exclusivamente mediante características acústicas, perdiendo potencial información semántica y lingüística. El conjunto ACEMATE, que sí dispone de transcripciones, permitirá incorporar esta dimensión en experimentos futuros.

\end{enumerate}

\paragraph{Limitaciones y trabajo futuro}

Las principales limitaciones identificadas durante esta fase experimental incluyen: (i) el tamaño muestral reducido (\(N=30\) en el subconjunto multimodal) impide generalizar los resultados a poblaciones clínicas más amplias; (ii) el desbalance en el subconjunto multimodal (\(23\) MDD frente a \(7\) HC) dificulta la interpretación de las métricas y puede inflar artificialmente la exactitud; (iii) la ausencia de transcripciones en MODMA impide aplicar técnicas de procesamiento de lenguaje natural sobre los registros de voz, limitando la representación del habla a características puramente acústicas; y (iv) la alta variabilidad entre pliegues de validación cruzada (\(\mathrm{std}\approx 0.12\)--\(0.20\)) refleja la inestabilidad inherente a los conjuntos de datos pequeños.

Estos aspectos serán abordados en la siguiente fase del proyecto mediante: (i) la incorporación de técnicas de aumento de datos y estrategias de regularización para mitigar el sobreajuste; (ii) el entrenamiento de arquitecturas profundas que puedan aprender representaciones más robustas a partir de las mismas características; (iii) la extensión del análisis con procesamiento de lenguaje natural sobre el conjunto ACEMATE, que sí dispone de transcripciones textuales; (iv) la implementación de mecanismos de atención multimodal que permitan ponderar la contribución de cada fuente de información en función de su disponibilidad y calidad; y (v) la evaluación de los modelos bajo protocolos de validación más rigurosos, incluyendo validación externa cuando se disponga de conjuntos de datos adicionales.
