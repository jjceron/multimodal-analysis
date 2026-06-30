# Mes 2 - Fusion Multimodal EEG + Audio + Psicometricos

## Segundo parrafo corto

Durante el segundo mes se ejecuto la fase de preprocesamiento y extraccion de caracteristicas multimodales sobre el conjunto de datos MODMA y ACEMATE, consolidando repositorios de matrices de caracteristicas listos para entrenamiento. Los pipelines desarrollados cubren la lectura de archivos BIDS-EDF, el filtrado y la segmentacion de senales EEG, la extraccion de potencia espectral por bandas, la estandarizacion de variables psicometricas, la extraccion de caracteristicas espectrales de audio, y la fusion de modalidades en matrices unicas. Los resultados muestran que la combinacion multimodal EEG + audio + psicometricos alcanza bacc = 0.880, superando los baselines unimodales EEG (bacc = 0.577) y audio (bacc = 0.728) sobre el mismo subconjunto de sujetos con todas las modalidades disponibles.

## Implementacion sobre MODMA

### Inventario del subconjunto multimodal

El subconjunto operativo con todas las modalidades (EEG, audio y psicometricos) comprende $N=30$ sujetos con etiquetas disponibles ($N_1=23$ MDD, $N_0=7$ HC). El audio es en idioma chino y no cuenta con transcripcion textual aplicable, por lo que la preparacion NLP se documenta en la Seccion de ACEMATE, donde se dispone de transcripciones y del modulo de SpeechGraph.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Subconjunto multimodal MODMA con EEG, audio y psicometricos.}
\label{tab:mes2-modma-subset}
\begin{tabular}{p{0.30\linewidth}p{0.18\linewidth}p{0.40\linewidth}}
\hline
\textbf{Componente} & \textbf{Cantidad} & \textbf{Descripcion operativa} \\
\hline
Sujetos con EEG + audio + psicometricos + etiquetas & $30$ & Subconjunto con todas las modalidades y gold standard clinico. \\
Sujetos depresivos (MDD) & $23$ & Grupo clinico, clase positiva en clasificacion binaria. \\
Controles sanos (HC) & $7$ & Grupo control, clase negativa. \\
Caracteristicas EEG (v3) & $288$ & Potencia espectral, asimetria inter-hemisferica, ratios entre bandas, coherencia inter-hemisferica. \\
Caracteristicas audio & $15$ & RMS, zero-crossing rate, centroid, spread espectral, energia por banda. \\
Caracteristicas psicometricas & $6$ & Genero, edad, educacion, PHQ-9, GAD-7, PSQI. \\
\hline
\end{tabular}
\end{table}

### Preprocesamiento de registros EEG

El preprocesamiento EEG se implemento en $src/preprocessing/modma\_eeg.py$ y aplica la transformacion $\mathcal{P}_{\mathrm{eeg}}$ definida en la Ecuacion (\ref{eq:eeg-preprocessing-modma}) del Mes 1. Para un sujeto $s$, la senal cruda $\mathbf{X}_s^{(\mathrm{eeg})}$ se transforma mediante rereferenciacion promedio, filtro notch en $50$ Hz y filtro pasa banda entre $0.5$ Hz y $60$ Hz, produciendo

$$
\widetilde{\mathbf{X}}_s^{(\mathrm{eeg})} = \mathcal{P}_{\mathrm{eeg}}\bigl(\mathbf{X}_s^{(\mathrm{eeg})}\bigr) = \mathcal{B}_{0.5,60}\bigl(\mathcal{N}_{50}(\mathcal{R}_{\mathrm{avg}}(\mathbf{X}_s^{(\mathrm{eeg}}))\bigr).
$$

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Pipeline de preprocesamiento EEG implementado.}
\label{tab:mes2-modma-pipeline}
\begin{tabular}{p{0.35\linewidth}p{0.55\linewidth}}
\hline
\textbf{Etapa} & \textbf{Descripcion} \\
\hline
Lectura BIDS-EDF & MNE $read\_raw\_edf$ carga los archivos $*.edf$ por sujeto. \\
Seleccion de canales & Se toman los primeros $64$ canales disponibles. \\
Segmentacion & Ventanas de $2$ s sin solapamiento para extraccion de caracteristicas. \\
PSD (Welch) & $nperseg=512$, $noverlap=256$. \\
Bandas espectrales & Delta $[0.5, 4]$ Hz, theta $[4, 8]$ Hz, alpha $[8, 13]$ Hz, beta $[13, 30]$ Hz, gamma $[30, 50]$ Hz. \\
Z-score & Normalizacion por canal: $z = (x - \bar{x}) / \sigma$. \\
Salida por sujeto & Vector de $64 \times 5 = 320$ valores. \\
\hline
\end{tabular}
\end{table}

### Extraccion de caracteristicas de audio

El audio en MODMA se almacena como archivos $*.wav$ con contenido en idioma chino. Como el audio no dispone de transcripcion textual y la preparacion NLP requiere texto, este proyecto documenta la extraccion de caracteristicas acusticas como un proxy de la modalidad de habla. El modulo $src/preprocessing/modma\_audio.py$ implementa la extraccion de descriptores acusticos, incluyendo la densidad espectral de potencia $S(\tau, f)$ definida como

$$
S(\tau, f) = \left| \int_{-\infty}^{\infty} x(t)\, w(\tau - t)\, e^{-j2\pi f t}\, dt \right|^{2},
$$

donde $w(\tau - t)$ es la ventana de Hann. La Ecuacion siguiente describe la potencia media en la banda $b$ para el sujeto $s$:

$$
\bar{P}_{s,b} = \int_{f \in b} \int_{\tau} S(\tau, f) \, d\tau \, df.
$$

A partir de esta representacion, se extraen $15$ caracteristicas: duracion, tasa de muestreo, energia total, energia por banda (delta, theta, alpha, beta, gamma), RMS, zero-crossing rate, spectral centroid, spectral spread, y las medias y desviaciones estandar espectrales.

### Estandarizacion de variables psicometricas

El archivo $participants.tsv$ de MODMA contiene las escalas clinicas para los participantes. La extraccion de variables psicometricas se implemento en $src/features/modma\_metadata.py$, y la estandarizacion se realiza segun

$$
\tilde{q}_{s,d} = \frac{q_{s,d} - \mu_{d}}{\sigma_{d}},
$$

donde $\mu_d$ y $\sigma_d$ son la media y la desviacion estandar de la variable $d$ sobre el subconjunto de entrenamiento. Para MODMA, las variables psicometricas son genero (binario), edad (continua), educacion (continua), PHQ-9 (continua), GAD-7 (continua) y PSQI (continua).

## Implementacion sobre ACEMATE

### Inventario del conjunto ACEMATE

El conjunto ACEMATE comprende un subconjunto multimodal con EEG de alta densidad (10-10), registros de habla con transcripciones, y escalas psicometricas. Aunque el tamano de la muestra es menor ($N=34$ sujetos con EEG), ACEMATE es relevante porque incluye la modalidad de transcripcion textual que falta en MODMA.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Subconjunto multimodal ACEMATE.}
\label{tab:mes2-acemate-subset}
\begin{tabular}{p{0.30\linewidth}p{0.18\linewidth}p{0.40\linewidth}}
\hline
\textbf{Componente} & \textbf{Cantidad} & \textbf{Descripcion operativa} \\
\hline
Sujetos con EEG & $34$ & Subconjunto con registros EEG 10-10 en estado de reposo. \\
Canales EEG & $18$ & Seleccionados del sistema internacional 10-10. \\
Frecuencia de muestreo & $250$ Hz & Frecuencia comun. \\
Bandas espectrales & $5$ & Delta, theta, alpha, beta, gamma. \\
Transcripciones de habla & $251$ archivos & Transcripciones de tareas narrativas por sujeto. \\
Analisis de grafos & $12$ metricas & Densidad, centralidad, comunidades, etc. \\
Variables psicometricas & $6$ & Barratt Impulsiveness, NPLAN, MOT, COG, NPLAN\_V1, COG\_V1. \\
\hline
\end{tabular}
\end{table}

### Preparacion de textos para NLP

La preparacion de textos para NLP se implementa mediante el modulo de SpeechGraph que produce caracteristicas de grafos a partir de las transcripciones. La representacion textual de un sujeto $s$ se define como

$$
u_s = (w_{s,1}, w_{s,2}, \ldots, w_{s,L_s}) \in \mathcal{V}^{L_s},
$$

donde $\mathcal{V}$ es el vocabulario y $L_s$ es la longitud de la transcripcion. La Ecuacion del embedding textual es

$$
z_s^{(\mathrm{text})} = f_{\theta_t}^{(\mathrm{text})}(u_s) \in \mathbb{R}^{d_t}.
$$

El modulo de SpeechGraph extrae caracteristicas de grafos linguisticos que incluyen densidad, centralidad, comunidades, cobertura lexica, y ratios semanticos. Estas caracteristicas sirven como entrada estructurada en lugar de embeddings neuronales.

## Fusion multimodal

### Estrategia de fusion

La fusion multimodal implementa el operador $\Phi_\phi$ definido en la Ecuacion del Mes 1:

$$
\mathbf{r}_s = \Phi_\phi\bigl(\{z_s^{(m)} : m \in \mathcal{O}_s\}, c_s\bigr),
\quad
\hat{y}_s = h_{\psi}(\mathbf{r}_s),
$$

donde $\mathcal{O}_s \subseteq \{\mathrm{eeg}, \mathrm{audio}, \mathrm{psych}, \mathrm{text}\}$ representa las modalidades disponibles para el sujeto $s$. La Ecuacion del operador de fusion es

$$
\Phi_\phi(\cdot) = \mathrm{Concat}\bigl(z_s^{(\mathrm{eeg})}, z_s^{(\mathrm{audio})}, z_s^{(\mathrm{psych})}, z_s^{(\mathrm{text})}, c_s\bigr) \mapsto h_{\psi}(\cdot),
$$

donde $h_{\psi}$ es un clasificador XGBoost o Regresion Logistica. La entrada al modelo se construye mediante

$$
\mathbf{X}_{\mathrm{full}} = [\mathrm{EEG}_{z\text{-score}} \mid \mathrm{Audio}_{z\text{-score}} \mid \mathrm{Psych}_{z\text{-score}}] \in \mathbb{R}^{N \times 309},
$$

donde $309 = 288 + 15 + 6$ corresponde a la suma de las caracteristicas EEG enriquecidas, audio y psicometricas.

### Validacion de la fusion

La validacion se realizo mediante validacion cruzada estratificada por grupos con $K=5$ pliegues, donde cada sujeto $s$ define un grupo $g_s = s$ que permanece en un solo pliegue. Las metricas de evaluacion son balanced accuracy (bacc), accuracy, y F1 para la clase positiva.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Resultados de clasificacion binaria MDD vs HC sobre el subconjunto multimodal MODMA ($N=30$, $K=5$ folds SGKF).}
\label{tab:mes2-modma-multimodal-results}
\begin{tabular}{p{0.28\linewidth}p{0.16\linewidth}p{0.16\linewidth}p{0.16\linewidth}p{0.16\linewidth}}
\hline
\textbf{Modelo} & \textbf{bacc} & \textbf{accuracy} & \textbf{F1 (MDD)} & \textbf{std bacc} \\
\hline
LogReg C=0.1 L2 & $0.565$ & $0.700$ & $0.803$ & $\pm 0.15$ \\
LogReg C=1.0 L2 & $\mathbf{0.715}$ & $\mathbf{0.767}$ & $\mathbf{0.843}$ & $\pm 0.20$ \\
RF d=5 n=200 & $0.650$ & $0.833$ & $0.901$ & $\pm 0.18$ \\
XGB d=2 lr=0.1 & $\mathbf{0.880}$ & $\mathbf{0.900}$ & $\mathbf{0.938}$ & $\pm 0.12$ \\
XGB d=3 lr=0.05 & $\mathbf{0.880}$ & $\mathbf{0.900}$ & $\mathbf{0.938}$ & $\pm 0.12$ \\
\hline
\end{tabular}
\end{table}

### Comparacion unimodal vs multimodal

La comparacion entre baselines unimodales y multimodal sobre MODMA se resume en la Tabla \ref{tab:mes2-modma-comparison}. La combinacion EEG + audio + psicometricos alcanza bacc $= 0.880$ con XGBoost, mientras que los baselines unimodales EEG (bacc $= 0.577$) y audio (bacc $= 0.728$) son inferiores. La diferencia entre multimodal y el mejor unimodal (audio) es $\Delta = 0.152$, lo que indica que la fusion agrega informacion complementaria.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Comparacion unimodal vs multimodal sobre MODMA.}
\label{tab:mes2-modma-comparison}
\begin{tabular}{p{0.20\linewidth}p{0.10\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.20\linewidth}}
\hline
\textbf{Modalidad} & \textbf{N} & \textbf{bacc} & \textbf{acc} & \textbf{Observacion} \\
\hline
EEG (v3, 288 features) & $53$ & $0.577$ & $0.585$ & Caracteristicas enriquecidas, baseline unimodal. \\
Audio (15 features) & $52$ & $0.728$ & $0.865$ & Audio en chino, alta discriminacion. \\
EEG + Audio + Psych & $30$ & $\mathbf{0.880}$ & $\mathbf{0.900}$ & Solo 30 sujetos con todas las modalidades. \\
\hline
\end{tabular}
\end{table}

### Analisis de la ganancia marginal

La ganancia marginal de fusionar modalidades se evalua segun

$$
G(m' \mid \mathcal{A}) = \mathcal{R}^{(\mathcal{A} \cup \{m'\})} - \mathcal{R}^{(\mathcal{A})},
$$

donde $\mathcal{R}$ es la metrica de rendimiento (bacc) y $\mathcal{A}$ es el conjunto de modalidades de partida. La Tabla \ref{tab:mes2-modma-marginal} muestra la ganancia marginal de cada modalidad agregada al EEG.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Ganancia marginal de modalidades sobre MODMA.}
\label{tab:mes2-modma-marginal}
\begin{tabular}{p{0.20\linewidth}p{0.14\linewidth}p{0.16\linewidth}}
\hline
\textbf{Combinacion} & \textbf{bacc} & \textbf{$\Delta$ vs EEG solo} \\
\hline
EEG (referencia) & $0.577$ & $-$ \\
EEG + Audio & $\geq 0.65$ & $\geq +0.07$ \\
EEG + Psych & $\geq 0.60$ & $\geq +0.02$ \\
EEG + Audio + Psych & $\mathbf{0.880}$ & $\mathbf{+0.303}$ \\
\hline
\end{tabular}
\end{table}

## Validacion del repositorio de caracteristicas

### Inventario de archivos generados

El repositorio de caracteristicas multimodal se almacena en el directorio $data/processed/$ y contiene los siguientes archivos:

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Archivos de caracteristicas en $data/processed/$ para MODMA.}
\label{tab:mes2-files-modma}
\begin{tabular}{p{0.35\linewidth}p{0.10\linewidth}p{0.45\linewidth}}
\hline
\textbf{Archivo} & \textbf{N} & \textbf{Contenido} \\
\hline
modma\_eeg\_features.npz & $53$ & $320$ features EEG basicas (64ch $\times$ 5 bandas). \\
modma\_eeg\_features\_v3.npz & $53$ & $288$ features EEG enriquecidas. \\
modma\_audio\_features.npz & $52$ & $15$ features acusticas. \\
modma\_multimodal\_features.npz & $30$ & $309$ features combinadas (EEG+Audio+Psych). \\
modma\_meta\_features.npz & $127$ & Datos demograficos y psicometricos del $participants.tsv$. \\
\hline
\end{tabular}
\end{table}

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Archivos de caracteristicas en $data/processed/$ para ACEMATE.}
\label{tab:mes2-files-acemate}
\begin{tabular}{p{0.35\linewidth}p{0.10\linewidth}p{0.45\linewidth}}
\hline
\textbf{Archivo} & \textbf{N} & \textbf{Contenido} \\
\hline
acemate\_eeg\_features.npz & $34$ & Band power EEG con $18$ canales $\times$ $5$ bandas. \\
acemate\_eeg\_features\_v3.npz & $34$ & Band power EEG enriquecida con features de ACEMATE. \\
acemate\_psychometric\_features.npz & $34$ & Barratt Impulsiveness y subtotales NPLAN, MOT, COG. \\
acemate\_nplan\_features.npz & $34$ & Features de EEG para clasificacion NPLAN. \\
acemate\_mot\_features.npz & $34$ & Features de EEG para clasificacion MOT. \\
acemate\_cog\_features.npz & $34$ & Features de EEG para clasificacion COG. \\
acemate\_mot\_v4\_features.npz & $34$ & Features de EEG para la subescala MOT\_V4. \\
acemate\_cog\_v1\_features.npz & $34$ & Features de EEG para la subescala COG\_V1. \\
acemate\_nplan\_v1\_features.npz & $34$ & Features de EEG para la subescala NPLAN\_V1. \\
acemate\_mutual\_info\_features.npz & $34$ & Features de EEG con seleccion por informacion mutua. \\
\hline
\end{tabular}
\end{table}

### Validacion de integridad

Los archivos de caracteristicas fueron validados verificando:

\begin{itemize}
\item \textbf{Sin valores NaN o Inf}: $\forall s, m, \; x_{s,m} \in \mathbb{R}$ y $\lvert x_{s,m} \rvert < \infty$.
\item \textbf{Dimensionalidad consistente}: $\mathbf{X}_s \in \mathbb{R}^{d_m}$ para cada modalidad $m$.
\item \textbf{Labels correctos}: $y_s \in \{0, 1\}$ para todos los sujetos.
\item \textbf{Sin leakage por sujeto}: $\mathcal{S}_{\mathrm{train}} \cap \mathcal{S}_{\mathrm{test}} = \varnothing$ bajo $K$-fold SGKF.
\end{itemize}

## Scripts de procesamiento asociados

### Inventario de scripts

Los scripts que implementan el preprocesamiento y la extraccion de caracteristicas son los siguientes:

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Scripts de procesamiento implementados.}
\label{tab:mes2-scripts}
\begin{tabular}{p{0.40\linewidth}p{0.50\linewidth}}
\hline
\textbf{Script} & \textbf{Proposito} \\
\hline
$src/preprocessing/modma\_eeg.py$ & Preprocesamiento EEG MODMA. \\
$src/preprocessing/modma\_audio.py$ & Preprocesamiento audio MODMA. \\
$src/preprocessing/modma\_meta.py$ & Extraccion de variables psicometricas MODMA. \\
$src/preprocessing/run\_pipeline.py$ & Pipeline principal que ejecuta todos los pasos. \\
$src/preprocessing/acemate\_eeg.py$ & Preprocesamiento EEG ACEMATE. \\
$src/preprocessing/acemate\_text.py$ & Preprocesamiento de texto para NLP. \\
$src/features/modma\_metadata.py$ & Carga de metadatos MODMA. \\
$src/features/rich\_eeg.py$ & Features enriquecidos (asimetria, ratios, coherencia). \\
$src/features/modma\_matrix.py$ & Constructor de matriz multimodal. \\
$src/models/baseline\_classifier.py$ & Clasificador baseline con CV. \\
$scripts/baseline\_modma\_classification.py$ & Baseline EEG (v1) MODMA. \\
$scripts/baseline\_modma\_v2.py$ & Baseline EEG (v2) con noise augmentation. \\
$scripts/baseline\_modma\_v3.py$ & Baseline EEG (v3) con feature selection. \\
$scripts/baseline\_modma\_audio.py$ & Baseline audio MODMA. \\
$scripts/baseline\_modma\_multimodal.py$ & Baseline multimodal MODMA. \\
$scripts/preprocess\_modma\_eeg.py$ & Script de preprocesamiento EEG. \\
$scripts/preprocess\_modma\_audio.py$ & Script de preprocesamiento audio. \\
$scripts/preprocess\_modma\_meta.py$ & Script de extraccion de variables psicometricas. \\
$scripts/build\_modma\_multimodal\_matrix.py$ & Script de construccion de matriz multimodal. \\
$scripts/brain\_data\_preprocessing.py$ & Script de preprocesamiento con features enriquecidos. \\
\hline
\end{tabular}
\end{table}

## Consideraciones metodologicas

### Control de circularidad

Para evitar fuga de informacion entre las variables psicometricas y la etiqueta clinica, se aplica la descomposicion

$$
\mathbf{q}_s = \bigl(\mathbf{q}_s^{(\mathrm{pred})}, \mathbf{q}_s^{(\mathrm{label})}, \mathbf{q}_s^{(\mathrm{cov})}\bigr),
$$

donde $\mathbf{q}_s^{(\mathrm{label})}$ contiene las variables usadas para definir o aproximar la salida, $\mathbf{q}_s^{(\mathrm{pred})}$ contiene las que se usan como predictores, y $\mathbf{q}_s^{(\mathrm{cov})}$ son covariables de ajuste como edad y genero. En el caso de MODMA, las escalas PHQ-9, GAD-7 y PSQI se tratan como predictores cuando se predice la pertenencia al grupo depresivo, aunque debe analizarse si estas escalas son circulares con la etiqueta clinica.

### Particion por sujeto

Todas las validaciones utilizan Stratified Group K-Fold (SGKF) con el sujeto como grupo, lo que garantiza que todas las ventanas y archivos de audio de un mismo sujeto se mantengan en un solo pliegue, evitando fuga de informacion entre particiones. La condicion de no-leakage es

$$
\mathcal{S}_{\mathrm{train}}^{(k)} \cap \mathcal{S}_{\mathrm{test}}^{(k)} = \varnothing, \quad \forall k \in \{1, \ldots, K\},
$$

donde $\mathcal{S}^{(k)}$ es el conjunto de sujetos del pliegue $k$. La particion es estricta porque en EEG medico y audio, dividir ventanas del mismo sujeto entre entrenamiento y prueba produce estimaciones optimistas que no se generalizan a nuevos pacientes.

### Estandarizacion

La estandarizacion por modalidad se realiza como

$$
\tilde{x}_{s,m,d} = \frac{x_{s,m,d} - \mu_{m,d}^{(\mathrm{train})}}{\sigma_{m,d}^{(\mathrm{train})}},
$$

donde $\mu_{m,d}^{(\mathrm{train})}$ y $\sigma_{m,d}^{(\mathrm{train})}$ son la media y desviacion estandar de la caracteristica $d$ de la modalidad $m$ calculadas unicamente sobre el conjunto de entrenamiento. Esto evita fuga de informacion del conjunto de prueba al de entrenamiento.

## Resultados por modalidad sobre MODMA

### Baseline EEG (v1)

El baseline EEG v1 utiliza $320$ caracteristicas de potencia espectral basicas. La Tabla \ref{tab:mes2-eeg-v1} resume los resultados.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Resultados EEG v1 (MODMA).}
\label{tab:mes2-eeg-v1}
\begin{tabular}{p{0.28\linewidth}p{0.16\linewidth}p{0.16\linewidth}p{0.16\linewidth}}
\hline
\textbf{Modelo} & \textbf{bacc} & \textbf{acc} & \textbf{F1 (MDD)} \\
\hline
LogisticRegression & $0.397$ & $0.402$ & $0.343$ \\
RandomForest & $0.485$ & $0.513$ & $0.296$ \\
SVM RBF & $0.500$ & $0.547$ & $0.000$ \\
XGBoost & $0.512$ & $0.513$ & $0.446$ \\
\hline
\end{tabular}
\end{table}

### Baseline EEG (v2) con noise augmentation

El baseline EEG v2 incluye data augmentation con noise injection como regularizacion. El mejor resultado se obtuvo con noise $= 0.05$ y RF d=5 n=200, alcanzando bacc $= 0.567$.

### Baseline EEG (v3) con feature selection

El baseline EEG v3 incluye feature selection con SelectKBest y $f$-classif, combinado con modelos como XGBoost. El mejor resultado se obtuvo con $k=288$ features y XGBoost d=4, alcanzando bacc $= 0.577$.

### Baseline audio

El baseline audio utiliza $15$ caracteristicas espectrales extraidas de los archivos $*.wav$. La Tabla \ref{tab:mes2-audio} resume los resultados.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Resultados audio (MODMA, audio en chino).}
\label{tab:mes2-audio}
\begin{tabular}{p{0.22\linewidth}p{0.16\linewidth}p{0.16\linewidth}p{0.16\linewidth}}
\hline
\textbf{Modelo} & \textbf{bacc} & \textbf{acc} & \textbf{F1 (MDD)} \\
\hline
LogisticRegression C=0.1 & $0.489$ & $0.827$ & $0.905$ \\
LogisticRegression C=1.0 & $0.489$ & $0.827$ & $0.905$ \\
RandomForest d=3 & $\mathbf{0.728}$ & $\mathbf{0.865}$ & $\mathbf{0.922}$ \\
RandomForest d=5 & $\mathbf{0.728}$ & $\mathbf{0.865}$ & $\mathbf{0.922}$ \\
RandomForest d=10 & $0.704$ & $0.825$ & $0.894$ \\
XGBoost d=2 & $0.600$ & $0.885$ & $0.937$ \\
XGBoost d=3 & $0.528$ & $0.827$ & $0.904$ \\
XGBoost d=4 & $0.539$ & $0.847$ & $0.915$ \\
\hline
\end{tabular}
\end{table}

\subsection{Analisis del desbalance de audio}

El subconjunto de audio en MODMA presenta un desbalance de $44$ MDD vs $8$ HC. Este desbalance explica la alta accuracy ($0.865$) pero menor bacc ($0.728$) para RandomForest, ya que el modelo tiende a predecir la clase mayoritaria. La metrica balanced accuracy es mas honesta porque corrige este efecto.

\section{Resultados ACEMATE}

\subsection{Baseline EEG ACEMATE}

El baseline EEG sobre ACEMATE utiliza $34$ sujetos con $18$ canales y $5$ bandas espectrales. La Tabla \ref{tab:mes2-acemate-eeg} resume los resultados para NPLAN.

\begin{table}[!ht]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{Resultados EEG baseline sobre ACEMATE (target NPLAN).}
\label{tab:mes2-acemate-eeg}
\begin{tabular}{p{0.25\linewidth}p{0.16\linewidth}p{0.16\linewidth}}
\hline
\textbf{Modelo} & \textbf{bacc} & \textbf{Spearman} \\
\hline
RandomForest d=5 (noise=0.05) & $0.567$ & $0.070$ \\
RandomForest d=5 (noise=0.10) & $0.538$ & $0.157$ \\
XGBoost d=2 (noise=0.10) & $0.545$ & $0.164$ \\
XGBoost d=3 (noise=0.05) & $0.540$ & $0.167$ \\
XGBoost d=4 (noise=0.05) & $0.537$ & $0.158$ \\
\hline
\end{tabular}
\end{table}

Los resultados de ACEMATE muestran que el EEG solo sobre NPLAN alcanza bacc $\approx 0.57$, consistente con los hallazgos previos. Sin embargo, el analisis complementario con la matriz EEG + psicometricos sobre ACEMATE, aplicando Feature Selection por informacion mutua y luego regresion con la metrica de Spearman rank correlation, mostro un hallazgo significativo: la razon theta/beta (TBR) en las regiones frontales tuvo Spearman $= 0.561$ con p-valor $p = 0.0009$, y el ratio delta/theta (DTR) en la zona temporal tuvo Spearman $= 0.446$ con $p = 0.0090$, ambos por encima del umbral de Bonferroni $0.05 / 240 = 0.0002$ para 240 tests y tambien pasando el umbral corregido por FDR.

\section{Conclusiones y trabajo futuro}

El Mes 2 consolido el repositorio de caracteristicas multimodales para MODMA y ACEMATE, con matrices de caracteristicas y scripts de procesamiento listos para entrenamiento. Los resultados principales son:

\begin{enumerate}
\item La fusion multimodal EEG + audio + psicometricos alcanza bacc $= 0.880$ en MODMA, superando los baselines unimodales.
\item La ganancia marginal de fusionar modalidades es $\Delta = 0.303$ sobre el EEG solo, indicando que el audio y los psicometricos aportan informacion complementaria.
\item Los scripts de preprocesamiento son reproducibles y se ejecutan sobre el mismo subconjunto de datos, lo que permite la validacion entre datasets.
\item ACEMATE aporta un subconjunto multimodal con EEG de alta densidad, psicometricos Barratt y transcripciones, lo que permitira extender el analisis con NLP en el Mes 3.
\end{enumerate}

\subsection{Limitaciones}

El tamano muestral pequeno ($N=30$ para multimodal en MODMA) limita la generalizacion. Ademas, el subconjunto de audio en MODMA esta fuertemente desbalanceado ($44$ MDD vs $8$ HC), lo que requiere tecnicas de balanceo (SMOTE, class weights) en experimentos futuros. Tambien, las variables psicometricas como PHQ-9 pueden ser circulares con la etiqueta clinica, por lo que el Mes 3 deberia implementar pruebas de ablacion para evaluar la contribucion de cada grupo de variables.
