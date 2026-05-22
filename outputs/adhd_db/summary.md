# Resumen de experimentos EEGNet en ADHD-DB

## Objetivo

Se evaluó una variante de EEGNet para clasificación binaria ADHD vs Control usando registros EEG preprocesados como tensores completos o como listas de sujetos. El análisis compara el efecto de la representación de entrada, la agregación temporal, la longitud del registro, el tipo de normalización y el parámetro `meanmax_alpha`.

## Pipeline

El pipeline carga archivos `.mat`, extrae la matriz EEG principal, orienta cada señal como canales por tiempo, corrige valores no finitos, aplica referencia promedio, filtro notch, filtrado pasa banda y, opcionalmente, remuestreo o recorte temporal. Después, los datos se entregan al modelo como `tensor` o como `list`.

Para cada ejecución se usó validación cruzada estratificada por sujeto con `k = 5`, `split_seed = 3407`, tres semillas de inicialización `2025`, `2026`, `2027`, `batch_size = 16`, `epochs = 100`, `patience = 35`, `lr = 1e-3`, `weight_decay = 0`, `dropout = 0`, estandarización por sujeto y canal, y pesos de clase activados.

La estandarización se aplica por sujeto y canal sobre la dimensión temporal, por lo que no usa estadísticas poblacionales de train/validation/test.

## Notación y shapes

La tabla está ordenada por `test_balanced_acc_mean`, de mayor a menor, primero los experimentos `tensor` y luego los experimentos `list`. Solo se incluyen filas con bloque `Overall` completo.

Cada sujeto se representa como:

| 1 | `eegnet_tensor_aggmean` | tensor | true | BatchNorm | 0.0 | full | 0.722000 | 0.722009 | 0.717305 | 0.597233 | 0.750000 | Mejor resultado global; la media temporal pura fue la agregación más estable. |
| 2 | `eegnet_tensor_noagg` | tensor | false | BatchNorm | — | full | 0.718889 | 0.719231 | 0.709601 | 0.640070 | 0.740000 | Muy cercano al mejor; preservar logits temporales y votar funciona bien. |
| 3 | `eegnet_tensor_62s_noagg` | tensor | false | BatchNorm | — | 62 s | 0.702222 | 0.702350 | 0.694846 | 0.633168 | 0.726667 | Recortar a 62 s reduce poco el rendimiento frente al tensor completo sin agregación. |
| 4 | `eegnet_tensor_aggmeanmax010` | tensor | true | BatchNorm | 0.1 | full | 0.635778 | 0.635684 | 0.618033 | 0.675798 | 0.680000 | Añadir componente máximo degrada el rendimiento frente a media pura. |
| 5 | `eegnet_tensor_aggmean_groupnorm` | tensor | true | GroupNorm | 0.0 | full | 0.616889 | 0.616026 | 0.594341 | 0.641366 | 0.690000 | En tensor, GroupNorm fue peor que BatchNorm. |
| 6 | `eegnet_tensor_aggmeanmax020` | tensor | true | BatchNorm | 0.2 | full | 0.572667 | 0.573291 | 0.552308 | 0.691571 | 0.653333 | Aumentar `alpha` empeora la clasificación. |
| 7 | `eegnet_tensor_aggmeanmax050` | tensor | true | BatchNorm | 0.5 | full | 0.495667 | 0.496154 | 0.460808 | 0.711105 | 0.586667 | La mezcla media-máximo con peso alto se aproxima a azar. |
| 8 | `eegnet_tensor_aggmeanmax100` | tensor | true | BatchNorm | 1.0 | full | 0.465556 | 0.465812 | 0.444730 | 0.782775 | 0.603333 | Usar máximo puro fue la peor configuración. |
| 9 | `eegnet_list_full_aggmean_groupnorm` | list | true | GroupNorm | 0.0 | full variable | 0.702444 | 0.702136 | 0.693267 | 0.607755 | 0.746666 | La agregación por media en lista con apilamiento cuando es posible queda muy cerca del mejor tensor. |
| 10 | `eegnet_list_62s_noagg_stackfix` | list | false | GroupNorm | — | 62 s | 0.699444 | 0.699573 | 0.692648 | 0.633873 | 0.730000 | La corrección de apilamiento mejora fuertemente el modo lista. |
| 11 | `eegnet_list_full_aggmean_batchnorm` | list | true | BatchNorm | 0.0 | full variable | 0.597000 | 0.597009 | 0.583097 | 0.697448 | 0.703333 | BatchNorm en lista completa quedó por debajo de GroupNorm en la misma configuración de agregación. |
| 12 | `eegnet_list_aggmean` | list | true | GroupNorm | 0.0 | full variable | 0.597000 | 0.597009 | 0.583097 | 0.697444 | 0.703333 | La agregación por media en lista sin apilamiento no alcanzó el rendimiento del mejor caso con GroupNorm. |
| 13 | `eegnet_list_62s_noagg` | list | false | GroupNorm | — | 62 s | 0.539222 | 0.540812 | 0.507884 | 0.696109 | 0.730000 | Antes del `stackfix`, el modo lista 62 s quedaba muy por debajo. |
| 14 | `eegnet_list_noagg` | list | false | GroupNorm | — | full variable | 0.509778 | 0.510256 | 0.469375 | 0.694999 | 0.670000 | La lista sin agregación ni corrección no fue competitiva. |

Cuando se usa `duration_sec = 62` y `default_fs = 128`, la entrada temporal queda en:

$$
T = 62 \cdot 128 = 7936,
\qquad
T' = 124
$$

| list full + BatchNorm + aggregate mean | 0.597009 |
## Representación `tensor`

En modo `tensor`, todos los sujetos se recortan a la mínima longitud temporal disponible para poder formar un tensor único:

El mejor modelo fue `tensor + BatchNorm + aggregate mean`, con `test_balanced_acc_mean = 0.722009`. La diferencia frente a `tensor + no aggregation` fue pequeña, lo que indica que tanto la media temporal pura como el voto mayoritario sobre logits temporales son estrategias válidas.
\bf {X} \in \rm R^{N \times C \times T}
$$

El modo `tensor` fue más estable que el modo `list` en las corridas principales. La ventaja se explica por el batch real, la longitud temporal común y el uso natural de `BatchNorm2d`.

El modo `list` requiere más cuidado. Cuando los sujetos tienen longitudes variables, `GroupNorm` es la opción correcta por no depender del tamaño efectivo del batch. En la corrida sin recorte y con agregación, `eegnet_list_full_aggmean_groupnorm` alcanzó `0.702136`, por lo que el registro completo con `GroupNorm` sí es competitivo y no debe omitirse. Además, la versión `list_62s_noagg_stackfix` llegó a `0.699573`, muy cerca de ese valor, lo que muestra que el apilamiento cuando las longitudes coinciden sigue siendo una mejora importante.
\bf {X}_B \in \rm R^{B \times C \times T}
El recorte a 62 segundos mantuvo un rendimiento competitivo. En tensor sin agregación, el resultado fue `0.702350`, solo por debajo del tensor completo sin agregación. En lista, el caso recortado con `stackfix` quedó en `0.699573`, mientras que el mejor caso de lista con registro completo y `GroupNorm` llegó a `0.702136`. Esto sugiere que una ventana fija de 62 s contiene información suficiente para la clasificación en este pipeline, pero no supera de forma consistente al uso del registro completo cuando la implementación de `list` está bien ajustada.

EEGNet añade una dimensión espacial interna:

La configuración recomendada sigue siendo `pp_as=tensor`, `aggregate=true`, `meanmax_alpha=0.0`, `norm=auto` y, por tanto, `BatchNorm2d`. Esta variante obtuvo el mejor rendimiento promedio y mantuvo una formulación simple: logits temporales agregados por media para producir una única predicción por sujeto.
\bf {X}_B \rightarrow \rm R^{B \times 1 \times C \times T}
Para registros de longitud variable, `pp_as=list` con `GroupNorm` es conceptualmente adecuado y, cuando se usa agregación sobre el registro completo, alcanzó `0.702136`, muy cerca del mejor tensor. La versión con `stackfix` para 62 s también fue competitiva con `0.699573`, así que conviene mencionar ambos resultados: el mejor caso de `list` completo con `GroupNorm` y la mejora práctica lograda por `stackfix` en la versión recortada.

Después de las capas convolucionales y de pooling, el clasificador produce logits temporales:

$$
\bf {Z} \in \rm R^{B \times T' \times L}
$$

Si `aggregate = true`, la agregación actúa después del clasificador temporal y antes de la pérdida:

$$
\bf {z}_s =
(1-\alpha)\frac{1}{T'}\sum_{t=1}^{T'} \bf {Z}_{s,t}
+
\alpha \max_{t} \bf {Z}_{s,t}
$$

Por tanto, la salida usada por `CrossEntropyLoss` es:

$$
\bf {z} \in \rm R^{B \times L},
\qquad
\bf {y} \in \rm R^{B}
$$

Si `aggregate = false`, no se aplica agregación dentro del modelo. La pérdida usa los logits temporales completos:

$$
\bf {Z} \in \rm R^{B \times T' \times L}
$$

En ese caso, las etiquetas originales siguen siendo por sujeto:

$$
\bf {y} \in \rm R^{B}
$$

pero para calcular la pérdida se expanden temporalmente:

$$
\tilde{\bf {y}} \in \rm R^{B \times T'}
$$

A nivel de dataset, esto equivale a:

$$
\tilde{\bf {y}} \in \rm R^{N \times T'}
$$

La predicción final por sujeto no se toma directamente de cada instante temporal, sino por voto mayoritario sobre las predicciones temporales.

## Representación `list`

En modo `list`, no se fuerza una longitud común entre sujetos. La entrada del dataset se mantiene como:

$$
[\bf {X}_1,\ldots,\bf {X}_N],
\qquad
\bf {X}_s \in \rm R^{C \times T_s}
$$

Cada batch contiene una lista de $B$ tensores:

$$
[\bf {X}_{s_1},\ldots,\bf {X}_{s_B}]
$$

Si todos los sujetos del batch tienen la misma longitud temporal, el modelo los apila temporalmente y ejecuta un forward equivalente a:

$$
\rm R^{B \times C \times T}
$$

Si las longitudes son distintas, el modelo procesa cada sujeto individualmente como:

$$
\rm R^{1 \times C \times T_s}
$$

y devuelve una lista de logits temporales:

$$
[\bf {Z}_1,\ldots,\bf {Z}_B],
\qquad
\bf {Z}_s \in \rm R^{T'_s \times L}
$$

Si `aggregate = true`, cada sujeto se agrega de forma independiente:

$$
\bf {z}_s \in \rm R^{L}
$$

y luego se concatena el batch:

$$
\bf {z} \in \rm R^{B \times L},
\qquad
\bf {y} \in \rm R^{B}
$$

Si `aggregate = false`, no existe necesariamente una matriz densa:

$$
\rm R^{B \times T' \times L}
$$

porque cada sujeto puede tener un $T'_s$ distinto. Por eso la salida se conserva como lista:

$$
[\bf {Z}_1,\ldots,\bf {Z}_B],
\qquad
\bf {Z}_s \in \rm R^{T'_s \times L}
$$

La etiqueta original sigue siendo una sola por sujeto:

$$
\bf {y} \in \rm R^{B}
$$

pero para la pérdida temporal se repite dentro de cada sujeto:

$$
\tilde{\bf {y}}_s \in \rm R^{T'_s}
$$

En este caso, no se debe describir la etiqueta expandida como una matriz única $\rm R^{N \times T'}$, salvo que todos los sujetos tengan la misma longitud temporal.

## Modelo

El modelo usa una arquitectura tipo EEGNet:

1. convolución temporal sobre la dimensión $T$;
2. convolución espacial depthwise sobre los $C$ canales;
3. bloque separable depthwise-pointwise;
4. clasificador convolucional $1 \times 1$;
5. logits temporales $\rm R^{B \times T' \times L}$ o lista de logits $\rm R^{T'_s \times L}$.

Con `norm = auto`, el modelo usa `BatchNorm2d` en modo `tensor` y `GroupNorm` en modo `list`.

## BatchNorm2d vs GroupNorm

En modo `tensor`, el modelo recibe batches con forma $\rm R^{B \times C \times T}$, donde $B > 1$ durante entrenamiento. Tras añadir la dimensión interna, el tensor pasa a $\rm R^{B \times 1 \times C \times T}$. En este caso, `BatchNorm2d` es la opción natural porque normaliza usando estadísticas del batch y aprovecha que las muestras se procesan juntas.

En modo `list`, en cambio, los sujetos pueden tener longitudes temporales distintas. Cuando no se pueden apilar, el forward se ejecuta sujeto por sujeto, con batch efectivo $B = 1$. En ese escenario, `BatchNorm2d` deja de ser una buena opción, porque sus estadísticas dependen del batch y no son estables con una sola muestra. Por eso `GroupNorm` es más adecuada para `list`: normaliza dentro de cada muestra y no depende del tamaño del batch.

## Versiones evaluadas

Se evaluaron las siguientes variantes:

- `tensor + BatchNorm + aggregate=false`: conserva logits temporales y predice por voto mayoritario.
- `tensor + BatchNorm + aggregate=true + alpha=0.0`: agregación por media temporal pura.
- `tensor + BatchNorm + aggregate=true + alpha>0`: combinación media-máximo.
- `tensor + GroupNorm + aggregate=true + alpha=0.0`: prueba directa del efecto de normalización.
- `list + GroupNorm + aggregate=false`: procesa sujetos como lista, sin agregación.
- `list + GroupNorm + aggregate=true + alpha=0.0`: procesa sujetos como lista con agregación por media.
- `list + 62s + stackfix`: versión corregida para apilar cuando las longitudes del batch coinciden.

## Results

La tabla está ordenada por `test_balanced_acc_mean`, de mejor a peor. Solo se incluyen filas con bloque `Overall` completo.

| Rank | Experimento | Entrada | Agregación | Norm | Alpha | Duración | Test acc | Test balanced acc | F1 macro | Test loss | Best val balanced acc | Hallazgo principal |
|---:|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `eegnet_tensor_aggmean` | tensor | true | BatchNorm | 0.0 | full | 0.722000 | 0.722009 | 0.717305 | 0.597233 | 0.750000 | Mejor resultado global; la media temporal pura fue la agregación más estable. |
| 2 | `eegnet_tensor_noagg` | tensor | false | BatchNorm | — | full | 0.718889 | 0.719231 | 0.709601 | 0.640070 | 0.740000 | Muy cercano al mejor; preservar logits temporales y votar funciona bien. |
| 3 | `eegnet_tensor_62s_noagg` | tensor | false | BatchNorm | — | 62 s | 0.702222 | 0.702350 | 0.694846 | 0.633168 | 0.726667 | Recortar a 62 s reduce poco el rendimiento frente al tensor completo sin agregación. |
| 4 | `eegnet_tensor_aggmeanmax010` | tensor | true | BatchNorm | 0.1 | full | 0.635778 | 0.635684 | 0.618033 | 0.675798 | 0.680000 | Añadir componente máximo degrada el rendimiento frente a media pura. |
| 5 | `eegnet_tensor_aggmean_groupnorm` | tensor | true | GroupNorm | 0.0 | full | 0.616889 | 0.616026 | 0.594341 | 0.641366 | 0.690000 | En tensor, GroupNorm fue peor que BatchNorm. |
| 6 | `eegnet_tensor_aggmeanmax020` | tensor | true | BatchNorm | 0.2 | full | 0.572667 | 0.573291 | 0.552308 | 0.691571 | 0.653333 | Aumentar `alpha` empeora la clasificación. |
| 7 | `eegnet_tensor_aggmeanmax050` | tensor | true | BatchNorm | 0.5 | full | 0.495667 | 0.496154 | 0.460808 | 0.711105 | 0.586667 | La mezcla media-máximo con peso alto se aproxima a azar. |
| 8 | `eegnet_tensor_aggmeanmax100` | tensor | true | BatchNorm | 1.0 | full | 0.465556 | 0.465812 | 0.444730 | 0.782775 | 0.603333 | Usar máximo puro fue la peor configuración. |
| 9 | `eegnet_list_full_aggmean_groupnorm` | list | true | GroupNorm | 0.0 | full variable | 0.702444 | 0.702136 | 0.693267 | 0.607755 | 0.746666 | La agregación por media en lista con apilamiento cuando es posible queda muy cerca del mejor tensor. |
| 10 | `eegnet_list_62s_noagg_stackfix` | list | false | GroupNorm | — | 62 s | 0.699444 | 0.699573 | 0.692648 | 0.633873 | 0.730000 | La corrección de apilamiento mejora fuertemente el modo lista. |
| 11 | `eegnet_list_full_aggmean_batchnorm` | list | true | BatchNorm | 0.0 | full variable | 0.597000 | 0.597009 | 0.583097 | 0.697448 | 0.703333 | BatchNorm en lista completa quedó por debajo de GroupNorm en la misma configuración de agregación. |
| 12 | `eegnet_list_aggmean` | list | true | GroupNorm | 0.0 | full variable | 0.597000 | 0.597009 | 0.583097 | 0.697444 | 0.703333 | La agregación por media en lista sin apilamiento no alcanzó el rendimiento del mejor caso con GroupNorm. |
| 13 | `eegnet_list_62s_noagg` | list | false | GroupNorm | — | 62 s | 0.539222 | 0.540812 | 0.507884 | 0.696109 | 0.730000 | Antes del `stackfix`, el modo lista 62 s quedaba muy por debajo. |
| 14 | `eegnet_list_noagg` | list | false | GroupNorm | — | full variable | 0.509778 | 0.510256 | 0.469375 | 0.694999 | 0.670000 | La lista sin agregación ni corrección no fue competitiva. |

## Comparación puntual de normalización

La comparación directa indica que, para entrada `tensor` con agregación por media, `BatchNorm2d` fue claramente superior a `GroupNorm`:

| Configuración | Test balanced acc |
|---|---:|
| tensor + BatchNorm + aggregate mean | 0.722009 |
| tensor + GroupNorm + aggregate mean | 0.616026 |
| list full + BatchNorm + aggregate mean | 0.597009 |
| list full + GroupNorm + aggregate mean | 0.702136 |

## Hallazgos

El mejor modelo fue `tensor + BatchNorm + aggregate mean`, con `test_balanced_acc_mean = 0.722009`. La diferencia frente a `tensor + no aggregation` fue pequeña, lo que indica que tanto la media temporal pura como el voto mayoritario sobre logits temporales son estrategias válidas.

La agregación `meanmax` no mejoró el rendimiento. Al aumentar `meanmax_alpha`, el desempeño cayó de forma monotónica aproximada: `0.635684` con `alpha = 0.1`, `0.573291` con `alpha = 0.2`, `0.496154` con `alpha = 0.5` y `0.465812` con `alpha = 1.0`. Esto sugiere que el máximo temporal introduce ruido o sobreenfatiza activaciones locales no representativas.

El modo `tensor` fue más estable que el modo `list` en las corridas principales. La ventaja se explica por el batch real, la longitud temporal común y el uso natural de `BatchNorm2d`.

El modo `list` requiere más cuidado. Cuando los sujetos tienen longitudes variables, `GroupNorm` es la opción correcta por no depender del tamaño efectivo del batch. En la corrida sin recorte y con agregación, `eegnet_list_full_aggmean_groupnorm` alcanzó `0.702136`, así que el registro completo con `GroupNorm` sí es competitivo y no debe omitirse. La versión `eegnet_list_62s_noagg_stackfix` subió hasta `0.699573`, muy cerca de `tensor_62s_noagg`, mientras que `eegnet_list_full_aggmean_batchnorm` quedó en `0.597009`, por debajo del caso con `GroupNorm`.

El recorte a 62 segundos mantuvo un rendimiento competitivo. En tensor sin agregación, el resultado fue `0.702350`, solo por debajo del tensor completo sin agregación. En lista, el caso recortado con `stackfix` quedó en `0.699573`, algo por debajo del mejor caso de lista completa con `GroupNorm` (`0.702136`). Esto sugiere que una ventana fija de 62 s contiene información suficiente para la clasificación en este pipeline, pero no supera de forma consistente al uso del registro completo cuando la implementación de `list` está bien ajustada.

## Conclusión

La configuración recomendada es `pp_as=tensor`, `aggregate=true`, `meanmax_alpha=0.0`, `norm=auto` y, por tanto, `BatchNorm2d`. Esta variante obtuvo el mejor rendimiento promedio y mantuvo una formulación simple: logits temporales agregados por media para producir una única predicción por sujeto.

Para registros de longitud variable, `pp_as=list` con `GroupNorm` es conceptualmente adecuado y, cuando se usa agregación sobre el registro completo, alcanzó `0.702136`, muy cerca del mejor tensor. La versión con `stackfix` para 62 s también fue competitiva con `0.699573`, así que conviene mencionar ambos resultados: el mejor caso de `list` completo con `GroupNorm` y la mejora práctica lograda por `stackfix` en la versión recortada. La corrida `list_full_aggmean_batchnorm` no mejora esa referencia y queda claramente por debajo.