# Resumen de resultados MDD con EEGNet

## Entrada de datos y metodología

Se evaluó una tarea de clasificación binaria a partir de EEG de reposo:

- `0 = H / control sano`
- `1 = MDD / depresión`

Los datos provienen de archivos EDF de la base `mdd_db`, separados por condición:

- `EC`: ojos cerrados
- `EO`: ojos abiertos

El pipeline de datos usado para los resultados reportados fue:

```text
EDF crudo
→ selección de 20 canales comunes
→ referencia promedio
→ filtro pasa banda 0.5–60 Hz
→ notch 50 Hz
→ frecuencia original 256 Hz
→ sin remuestreo: target_fs=None
→ sin recorte manual: duration_sec=None
→ construcción de dataset en modo tensor o list
→ validación cruzada estratificada por sujeto/grupo
```

Para el modo `tensor`, cada condición se convierte a una matriz de tamaño fijo:

```text
EC tensor: N=58, C=20, T=46080  ≈ 180 s a 256 Hz
EO tensor: C=20, T=48384        ≈ 189 s a 256 Hz
```

Para el modo `list`, cada muestra conserva su longitud temporal individual, por lo que la entrada es una lista de tensores `Tensor[C, T_i]`.

Las auditorías del dataset y dataloaders verificaron:

```text
sin solapamiento de sujetos entre train/val/test
sin solapamiento de archivos entre train/val/test
sin muestras faltantes en outer test
sin muestras repetidas en outer test
sin sujetos en múltiples folds de test
shapes consistentes
sin NaN/Inf
forward válido por EEGNet
```

En EO se detectó inicialmente un sujeto duplicado representado por dos archivos. Para reportar limpio, se debe usar la versión EO sin duplicado, es decir, con un archivo por sujeto.

---

## Entrenamiento

El modelo usado fue `EEGNet` para clasificación binaria sujeto/archivo:

```text
entrada:     EEG Tensor[B, 20, T] o list[Tensor[20, T_i]]
modelo:      EEGNet
salida:      logits_time [B, T', 2]
agregación:  promedio temporal puro
salida final: logits [B, 2]
predicción: argmax(logits)
```

Configuración común de entrenamiento:

```text
aggregate=True
meanmax_alpha=0.0
standardize=True
class_weights=True
epochs=60
patience=20
batch_size=16
init_seeds=[3001, 2025, 2026]
split_seed=3407
k=5 folds
inner_splits=5
```

En modo `tensor`, `norm=auto` usa BatchNorm.  
En modo `list`, `norm=auto` usa GroupNorm.

La métrica principal para comparar modelos es `test_balanced_acc_mean`, porque las clases están levemente desbalanceadas.

---

## Resultados de las cuatro pruebas

| Experimento | Condición | pp_as | Norm real | Accuracy mean | **Balanced Acc mean** | F1 macro mean | Test loss mean | Best val BAcc mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `eegnet_ec_tensor_agg_auto_3init` | EC | tensor | BatchNorm | 0.828283 | **0.825556** | 0.806975 | 0.460998 | 0.915000 |
| `eegnet_eo_tensor_agg_auto_3init` | EO | tensor | BatchNorm | 0.866239 | **0.865714** | 0.861888 | 0.355107 | 0.913333 |
| `eegnet_ec_list_agg_auto_3init` | EC | list | GroupNorm | 0.701515 | **0.697778** | 0.677637 | 0.611843 | 0.845000 |
| `eegnet_eo_list_agg_auto_3init` | EO | list | GroupNorm | 0.838462 | **0.833333** | 0.834241 | 0.476328 | 0.933333 |

### Comparación principal

El mejor resultado global fue:

```text
EO + tensor + EEGNet + promedio temporal puro
test_balanced_acc_mean = 0.865714
test_f1_macro_mean     = 0.861888
```

El segundo mejor resultado fue:

```text
EO + list + EEGNet + promedio temporal puro
test_balanced_acc_mean = 0.833333
test_f1_macro_mean     = 0.834241
```

En EC, el modo tensor fue claramente superior al modo list:

```text
EC tensor balanced acc = 0.825556
EC list   balanced acc = 0.697778
```

En EO, ambos modos funcionaron bien, pero tensor fue superior:

```text
EO tensor balanced acc = 0.865714
EO list   balanced acc = 0.833333
```

---

## Conclusión

Los resultados apoyan usar como pipeline principal:

```text
pp_as=tensor
norm=auto → BatchNorm
aggregate=True
meanmax_alpha=0.0
target_fs=None
duration_sec=None
fs original=256 Hz
```

La configuración `tensor` fue la más consistente y reportable, especialmente porque funciona bien en ambas condiciones. La condición EO obtuvo el mejor rendimiento con EEGNet, alcanzando una balanced accuracy media de aproximadamente `0.866`.

La variante `list`, aunque conserva longitudes temporales variables, no mejoró el desempeño general. En EO fue competitiva, pero en EC redujo notablemente el rendimiento. Por tanto, `list` debe reportarse como ablación secundaria y no como pipeline principal.

Conclusión reportable:

```text
EEGNet permitió discriminar sujetos MDD frente a controles sanos usando EEG de reposo.
El mejor desempeño se obtuvo en EO con modo tensor, sin remuestreo ni recorte manual,
usando 20 canales comunes y agregación temporal por promedio puro de logits.
```

Este resultado debe interpretarse como clasificación experimental MDD vs control en esta base EEG, no como diagnóstico clínico.
