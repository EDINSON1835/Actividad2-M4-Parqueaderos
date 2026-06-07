# 🏢 Simulación de Parqueaderos — Centro Comercial Supercentro
## Laboratorio Final — Sistema de Colas M/M/1

---

## 📋 Descripción

Simulación completa de un sistema de pago de parqueaderos con 3 cajeros independientes usando el modelo M/M/1. Implementa todos los puntos del laboratorio:

| Punto | Descripción | Archivo de salida |
|-------|-------------|-------------------|
| P1 | Estadísticas por cajero (IC 95%) | `estadisticas_cajeros.png` |
| P2 | Promedio de usuarios por tipo | `usuarios_por_tipo.png` |
| P3 | Estrategia de mejora / decisión | `estrategia_mejora.png` |
| P4 | Verificación, Calibración, Validación | `diagrama_vv.png` + `validacion_mm1.png` |
| P5 | Eliminación del estado transitorio | `estado_estable.png` |

---

## ⚙️ Instalación y Ejecución en VS Code

### 1. Instalar dependencias

Abre una terminal en VS Code (`Ctrl+` ` `) y ejecuta:

```bash
pip install -r requirements.txt
```

O en sistemas con Python 3:
```bash
pip3 install -r requirements.txt
```

### 2. Ejecutar la simulación

```bash
python simulacion_parqueadero.py
```

O presiona **F5** en VS Code con el archivo abierto.

---

## 📊 Parámetros del Sistema

| Tipo Usuario | Tiempo Servicio (μ) | Tiempo Llegada (λ) | Distribución | % |
|---|---|---|---|---|
| 🟢 Rápido | 1 min | 3 min | Exponencial | 25% |
| 🔵 Normal | 3 min | 3 min | Exponencial | 20% |
| 🟠 Lento | 4 min | 5 min | Exponencial | 27.5% |
| 🔴 Muy Lento | 6 min | 7 min | Exponencial | 27.5% |

### Factores de Utilización (ρ)

| Tipo | ρ | Estado |
|------|---|--------|
| 🟢 Rápido | 0.333 | ✅ Estable |
| 🔵 Normal | 1.000 | ⚠️ Crítico |
| 🟠 Lento | 0.800 | ✅ Estable |
| 🔴 Muy Lento | 0.857 | ✅ Estable |

---

## 🏗️ Estructura del Código

```
simulacion_parqueadero.py
│
├── MÓDULO 1 — Simulación M/M/1
│   ├── simular_mm1()           → Eventos discretos por tipo puro
│   └── run_replica_cajeros()   → 3 cajeros con mezcla de usuarios
│
├── MÓDULO 2 — Estado Estable
│   ├── determinar_estado_estable()     → Promedio móvil + varianza
│   ├── calcular_metricas_mm1_teoricas() → Fórmulas analíticas
│   └── intervalo_confianza()           → IC t-Student
│
├── MÓDULO 3 — Experimento
│   └── ejecutar_experimento_completo() → 30 réplicas × 2000 min
│
├── MÓDULO 4 — Visualizaciones
│   ├── plot_portada()              → Portada del proyecto
│   ├── plot_estado_estable()       → PUNTO 5
│   ├── plot_estadisticas_cajeros() → PUNTO 1
│   ├── plot_usuarios_por_tipo()    → PUNTO 2
│   ├── plot_estrategia_mejora()    → PUNTO 3
│   ├── plot_diagrama_vv()          → PUNTO 4 (diagrama)
│   ├── plot_validacion_mm1()       → PUNTO 4 (validación)
│   └── plot_resumen_ejecutivo()    → Dashboard final
│
└── MÓDULO 5 — main()
```

---

## 📈 Figuras Generadas

Al ejecutar el script se generan automáticamente **8 figuras PNG**:

1. `portada.png` — Portada visual del proyecto
2. `diagrama_vv.png` — Diagrama de flujo V&V
3. `estado_estable.png` — Warm-up period (antes/después)
4. `estadisticas_cajeros.png` — Violin plots + IC por cajero
5. `usuarios_por_tipo.png` — Proporciones observadas vs esperadas
6. `estrategia_mejora.png` — Heatmap y tabla de decisión
7. `validacion_mm1.png` — QQ plots + test KS
8. `resumen_ejecutivo.png` — Dashboard consolidado

---

## 🔬 Metodología

### Modelo M/M/1 por Cajero
- Llegadas: proceso de Poisson (tiempos exponenciales)
- Servicio: distribución exponencial
- Disciplina: FIFO (First In, First Out)
- Capacidad: cola infinita

### Determinación del Estado Estable
1. Simular serie temporal de tiempos en sistema
2. Calcular promedio móvil con ventana = 100 obs.
3. Detectar punto de mínima varianza del promedio móvil
4. Usar solo datos posteriores al punto de corte

### Validación
- Test de Kolmogorov-Smirnov vs distribución teórica
- Comparación W_simulado vs W_teórico
- Nivel de significancia α = 0.05

---

## 📐 Fórmulas M/M/1

```
ρ  = λ/μ              (factor de utilización)
L  = ρ/(1-ρ)          (clientes en sistema)
Lq = ρ²/(1-ρ)         (clientes en cola)
W  = 1/(μ-λ)          (tiempo en sistema)
Wq = ρ/(μ-λ)          (tiempo en cola)
P₀ = 1-ρ              (probabilidad sistema vacío)
```

---

## ⏱️ Tiempo de Ejecución Estimado

- Con configuración estándar (30 réplicas × 2000 min): **~30-60 segundos**
- Para pruebas rápidas, reducir `NUM_REPLICAS = 10` y `TIEMPO_SIMULACION = 500`

---

## 🛠️ Configuración Recomendada VS Code

Extensiones sugeridas:
- **Python** (Microsoft)
- **Pylance**
- **Jupyter** (para exploración interactiva)

Configuración en `settings.json`:
```json
{
    "python.defaultInterpreterPath": "python",
    "python.terminal.activateEnvironment": true
}
```
