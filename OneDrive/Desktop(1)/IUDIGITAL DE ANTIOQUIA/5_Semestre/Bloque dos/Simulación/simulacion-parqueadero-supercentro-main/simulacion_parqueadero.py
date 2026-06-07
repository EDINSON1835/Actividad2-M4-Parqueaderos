"""
=============================================================================
SIMULACIÓN DE PARQUEADEROS - CENTRO COMERCIAL SUPERCENTRO
Sistema de Pago M/M/1 - Laboratorio Final - Actividad Didáctica 2
=============================================================================
Autores: Juan David Calle Correa - Edinson Mena Arroyo
Modelo: M/M/1 independiente por cajero
Herramientas: Python, NumPy, Matplotlib, SciPy
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN GLOBAL Y PARÁMETROS DEL SISTEMA
# ============================================================

np.random.seed(42)  # Semilla para reproducibilidad

# Parámetros de tipos de usuarios
USUARIOS = {
    "rapido": {
        "servicio": 1,
        "llegada": 3,
        "prob": 0.25,
        "color": "#2ecc71",
        "emoji": "🟢",
        "label": "Rápido",
    },
    "normal": {
        "servicio": 3,
        "llegada": 3,
        "prob": 0.20,
        "color": "#3498db",
        "emoji": "🔵",
        "label": "Normal",
    },
    "lento": {
        "servicio": 4,
        "llegada": 5,
        "prob": 0.275,
        "color": "#e67e22",
        "emoji": "🟠",
        "label": "Lento",
    },
    "muy_lento": {
        "servicio": 6,
        "llegada": 7,
        "prob": 0.275,
        "color": "#e74c3c",
        "emoji": "🔴",
        "label": "Muy Lento",
    },
}

NUM_CAJEROS = 3
TIEMPO_SIMULACION = 2000  # minutos por réplica
NUM_REPLICAS = 30  # réplicas para análisis estadístico
VENTANA_MOVIL = 100  # ventana para promedio móvil
CONFIANZA = 0.95  # nivel de confianza para IC

# Colores del tema
COLORS = {
    "bg": "#0f1117",
    "panel": "#1a1d2e",
    "accent1": "#7c3aed",
    "accent2": "#06b6d4",
    "accent3": "#f59e0b",
    "text": "#e2e8f0",
    "green": "#10b981",
    "red": "#ef4444",
    "orange": "#f97316",
}

plt.rcParams.update(
    {
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": "#334155",
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "grid.color": "#334155",
        "grid.alpha": 0.4,
        "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
    }
)

# ============================================================
# MÓDULO 1 - SIMULACIÓN M/M/1
# ============================================================


def simular_mm1(lambda_llegada, mu_servicio, tiempo_sim, tipo_usuario="desconocido"):
    """
    Simula un sistema M/M/1 usando método de eventos discretos.

    Parámetros:
        lambda_llegada: tasa de llegadas (clientes/min)
        mu_servicio:    tasa de servicio (clientes/min)
        tiempo_sim:     duración total de simulación (min)
        tipo_usuario:   etiqueta del tipo de usuario

    Retorna:
        dict con métricas de la simulación
    """
    tiempo_actual = 0.0
    # Primer evento de llegada
    t_llegada = np.random.exponential(1.0 / lambda_llegada)
    t_salida = float("inf")  # No hay cliente siendo atendido aún

    clientes_sistema = 0
    cola = []  # tiempos de llegada de clientes en cola

    tiempos_espera = []  # Wq por cliente
    tiempos_sistema = []  # W  por cliente
    ocupacion_serie = []  # ρ instantáneo en el tiempo

    clientes_atendidos = 0
    area_sistema = 0.0  # área bajo curva N(t) para L
    tiempo_anterior = 0.0

    inicio_servicio = None

    while tiempo_actual < tiempo_sim:
        # ---- Acumular área antes del próximo evento ----
        area_sistema += clientes_sistema * (min(t_llegada, t_salida) - tiempo_anterior)
        tiempo_anterior = min(t_llegada, t_salida)

        if t_llegada < t_salida:
            # ----- EVENTO: LLEGADA -----
            tiempo_actual = t_llegada
            t_llegada = tiempo_actual + np.random.exponential(1.0 / lambda_llegada)

            if clientes_sistema == 0:
                # Servidor libre → atención inmediata
                inicio_servicio = tiempo_actual
                t_salida = tiempo_actual + np.random.exponential(1.0 / mu_servicio)
                tiempos_espera.append(0.0)
            else:
                # Servidor ocupado → encolar con tiempo de llegada
                cola.append(tiempo_actual)

            clientes_sistema += 1
            ocupacion_serie.append((tiempo_actual, clientes_sistema))

        else:
            # ----- EVENTO: SALIDA -----
            tiempo_actual = t_salida
            clientes_atendidos += 1

            # Registrar tiempo en sistema para el cliente que sale
            if inicio_servicio is not None:
                w = (
                    tiempo_actual - inicio_servicio + tiempos_espera[-1]
                    if tiempos_espera
                    else tiempo_actual - inicio_servicio
                )
                # Re-calcular correctamente: W = tiempo en cola + tiempo de servicio

            clientes_sistema -= 1

            if cola:
                # Siguiente cliente en cola
                t_arr = cola.pop(0)
                wq = tiempo_actual - t_arr
                tiempos_espera.append(wq)
                t_servicio = np.random.exponential(1.0 / mu_servicio)
                tiempos_sistema.append(wq + t_servicio)
                inicio_servicio = tiempo_actual
                t_salida = tiempo_actual + t_servicio
            else:
                t_salida = float("inf")
                inicio_servicio = None

            ocupacion_serie.append((tiempo_actual, clientes_sistema))

    # ---- Métricas finales ----
    if not tiempos_espera:
        tiempos_espera = [0.0]
    if not tiempos_sistema:
        tiempos_sistema = [1.0 / mu_servicio]

    L_simulado = area_sistema / tiempo_sim if tiempo_sim > 0 else 0

    return {
        "tiempos_espera": np.array(tiempos_espera),
        "tiempos_sistema": np.array(tiempos_sistema),
        "ocupacion_serie": ocupacion_serie,
        "clientes_atendidos": clientes_atendidos,
        "L_simulado": L_simulado,
        "tipo": tipo_usuario,
        "lambda": lambda_llegada,
        "mu": mu_servicio,
        "rho": lambda_llegada / mu_servicio,
    }


def run_replica_cajeros(tiempo_sim=TIEMPO_SIMULACION):
    """
    Ejecuta una réplica completa con 3 cajeros y mezcla de usuarios.
    Cada cajero recibe clientes de acuerdo a las proporciones definidas.

    Retorna lista de resultados por cajero.
    """
    tipos = list(USUARIOS.keys())
    probs = [USUARIOS[t]["prob"] for t in tipos]
    resultados_cajeros = []

    for cajero_id in range(NUM_CAJEROS):
        # Cada cajero tiene su propia cola con mezcla de usuarios
        tiempos_espera_total = []
        tiempos_sistema_total = []
        conteo_tipos = {t: 0 for t in tipos}

        tiempo_actual = 0.0
        t_llegada_prox = np.random.exponential(3.0)  # llegada general cada ~3 min
        t_salida_prox = float("inf")
        cola_cajero = []  # (t_llegada, tipo)
        tipo_en_servicio = None
        clientes_sistema = 0
        inicio_serv = None

        while tiempo_actual < tiempo_sim:
            if t_llegada_prox <= t_salida_prox:
                tiempo_actual = t_llegada_prox
                tipo_nuevo = np.random.choice(tipos, p=probs)
                conteo_tipos[tipo_nuevo] += 1
                mu_nuevo = 1.0 / USUARIOS[tipo_nuevo]["servicio"]

                if clientes_sistema == 0:
                    tipo_en_servicio = tipo_nuevo
                    t_salida_prox = tiempo_actual + np.random.exponential(
                        1.0 / mu_nuevo
                    )
                    inicio_serv = tiempo_actual
                    tiempos_espera_total.append(0.0)
                else:
                    cola_cajero.append((tiempo_actual, tipo_nuevo, mu_nuevo))

                clientes_sistema += 1
                t_llegada_prox = tiempo_actual + np.random.exponential(3.0)

            else:
                tiempo_actual = t_salida_prox
                clientes_sistema -= 1

                if inicio_serv is not None:
                    w_total = tiempo_actual - inicio_serv
                    tiempos_sistema_total.append(w_total)

                if cola_cajero:
                    t_arr, tipo_sig, mu_sig = cola_cajero.pop(0)
                    wq = tiempo_actual - t_arr
                    tiempos_espera_total.append(wq)
                    t_serv = np.random.exponential(1.0 / mu_sig)
                    tiempos_sistema_total.append(wq + t_serv)
                    tipo_en_servicio = tipo_sig
                    t_salida_prox = tiempo_actual + t_serv
                    inicio_serv = tiempo_actual
                else:
                    t_salida_prox = float("inf")
                    tipo_en_servicio = None
                    inicio_serv = None

        resultados_cajeros.append(
            {
                "cajero_id": cajero_id + 1,
                "tiempos_espera": (
                    np.array(tiempos_espera_total)
                    if tiempos_espera_total
                    else np.array([0.0])
                ),
                "tiempos_sistema": (
                    np.array(tiempos_sistema_total)
                    if tiempos_sistema_total
                    else np.array([1.0])
                ),
                "conteo_tipos": conteo_tipos,
            }
        )

    return resultados_cajeros


# ============================================================
# MÓDULO 2 - ESTADO ESTABLE Y TRANSITORIO
# ============================================================


def determinar_estado_estable(datos, ventana=VENTANA_MOVIL):
    """
    Detecta el warm-up period usando promedio móvil y varianza.

    Parámetros:
        datos:   serie temporal de tiempos de servicio
        ventana: tamaño de ventana del promedio móvil

    Retorna:
        punto_corte, promedios_moviles
    """
    n = len(datos)
    if n < ventana * 2:
        return n // 4, np.array([np.mean(datos)] * n)

    promedios_moviles = np.array(
        [np.mean(datos[i : i + ventana]) for i in range(n - ventana + 1)]
    )

    # Calcular varianza del promedio móvil en ventanas sucesivas
    varianzas = np.array(
        [
            np.var(promedios_moviles[i : i + ventana])
            for i in range(len(promedios_moviles) - ventana + 1)
        ]
    )

    # Punto donde la varianza es mínima (sistema estabilizado)
    idx_min_var = np.argmin(varianzas)
    punto_corte = min(idx_min_var + ventana, n // 2)

    return punto_corte, promedios_moviles


def calcular_metricas_mm1_teoricas(lambda_arr, mu_serv):
    """Calcula métricas teóricas del modelo M/M/1."""
    rho = lambda_arr / mu_serv
    if rho >= 1.0:
        return {
            "rho": rho,
            "L": np.inf,
            "Lq": np.inf,
            "W": np.inf,
            "Wq": np.inf,
            "P0": 0,
        }

    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1 / (mu_serv - lambda_arr)
    Wq = rho / (mu_serv - lambda_arr)
    P0 = 1 - rho

    return {"rho": rho, "L": L, "Lq": Lq, "W": W, "Wq": Wq, "P0": P0}


def intervalo_confianza(datos, confianza=CONFIANZA):
    """Calcula media e IC para un vector de datos."""
    n = len(datos)
    if n < 2:
        return np.mean(datos), 0, 0
    media = np.mean(datos)
    sem = stats.sem(datos)
    ic = stats.t.interval(confianza, df=n - 1, loc=media, scale=sem)
    return media, ic[0], ic[1]


# ============================================================
# MÓDULO 3 - EJECUCIÓN COMPLETA DE RÉPLICAS
# ============================================================


def ejecutar_experimento_completo():
    """
    Ejecuta NUM_REPLICAS réplicas y recopila todas las métricas.
    Retorna diccionarios con resultados agregados.
    """
    print("=" * 60)
    print("  SIMULACIÓN PARQUEADERO SUPERCENTRO")
    print("  Ejecutando experimento completo...")
    print("=" * 60)

    # ----- Simulaciones por tipo de usuario (puro) -----
    resultados_por_tipo = {tipo: [] for tipo in USUARIOS}

    for tipo, params in USUARIOS.items():
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]
        print(f"  Simulando usuario {params['label']} ({NUM_REPLICAS} réplicas)...")
        for rep in range(NUM_REPLICAS):
            res = simular_mm1(la, mu, TIEMPO_SIMULACION, tipo)
            resultados_por_tipo[tipo].append(res)

    # ----- Simulaciones de 3 cajeros con mezcla -----
    print(f"  Simulando 3 cajeros mixtos ({NUM_REPLICAS} réplicas)...")
    resultados_cajeros = [[] for _ in range(NUM_CAJEROS)]
    conteos_tipos_global = {t: [] for t in USUARIOS}

    for rep in range(NUM_REPLICAS):
        cajeros_rep = run_replica_cajeros()
        for caj_idx, caj_data in enumerate(cajeros_rep):
            resultados_cajeros[caj_idx].append(caj_data)
            for tipo in USUARIOS:
                conteos_tipos_global[tipo].append(caj_data["conteo_tipos"][tipo])

    print("  ✓ Experimento completado.\n")
    return resultados_por_tipo, resultados_cajeros, conteos_tipos_global


# ============================================================
# MÓDULO 4 - VISUALIZACIONES
# ============================================================


def plot_portada():
    """Genera portada visual del proyecto."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["bg"])

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Fondo con gradiente simulado mediante rectángulos
    for i, alpha in enumerate(np.linspace(0.02, 0.08, 20)):
        rect = plt.Rectangle(
            (0, i / 20), 1, 1 / 20, color=COLORS["accent1"], alpha=alpha
        )
        ax.add_patch(rect)

    # Título principal
    ax.text(
        0.5,
        0.82,
        "SIMULACIÓN DE PARQUEADEROS",
        ha="center",
        va="center",
        fontsize=32,
        fontweight="bold",
        color="white",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.73,
        "Centro Comercial SUPERCENTRO",
        ha="center",
        va="center",
        fontsize=22,
        color=COLORS["accent2"],
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.65,
        "Sistema de Cola M/M/1 — Laboratorio Final",
        ha="center",
        va="center",
        fontsize=14,
        color="#94a3b8",
        transform=ax.transAxes,
    )

    # Tarjetas de parámetros
    colores_tipo = [
        COLORS["green"],
        COLORS["accent2"],
        COLORS["accent3"],
        COLORS["red"],
    ]
    labels_cortos = [
        "Rápido\nλ=1/3  μ=1/1\nρ=0.333",
        "Normal\nλ=1/3  μ=1/3\nρ=1.000 ⚠️",
        "Lento\nλ=1/5  μ=1/4\nρ=0.800",
        "Muy Lento\nλ=1/7  μ=1/6\nρ=0.857",
    ]

    for i, (col, lbl) in enumerate(zip(colores_tipo, labels_cortos)):
        x = 0.12 + i * 0.22
        rect = FancyBboxPatch(
            (x - 0.09, 0.32),
            0.18,
            0.22,
            boxstyle="round,pad=0.01",
            facecolor=col,
            alpha=0.15,
            edgecolor=col,
            linewidth=2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            0.44,
            lbl,
            ha="center",
            va="center",
            fontsize=9.5,
            color="white",
            transform=ax.transAxes,
            linespacing=1.5,
        )

    # Métricas del sistema
    metricas_txt = [
        ("3", "Cajeros\npor Salida"),
        ("4", "Tipos de\nUsuarios"),
        ("30", "Réplicas\nde Simulación"),
        ("M/M/1", "Modelo de\nColas"),
    ]
    for i, (val, lbl) in enumerate(metricas_txt):
        x = 0.12 + i * 0.22
        ax.text(
            x,
            0.22,
            val,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=COLORS["accent1"],
            transform=ax.transAxes,
        )
        ax.text(
            x,
            0.14,
            lbl,
            ha="center",
            va="center",
            fontsize=9,
            color="#94a3b8",
            transform=ax.transAxes,
            linespacing=1.4,
        )

    ax.text(
        0.5,
        0.05,
        "Python  •  NumPy  •  Matplotlib  •  SciPy  •  Eventos Discretos",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.savefig("portada.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.show()
    print("  [Figura] Portada generada")


def plot_estado_estable(resultados_por_tipo):
    """
    Punto 5: Gráfica del estado estable antes y después del transitorio.
    Muestra promedio móvil y punto de corte para cada tipo de usuario.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "PUNTO 5 — Eliminación del Estado Transitorio\nAnálisis de Warm-Up Period por Tipo de Usuario",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=1.01,
    )
    axes = axes.flatten()

    puntos_corte = {}

    for idx, (tipo, params) in enumerate(USUARIOS.items()):
        ax = axes[idx]
        color = params["color"]

        # Tomar primera réplica como ejemplo representativo
        res = resultados_por_tipo[tipo][0]
        datos = res["tiempos_sistema"]

        if len(datos) < 10:
            datos = np.random.exponential(params["servicio"], 500)

        punto_corte, prom_mov = determinar_estado_estable(
            datos, ventana=min(50, len(datos) // 4)
        )
        puntos_corte[tipo] = punto_corte

        x_datos = np.arange(len(datos))
        x_prom = np.arange(len(prom_mov))

        # Datos originales
        ax.plot(
            x_datos, datos, color=color, alpha=0.25, linewidth=0.7, label="Datos brutos"
        )

        # Promedio móvil
        ax.plot(x_prom, prom_mov, color=color, linewidth=2.5, label="Promedio móvil")

        # Línea de corte
        ax.axvline(
            x=punto_corte,
            color="white",
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label=f"Corte: obs {punto_corte}",
        )

        # Sombreado transitorio vs estable
        ax.axvspan(0, punto_corte, alpha=0.12, color="red", label="Transitorio")
        ax.axvspan(
            punto_corte, len(datos), alpha=0.08, color="green", label="Estado estable"
        )

        # Media antes y después
        if punto_corte < len(datos):
            media_antes = np.mean(datos[:punto_corte]) if punto_corte > 0 else 0
            media_despues = np.mean(datos[punto_corte:])
            ax.axhline(
                y=media_antes,
                color="#ef4444",
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"Media transitoria: {media_antes:.2f}",
            )
            ax.axhline(
                y=media_despues,
                color="#10b981",
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label=f"Media estable: {media_despues:.2f}",
            )

        # Teoría M/M/1
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]
        teorica = calcular_metricas_mm1_teoricas(la, mu)
        if teorica["W"] != np.inf:
            ax.axhline(
                y=teorica["W"],
                color="yellow",
                linestyle="-.",
                linewidth=1.5,
                alpha=0.7,
                label=f'W teórico: {teorica["W"]:.2f}',
            )

        ax.set_title(
            f'{params["emoji"]} {params["label"]}  (ρ={teorica["rho"]:.3f})',
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        ax.set_xlabel("Número de observación", fontsize=9)
        ax.set_ylabel("Tiempo en sistema (min)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "estado_estable.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"]
    )
    plt.show()
    print("  [Figura] Estado estable y transitorio generado")
    return puntos_corte


def plot_estadisticas_cajeros(resultados_cajeros, puntos_corte=None):
    """
    Punto 1: Estadísticas por cajero con intervalos de confianza.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "PUNTO 1 — Análisis Estadístico de Cajeros\nMétricas con Intervalos de Confianza al 95%",
        fontsize=14,
        fontweight="bold",
        color="white",
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colores_cajeros = [COLORS["accent1"], COLORS["accent2"], COLORS["accent3"]]
    medias_cajeros = []
    ics_cajeros = []
    stds_cajeros = []

    # ---- Panel 1: Distribución por cajero (violin/box) ----
    ax1 = fig.add_subplot(gs[0, :2])
    datos_violin = []
    labels_violin = []

    for caj_idx in range(NUM_CAJEROS):
        tiempos_all = []
        for rep_data in resultados_cajeros[caj_idx]:
            tiempos_all.extend(rep_data["tiempos_sistema"].tolist())

        tiempos_all = np.array(tiempos_all)
        tiempos_all = tiempos_all[
            tiempos_all < np.percentile(tiempos_all, 99)
        ]  # quitar outliers extremos
        datos_violin.append(tiempos_all)
        labels_violin.append(f"Cajero {caj_idx + 1}")

        media, ic_low, ic_high = intervalo_confianza(tiempos_all)
        medias_cajeros.append(media)
        ics_cajeros.append((ic_low, ic_high))
        stds_cajeros.append(np.std(tiempos_all, ddof=1))

    parts = ax1.violinplot(
        datos_violin, positions=[1, 2, 3], showmeans=True, showmedians=True
    )
    for i, (pc, col) in enumerate(zip(parts["bodies"], colores_cajeros)):
        pc.set_facecolor(col)
        pc.set_alpha(0.4)
    parts["cmeans"].set_color("white")
    parts["cmedians"].set_color("yellow")

    for i, (med, (ic_l, ic_h)) in enumerate(zip(medias_cajeros, ics_cajeros)):
        ax1.errorbar(
            i + 1,
            med,
            yerr=[[med - ic_l], [ic_h - med]],
            fmt="o",
            color="white",
            capsize=6,
            capthick=2,
            elinewidth=2,
            markersize=8,
        )
        ax1.annotate(
            f"{med:.2f}±{(ic_h-med):.2f}",
            (i + 1, ic_h + 0.2),
            ha="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(labels_violin, fontsize=11)
    ax1.set_title(
        "Distribución Tiempos en Sistema por Cajero (Violin + IC 95%)", fontsize=11
    )
    ax1.set_ylabel("Tiempo en sistema (min)")
    ax1.grid(True, alpha=0.3, axis="y")

    # ---- Panel 2: Comparativa de medias ----
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(
        [1, 2, 3], medias_cajeros, color=colores_cajeros, alpha=0.8, width=0.5
    )
    ax2.errorbar(
        [1, 2, 3],
        medias_cajeros,
        yerr=[
            [m - ic[0] for m, ic in zip(medias_cajeros, ics_cajeros)],
            [ic[1] - m for m, ic in zip(medias_cajeros, ics_cajeros)],
        ],
        fmt="none",
        color="white",
        capsize=6,
        elinewidth=2,
    )

    idx_min = np.argmin(medias_cajeros)
    idx_max = np.argmax(medias_cajeros)
    ax2.annotate(
        "✓ Más eficiente",
        (idx_min + 1, medias_cajeros[idx_min]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        color=COLORS["green"],
        fontsize=8,
        fontweight="bold",
    )
    ax2.annotate(
        "✗ Menos eficiente",
        (idx_max + 1, medias_cajeros[idx_max]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        color=COLORS["red"],
        fontsize=8,
        fontweight="bold",
    )

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(labels_violin)
    ax2.set_title("Comparativa\nde Medias", fontsize=10)
    ax2.set_ylabel("Tiempo promedio (min)")
    ax2.grid(True, alpha=0.3, axis="y")

    # ---- Panel 3: Tabla resumen ----
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")

    col_labels = [
        "Cajero",
        "Media (min)",
        "Std Dev",
        "Mínimo",
        "Máximo",
        "IC 95% Inferior",
        "IC 95% Superior",
        "Estado",
    ]
    tabla_datos = []

    for caj_idx in range(NUM_CAJEROS):
        tiempos_all = []
        for rep_data in resultados_cajeros[caj_idx]:
            tiempos_all.extend(rep_data["tiempos_sistema"].tolist())
        tiempos_all = np.array(tiempos_all)
        media, ic_l, ic_h = intervalo_confianza(tiempos_all)
        estado = (
            "✓ Óptimo"
            if caj_idx == idx_min
            else ("⚠ Revisar" if caj_idx == idx_max else "— Normal")
        )
        tabla_datos.append(
            [
                f"Cajero {caj_idx + 1}",
                f"{media:.3f}",
                f"{np.std(tiempos_all, ddof=1):.3f}",
                f"{np.min(tiempos_all):.3f}",
                f"{np.percentile(tiempos_all, 99):.3f}",
                f"{ic_l:.3f}",
                f"{ic_h:.3f}",
                estado,
            ]
        )

    tabla = ax3.table(
        cellText=tabla_datos, colLabels=col_labels, loc="center", cellLoc="center"
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 2.2)

    for j in range(len(col_labels)):
        tabla[0, j].set_facecolor(COLORS["accent1"])
        tabla[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(tabla_datos) + 1):
        col = colores_cajeros[i - 1]
        for j in range(len(col_labels)):
            tabla[i, j].set_facecolor(COLORS["panel"])
            tabla[i, j].set_text_props(color=COLORS["text"])

    ax3.set_title("Tabla Resumen — Estadísticas por Cajero", fontsize=11, pad=10)

    plt.savefig(
        "estadisticas_cajeros.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"]
    )
    plt.show()
    print("  [Figura] Estadísticas por cajero generadas")

    # Imprimir resumen en consola
    print("\n  PUNTO 1 — RESUMEN ESTADÍSTICO POR CAJERO:")
    print(
        f"  {'Cajero':<10} {'Media':>8} {'Std':>8} {'IC 95% Inf':>12} {'IC 95% Sup':>12}"
    )
    print("  " + "-" * 54)
    for caj_idx in range(NUM_CAJEROS):
        tiempos_all = []
        for rep_data in resultados_cajeros[caj_idx]:
            tiempos_all.extend(rep_data["tiempos_sistema"].tolist())
        tiempos_all = np.array(tiempos_all)
        media, ic_l, ic_h = intervalo_confianza(tiempos_all)
        print(
            f"  {'Cajero ' + str(caj_idx+1):<10} {media:>8.3f} {np.std(tiempos_all,ddof=1):>8.3f} {ic_l:>12.3f} {ic_h:>12.3f}"
        )
    print(
        f"\n  ✓ Cajero más eficiente  : Cajero {idx_min+1} (media={medias_cajeros[idx_min]:.3f} min)"
    )
    print(
        f"  ✗ Cajero menos eficiente: Cajero {idx_max+1} (media={medias_cajeros[idx_max]:.3f} min)"
    )


def plot_usuarios_por_tipo(resultados_cajeros, conteos_tipos_global):
    """
    Punto 2: Promedio de usuarios por tipo en los 3 cajeros.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        "PUNTO 2 — Análisis de Usuarios por Tipo\nDistribución y Verificación de Proporciones",
        fontsize=14,
        fontweight="bold",
        color="white",
    )

    tipos_lista = list(USUARIOS.keys())
    colores_tipos = [USUARIOS[t]["color"] for t in tipos_lista]
    labels_tipos = [USUARIOS[t]["label"] for t in tipos_lista]
    probs_esperadas = [USUARIOS[t]["prob"] for t in tipos_lista]

    # ---- Promedio de conteos por tipo ----
    promedios_conteo = [np.mean(conteos_tipos_global[t]) for t in tipos_lista]
    total_prom = sum(promedios_conteo)
    proporciones_obs = [p / total_prom for p in promedios_conteo]

    # Panel 1: Gráfico de barras de proporciones
    ax1 = axes[0]
    x = np.arange(len(tipos_lista))
    width = 0.35
    bars1 = ax1.bar(
        x - width / 2,
        probs_esperadas,
        width,
        label="Proporción esperada",
        color=colores_tipos,
        alpha=0.9,
    )
    bars2 = ax1.bar(
        x + width / 2,
        proporciones_obs,
        width,
        label="Proporción observada",
        color=colores_tipos,
        alpha=0.45,
        edgecolor="white",
        linewidth=1.5,
    )

    for i, (esp, obs) in enumerate(zip(probs_esperadas, proporciones_obs)):
        ax1.annotate(
            f"{esp:.1%}",
            (x[i] - width / 2, esp + 0.005),
            ha="center",
            fontsize=8,
            color="white",
        )
        ax1.annotate(
            f"{obs:.1%}",
            (x[i] + width / 2, obs + 0.005),
            ha="center",
            fontsize=8,
            color=colores_tipos[i],
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_tipos, fontsize=9)
    ax1.set_title("Proporción Esperada vs Observada", fontsize=10)
    ax1.set_ylabel("Proporción")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Pie chart de distribución observada
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        proporciones_obs,
        labels=labels_tipos,
        autopct="%1.1f%%",
        colors=colores_tipos,
        startangle=90,
        pctdistance=0.8,
        wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(9)
    ax2.set_title("Distribución Observada\nen los 3 Cajeros", fontsize=10)

    # Panel 3: Box plot de conteos por réplica
    ax3 = axes[2]
    datos_box = [conteos_tipos_global[t] for t in tipos_lista]
    bp = ax3.boxplot(
        datos_box,
        labels=labels_tipos,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch, col in zip(bp["boxes"], colores_tipos):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color("#94a3b8")

    ax3.set_title("Variabilidad de Conteos\npor Réplica", fontsize=10)
    ax3.set_ylabel("Número de usuarios por cajero")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        "usuarios_por_tipo.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"]
    )
    plt.show()
    print("  [Figura] Usuarios por tipo generados")

    # Imprimir resumen
    print("\n  PUNTO 2 — PROMEDIOS DE USUARIOS POR TIPO:")
    print(f"  {'Tipo':<12} {'Esperado':>10} {'Observado':>10} {'Diferencia':>12}")
    print("  " + "-" * 46)
    for t, esp, obs in zip(tipos_lista, probs_esperadas, proporciones_obs):
        dif = obs - esp
        signo = "+" if dif > 0 else ""
        print(
            f"  {USUARIOS[t]['label']:<12} {esp:>10.1%} {obs:>10.1%} {signo}{dif:>11.1%}"
        )


def plot_validacion_mm1(resultados_por_tipo):
    """
    Punto 4: Verificación, calibración y validación vs teoría M/M/1.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "PUNTO 4 — Verificación, Calibración y Validación\nComparación Simulación vs Teoría M/M/1",
        fontsize=14,
        fontweight="bold",
        color="white",
    )
    axes = axes.flatten()

    print("\n  PUNTO 4 — VALIDACIÓN M/M/1:")
    print(
        f"  {'Tipo':<12} {'W_teórico':>10} {'W_simulado':>10} {'Error%':>8} {'p-value':>10} {'Válido':>8}"
    )
    print("  " + "-" * 62)

    for idx, (tipo, params) in enumerate(USUARIOS.items()):
        ax = axes[idx]
        color = params["color"]

        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]
        teorica = calcular_metricas_mm1_teoricas(la, mu)

        # Tiempos en sistema de todas las réplicas
        w_simulados = []
        for res in resultados_por_tipo[tipo]:
            datos = res["tiempos_sistema"]
            if len(datos) > 5:
                pc, _ = determinar_estado_estable(
                    datos, ventana=min(30, len(datos) // 4)
                )
                w_simulados.extend(datos[pc:].tolist())

        w_simulados = np.array(w_simulados)
        w_sim_media = np.mean(w_simulados)

        # ---- QQ Plot para verificar distribución ----
        if teorica["rho"] < 1.0 and len(w_simulados) > 10:
            # Esperamos distribución exponencial con media W_teorico
            scale_esperado = teorica["W"]
            datos_teoricos = np.random.exponential(
                scale=scale_esperado, size=len(w_simulados)
            )

            q_sim = np.percentile(w_simulados, np.linspace(5, 95, 50))
            q_teo = np.percentile(datos_teoricos, np.linspace(5, 95, 50))

            ax.scatter(
                q_teo,
                q_sim,
                color=color,
                alpha=0.7,
                s=30,
                label="Datos sim. vs teórico",
            )
            lim_max = max(q_teo.max(), q_sim.max()) * 1.05
            ax.plot(
                [0, lim_max],
                [0, lim_max],
                "w--",
                linewidth=1.5,
                alpha=0.7,
                label="Línea ideal (y=x)",
            )

            # Test KS
            ks_stat, p_value = stats.ks_1samp(
                w_simulados, stats.expon(scale=scale_esperado).cdf
            )

            error_pct = (
                abs(w_sim_media - teorica["W"]) / teorica["W"] * 100
                if teorica["W"] > 0
                else 0
            )
            valido = "✓ Sí" if p_value > 0.05 else "✗ No"

            ax.set_xlabel(f'Cuantiles teóricos (W={teorica["W"]:.2f} min)', fontsize=9)
            ax.set_ylabel("Cuantiles simulados (min)", fontsize=9)

            color_texto = COLORS["green"] if p_value > 0.05 else COLORS["red"]
            ax.text(
                0.05,
                0.92,
                f'W teórico:  {teorica["W"]:.3f} min\n'
                f"W simulado: {w_sim_media:.3f} min\n"
                f"Error:      {error_pct:.1f}%\n"
                f"KS p-value: {p_value:.4f}\n"
                f"Validado:   {valido}",
                transform=ax.transAxes,
                fontsize=8.5,
                color=color_texto,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.5),
            )

            print(
                f"  {params['label']:<12} {teorica['W']:>10.3f} {w_sim_media:>10.3f} "
                f"{error_pct:>7.1f}% {p_value:>10.4f} {valido:>8}"
            )
        else:
            ax.text(
                0.5,
                0.5,
                f'ρ = {teorica["rho"]:.3f}\nSistema Crítico\n(ρ ≥ 1)',
                ha="center",
                va="center",
                fontsize=14,
                color="#ef4444",
                transform=ax.transAxes,
                fontweight="bold",
            )
            print(
                f"  {params['label']:<12} {'∞':>10} {w_sim_media:>10.3f} {'N/A':>8} {'N/A':>10} {'⚠️ Crit':>8}"
            )

        ax.set_title(
            f'{params["emoji"]} {params["label"]} — QQ Plot (ρ={teorica["rho"]:.3f})',
            fontsize=10,
            color=color,
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "validacion_mm1.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"]
    )
    plt.show()
    print("  [Figura] Validación M/M/1 generada")


def plot_estrategia_mejora(resultados_cajeros):
    """
    Punto 3: Estrategia de mejora — análisis de suficiencia de 3 cajeros.
    Simula escenarios con 1, 2, 3 y 4 cajeros.
    """
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "PUNTO 3 — Estrategia de Mejora\n¿Son Suficientes 3 Cajeros? Análisis de Escenarios",
        fontsize=14,
        fontweight="bold",
        color="white",
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Criterios de decisión
    CRITERIOS = {
        "Wq_max": 3.0,  # min en cola aceptable
        "W_max": 5.0,  # min en sistema aceptable
        "rho_max": 0.85,  # utilización máxima
        "Lq_max": 3.0,  # clientes en cola aceptable
    }

    # ---- Métricas teóricas por tipo (ponderadas) ----
    # Calcular tasa de llegada efectiva promedio ponderada
    tipos = list(USUARIOS.keys())

    # Para 3 cajeros, calcular métricas agregadas
    # Usar el tipo más crítico (Normal) como referencia
    escenarios_cajeros = [1, 2, 3, 4, 5]

    ax1 = fig.add_subplot(gs[0, :2])

    # Para cada tipo, calcular Wq y W teóricos y variar "capacidad efectiva"
    wq_por_tipo = {}
    w_por_tipo = {}
    rho_por_tipo = {}

    for tipo, params in USUARIOS.items():
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]

        wqs = []
        ws = []
        rhos = []
        for n_caj in escenarios_cajeros:
            # Con n cajeros: capacidad = n * mu (simplificado M/M/n)
            mu_eff = n_caj * mu
            if la < mu_eff:
                rho_eff = la / mu_eff
                wq = (la / (mu_eff**2)) / (1 - rho_eff)
                w = wq + 1.0 / mu
            else:
                rho_eff = 1.0
                wq = 999.0
                w = 999.0
            wqs.append(min(wq, 20))
            ws.append(min(w, 25))
            rhos.append(rho_eff)

        wq_por_tipo[tipo] = wqs
        w_por_tipo[tipo] = ws
        rho_por_tipo[tipo] = rhos

    # Plot Wq vs número de cajeros
    for tipo, params in USUARIOS.items():
        ax1.plot(
            escenarios_cajeros,
            wq_por_tipo[tipo],
            "o-",
            color=params["color"],
            linewidth=2.5,
            markersize=7,
            label=f'{params["emoji"]} {params["label"]}',
        )

    ax1.axhline(
        y=CRITERIOS["Wq_max"],
        color="white",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f'Límite aceptable Wq={CRITERIOS["Wq_max"]} min',
    )
    ax1.axhline(
        y=5.0,
        color="red",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Valor crítico Wq=5 min",
    )
    ax1.axvline(
        x=3,
        color=COLORS["accent3"],
        linestyle="-.",
        linewidth=2,
        alpha=0.7,
        label="Escenario actual (3 cajeros)",
    )

    ax1.set_xlabel("Número de Cajeros", fontsize=10)
    ax1.set_ylabel("Tiempo promedio en cola Wq (min)", fontsize=10)
    ax1.set_title("Tiempo en Cola vs Número de Cajeros (Teoría M/M/n)", fontsize=10)
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_xticks(escenarios_cajeros)
    ax1.grid(True, alpha=0.3)

    # ---- Gráfico de utilización ----
    ax2 = fig.add_subplot(gs[0, 2])
    for tipo, params in USUARIOS.items():
        ax2.plot(
            escenarios_cajeros,
            rho_por_tipo[tipo],
            "o-",
            color=params["color"],
            linewidth=2,
            markersize=6,
            label=params["label"],
        )
    ax2.axhline(
        y=CRITERIOS["rho_max"],
        color=COLORS["accent3"],
        linestyle="--",
        linewidth=2,
        label=f'ρ máx = {CRITERIOS["rho_max"]}',
    )
    ax2.axhline(
        y=0.6,
        color=COLORS["green"],
        linestyle=":",
        linewidth=1.5,
        label="ρ mín óptimo = 0.60",
    )
    ax2.axvline(x=3, color=COLORS["accent3"], linestyle="-.", linewidth=1.5)
    ax2.set_xlabel("N° Cajeros", fontsize=9)
    ax2.set_ylabel("Factor de utilización ρ", fontsize=9)
    ax2.set_title("Utilización\nvs Cajeros", fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ---- Heatmap de métricas ----
    ax3 = fig.add_subplot(gs[1, :])

    metricas_tabla = []
    tipos_nombres = []
    cols_escenarios = [f'{n} Cajero{"s" if n>1 else ""}' for n in escenarios_cajeros]

    for tipo, params in USUARIOS.items():
        fila = wq_por_tipo[tipo]
        metricas_tabla.append(fila)
        tipos_nombres.append(f'{params["emoji"]} {params["label"]}')

    metricas_arr = np.array(metricas_tabla)
    im = ax3.imshow(metricas_arr, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=8)
    ax3.set_xticks(range(len(escenarios_cajeros)))
    ax3.set_xticklabels(cols_escenarios, fontsize=9)
    ax3.set_yticks(range(len(tipos_nombres)))
    ax3.set_yticklabels(tipos_nombres, fontsize=10)
    ax3.set_title(
        "Heatmap: Tiempo en Cola Wq (min) por Tipo de Usuario y Número de Cajeros",
        fontsize=10,
    )

    for i in range(len(tipos_nombres)):
        for j in range(len(escenarios_cajeros)):
            val = metricas_arr[i, j]
            color_txt = "white" if val > 4 else "black"
            ax3.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=color_txt,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax3, label="Wq (min)", shrink=0.8)

    # Recuadro en columna de 3 cajeros
    from matplotlib.patches import Rectangle

    rect_col = Rectangle(
        (1.5, -0.5),
        1,
        len(tipos_nombres),
        linewidth=3,
        edgecolor=COLORS["accent3"],
        facecolor="none",
    )
    ax3.add_patch(rect_col)

    # ---- Tabla de decisión ----
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")

    decision_data = []
    for tipo, params in USUARIOS.items():
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]

        # Con 3 cajeros
        mu_eff = 3 * mu
        if la < mu_eff:
            rho_3 = la / mu_eff
            wq_3 = (la / (mu_eff**2)) / (1 - rho_3)
            w_3 = wq_3 + 1.0 / mu
            lq_3 = la * wq_3
        else:
            rho_3, wq_3, w_3, lq_3 = 1.0, 999, 999, 999

        crit_wq = "✓" if wq_3 < CRITERIOS["Wq_max"] else "✗"
        crit_w = "✓" if w_3 < CRITERIOS["W_max"] else "✗"
        crit_rho = "✓" if rho_3 < CRITERIOS["rho_max"] else "✗"
        crit_lq = "✓" if lq_3 < CRITERIOS["Lq_max"] else "✗"

        ok_count = [crit_wq, crit_w, crit_rho, crit_lq].count("✓")
        suficiente = "✅ SUFICIENTE" if ok_count >= 3 else "❌ INSUFICIENTE"

        decision_data.append(
            [
                f'{params["emoji"]} {params["label"]}',
                f"{rho_3:.3f}",
                f"{wq_3:.3f}",
                f"{w_3:.3f}",
                f"{lq_3:.3f}",
                crit_wq,
                crit_w,
                crit_rho,
                crit_lq,
                suficiente,
            ]
        )

    col_headers = [
        "Tipo Usuario",
        "ρ (3 caj)",
        "Wq (min)",
        "W (min)",
        "Lq",
        "Wq<3?",
        "W<5?",
        "ρ<85%?",
        "Lq<3?",
        "Decisión",
    ]

    tabla = ax4.table(
        cellText=decision_data, colLabels=col_headers, loc="center", cellLoc="center"
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 2.0)

    for j in range(len(col_headers)):
        tabla[0, j].set_facecolor(COLORS["accent1"])
        tabla[0, j].set_text_props(color="white", fontweight="bold")

    for i, (tipo, params) in enumerate(USUARIOS.items()):
        for j in range(len(col_headers)):
            tabla[i + 1, j].set_facecolor(COLORS["panel"])
            tabla[i + 1, j].set_text_props(color=COLORS["text"])
        # Colorear decisión
        dec_cell = tabla[i + 1, len(col_headers) - 1]
        if "✅" in decision_data[i][-1]:
            dec_cell.set_facecolor("#064e3b")
            dec_cell.set_text_props(color="#10b981", fontweight="bold")
        else:
            dec_cell.set_facecolor("#7f1d1d")
            dec_cell.set_text_props(color="#ef4444", fontweight="bold")

    ax4.set_title(
        "Tabla de Decisión — Evaluación de Suficiencia de 3 Cajeros",
        fontsize=11,
        pad=10,
    )

    plt.savefig(
        "estrategia_mejora.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"]
    )
    plt.show()
    print("  [Figura] Estrategia de mejora generada")

    # Resumen en consola
    print("\n  PUNTO 3 — DECISIÓN FINAL (3 cajeros):")
    for fila in decision_data:
        print(
            f"  {fila[0]:<18} Wq={fila[2]} min  W={fila[3]} min  ρ={fila[1]}  → {fila[-1]}"
        )


def plot_diagrama_vv():
    """
    Punto 4: Diagrama de flujo del proceso Verificación, Calibración y Validación.
    """
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_title(
        "PUNTO 4 — Diagrama de Flujo\nProceso de Verificación, Calibración y Validación",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=15,
    )

    def draw_box(ax, x, y, w, h, text, color, fontsize=9.5, shape="rect"):
        if shape == "diamond":
            diamond_x = [x, x + w / 2, x + w, x + w / 2, x]
            diamond_y = [y + h / 2, y + h, y + h / 2, y, y + h / 2]
            ax.fill(diamond_x, diamond_y, color=color, alpha=0.85, zorder=2)
            ax.plot(diamond_x, diamond_y, color="white", linewidth=1.5, zorder=3)
        else:
            rect = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08",
                facecolor=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
            zorder=4,
            wrap=True,
            multialignment="center",
        )

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->", color="white", lw=2, connectionstyle="arc3,rad=0"
            ),
        )

    def draw_arrow_side(ax, x1, y1, x2, y2, label="", color="red"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->", color=color, lw=1.8, connectionstyle="arc3,rad=0.3"
            ),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.3, my, label, fontsize=8, color=color, fontweight="bold")

    # ---- Bloques del diagrama ----
    cx = 3.5  # centro x
    bw = 3.0  # ancho de box
    bh = 0.7  # alto de box

    # INICIO
    draw_box(
        ax,
        cx,
        16.5,
        bw,
        bh,
        "INICIO\nDefinición del Problema",
        COLORS["accent2"],
        fontsize=9,
    )
    draw_arrow(ax, cx + bw / 2, 16.5, cx + bw / 2, 15.5)

    # Construcción del modelo
    draw_box(
        ax,
        cx,
        14.7,
        bw,
        bh,
        "Construcción del Modelo\nM/M/1 (Python + NumPy)",
        COLORS["accent1"],
        fontsize=8.5,
    )
    draw_arrow(ax, cx + bw / 2, 14.7, cx + bw / 2, 13.5)

    # --- VERIFICACIÓN ---
    draw_box(
        ax,
        cx,
        12.7,
        bw,
        bh,
        "VERIFICACIÓN\n¿Código correcto?",
        "#c0392b",
        fontsize=9,
        shape="diamond",
    )
    draw_arrow(ax, cx + bw / 2, 12.7, cx + bw / 2, 11.7)
    ax.text(
        cx + bw / 2 + 0.15,
        12.18,
        "SÍ →",
        fontsize=9,
        color=COLORS["green"],
        fontweight="bold",
    )

    # Flecha NO → Corregir
    draw_box(
        ax, 7.2, 12.5, 2.0, bh * 0.9, "NO\n→ Corregir\nCódigo", "#922b21", fontsize=8
    )
    ax.annotate(
        "",
        xy=(7.2, 12.85),
        xytext=(cx + bw, 13.05),
        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.8),
    )
    ax.text(7.35, 13.15, "NO", fontsize=9, color="#ef4444", fontweight="bold")
    ax.annotate(
        "",
        xy=(cx + bw, 12.85),
        xytext=(7.2, 12.85),
        arrowprops=dict(
            arrowstyle="->",
            color="#ef4444",
            lw=1.5,
            connectionstyle="angle,angleA=0,angleB=90",
        ),
    )

    # --- CALIBRACIÓN ---
    draw_box(
        ax,
        cx,
        11.0,
        bw,
        bh,
        "CALIBRACIÓN\nAjustar Parámetros λ y μ",
        "#d68910",
        fontsize=8.5,
    )
    draw_arrow(ax, cx + bw / 2, 11.0, cx + bw / 2, 9.9)

    # --- VALIDACIÓN ---
    draw_box(
        ax,
        cx,
        9.2,
        bw,
        bh,
        "VALIDACIÓN\n¿Modelo ≈ Realidad?",
        "#1a5276",
        fontsize=9,
        shape="diamond",
    )
    draw_arrow(ax, cx + bw / 2, 9.2, cx + bw / 2, 8.2)
    ax.text(
        cx + bw / 2 + 0.15,
        8.7,
        "SÍ →",
        fontsize=9,
        color=COLORS["green"],
        fontweight="bold",
    )

    # Flecha NO → Revisar
    draw_box(
        ax, 7.2, 9.0, 2.0, bh * 0.9, "NO\n→ Revisar\nModelo", "#6c3483", fontsize=8
    )
    ax.annotate(
        "",
        xy=(7.2, 9.35),
        xytext=(cx + bw, 9.55),
        arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=1.8),
    )
    ax.text(7.35, 9.65, "NO", fontsize=9, color="#9b59b6", fontweight="bold")
    ax.annotate(
        "",
        xy=(cx + bw, 9.35),
        xytext=(7.2, 9.35),
        arrowprops=dict(
            arrowstyle="->",
            color="#9b59b6",
            lw=1.5,
            connectionstyle="angle,angleA=0,angleB=90",
        ),
    )

    # --- MODELO VALIDADO ---
    draw_box(
        ax,
        cx,
        7.5,
        bw,
        bh,
        "MODELO VALIDADO\nAnálisis y Resultados",
        "#1e8449",
        fontsize=8.5,
    )
    draw_arrow(ax, cx + bw / 2, 7.5, cx + bw / 2, 6.5)

    # --- SIMULACIÓN COMPLETA ---
    draw_box(
        ax,
        cx,
        5.7,
        bw,
        bh,
        "Simulación Completa\n30 Réplicas × 2000 min",
        COLORS["accent1"],
        fontsize=8.5,
    )
    draw_arrow(ax, cx + bw / 2, 5.7, cx + bw / 2, 4.7)

    # --- ANÁLISIS ---
    draw_box(
        ax,
        cx,
        3.9,
        bw,
        bh,
        "Análisis:\nPuntos 1-3 y 5",
        COLORS["accent2"],
        fontsize=8.5,
    )
    draw_arrow(ax, cx + bw / 2, 3.9, cx + bw / 2, 2.9)

    # --- FIN ---
    draw_box(
        ax,
        cx,
        2.2,
        bw,
        bh,
        "CONCLUSIONES\ny Recomendaciones",
        COLORS["green"],
        fontsize=8.5,
    )

    # Leyenda
    leyenda_items = [
        (COLORS["accent2"], "██ Inicio/Fin"),
        (COLORS["accent1"], "██ Proceso"),
        ("#c0392b", "◇  Verificación"),
        ("#1a5276", "◇  Validación"),
        ("#d68910", "██ Calibración"),
    ]
    for i, (col, lbl) in enumerate(leyenda_items):
        ax.text(0.4, 16.8 - i * 0.6, lbl, fontsize=8.5, color=col, fontweight="bold")

    plt.tight_layout()
    plt.savefig("diagrama_vv.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.show()
    print("  [Figura] Diagrama V&V generado")


def plot_resumen_ejecutivo(resultados_por_tipo, resultados_cajeros):
    """
    Dashboard final con resumen ejecutivo de todos los resultados.
    """
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        "RESUMEN EJECUTIVO — Simulación Parqueadero Supercentro\nDashboard de Resultados Principales",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    # ---- Panel 1: Métricas teóricas por tipo ----
    ax1 = fig.add_subplot(gs[0, :2])
    tipos_lista = list(USUARIOS.keys())
    labels = [USUARIOS[t]["label"] for t in tipos_lista]
    colores = [USUARIOS[t]["color"] for t in tipos_lista]

    metricas_names = ["W (min)", "Wq (min)", "L", "Lq", "ρ"]
    valores = []
    for tipo, params in USUARIOS.items():
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]
        m = calcular_metricas_mm1_teoricas(la, mu)
        v = [
            min(m["W"], 30),
            min(m["Wq"], 30),
            min(m["L"], 15),
            min(m["Lq"], 15),
            m["rho"],
        ]
        valores.append(v)

    x = np.arange(len(metricas_names))
    width = 0.2
    for i, (vals, col, lbl) in enumerate(zip(valores, colores, labels)):
        ax1.bar(x + i * width, vals, width, label=lbl, color=col, alpha=0.85)

    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels(metricas_names, fontsize=9)
    ax1.set_title("Métricas Teóricas M/M/1 por Tipo de Usuario", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylabel("Valor")

    # ---- Panel 2: Rho gauge / semáforo ----
    ax2 = fig.add_subplot(gs[0, 2])
    rhos = []
    for tipo, params in USUARIOS.items():
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]
        rhos.append(la / mu)

    bars = ax2.barh(labels, rhos, color=colores, alpha=0.85, height=0.5)
    ax2.axvline(
        x=1.0, color="red", linestyle="--", linewidth=2, label="ρ=1 (inestable)"
    )
    ax2.axvline(
        x=0.85, color="orange", linestyle=":", linewidth=1.5, label="ρ=0.85 (crítico)"
    )
    ax2.axvline(
        x=0.6,
        color="green",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="ρ=0.6 (óptimo)",
    )

    for i, (rho, col) in enumerate(zip(rhos, colores)):
        ax2.text(
            rho + 0.02,
            i,
            f"{rho:.3f}",
            va="center",
            fontsize=9,
            color=col,
            fontweight="bold",
        )

    ax2.set_title("Factor de\nUtilización ρ", fontsize=10)
    ax2.set_xlabel("ρ")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_xlim(0, 1.3)

    # ---- Panel 3: Resumen de decisión ----
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")

    decisiones = []
    for tipo, params in USUARIOS.items():
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]
        mu_eff = 3 * mu
        if la < mu_eff:
            rho_3 = la / mu_eff
            wq_3 = (la / (mu_eff**2)) / (1 - rho_3)
            suf = wq_3 < 3.0
        else:
            suf = False
        icon = "✅" if suf else "❌"
        decisiones.append((params["label"], icon))

    ax3.text(
        0.5,
        0.95,
        "¿3 Cajeros\nSuficientes?",
        ha="center",
        va="top",
        fontsize=13,
        color="white",
        fontweight="bold",
        transform=ax3.transAxes,
    )

    for i, (lbl, icon) in enumerate(decisiones):
        y_pos = 0.75 - i * 0.15
        ax3.text(
            0.15,
            y_pos,
            icon,
            ha="center",
            va="center",
            fontsize=18,
            transform=ax3.transAxes,
        )
        ax3.text(
            0.55,
            y_pos,
            lbl,
            ha="left",
            va="center",
            fontsize=11,
            color="white",
            transform=ax3.transAxes,
        )

    # Veredicto final
    todos_ok = all(d[1] == "✅" for d in decisiones)
    mitad_ok = sum(1 for d in decisiones if d[1] == "✅") >= 2
    veredicto = (
        "✅ SUFICIENTES"
        if todos_ok
        else ("⚠️ PARCIALMENTE" if mitad_ok else "❌ INSUFICIENTES")
    )
    color_verd = (
        COLORS["green"]
        if todos_ok
        else (COLORS["accent3"] if mitad_ok else COLORS["red"])
    )

    ax3.text(
        0.5,
        0.12,
        veredicto,
        ha="center",
        va="center",
        fontsize=12,
        color=color_verd,
        fontweight="bold",
        transform=ax3.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=COLORS["panel"],
            edgecolor=color_verd,
            linewidth=2,
        ),
    )

    # ---- Panel 4: Comparación simulación vs teoría (W) ----
    ax4 = fig.add_subplot(gs[1, :3])

    w_teoricos = []
    w_simulados_med = []
    w_simulados_err = []

    for tipo, params in USUARIOS.items():
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]
        m = calcular_metricas_mm1_teoricas(la, mu)
        w_teoricos.append(min(m["W"], 40) if m["W"] != np.inf else 40)

        ws_rep = []
        for res in resultados_por_tipo[tipo]:
            datos = res["tiempos_sistema"]
            if len(datos) > 5:
                pc, _ = determinar_estado_estable(
                    datos, ventana=min(30, len(datos) // 4)
                )
                ws_rep.extend(datos[pc:].tolist())

        if ws_rep:
            m2, ic_l, ic_h = intervalo_confianza(np.array(ws_rep))
            w_simulados_med.append(m2)
            w_simulados_err.append(m2 - ic_l)
        else:
            w_simulados_med.append(0)
            w_simulados_err.append(0)

    x = np.arange(len(labels))
    width = 0.35
    ax4.bar(
        x - width / 2,
        w_teoricos,
        width,
        label="W teórico M/M/1",
        color=[USUARIOS[t]["color"] for t in tipos_lista],
        alpha=0.9,
    )
    ax4.bar(
        x + width / 2,
        w_simulados_med,
        width,
        label="W simulado (media ± IC95%)",
        color=[USUARIOS[t]["color"] for t in tipos_lista],
        alpha=0.45,
        edgecolor="white",
        linewidth=1.5,
    )
    ax4.errorbar(
        x + width / 2,
        w_simulados_med,
        yerr=w_simulados_err,
        fmt="none",
        color="white",
        capsize=5,
        elinewidth=1.5,
    )

    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.set_title(
        "Validación: W Teórico vs W Simulado por Tipo de Usuario", fontsize=10
    )
    ax4.set_ylabel("Tiempo promedio en sistema W (min)")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # ---- Panel 5: Estadísticas cajeros (box simple) ----
    ax5 = fig.add_subplot(gs[1, 3])

    datos_cajeros = []
    for caj_idx in range(NUM_CAJEROS):
        ts = []
        for rep in resultados_cajeros[caj_idx]:
            ts.extend(rep["tiempos_sistema"].tolist())
        datos_cajeros.append(np.array(ts))

    bp = ax5.boxplot(
        datos_cajeros,
        labels=[f"Caj.{i+1}" for i in range(NUM_CAJEROS)],
        patch_artist=True,
        notch=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    cajero_colors = [COLORS["accent1"], COLORS["accent2"], COLORS["accent3"]]
    for patch, col in zip(bp["boxes"], cajero_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax5.set_title("Distribución\nTiempos por Cajero", fontsize=10)
    ax5.set_ylabel("Tiempo en sistema (min)")
    ax5.grid(True, alpha=0.3, axis="y")

    plt.savefig(
        "resumen_ejecutivo.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"]
    )
    plt.show()
    print("  [Figura] Resumen ejecutivo generado")


# ============================================================
# MÓDULO 5 - FUNCIÓN PRINCIPAL
# ============================================================


def main():
    print("\n" + "=" * 60)
    print("  LABORATORIO FINAL — SIMULACIÓN DE PARQUEADEROS")
    print("  Centro Comercial Supercentro")
    print("  Modelo M/M/1 | Python + NumPy + Matplotlib + SciPy")
    print("=" * 60 + "\n")

    # 0. Portada
    print("  Generando figuras...\n")
    plot_portada()

    # Ejecutar experimento principal
    resultados_por_tipo, resultados_cajeros, conteos_tipos_global = (
        ejecutar_experimento_completo()
    )

    # 1. Diagrama V&V (independiente de datos)
    print("  Generando diagrama V&V...")
    plot_diagrama_vv()

    # 2. Punto 5 — Estado estable (antes de analizar resultados)
    print("  Analizando estado estable...")
    puntos_corte = plot_estado_estable(resultados_por_tipo)

    # 3. Punto 1 — Estadísticas por cajero
    print("  Calculando estadísticas por cajero...")
    plot_estadisticas_cajeros(resultados_cajeros, puntos_corte)

    # 4. Punto 2 — Usuarios por tipo
    print("  Analizando usuarios por tipo...")
    plot_usuarios_por_tipo(resultados_cajeros, conteos_tipos_global)

    # 5. Punto 3 — Estrategia de mejora
    print("  Evaluando estrategia de mejora...")
    plot_estrategia_mejora(resultados_cajeros)

    # 6. Punto 4 — Validación M/M/1
    print("  Ejecutando validación M/M/1...")
    plot_validacion_mm1(resultados_por_tipo)

    # 7. Resumen ejecutivo
    print("  Generando resumen ejecutivo...")
    plot_resumen_ejecutivo(resultados_por_tipo, resultados_cajeros)

    # ---- Impresión de métricas teóricas ----
    print("\n" + "=" * 60)
    print("  MÉTRICAS TEÓRICAS M/M/1 POR TIPO DE USUARIO")
    print("=" * 60)
    print(
        f"  {'Tipo':<12} {'λ':>6} {'μ':>6} {'ρ':>6} {'L':>8} {'Lq':>8} {'W(min)':>9} {'Wq(min)':>9} {'P₀':>6}"
    )
    print("  " + "-" * 75)
    for tipo, params in USUARIOS.items():
        la = 1.0 / params["llegada"]
        mu = 1.0 / params["servicio"]
        m = calcular_metricas_mm1_teoricas(la, mu)
        w_str = f'{m["W"]:.3f}' if m["W"] != np.inf else "∞"
        wq_str = f'{m["Wq"]:.3f}' if m["Wq"] != np.inf else "∞"
        l_str = f'{m["L"]:.3f}' if m["L"] != np.inf else "∞"
        lq_str = f'{m["Lq"]:.3f}' if m["Lq"] != np.inf else "∞"
        print(
            f"  {params['label']:<12} {la:>6.4f} {mu:>6.4f} {m['rho']:>6.3f} "
            f"{l_str:>8} {lq_str:>8} {w_str:>9} {wq_str:>9} {m['P0']:>6.3f}"
        )

    print("\n" + "=" * 60)
    print("  SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print("  Archivos generados:")
    imgs = [
        "portada.png",
        "diagrama_vv.png",
        "estado_estable.png",
        "estadisticas_cajeros.png",
        "usuarios_por_tipo.png",
        "estrategia_mejora.png",
        "validacion_mm1.png",
        "resumen_ejecutivo.png",
    ]
    for img in imgs:
        print(f"    • {img}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
