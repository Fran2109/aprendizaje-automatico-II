# -*- coding: utf-8 -*-
"""
Script de Auditoría y Métricas para Identificar Bancos Líderes en el Dataset de AML.
Calcula métricas operativas, de riesgo criminal, de volumen monetario y de conectividad de red.
Genera un ranking multidimensional y exporta la recomendación de bancos a CSV.
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

def buscar_dataset():
    """Localiza el archivo CSV del dataset."""
    candidatos = [
        Path(r"C:\content\aml_large\HI-Large_Trans.csv"),
        Path("/content/aml_large/HI-Large_Trans.csv"),
        Path("HI-Large_Trans.csv"),
        Path(r"C:\content\aml_medium\HI-Medium_Trans.csv"),
        Path(r"C:\colab\HI-Large_Trans.csv")
    ]
    for c in candidatos:
        if c.exists():
            return c
    return None

def analizar_bancos(ruta_csv, fecha_ini="2022/08/01", fecha_fin="2022/10/01", chunk_size=2000000):
    """
    Lee en streaming de chunks para máxima velocidad y cero saturación de RAM.
    Calcula métricas para cada banco tanto como emisor (From Bank) como receptor (To Bank).
    """
    print(f"\n Procesando archivo completo en streaming (bloques de {chunk_size:,}); ventana: {fecha_ini} a {fecha_fin}...")
    t0 = time.time()

    # Estructuras de acumulación
    trx_emitidas = defaultdict(int)
    trx_recibidas = defaultdict(int)
    lavado_emitido = defaultdict(int)
    lavado_recibido = defaultdict(int)
    monto_emitido = defaultdict(float)
    monto_lavado = defaultdict(float)
    contrapartes = defaultdict(set)

    filas_leidas = 0
    pares_chunks = []

    cols = ["Timestamp", "From Bank", "To Bank", "Amount Paid", "Is Laundering"]
    filas_totales = 0

    for chunk in pd.read_csv(ruta_csv, usecols=cols, chunksize=chunk_size):
        filas_totales += len(chunk)
        # Filtro de ventana temporal (formato 'yyyy/MM/dd HH:mm' -> comparable como texto)
        if fecha_ini or fecha_fin:
            mask = pd.Series(True, index=chunk.index)
            if fecha_ini:
                mask &= chunk["Timestamp"] >= fecha_ini
            if fecha_fin:
                mask &= chunk["Timestamp"] < fecha_fin
            chunk = chunk[mask]
            if chunk.empty:
                print(f"   -> Leídas {filas_totales:,} filas totales, 0 en ventana ({time.time() - t0:.1f}s)...")
                continue
        # 1. Conteo de transacciones
        for b, c in chunk["From Bank"].value_counts().items():
            trx_emitidas[b] += c
        for b, c in chunk["To Bank"].value_counts().items():
            trx_recibidas[b] += c

        # 2. Montos
        montos_orig = chunk.groupby("From Bank")["Amount Paid"].sum()
        for b, val in montos_orig.items():
            monto_emitido[b] += val

        # 3. Lavado (positivos)
        chunk_lavado = chunk[chunk["Is Laundering"] == 1]
        if not chunk_lavado.empty:
            for b, c in chunk_lavado["From Bank"].value_counts().items():
                lavado_emitido[b] += c
            for b, c in chunk_lavado["To Bank"].value_counts().items():
                lavado_recibido[b] += c

            montos_lav = chunk_lavado.groupby("From Bank")["Amount Paid"].sum()
            for b, val in montos_lav.items():
                monto_lavado[b] += val

        # 5. Agregado por par (From, To) para analizar relaciones entre bancos
        g = chunk.groupby(["From Bank", "To Bank"]).agg(
            Trx=("Is Laundering", "size"),
            Lavado=("Is Laundering", "sum"),
            Monto=("Amount Paid", "sum"),
        ).reset_index()
        pares_chunks.append(g)
        if len(pares_chunks) >= 10:  # consolidar para limitar memoria
            pares_chunks = [pd.concat(pares_chunks, ignore_index=True)
                            .groupby(["From Bank", "To Bank"], as_index=False)[["Trx", "Lavado", "Monto"]].sum()]

        filas_leidas += len(chunk)
        print(f"   -> Leídas {filas_totales:,} filas totales, {filas_leidas:,} en ventana ({time.time() - t0:.1f}s)...")

    print(f" Análisis completado en {time.time() - t0:.2f} segundos.")

    # Conectividad interbancaria a partir de los pares únicos
    pares_tmp = pd.concat(pares_chunks, ignore_index=True)
    inter_tmp = pares_tmp[pares_tmp["From Bank"] != pares_tmp["To Bank"]][["From Bank", "To Bank"]].drop_duplicates()
    for a, b in zip(inter_tmp["From Bank"].to_numpy(), inter_tmp["To Bank"].to_numpy()):
        contrapartes[a].add(b)
        contrapartes[b].add(a)
    del pares_tmp, inter_tmp

    # Consolidar todas las métricas en un DataFrame unificado por banco
    todos_los_bancos = set(trx_emitidas.keys()).union(set(trx_recibidas.keys()))
    registros = []

    for b in todos_los_bancos:
        em = trx_emitidas[b]
        rec = trx_recibidas[b]
        tot = em + rec
        lav_em = lavado_emitido[b]
        lav_rec = lavado_recibido[b]
        lav_tot = lav_em + lav_rec
        tasa_lav = (lav_tot / tot * 100) if tot > 0 else 0.0
        m_tot = monto_emitido[b]
        m_lav = monto_lavado[b]
        conectividad = len(contrapartes[b])

        registros.append({
            "Banco_ID": b,
            "Trx_Total": tot,
            "Trx_Emitidas": em,
            "Trx_Recibidas": rec,
            "Lavado_Casos_Total": lav_tot,
            "Lavado_Emitido": lav_em,
            "Lavado_Recibido": lav_rec,
            "Tasa_Lavado_Pct": tasa_lav,
            "Monto_Total_USD": m_tot,
            "Monto_Lavado_USD": m_lav,
            "Conectividad_Red_Grado": conectividad
        })

    df = pd.DataFrame(registros)

    # Cálculo de un Score de Liderazgo Multidimensional (0 a 100)
    # 40% Casos de Lavado (crucial para que el ML tenga etiquetas positivas)
    # 30% Volumen de Transacciones (relevancia y representatividad)
    # 30% Conectividad de Red (relevancia para los Grafos GAT/GraphSAGE)
    max_lav = df["Lavado_Casos_Total"].max() or 1
    max_trx = df["Trx_Total"].max() or 1
    max_con = df["Conectividad_Red_Grado"].max() or 1

    df["Score_Liderazgo"] = (
        0.40 * (df["Lavado_Casos_Total"] / max_lav) +
        0.30 * (df["Trx_Total"] / max_trx) +
        0.30 * (df["Conectividad_Red_Grado"] / max_con)
    ) * 100

    pares = (pd.concat(pares_chunks, ignore_index=True)
             .groupby(["From Bank", "To Bank"], as_index=False)[["Trx", "Lavado", "Monto"]].sum())
    return df, pares

def analizar_conectividad_top(df, pares, top_n=10):
    """
    Analiza la relación ENTRE los bancos del top: matriz de transacciones/lavado,
    variables de conectividad interna y comparación de filtros OR vs AND.
    Agrega a df: Conectividad_Interna, Pct_Trx_Interna, Trx_Interna.
    """
    top = df.sort_values("Score_Liderazgo", ascending=False).head(top_n)["Banco_ID"].tolist()
    top_set = set(top)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n" + "=" * 90)
    print(f" 5. CONECTIVIDAD ENTRE LOS TOP {top_n} BANCOS (coherencia del grafo)")
    print("=" * 90)

    inter = pares[pares["From Bank"].isin(top_set) & pares["To Bank"].isin(top_set)]

    mat_trx = inter.pivot_table(index="From Bank", columns="To Bank", values="Trx",
                                aggfunc="sum", fill_value=0).reindex(index=top, columns=top, fill_value=0)
    mat_lav = inter.pivot_table(index="From Bank", columns="To Bank", values="Lavado",
                                aggfunc="sum", fill_value=0).reindex(index=top, columns=top, fill_value=0)
    print("\n Matriz de transacciones (filas = emisor, columnas = receptor):")
    print(mat_trx.to_string())
    print("\n Matriz de casos de lavado (filas = emisor, columnas = receptor):")
    print(mat_lav.astype(int).to_string())

    # Variables por banco: conectividad interna al top
    filas = []
    for b in top:
        sal = pares[pares["From Bank"] == b]
        ent = pares[pares["To Bank"] == b]
        total = sal["Trx"].sum() + ent["Trx"].sum()
        # excluye autoloops (mismo banco) del cálculo de vínculos con otros líderes
        sal_top = sal[sal["To Bank"].isin(top_set) & (sal["To Bank"] != b)]
        ent_top = ent[ent["From Bank"].isin(top_set) & (ent["From Bank"] != b)]
        vecinos = set(sal_top["To Bank"]).union(ent_top["From Bank"])
        trx_int = sal_top["Trx"].sum() + ent_top["Trx"].sum()
        filas.append({
            "Banco_ID": b,
            "Conectividad_Interna": len(vecinos),
            f"de_{top_n - 1}_posibles": top_n - 1,
            "Trx_Interna_Top": int(trx_int),
            "Pct_Trx_Interna": round(trx_int / total * 100, 2) if total else 0.0,
            "Lavado_Interno_Top": int(sal_top["Lavado"].sum() + ent_top["Lavado"].sum()),
        })
    df_int = pd.DataFrame(filas)
    print("\n Conectividad interna por banco líder:")
    print(df_int.to_string(index=False))

    # Densidad de la subred de líderes
    aristas = inter[inter["From Bank"] != inter["To Bank"]][["From Bank", "To Bank"]].drop_duplicates()
    posibles = top_n * (top_n - 1)
    densidad = len(aristas) / posibles if posibles else 0
    print(f"\n Densidad de la subred top (aristas dirigidas / posibles): {len(aristas)}/{posibles} = {densidad:.2%}")

    # Comparación de filtros
    or_mask = pares["From Bank"].isin(top_set) | pares["To Bank"].isin(top_set)
    and_mask = pares["From Bank"].isin(top_set) & pares["To Bank"].isin(top_set)
    res = pd.DataFrame({
        "Filtro": ["OR (toca algún líder)", "AND (ambos extremos líderes)"],
        "Transacciones": [int(pares[or_mask]["Trx"].sum()), int(pares[and_mask]["Trx"].sum())],
        "Lavado": [int(pares[or_mask]["Lavado"].sum()), int(pares[and_mask]["Lavado"].sum())],
        "Pares_unicos": [int(or_mask.sum()), int(and_mask.sum())],
        "Bancos_en_grafo": [
            len(set(pares[or_mask]["From Bank"]).union(pares[or_mask]["To Bank"])),
            len(set(pares[and_mask]["From Bank"]).union(pares[and_mask]["To Bank"])),
        ],
    })
    print("\n Comparación de filtros para construir el grafo:")
    print(res.to_string(index=False))

    nodos_or = res.loc[0, "Bancos_en_grafo"]
    print(f"\n Con OR el grafo incluye {nodos_or:,} bancos, de los cuales solo {top_n} son líderes "
          f"({top_n / nodos_or:.2%}); el resto son bancos periféricos.")
    if densidad < 0.5:
        print(" [AVISO] Subred de líderes poco densa: el filtro AND dejaría un grafo disperso; "
              "OR conserva más contexto pero con muchos nodos periféricos.")

    # Agregar variables al DataFrame principal
    df = df.merge(df_int[["Banco_ID", "Conectividad_Interna", "Trx_Interna_Top", "Pct_Trx_Interna"]],
                  on="Banco_ID", how="left")
    return df

def mostrar_reporte(df, pares):
    """Muestra rankings y tablas comparativas."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n" + "=" * 90)
    print(" REPORTE Y MÉTRICAS DE DEFINICIÓN DE BANCOS LÍDERES")
    print("=" * 90)
    print(f"Total de bancos únicos identificados: {len(df):,}")

    # 1. Top 10 por Casos de Lavado
    print("\n" + "-" * 90)
    print(" 1. TOP 10 BANCOS POR CASOS DE LAVADO (Métrica Crítica para Machine Learning)")
    print("    -> Garantiza que los modelos tengan suficientes positivos en Train, Val y Test.")
    print("-" * 90)
    top_lavado = df.sort_values(by="Lavado_Casos_Total", ascending=False).head(10)
    print(top_lavado[["Banco_ID", "Lavado_Casos_Total", "Lavado_Emitido", "Lavado_Recibido", "Trx_Total", "Tasa_Lavado_Pct"]].to_string(index=False))

    # 2. Top 10 por Volumen Operativo
    print("\n" + "-" * 90)
    print(" 2. TOP 10 BANCOS POR VOLUMEN DE TRANSACCIONES (Importancia Sistémica en el Mercado)")
    print("    -> Mide los bancos con mayor presencia operativa en el ecosistema.")
    print("-" * 90)
    top_volumen = df.sort_values(by="Trx_Total", ascending=False).head(10)
    print(top_volumen[["Banco_ID", "Trx_Total", "Trx_Emitidas", "Trx_Recibidas", "Monto_Total_USD", "Lavado_Casos_Total"]].to_string(index=False))

    # 3. Top 10 por Conectividad de Red
    print("\n" + "-" * 90)
    print(" 3. TOP 10 BANCOS POR CONECTIVIDAD DE RED (Métrica Crítica para Redes de Grafos)")
    print("    -> Cantidad de bancos contrapartes distintos con los que intercambian transferencias.")
    print("-" * 90)
    top_conectividad = df.sort_values(by="Conectividad_Red_Grado", ascending=False).head(10)
    print(top_conectividad[["Banco_ID", "Conectividad_Red_Grado", "Trx_Total", "Lavado_Casos_Total"]].to_string(index=False))

    # 4. Ranking Final de Liderazgo Multidimensional
    print("\n" + "=" * 90)
    print(" 4. RANKING COMPUESTO: TOP 10 BANCOS LÍDERES SUGERIDOS")
    print("    (Ponderación: 40% Casos Lavado + 30% Volumen Trx + 30% Conectividad de Red)")
    print("=" * 90)
    top_lideres = df.sort_values(by="Score_Liderazgo", ascending=False).head(10)
    print(top_lideres[["Banco_ID", "Score_Liderazgo", "Lavado_Casos_Total", "Trx_Total", "Conectividad_Red_Grado", "Tasa_Lavado_Pct"]].to_string(index=False))

    # 5. Conectividad entre los bancos del top
    df = analizar_conectividad_top(df, pares, top_n=10)

    # Guardar en archivo
    salida_csv = Path(r"c:\colab\ranking_bancos_metricas.csv")
    df.sort_values(by="Score_Liderazgo", ascending=False).to_csv(salida_csv, index=False)
    print(f"\n Ranking completo exportado a: {salida_csv.resolve()}")

    # Código listo para copiar y pegar en el notebook
    bancos_top5 = top_lideres["Banco_ID"].head(5).tolist()
    bancos_top10 = top_lideres["Banco_ID"].head(10).tolist()

    print("\n" + "=" * 90)
    print(" CÓDIGO SUGERIDO PARA FILTRAR EN TU NOTEBOOK (CELDA 9):")
    print("=" * 90)
    print("# Lista de Bancos Líderes seleccionados:")
    print(f"bancos_lideres = {bancos_top5}  # Top 5 Bancos Líderes")
    print("\n# Filtro en Spark:")
    print("df_3m = (")
    print("    df_large")
    print("    .withColumn('timestamp_parsed', F.to_timestamp('Timestamp', 'yyyy/MM/dd HH:mm'))")
    print("    .filter(")
    print("        (F.col('timestamp_parsed') >= F.lit('2022-08-01 00:00:00')) &")
    print("        (F.col('timestamp_parsed') < F.lit('2022-10-01 00:00:00')) &")
    print("        ((F.col('From Bank').isin(bancos_lideres)) | (F.col('To Bank').isin(bancos_lideres)))")
    print("    )")
    print("    .drop('timestamp_parsed')")
    print(")")
    print("=" * 90)

def main():
    print("=" * 90)
    print(" AUDITORÍA Y DEFINICIÓN DE BANCOS LÍDERES EN EL DATASET")
    print("=" * 90)

    ruta = buscar_dataset()
    if not ruta:
        print("[ERROR] No se encontró el dataset en las rutas estándar.")
        return

    print(f"Dataset encontrado: {ruta}")
    # Archivo completo, restringido a una ventana de 2 meses (ago-sep 2022).
    # Usar fecha_ini=None, fecha_fin=None para analizar todo el período.
    df_ranking, pares = analizar_bancos(ruta, fecha_ini="2022/08/01", fecha_fin="2022/10/01")
    mostrar_reporte(df_ranking, pares)

if __name__ == "__main__":
    main()
