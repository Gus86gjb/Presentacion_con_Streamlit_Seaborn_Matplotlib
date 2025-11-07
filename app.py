import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Propinas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Análisis de Comportamiento de Propinas")
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    tips = sns.load_dataset("tips")
    # Crear características adicionales para análisis
    tips['tip_percentage'] = (tips['tip'] / tips['total_bill']) * 100
    tips['day_order'] = tips['day'].map({'Thur': 1, 'Fri': 2, 'Sat': 3, 'Sun': 4})
    tips['meal_type'] = tips['time'].map({'Lunch': 'Almuerzo', 'Dinner': 'Cena'})
    return tips

tips = load_data()

# Sidebar para controles
st.sidebar.header("🎛️ Controles de Análisis")

# Filtros interactivos
st.sidebar.subheader("🔍 Filtros")
selected_days = st.sidebar.multiselect(
    "Días de la semana:",
    options=tips['day'].unique(),
    default=tips['day'].unique()
)

selected_time = st.sidebar.multiselect(
    "Horario:",
    options=tips['time'].unique(),
    default=tips['time'].unique()
)

# Filtrar datos
filtered_tips = tips[
    (tips['day'].isin(selected_days)) & 
    (tips['time'].isin(selected_time))
]

# Métricas principales
st.subheader("📈 Métricas Clave del Dataset")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Registros", f"{len(filtered_tips):,}")
with col2:
    st.metric("Facturación Total", f"${filtered_tips['total_bill'].sum():,.2f}")
with col3:
    st.metric("Propina Promedio", f"${filtered_tips['tip'].mean():.2f}")
with col4:
    st.metric("% Propina Promedio", f"{filtered_tips['tip_percentage'].mean():.1f}%")

st.markdown("---")

# SECCIÓN 1: ANÁLISIS DEMOGRÁFICO Y TEMPORAL
st.header("👥 Análisis Demográfico y Temporal")

col1, col2 = st.columns(2)

with col1:
    # Distribución por género
    fig, ax = plt.subplots(figsize=(10, 6))
    gender_counts = filtered_tips['sex'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    wedges, texts, autotexts = ax.pie(
        gender_counts.values, 
        labels=gender_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax.set_title('Distribución por Género', fontsize=14, fontweight='bold')
    plt.setp(autotexts, size=12, weight="bold", color='white')
    st.pyplot(fig)
    
    # Insights
    st.info(f"**Insight:** {gender_counts.idxmax()} representa el {gender_counts.max()/len(filtered_tips)*100:.1f}% de los clientes")

with col2:
    # Distribución por día y hora
    fig, ax = plt.subplots(figsize=(10, 6))
    day_time_counts = pd.crosstab(filtered_tips['day'], filtered_tips['time'])
    day_time_counts.plot(kind='bar', ax=ax, color=['#FFD166', '#06D6A0'])
    ax.set_title('Distribución por Día y Horario', fontsize=14, fontweight='bold')
    ax.set_xlabel('Día de la Semana')
    ax.set_ylabel('Número de Transacciones')
    ax.legend(title='Horario')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Encontrar combinación más popular
    most_common = day_time_counts.stack().idxmax()
    st.info(f"**Insight:** {most_common[1]} los {most_common[0]} es el horario más concurrido")

# SECCIÓN 2: ANÁLISIS FINANCIERO
st.header("💰 Análisis Financiero")

col1, col2 = st.columns(2)

with col1:
    # Relación factura-propina con análisis de segmentos
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Crear segmentos por monto de factura
    filtered_tips['bill_segment'] = pd.cut(
        filtered_tips['total_bill'], 
        bins=[0, 20, 40, 60, 100],
        labels=['<20', '20-40', '40-60', '>60']
    )
    
    scatter = sns.scatterplot(
        data=filtered_tips, 
        x='total_bill', 
        y='tip', 
        hue='bill_segment',
        size='size',
        sizes=(20, 200),
        alpha=0.7,
        ax=ax
    )
    ax.set_title('Relación: Factura Total vs Propina', fontsize=14, fontweight='bold')
    ax.set_xlabel('Factura Total ($)')
    ax.set_ylabel('Propina ($)')
    st.pyplot(fig)
    
    # Calcular correlación
    correlation = filtered_tips['total_bill'].corr(filtered_tips['tip'])
    st.success(f"**Correlación:** {correlation:.3f} - Relación positiva fuerte entre factura y propina")

with col2:
    # Porcentaje de propina por categorías
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calcular porcentaje promedio por día y horario
    tip_percentage_by_category = filtered_tips.groupby(['day', 'time'])['tip_percentage'].mean().unstack()
    
    sns.heatmap(
        tip_percentage_by_category, 
        annot=True, 
        fmt='.1f', 
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': '% de Propina'}
    )
    ax.set_title('Porcentaje de Propina Promedio por Día y Horario', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    # Encontrar mejor y peor porcentaje
    max_percentage = tip_percentage_by_category.max().max()
    min_percentage = tip_percentage_by_category.min().min()
    st.info(f"**Rango de % propina:** {min_percentage:.1f}% a {max_percentage:.1f}%")

# SECCIÓN 3: COMPORTAMIENTO DE PROPINAS
st.header("🎯 Análisis de Comportamiento de Propinas")

col1, col2 = st.columns(2)

with col1:
    # Distribución del porcentaje de propina
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(
        data=filtered_tips, 
        x='tip_percentage', 
        hue='sex',
        kde=True,
        bins=20,
        ax=ax
    )
    ax.set_title('Distribución del Porcentaje de Propina por Género', fontsize=14, fontweight='bold')
    ax.set_xlabel('Porcentaje de Propina (%)')
    ax.set_ylabel('Frecuencia')
    
    # Añadir líneas de referencia
    mean_percentage = filtered_tips['tip_percentage'].mean()
    ax.axvline(mean_percentage, color='red', linestyle='--', label=f'Promedio: {mean_percentage:.1f}%')
    ax.legend()
    
    st.pyplot(fig)
    
    # Estadísticas de porcentaje
    st.metric("Porcentaje de Propina Promedio", f"{mean_percentage:.1f}%")

with col2:
    # Comportamiento de fumadores vs no fumadores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Propina promedio por fumador
    smoker_tips = filtered_tips.groupby('smoker')['tip_percentage'].mean()
    colors = ['#FF9F1C', '#2EC4B6']
    ax1.bar(smoker_tips.index, smoker_tips.values, color=colors)
    ax1.set_title('Propina Promedio: Fumadores vs No Fumadores')
    ax1.set_ylabel('% de Propina Promedio')
    
    # Tamaño de grupo por fumador
    group_size_smoker = filtered_tips.groupby('smoker')['size'].mean()
    ax2.bar(group_size_smoker.index, group_size_smoker.values, color=colors)
    ax2.set_title('Tamaño Promedio de Grupo')
    ax2.set_ylabel('Personas por Grupo')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Insight sobre fumadores
    smoker_diff = smoker_tips['Yes'] - smoker_tips['No']
    if smoker_diff > 0:
        st.warning(f"**Insight:** Los fumadores dan {smoker_diff:.1f}% más de propina en promedio")

# SECCIÓN 4: ANÁLISIS ESTADÍSTICO AVANZADO
st.header("📊 Análisis Estadístico Avanzado")

col1, col2 = st.columns(2)

with col1:
    # Boxplot de propinas por día
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=filtered_tips, 
        x='day', 
        y='tip_percentage',
        hue='sex',
        ax=ax
    )
    ax.set_title('Distribución de % de Propina por Día y Género', fontsize=14, fontweight='bold')
    ax.set_xlabel('Día de la Semana')
    ax.set_ylabel('Porcentaje de Propina (%)')
    
    st.pyplot(fig)
    
    # Análisis de outliers
    Q1 = filtered_tips['tip_percentage'].quantile(0.25)
    Q3 = filtered_tips['tip_percentage'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = filtered_tips[
        (filtered_tips['tip_percentage'] < Q1 - 1.5*IQR) | 
        (filtered_tips['tip_percentage'] > Q3 + 1.5*IQR)
    ]
    st.info(f"**Outliers detectados:** {len(outliers)} registros con % de propina atípico")

with col2:
    # Análisis de tamaño de grupo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribución de tamaños de grupo
    size_distribution = filtered_tips['size'].value_counts().sort_index()
    ax1.bar(size_distribution.index, size_distribution.values, color='skyblue')
    ax1.set_title('Distribución de Tamaños de Grupo')
    ax1.set_xlabel('Tamaño del Grupo')
    ax1.set_ylabel('Frecuencia')
    
    # Propina promedio por tamaño de grupo
    tip_by_size = filtered_tips.groupby('size')['tip_percentage'].mean()
    ax2.plot(tip_by_size.index, tip_by_size.values, marker='o', linewidth=2, markersize=8)
    ax2.set_title('Propina Promedio por Tamaño de Grupo')
    ax2.set_xlabel('Tamaño del Grupo')
    ax2.set_ylabel('% de Propina Promedio')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Insight sobre tamaño de grupo
    if len(tip_by_size) > 1:
        size_corr = filtered_tips['size'].corr(filtered_tips['tip_percentage'])
        st.success(f"**Correlación tamaño-propina:** {size_corr:.3f}")

# SECCIÓN 5: RESUMEN EJECUTIVO
st.markdown("---")
st.header("🎯 Resumen Ejecutivo de Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Hallazgos Principales")
    
    insights = [
        f"• **Factura promedio:** ${filtered_tips['total_bill'].mean():.2f}",
        f"• **Propina promedio:** ${filtered_tips['tip'].mean():.2f} ({filtered_tips['tip_percentage'].mean():.1f}%)",
        f"• **Día más popular:** {filtered_tips['day'].mode().iloc[0]}",
        f"• **Horario más concurrido:** {filtered_tips['time'].mode().iloc[0]}",
        f"• **Tamaño promedio de grupo:** {filtered_tips['size'].mean():.1f} personas",
        f"• **Correlación factura-propina:** {filtered_tips['total_bill'].corr(filtered_tips['tip']):.3f}"
    ]
    
    for insight in insights:
        st.write(insight)

with col2:
    st.subheader("💡 Recomendaciones de Negocio")
    
    recommendations = [
        "🎯 **Enfoque en cenas de fin de semana** - Mayor volumen y propinas",
        "👥 **Grupos grandes** - Tienden a dar porcentajes de propina similares",
        "⏰ **Optimizar horarios** - Viernes y sábados por la noche son críticos",
        "📊 **Monitorear consistencia** - Variación significativa en % de propina entre días",
        "🎪 **Experiencia para fumadores** - Generan mayor % de propina en promedio"
    ]
    
    for rec in recommendations:
        st.write(rec)

# SECCIÓN 6: DATOS DETALLADOS
st.markdown("---")
st.header("📋 Datos Detallados")

with st.expander("🔍 Ver Dataset Completo con Análisis"):
    tab1, tab2, tab3 = st.tabs(["Datos Originales", "Estadísticas", "Top Registros"])
    
    with tab1:
        st.dataframe(filtered_tips, use_container_width=True)
    
    with tab2:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(filtered_tips.describe(), use_container_width=True)
    
    with tab3:
        st.subheader("Top 10 Propinas Más Generosas")
        top_tips = filtered_tips.nlargest(10, 'tip_percentage')[['total_bill', 'tip', 'tip_percentage', 'sex', 'day', 'time']]
        st.dataframe(top_tips, use_container_width=True)

# Pie de página
st.markdown("---")
st.markdown(
    "**Análisis creado con Streamlit, Seaborn y Matplotlib** • "
    "Dataset: 'tips' de Seaborn • "
    "📊 **Insights para optimización de negocio**"
)