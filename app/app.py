import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración premium de la página de Streamlit
st.set_page_config(
    page_title="Centralidad de Redes - YouTube-8M",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para dar una apariencia moderna y limpia
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)

# 1. Carga optimizada de datos con caché
@st.cache_data
def load_data():
    paths_to_try = [
        "data/vocabulary.csv",
        "../data/vocabulary.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "vocabulary.csv"),
        os.path.join(os.path.dirname(__file__), "data", "vocabulary.csv")
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Carga robusta limpiando nulos/vacíos en la columna clave de nodos: Name
            df = df.dropna(subset=['Name'])
            df = df[df['Name'].astype(str).str.strip() != '']
            
            # Rellenar valores vacíos en las clasificaciones
            df['Vertical1'] = df['Vertical1'].fillna('Unknown')
            df['Vertical2'] = df['Vertical2'].fillna('')
            df['Vertical3'] = df['Vertical3'].fillna('')
            return df
    raise FileNotFoundError("No se encontró vocabulary.csv en las rutas configuradas.")

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.info("Asegúrate de que el archivo 'vocabulary.csv' esté en la carpeta 'data/' del proyecto.")
    st.stop()

# Títulos de la Aplicación
st.markdown('<div class="main-title">🕸️ Métricas de Centralidad y Visualización de Redes</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análisis de centralidades del vocabulario de etiquetas de YouTube-8M usando NetworkX y Matplotlib</div>', unsafe_allow_html=True)

# Explicación breve de las métricas para aportar valor educativo
with st.expander("ℹ️ ¿Qué significan las Métricas de Centralidad?"):
    st.markdown("""
    Las métricas de centralidad identifican los nodos más importantes o influyentes dentro de una red desde diferentes perspectivas:
    *   **Centralidad de Grado (Degree Centrality):** Mide la cantidad de conexiones directas que tiene un nodo. En nuestro caso, cuántas categorías secundarias comparte directamente con otras etiquetas populares.
    *   **Centralidad de Cercanía (Closeness Centrality):** Mide qué tan rápido se puede llegar desde un nodo a todos los demás de la red. Los nodos con alta cercanía están en el 'centro geométrico' del grafo.
    *   **Centralidad de Intermediación (Betweenness Centrality):** Mide cuántas veces un nodo actúa como 'puente' o paso obligado en los caminos más cortos entre cualquier par de nodos. Un valor alto indica que el nodo controla el flujo de información.
    *   **Centralidad de Eigenvector:** Mide la influencia de un nodo considerando no solo cuántos amigos tiene, sino qué tan influyentes son esos amigos. Ideal para identificar nodos conectados a grandes hubs.
    """)

# 2. Barra lateral de control (Parámetros interactivos)
st.sidebar.header("🛠️ Configuración de la Red")

# Obtener categorías únicas excluyendo vacíos
categories = sorted([cat for cat in df_raw['Vertical1'].unique() if cat != 'Unknown'])
selected_category = st.sidebar.selectbox(
    "1. Categoría Principal (Vertical)",
    categories,
    index=categories.index("Games") if "Games" in categories else 0
)

# Slider para controlar la cantidad de nodos (Top N por popularidad)
num_nodes = st.sidebar.slider(
    "2. Cantidad de Nodos (Top por videos)",
    min_value=15,
    max_value=100,
    value=45,
    step=5,
    help="Limita los nodos a los Top N más populares de la categoría elegida para garantizar legibilidad y velocidad."
)

# Selección de la métrica activa para la visualización del grafo
selected_metric = st.sidebar.selectbox(
    "3. Métrica de Enfoque (Visual)",
    ["Grado (Degree)", "Cercanía (Closeness)", "Intermediación (Betweenness)", "Eigenvector"],
    index=0
)

# Parámetros avanzados del layout de red
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parámetros del Layout")
layout_k = st.sidebar.slider(
    "Separación de nodos (k)",
    min_value=0.05,
    max_value=0.5,
    value=0.18,
    step=0.01,
    help="Controla la fuerza de repulsión entre los nodos en el Spring Layout."
)

hide_low_labels = st.sidebar.checkbox(
    "Ocultar etiquetas de baja centralidad",
    value=True,
    help="Oculta las etiquetas de los nodos menos influyentes para evitar la saturación visual del gráfico."
)

# 3. Preparación de datos y filtrado
df_filtered = df_raw[df_raw['Vertical1'] == selected_category]
df_subset = df_filtered.nlargest(num_nodes, 'TrainVideoCount').copy()

# 4. Construcción óptima del Grafo con NetworkX
@st.cache_data
def build_network(df_records):
    G = nx.Graph()
    records = df_records.to_dict('records')
    
    # Agregar todos los nodos
    for r in records:
        G.add_node(
            r['Name'],
            label=r['Name'],
            popularity=int(r['TrainVideoCount']),
            v2=r['Vertical2'],
            v3=r['Vertical3']
        )
        
    n = len(records)
    # Conectar si comparten subcategorías secundarias (Vertical2 o Vertical3)
    for i in range(n):
        for j in range(i + 1, n):
            u = records[i]
            v = records[j]
            shared_v2 = (u['Vertical2'] == v['Vertical2'] and u['Vertical2'] != '')
            shared_v3 = (u['Vertical3'] == v['Vertical3'] and u['Vertical3'] != '')
            if shared_v2 or shared_v3:
                G.add_edge(u['Name'], v['Name'])
                
    # Conectar nodos adyacentes en orden de popularidad para asegurar conectividad
    # y evitar islas desconectadas, permitiendo que Closeness y Betweenness tengan sentido completo
    sorted_records = sorted(records, key=lambda x: x['TrainVideoCount'], reverse=True)
    for i in range(len(sorted_records) - 1):
        G.add_edge(sorted_records[i]['Name'], sorted_records[i+1]['Name'])
        
    return G

G = build_network(df_subset)

# 5. Cálculo de las 4 métricas de centralidad obligatorias
degree_cent = nx.degree_centrality(G)
closeness_cent = nx.closeness_centrality(G)
betweenness_cent = nx.betweenness_centrality(G)
# max_iter=1000 previene problemas de convergencia matemática
try:
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    # Fallback robusto en caso de que no converja el método de potencia
    eigenvector_cent = nx.degree_centrality(G)

# Consolidar los resultados en el DataFrame para visualización
df_metrics = pd.DataFrame({
    'Etiqueta (Nodo)': list(G.nodes()),
    'Popularidad (Videos)': [G.nodes[node]['popularity'] for node in G.nodes()],
    'Categoría Secundaria': [G.nodes[node]['v2'] if G.nodes[node]['v2'] != '' else 'N/A' for node in G.nodes()],
    'Centralidad de Grado': [degree_cent[node] for node in G.nodes()],
    'Centralidad de Cercanía': [closeness_cent[node] for node in G.nodes()],
    'Centralidad de Intermediación': [betweenness_cent[node] for node in G.nodes()],
    'Centralidad de Eigenvector': [eigenvector_cent[node] for node in G.nodes()]
})

# Mapear la selección de métrica activa a la columna correspondiente
metric_map = {
    "Grado (Degree)": 'Centralidad de Grado',
    "Cercanía (Closeness)": 'Centralidad de Cercanía',
    "Intermediación (Betweenness)": 'Centralidad de Intermediación',
    "Eigenvector": 'Centralidad de Eigenvector'
}
active_metric_col = metric_map[selected_metric]

# Ordenar la tabla por la métrica activa por defecto
df_metrics_sorted = df_metrics.sort_values(by=active_metric_col, ascending=False).reset_index(drop=True)

# 6. Interfaz Principal con Pestañas (Tabs)
tab1, tab2 = st.tabs(["📊 Visualización de la Red", "📈 Tabla de Métricas y Estadísticas"])

# --- PESTAÑA 1: VISUALIZACIÓN ---
with tab1:
    st.subheader(f"Estructura de la Red: {selected_category}")
    st.caption(f"El tamaño de los nodos varía según su **{selected_metric}** y los colores representan el nivel de centralidad.")
    
    # Crear la figura de Matplotlib de forma limpia y moderna
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Distribución de nodos fija mediante semilla para evitar saltos al interactuar
    pos = nx.spring_layout(G, k=layout_k, seed=42)
    
    # Obtener valores de la métrica activa para escalar los nodos
    metric_values = np.array([df_metrics.loc[df_metrics['Etiqueta (Nodo)'] == node, active_metric_col].values[0] for node in G.nodes()])
    
    # Normalizar los valores para un escalado de tamaño armonioso (entre 200 y 1600 px)
    if metric_values.max() == metric_values.min():
        node_sizes = [600] * len(G.nodes())
    else:
        node_sizes = 200 + 1400 * (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min())
        
    # Dibujar aristas con transparencia elegante para evitar saturar el fondo
    nx.draw_networkx_edges(
        G, pos, 
        ax=ax, 
        edge_color='#D1D5DB', 
        width=1.2, 
        alpha=0.6
    )
    
    # Dibujar los nodos coloreados mediante mapa de calor según el valor de centralidad
    nodes_draw = nx.draw_networkx_nodes(
        G, pos, 
        ax=ax, 
        node_size=node_sizes, 
        node_color=metric_values, 
        cmap=plt.cm.plasma, 
        alpha=0.9,
        edgecolors='#4B5563',
        linewidths=1.0
    )
    
    # Crear etiquetas limpias
    labels = {}
    threshold = np.percentile(metric_values, 50) if hide_low_labels else -1.0
    
    for node in G.nodes():
        val = df_metrics.loc[df_metrics['Etiqueta (Nodo)'] == node, active_metric_col].values[0]
        if val >= threshold:
            labels[node] = node
            
    # Dibujar las etiquetas sobre los nodos
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels, 
        ax=ax, 
        font_size=9, 
        font_weight='bold', 
        font_color='#111827',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7)
    )
    
    # Añadir barra de color estilizada para interpretar las métricas
    cbar = plt.colorbar(nodes_draw, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f'Intensidad de {selected_metric}', fontsize=10, weight='bold', color='#374151')
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    
    # Limpiar bordes del eje de Matplotlib
    ax.axis('off')
    plt.tight_layout()
    
    # Mostrar el gráfico en Streamlit de forma fluida
    st.pyplot(fig)
    plt.close(fig)

# --- PESTAÑA 2: ESTADÍSTICAS Y DATOS ---
with tab2:
    col1, col2, col3, col4 = st.columns(4)
    
    # Métricas Globales del Grafo
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Nodos (Etiquetas)</div>
            <div class="metric-value">{G.number_of_nodes()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Conexiones (Aristas)</div>
            <div class="metric-value">{G.number_of_edges()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Densidad del Grafo</div>
            <div class="metric-value">{nx.density(G):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        # Al asegurar conectividad, el grafo siempre es conexo, por lo que el diámetro es válido
        try:
            diameter = nx.diameter(G)
        except nx.NetworkXError:
            diameter = "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Diámetro (Distancia Máx)</div>
            <div class="metric-value">{diameter}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🏆 Tabla Detallada de Métricas de Centralidad")
    st.caption("Podés hacer clic en los encabezados de las columnas para ordenar la red de acuerdo a cada tipo de centralidad.")
    
    # Mostrar el DataFrame formateado de manera elegante
    st.dataframe(
        df_metrics_sorted.style.format({
            'Centralidad de Grado': '{:.4f}',
            'Centralidad de Cercanía': '{:.4f}',
            'Centralidad de Intermediación': '{:.4f}',
            'Centralidad de Eigenvector': '{:.4f}',
            'Popularidad (Videos)': '{:,}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Resumen visual rápido de los Top 5 líderes
    st.markdown("---")
    st.markdown(f"### 🎯 Top 5 Nodos más Influyentes por **{selected_metric}**")
    
    top_5 = df_metrics_sorted.head(5)
    
    # Crear gráfico de barras estilizado de los Top 5
    fig_bar, ax_bar = plt.subplots(figsize=(10, 3.5))
    fig_bar.patch.set_facecolor('#FFFFFF')
    ax_bar.set_facecolor('#FFFFFF')
    
    y_pos = np.arange(len(top_5))
    bars = ax_bar.barh(y_pos, top_5[active_metric_col], align='center', color='#4F46E5', alpha=0.85, edgecolor='#312E81')
    
    # Personalizar ejes del gráfico de barra
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top_5['Etiqueta (Nodo)'], fontsize=10, weight='bold', color='#1F2937')
    ax_bar.invert_yaxis()  # El más alto arriba
    ax_bar.set_xlabel(f'Valor de {selected_metric}', fontsize=10, color='#374151')
    
    # Eliminar espinas para diseño ultra plano (flat design)
    for spine in ['top', 'right', 'bottom']:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines['left'].set_color('#9CA3AF')
    
    # Agregar etiquetas de valor al final de cada barra
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(
            width + (top_5[active_metric_col].max() * 0.01),
            bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            ha='left', va='center', fontsize=9, weight='bold', color='#4F46E5'
        )
        
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)
