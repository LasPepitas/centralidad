import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# Config basica de la pagina
st.set_page_config(
    page_title="YouTube-8M - Network Centrality",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS rapidos para emprolijar la UI
st.markdown("""
<style>
    .titulo-principal {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.3rem;
    }
    .subtitulo {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .tarjeta-metrica {
        background-color: #f3f4f6;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
    }
    .metrica-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metrica-valor {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# cacheamos la lectura para no leer el CSV en cada interaccion
@st.cache_data
def cargar_datos():
    paths = [
        "data/vocabulary.csv",
        "../data/vocabulary.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "vocabulary.csv"),
        os.path.join(os.path.dirname(__file__), "data", "vocabulary.csv")
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            
            # PARCHE: el csv original tiene 56 filas con Name vacio (ej. linea 120).
            # Pandas los lee como NaN float y rompe los dicts de adyacencia de NetworkX (nan != nan).
            df = df.dropna(subset=['Name'])
            df = df[df['Name'].astype(str).str.strip() != '']
            
            # rellenar clasificaciones nulas
            df['Vertical1'] = df['Vertical1'].fillna('Otros')
            df['Vertical2'] = df['Vertical2'].fillna('')
            df['Vertical3'] = df['Vertical3'].fillna('')
            return df
    raise FileNotFoundError("No se encontro vocabulary.csv en las rutas tipicas.")

try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar dataset: {e}")
    st.stop()

# Header de la app
st.markdown('<div class="titulo-principal">🕸️ Analizador de Centralidad de Redes</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Visualizacion de grafos y calculo de metricas en base a etiquetas de YouTube-8M</div>', unsafe_allow_html=True)

# Info rapida de ayuda
with st.expander("❓ Guia rapida de las Metricas"):
    st.markdown("""
    - **Grado (Degree):** Cuantas conexiones directas tiene el nodo.
    - **Cercania (Closeness):** Que tan cerca esta un nodo del resto (centro geometrico).
    - **Intermediacion (Betweenness):** Cuantas veces el nodo es paso obligado (puente) en los caminos mas cortos.
    - **Eigenvector:** Influencia del nodo basada en la importancia de sus vecinos directos.
    """)

# Sidebar de controles
st.sidebar.header("🛠️ Controles del Grafo")

# Filtro por categoria
categorias = sorted([c for c in df['Vertical1'].unique() if c != 'Otros'])
cat_seleccionada = st.sidebar.selectbox(
    "Categoria (Vertical1)",
    categorias,
    index=categorias.index("Games") if "Games" in categorias else 0
)

# limite de nodos para que matplotlib no dibuje una mancha negra (hairball)
limite_nodos = st.sidebar.slider(
    "Nodos a mostrar (Top popularidad)",
    min_value=15,
    max_value=100,
    value=45,
    step=5
)

# metrica activa para el tamaño y color en el layout
metrica_activa = st.sidebar.selectbox(
    "Metrica de enfoque visual",
    ["Grado", "Cercania", "Intermediacion", "Eigenvector"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Ajustes del Spring Layout")

k_layout = st.sidebar.slider(
    "Fuerza de repulsion (k)",
    min_value=0.05,
    max_value=0.5,
    value=0.18,
    step=0.01
)

filtrar_etiquetas = st.sidebar.checkbox(
    "Limpiar etiquetas (ocultar de menor relevancia)",
    value=True
)

# Filtrado y slice de datos
df_filtrado = df[df['Vertical1'] == cat_seleccionada]
df_subset = df_filtrado.nlargest(limite_nodos, 'TrainVideoCount').copy()

@st.cache_data
def generar_grafo(datos_df):
    G = nx.Graph()
    records = datos_df.to_dict('records')
    
    # meter nodos
    for r in records:
        G.add_node(
            r['Name'],
            popularity=int(r['TrainVideoCount']),
            v2=r['Vertical2'],
            v3=r['Vertical3']
        )
        
    n = len(records)
    # conexion por verticals secundarios compartidos
    for i in range(n):
        for j in range(i + 1, n):
            u, v = records[i], records[j]
            if (u['Vertical2'] == v['Vertical2'] and u['Vertical2'] != '') or \
               (u['Vertical3'] == v['Vertical3'] and u['Vertical3'] != ''):
                G.add_edge(u['Name'], v['Name'])
                
    # conectar vecinos en popularidad para forzar un componente conexo minimo.
    # Evita islas aisladas que rompen closeness/eigenvector (matematica de grafos).
    ordenados = sorted(records, key=lambda x: x['TrainVideoCount'], reverse=True)
    for i in range(len(ordenados) - 1):
        G.add_edge(ordenados[i]['Name'], ordenados[i+1]['Name'])
        
    return G

G = generar_grafo(df_subset)

# Calculo de centralidades con networkx
deg_c = nx.degree_centrality(G)
clo_c = nx.closeness_centrality(G)
bet_c = nx.betweenness_centrality(G)

# control de convergencia para eigenvector
try:
    eig_c = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    eig_c = nx.degree_centrality(G) # fallback simple

# Armar dataframe final consolidado
df_res = pd.DataFrame({
    'Nodo': list(G.nodes()),
    'Popularidad (Videos)': [G.nodes[node]['popularity'] for node in G.nodes()],
    'Vertical 2': [G.nodes[node]['v2'] if G.nodes[node]['v2'] != '' else 'N/A' for node in G.nodes()],
    'Grado': [deg_c[node] for node in G.nodes()],
    'Cercania': [clo_c[node] for node in G.nodes()],
    'Intermediacion': [bet_c[node] for node in G.nodes()],
    'Eigenvector': [eig_c[node] for node in G.nodes()]
})

df_res_ordenado = df_res.sort_values(by=metrica_activa, ascending=False).reset_index(drop=True)

# Tabs principales
tab_red, tab_datos = st.tabs(["📊 Red Visual", "📈 Metricas y Resumen"])

with tab_red:
    st.subheader(f"Grafo de la red: {cat_seleccionada}")
    st.caption(f"Los nodos mas grandes tienen mayor {metrica_activa}. Colores en base a intensidad de centralidad.")
    
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=100)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    # semilla fija para que la red no salte de lado en cada click
    pos = nx.spring_layout(G, k=k_layout, seed=42)
    
    valores_metrica = np.array([df_res.loc[df_res['Nodo'] == node, metrica_activa].values[0] for node in G.nodes()])
    
    # escalar el tamaño entre un minimo y maximo visible
    if valores_metrica.max() == valores_metrica.min():
        node_sizes = [500] * len(G.nodes())
    else:
        node_sizes = 180 + 1320 * (valores_metrica - valores_metrica.min()) / (valores_metrica.max() - valores_metrica.min())
        
    # aristas finas y un poco traslucidas
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#d1d5db', width=1.1, alpha=0.55)
    
    # dibujar nodos usando colormap plasma
    nodos = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=valores_metrica,
        cmap=plt.cm.plasma,
        alpha=0.9,
        edgecolors='#4b5563',
        linewidths=0.8
    )
    
    # armar las etiquetas de texto
    labels = {}
    umbral = np.percentile(valores_metrica, 50) if filtrar_etiquetas else -1.0
    
    for node in G.nodes():
        val = df_res.loc[df_res['Nodo'] == node, metrica_activa].values[0]
        if val >= umbral:
            labels[node] = node
            
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8,
        font_weight='bold',
        font_color='#111827',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7)
    )
    
    # barrita de colores
    cbar = plt.colorbar(nodos, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(f'Escala de {metrica_activa}', fontsize=9, weight='bold', color='#374151')
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_datos:
    # 4 columnas de kpis globales
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="tarjeta-metrica">
            <div class="metrica-label">Nodos</div>
            <div class="metrica-value">{G.number_of_nodes()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="tarjeta-metrica">
            <div class="metrica-label">Aristas</div>
            <div class="metrica-value">{G.number_of_edges()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="tarjeta-metrica">
            <div class="metrica-label">Densidad</div>
            <div class="metrica-value">{nx.density(G):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        try:
            d = nx.diameter(G)
        except nx.NetworkXError:
            d = "N/A"
        st.markdown(f"""
        <div class="tarjeta-metrica">
            <div class="metrica-label">Diametro</div>
            <div class="metrica-value">{d}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🏆 Coeficientes e Indices de Centralidad")
    st.caption("Ordena la tabla haciendo clic en las cabeceras.")
    
    st.dataframe(
        df_res_ordenado.style.format({
            'Grado': '{:.4f}',
            'Cercania': '{:.4f}',
            'Intermediacion': '{:.4f}',
            'Eigenvector': '{:.4f}',
            'Popularidad (Videos)': '{:,}'
        }),
        use_container_width=True,
        height=380
    )
    
    st.markdown("---")
    st.markdown(f"### 🎯 Top 5 Nodos con mayor **{metrica_activa}**")
    
    top5 = df_res_ordenado.head(5)
    
    # Grafico de barra simple de matplotlib
    fig_bar, ax_bar = plt.subplots(figsize=(9, 3.2))
    fig_bar.patch.set_facecolor('#ffffff')
    ax_bar.set_facecolor('#ffffff')
    
    y_pos = np.arange(len(top5))
    bars = ax_bar.barh(y_pos, top5[metrica_activa], align='center', color='#4f46e5', alpha=0.8, edgecolor='#312e81')
    
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top5['Nodo'], fontsize=9, weight='bold', color='#1f2937')
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(f'Valor de {metrica_activa}', fontsize=9, color='#374151')
    
    # estetica minimalista para la barra
    for spine in ['top', 'right', 'bottom']:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines['left'].set_color('#9ca3af')
    
    # meter valores al final de la barra
    for b in bars:
        w = b.get_width()
        ax_bar.text(
            w + (top5[metrica_activa].max() * 0.01),
            b.get_y() + b.get_height()/2,
            f'{w:.4f}',
            ha='left', va='center', fontsize=9, weight='bold', color='#4f46e5'
        )
        
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)
