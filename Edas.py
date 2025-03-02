import streamlit as st 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 

st.set_page_config(
    page_title='EDAs', 
    page_icon=':bar_chart:'
)

st.title('Página 1 - EDAs')

if st.button('Volver a inicio'):
    st.switch_page('Home.py')


@st.cache_resource
def load_dataset():
    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/diamonds.csv'
    return pd.read_csv(url)

# Carga de datos
df = sns.load_dataset('diamonds').dropna()
st.title('Edas de Seaborn y Plotly en Streanlit')
st.dataframe(df)

# Filtros
st.header('Filtros globales')
st.subheader('Filtros categóricas')

# Filtro por color
color = df['color'].unique()
selected_color = st.multiselect('Selecciona el color deseado', options=color, default=color)

# Filtro por cut
cut = df['cut'].unique().tolist()
selected_cut = st.multiselect('Selecciona la categória deseada', options=cut, default=cut)

# Filtro por clarity

clarity = df['clarity'].unique()
selected_clarity = st.multiselect('Selecciona claridad deseada', options=clarity, default=clarity)

st.subheader('Filtros numéricos')

# Rango de precio
price_min, price_max = st.slider(
    'Selecciona rango  de precio (máximo y mínimo):', 
    min_value=df['price'].min(),
    max_value=df['price'].max(),
    value=(df['price'].min(), df['price'].max())

)

# Rango de peso
carat_min, carat_max = st.slider(
    'Selecciona rango  de peso (máximo y mínimo):', 
    min_value=df['carat'].min(),
    max_value=df['carat'].max(),
    value=(df['carat'].min(), df['carat'].max())

)


# Rango de profundidad
depth_min, depth_max = st.slider(
    'Selecciona rango  de profundidad (máximo y mínimo):', 
    min_value=df['depth'].min(),
    max_value=df['depth'].max(),
    value=(df['depth'].min(), df['depth'].max())

)

table_min, table_max = st.slider(
    'Selecciona rango de table', 
    min_value=df['table'].min(),
    max_value=df['table'].max(),
    value=(df['table'].min(), df['table'].max())
)

# Aplicar filtros
df_filtered = df[
    (df['cut'].isin(selected_cut)) &
    (df['color'].isin(selected_color)) &
    (df['clarity'].isin(selected_clarity)) &
    (df['carat'] >= carat_min) & (df['carat'] <= carat_max) &
    (df['depth'] >= depth_min) & (df['depth'] <= depth_max) &
    (df['table'] >= table_min) & (df['table'] <= table_max) &
    (df['price'] >= price_min) & (df['price'] <= price_max)
]

st.dataframe(df_filtered, use_container_width=True)


st.table(df_filtered.head())
st.write(f'Nº filas antes de filtrar: **{df.shape[0]}**')
st.write(f'Nº filas después de filtrar: **{df_filtered.shape[0]}**')
st.write(f'Nº filas eliminadas por filtro: **{df.shape[0] - df_filtered.shape[0]}**')




# Matplotlib
st.header('1. Gráficos de Matplotlib')
st.header('')

st.subheader('Histograma')
fig, ax = plt.subplots(figsize=(12,6))
ax.hist(df_filtered['price'], bins=20, color='skyblue', edgecolor= 'black')
ax.set_title('Histograma de price')
ax.set_xlabel('price')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

st.header('')

st.subheader('Boxplot')
fig, ax = plt.subplots(figsize=(12,6))
ax.boxplot(x=df_filtered['price'], showmeans=True)
ax.set_title('Boxplot de cut')
ax.set_xlabel('price')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

st.header('')

st.subheader('Subplots')
class_cut = {'Ideal': 'blue', 'Very Good': 'green', 'Premium': 'red', 'Good': 'yellow', 'Fair': 'orange'}
fig, ax = plt.subplots(figsize=(12,6))
for cut, color in class_cut.items():
    df_cut = df_filtered[df_filtered['cut'] == cut]
    ax.scatter(x=df_cut['price'], y=df_cut['color'], color=color, label=cut)
    
ax.set_title('Relación entre Precio y Color')
ax.set_xlabel('precio')
ax.set_ylabel('color')
ax.legend()
ax.grid()
st.pyplot(fig)

st.header('')

st.subheader('Subplots')
fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot(df_filtered, x='price', y='color', hue='cut')
ax.set_title('Relación entre price y color')
ax.set_xlabel('price')
ax.set_ylabel('color')
ax.legend()
ax.grid()
st.pyplot(fig)


st.header('')

# Seaborn
st.header('2. Gráficos de Seaborn')


fig, ax = plt.subplots(figsize=(6,4))
sns.kdeplot(df_filtered, x='price')
sns.rugplot(df_filtered, x='price')
sns.histplot(df_filtered, x='price')
ax.set_title('Análisis univariante price')
st.pyplot(fig)


st.header('')

st.header('Heatmap')
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(df_filtered.corr(numeric_only=True).round(2), annot=True, cmap='viridis')
ax.set_title('Correlaciones')
st.pyplot(fig)


st.header('')

# Plotly:
st.header('3. Gráficos de Plotly')
df_grouped = df_filtered.groupby('price', as_index=False)['carat'].mean()
st.table(df_grouped[:10])
# fig = px.bar(df_grouped)
st.header('')
st.header('Precio medio por peso del diamante')
fig = px.bar(df_grouped, x='price', y='carat', color='price')
st.plotly_chart(fig)

st.header('')

st.header('Gráfico scatter con facet_col: cut')
fig = px.scatter(df_filtered, x='depth', y='table', color='price', facet_col='cut', opacity=1)
st.plotly_chart(fig)

st.header('')

st.header('Gráfico con scatter')
fig = px.scatter(df_filtered, x='cut', y='clarity', color='price',size='carat', width=800, height=700)
st.plotly_chart(fig)

st.header('')

fig = px.scatter(df_filtered, x='depth', y='table', color='price', opacity=1)
st.plotly_chart(fig)

st.header('')

st.header('Descargar datos')
st.write('Descargar dataset original o con los filtros actuales')

col1, col2 = st.columns(2)

with col1:
    st.download_button(
    'Descargar datos originales',
    data=df.to_csv(index=False),
    file_name='diamonds_filtered.csv',
    mime='text/csv'
)
    
with col2:
    st.download_button(
    'Descargar datos filtrados',
    data=df_filtered.to_csv(index=False),
    file_name='diamonds_filtered.csv',
    mime='text/csv'
)

st.write()
