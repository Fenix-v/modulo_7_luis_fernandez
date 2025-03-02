import streamlit as st

st.set_page_config(
    page_title='Home', 
    page_icon=':house:'
)


st.title('MODULO 7 - STREAMLIT')
st.write('''
         Aplicacion de Streamlit para EDA, regresión y clasificación.
        ''')

col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Ir a EDAs'):
        st.switch_page('pages/1 - Edas.py')
        
with col2:
    if st.button('Ir a Regresión'):
        st.switch_page('pages/2 - Regresion.py')
        
with col3:
    if st.button('Ir a Clasificación'):
        st.switch_page('pages/3 - Clasificacion.py')
        
        
        