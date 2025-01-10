import streamlit as st
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
print(sys.executable)

# Configuration globale
st.set_page_config(
  page_title="NBA Advanced EDA",
  page_icon='nba.svg',
  layout="centered",
  initial_sidebar_state="auto",
)

# Configuration personnalisée
st.markdown("""
    <style>
        [data-testid="stRadio"] label {
            color: white !important;  /* Changer la couleur du texte des labels */
        }
        [role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
            color: white !important;  /* Texte des éléments de la sidebar en blanc */
        }
        .block-container {
            max-width: 1400px !important;  /* Modifier cette valeur pour augmenter/diminuer la largeur */
        }
    </style>
""", unsafe_allow_html=True)

pages=[
"Home", 
"Introduction", 
"Exploratory Data Analysis", 
"Feature Engineering & preprocessing", 
"Modélisation",
"Players dashboard",
"Playground",
"Conclusion"
]

page = st.sidebar.radio("Navigation", pages)


if page == pages[0]:
    import home
    home.main()

elif page == pages[1]:
    import introduction
    introduction.main()

elif page == pages[2]:
    import eda
    eda.main()

elif page == pages[3]:
    import feature_engineering
    feature_engineering.main()

elif page == pages[4]:
    import models
    models.main()

elif page == pages[5]:
    import players_dashboard
    players_dashboard.main()

elif page == pages[6]:
    import playground
    playground.main()

elif page == pages[7]:
    import conclusion
    conclusion.main()
