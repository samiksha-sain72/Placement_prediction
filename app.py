import streamlit as st
# CSS to hide the default sidebar (replace with the actual class or ID)

st.markdown("""
<style>
.st-emotion-cache-79elbk {
    # visibility: hidden; 
    display: none;
}
# #stSitebarNav{
#     # visibility: hidden;
#     display:none;
# }
# # #stSitebarUserContent{
# #     position: top;
# }
</style>
""", unsafe_allow_html=True)

from pages import Home, Prediction, Research, Feedback  # Corrected import statement

# Set page configuration to remove default sidebar


def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "Prediction", "Research", "Feedback"] 
    page = st.sidebar.selectbox("Go to", pages)

    if page == "Home":
        Home.show()
    elif page == "Prediction":
        Prediction.show()
    elif page == "Research":
        Research.show()
    # elif page == "About":
    #     about.show()
    elif page == "Feedback":
        Feedback.show()

if __name__ == "__main__":
    main()