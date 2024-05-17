import streamlit as st

st.set_page_config(
    page_title="photojam-ARA-GROUP",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Add custom CSS to set the background image
st.markdown(
    """
    <style>
    body {
        background-image: url('photos/background.png');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# # Display logo at the top of the sidebar
# st.markdown(
#     """
#     <div style="display: flex; justify-content: center; padding-top: 10px;">
#         <img src="../photos/logo1.png" style="width: 200px;">
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# Display the logo image
st.sidebar.image("photos/logo1.png", width=200)

st.title("Welcome to my French Festival App")
st.header("By ARA group")
st.subheader("Arash, Ratha, Ahmed")


st.title("Image Gallery")

# Define a list of image file paths
image_paths = [
    "photos/background.png",
    # Add more image paths as needed
]

# Display each image
for image_path in image_paths:
    st.image(image_path, caption="Image", use_column_width=True)
