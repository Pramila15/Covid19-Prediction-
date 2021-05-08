import streamlit as st
import pickle
import numpy as np

filename = 'finalized_model.sav'
# load the model from disk
model = pickle.load(open(filename, 'rb'))


def predict_corona(sore_throat, sense_of_taste, contact_indication):
    input = np.array([[sore_throat, sense_of_taste, contact_indication]]).astype(np.float64)
    prediction = model.predict_proba(input)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    st.title("Covid-19 Prediction Model")
    html_temp = """
    <div style="background-color:#ffffff ;padding:10px;">
    <h2 style="color:black;">Symtoms here - 0 - Yes , 1 - No</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    sorethroat = st.text_input("Sore Throat", "")
    senseoftaste = st.text_input("Sense of taste", "")
    contactindication = st.text_input("Fever", "")
    safe_html = """  
      <div style="background-color:#00FF00;padding:10px >
       <h2 style="color:white;text-align:center;">Negative</h2>
       </div>
    """
    danger_html = """  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;">Positive</h2>
       </div>
    """

    if st.button("Predict"):
        output = predict_corona(sorethroat, senseoftaste, contactindication)
        st.success('Results {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html, unsafe_allow_html=True)
        else:
            st.markdown(safe_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))