# 📧 Email_Generator_Using_Langchain
This tool is a cold email generator designed specifically for service companies, using Groq, Langchain, and Streamlit. Users can input the URL of a company's careers page, and the tool extracts job listings from that page. It then generates personalized cold emails that include relevant portfolio links sourced from a vector database, tailored to the specific job descriptions.

![image](https://github.com/user-attachments/assets/942e656f-28cb-49a2-851f-0018a6adc364)

##Set-up
To get started we first need to get an API_KEY from here: https://console.groq.com/keys.  update the value of GROQ_API_KEY with the API_KEY you created.

To get started, first install the dependencies using:

 pip install -r requirements.txt
Run the streamlit app:

streamlit run app/main.py

