import streamlit as st
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
import datetime
import os
from backend.core import run_llm

load_dotenv()

def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()},{question}->{answer}\n")

def load_history():
    if os.path.exist("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()
    return[]

def main():
    st.set_page_config(page_title="Agente de Python Interactivo",
                       layout="wide")
    st.title("Agente de Python Interactivo")
    st.markdown(
        """
        <style>
        .stApp{background-color:black;}
        .title{color:white;}
        .button{background-color:white; color:black; border-radius:5px;}
        .input{border 1px solid black; border radius:5px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    instrucciones = """
    - Siempre usa la herramienta, incluso si sabes la respuesta.
    - Debes usar codigo de Python para responder.
    - Eres un agente que puede escribir codigo
    - Solo responde la pregunta escribiendo codigo, incluso si sabes la respuesta.
    - Si no sabes la respuesta escribe "No se la respuesta".
    """
    st.markdown(instrucciones)

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instrucciones=instrucciones)
    st.write("Prompt cargando...")

    tools = [PythonREPLTool()]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agente = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agente_executor = AgentExecutor(
        agent=agente,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    st.markdown("### Ejemplos:")
    ejemplos = [
        "Calcula la suma de 2 y 3",
        "Haz una lista del 20 al 30",
        "Crea una funcion que calcule el factorial de un numero",
    ]

    example = st.selectbox("Selecciona un ejemplo:", ejemplos)

    if st.button("Ejecutar ejemplo"):
        user_input = example
        try:
            respuesta = agente_executor.invoke(input={"input": user_input, "instructions":instrucciones, "agent_scratchpad": ""})
            st.markdown("### Respuesta del agente: ")
            st.code(respuesta["output"], language="python")
        except ValueError as e:
            st.error(f"Error en el agente: {str(e)}")

    pregunta = st.text_input("Question", placeholder="Haz tu pregunta aqui...")

    if st.button("Ejecutar"):
        user_input2 = pregunta
        try:
            resultado = run_llm(query=user_input2)
            st.markdown("### Respuesta del agente: ")
            st.code(resultado["result"], language="python")
        except ValueError as e:
            st.error(f"Error en el agente: {str(e)}")

if __name__ == "__main__":
    main()