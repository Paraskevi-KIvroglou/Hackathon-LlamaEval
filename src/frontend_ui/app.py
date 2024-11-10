import streamlit as st
import data_provider as dp
import together_api 

models_dict = dp.get_Llama_Models()
names_models = dp.get_Llama_Models().keys()
api_models = dp.get_Llama_Models().values()

tasks_dict = dp.get_evaluation_tasks()
tasks_names = dp.get_evaluation_tasks().keys()
tasks = dp.get_evaluation_tasks().values()

st.set_page_config(page_title="LlamaEval", page_icon=":llama:", layout="wide")

# CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- HEADER  ----
with st.container():
    st.subheader("Welcome to :llama:")
    st.title("LlamaEval: Quick Evaluation Dashboard")
    st.write(
        "This tool allows you to input the Llama model output, and it will give you a basic evaluation score."
    )

with st.container():
    st.write("---")
    

# main
st.write("Select Llama models to evaluate and specify a task.")

# combo box for model selection
model_name = st.selectbox("Select Llama Models", names_models)
model = models_dict[model_name]

task_name = st.selectbox("Enter Evaluation Task", tasks_names)
task = tasks_dict[task_name]

# evaluation button
if st.button("Run Evaluation") and model and task:
    st.write("Evaluating selected models on the specified task...")
    
    evaluation_scores = together_api.evaluate_benchmarks(model, task)
    # Panel to show models response and output
    with st.expander(f"Model: {model} - Task: '{task}'"):  
        # Score Board Panel
        st.write("## Evaluation Scores")
        st.write("---")
    
        # scores, get them from the API response
        scores = dp.get_metrics_to_print(task=task, results= evaluation_scores)
        st.table(scores)
    
    #     # Display
    # score_board = {
    #     "Model": list(scores.keys()),
    #     "Score": list(scores.values())
    # }
    
    # # score table
    # st.table(score_board)