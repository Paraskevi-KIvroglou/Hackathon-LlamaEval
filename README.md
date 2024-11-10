# Llama Impact Hackathon

## LlamaEval: Quick Evaluation Dashboard
LlamaEval is a rapid prototype developed during a hackathon to provide a user-friendly dashboard for evaluating and comparing Llama models using the TogetherAI API.

Features

  Model Selection: Choose from various Llama models available through TogetherAI.
  Benchmark Tasks: Evaluate models on predefined tasks such as question answering and text summarization. 
  Performance Metrics: View accuracy, BLEU scores, and other relevant metrics for each model.
  User-Friendly Interface: Simple web interface for inputting prompts and viewing results.
  Quick Comparison: Easily compare the performance of different Llama models side-by-side.

Note

For the prototype, we kept the size of the benchmark small. In later, steps we plan to iterate on top of real-world datasets. 

Installation

  1.Clone the repository
     
  2.Install dependencies

    bash
    pip install -r requirements.txt
  3. Set up your TogetherAI API key:
    
    bash
    export TOGETHERAI_API_KEY=your_api_key_here

  Usage
  
  Run the application:

    bash
    streamlit run app.py

Open your web browser and navigate to http://localhost:8501.
Select a Llama model, choose a benchmark task, and input your prompt.
View the results and performance metrics on the dashboard.

Future Development

  - Custom dataset uploads
  
  - Support for additional AI models
  
  - Advanced visualization of performance metrics
  
  - Integration with other AI model providers

Contributors
  - [Paraskevi Kivroglou] [https://www.linkedin.com/in/paraskevi-kivroglou-925881292/] 
  - [Mayank Varshney]  [LinkedIn : https://www.linkedin.com/in/varsh-mayank/] 
  - [Amina Asif] [LinkedIn : https://www.linkedin.com/in/amina-work/] [GitHub:https://github.com/AminaAsif9]

License

This project is licensed under the MIT License - see the LICENSE file for details.

    
