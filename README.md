
# 👨‍👩‍👦‍👦 InteractEval 👨‍👩‍👦‍👦 


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code](https://img.shields.io/badge/Code-Python3.9-blue)](https://docs.python.org/3/license.html)
[![data](https://img.shields.io/badge/Data-SummEval-green)](https://github.com/Yale-LILY/SummEval.git)






## 📖 Overview

This study introduces **InteractEval**, a framework that integrates human expertise and Large Language Models (LLMs) using the Think-Aloud (TA) method to generate attributes for checklist-based text evaluation. By combining human flexibility and reasoning with LLM consistency, InteractEval outperforms other non-LLM based and LLM-based baselines across four distinct dimensions, including Coherence, Fluency, Consistency, and Relevance. The experiment also investigates the effectiveness of the TA method, showing that it promotes divergent thinking in both humans and LLMs, leading to the generation of a wider range of relevant attributes and enhance text evaluation performance. Comparative analysis reveals that humans excel at identifying attributes related to internal quality (Coherence and Fluency), while LLMs perform better in external alignment (Consistency and Relevance). Consequently, leveraging both humans and LLMs together yields superior evaluation outcomes. In other words, this study emphasizes the necessity of combining humans and LLMs in an automated checklist-based text evaluation framework.

## 📑 Paper
TBD

## ⭐ Main Feature

### Human-LLM Combination
- Combination of humans' thoughts and LLMs' thoughts

### Think Aloud (TA)
- Checklist construction based on Think Aloud process


## 💻 Getting Started


### Installation
```
accelerate
git+https://github.com/huggingface/transformers
jinja2>=3.1.0
openai==0.28.0
pandas
tiktoken
scipy
prettytable
google-generativeai
jupyter
anthropic
```

### How to Run (Evaluator: GPT-3.5-Turbo / Dimension: Coherence)
<pre>
pip install -r requirements.txt
python main.py --model_name gpt-3.5-Turbo --dimension coherence
</pre>

## 🔧 Stack
- **Language**: Python
- **Utilized LLMs**: GPT-4/3.5-Turbo, Gemini-1.5-Pro, Llama-3.1-8B-Instruct, Claude-3.5-Sonnet
- **Dependencies** : Refer to "requirements.txt"
- **Dataset** : SummEval


## Project Structure

<!-- ```markdown -->
<pre>
InteractEval
├──assets
├──data
│   ├── coherence
│   ├── consistency
│   ├── fluency
│   └── relevance
├──prompts
│   ├── checklist_construction
│   │   ├── attributes_clustering
│   │   ├── component_extraction
│   │   ├── question_generation
│   │   ├── question_validation
│   │   ├── sub_question_generation
│   ├── evaluation
│   ├── think_aloud
│   │   ├── claude
│   │   ├── gemini
│   │   ├── gpt
│   │   └── llama
├──src
└── think_aloud
</pre>


## <img width="24" height="24" src="https://img.icons8.com/emoji/48/llama-emoji.png" alt="llama-emoji"/> How to Run Think Aloud with LLama-3.1
<pre>
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
</pre>

