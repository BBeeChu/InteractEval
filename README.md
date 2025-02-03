
# 👨‍👩‍👦‍👦 InteractEval 👨‍👩‍👦‍👦 


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code](https://img.shields.io/badge/Code-Python3.9-blue)](https://docs.python.org/3/license.html)
[![data](https://img.shields.io/badge/Data-SummEval-green)](https://github.com/Yale-LILY/SummEval.git)
[![data](https://img.shields.io/badge/Data-ELLIPSE-red)](https://github.com/scrosseye/ELLIPSE-Corpus.git)






## 📖 Overview
![Image](https://github.com/user-attachments/assets/6a8d587e-26bb-4f6b-8ef7-2d9e7eac1920)
This study introduces **InteractEval**, a framework that integrates the outcomes of Think-Aloud (TA) conducted by humans and LLMs to generate attributes for checklist-based text evaluation. By combining humans' flexibility and high-level reasoning with LLMs' consistency and extensive knowledge, InteractEval outperforms text evaluation baselines on a text summarization benchmark (SummEval) and an essay scoring benchmark (ELLIPSE). Furthermore, an in-depth analysis shows that it promotes divergent thinking in both humans and LLMs, leading to the generation of a wider range of relevant attributes and enhancement of text evaluation performance. A subsequent comparative analysis reveals that humans excel at identifying attributes related to internal quality (Coherence and Fluency), but LLMs perform better at those attributes related to external alignment (Consistency and Relevance). Consequently, leveraging both humans and LLMs together produces the best evaluation outcomes, highlighting the necessity of effectively combining humans and LLMs in an automated checklist-based text evaluation.

## 📑 Paper
**Think Together and Work Better: Combining Humans' and LLMs' Think-Aloud Outcomes for Effective Text Evaluation**  
*Seong Yeub Chu, Jong Woo Kim, Mun Yong Yi*  
CHI, 2025. [`arXiv`](https://arxiv.org/abs/2409.07355)

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

### How to Run (Evaluator: GPT-3.5-Turbo / Data: SummEval / Dimension: Coherence)
<pre>
pip install -r requirements.txt
generate a checklist by "./summeval_checklist_construction.ipynb"
python ./src/main.py --model_name gpt-3.5-Turbo --dimension coherence"
</pre>

## 🔧 Stack
- **Language**: Python
- **Utilized LLMs**: GPT-4/3.5-Turbo, Gemini-1.5-Pro, Llama-3.1-8B-Instruct, Claude-3.5-Sonnet
- **Dependencies** : Refer to "requirements.txt"
- **Dataset** : SummEval, ELLIPSE


## Project Structure

<!-- ```markdown -->
<pre>
InteractEval
├──assets
├──data
│   ├──ellipse
│   └──summeval
├──prompts
│   ├──ellipse
│   │   ├──checklist_construction
│   │   │   ├──attributes_clustering
│   │   │   ├──component_extraction
│   │   │   ├──question_generation
│   │   │   ├──question_validation
│   │   │   ├──sub_question_generation
│   │   ├──evaluation
│   │   ├──think_aloud
│   │   │   ├──claude
│   │   │   ├──gemini
│   │   │   ├──gpt
│   │   │   └──llama
│   └──summeval
├──src
├──summeval_think_aloud
└──ellipse_think_aloud
</pre>


## <img width="24" height="24" src="https://img.icons8.com/emoji/48/llama-emoji.png" alt="llama-emoji"/> How to Run Think Aloud with LLama-3.1
<pre>
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
</pre>

