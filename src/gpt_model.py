import re
import tiktoken
import openai



class Model():
    def __init__(self, model, temperature, api_key) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        print(self.model)
    
    def ask_chatgpt(self, prompt, repeat=None):
        if repeat:
            
            response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=prompt,
                    temperature=self.temperature,
                    api_key=self.api_key,
                    n=repeat
                )
            return response
        else:
            
            response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=prompt,
                    temperature=self.temperature,
                    api_key=self.api_key,
                )

            return response.choices[0].message.content
       
        
    
    def post_process(self, answer):
        answer = answer.replace('\n', ' ').replace('sql','').replace('```','')
        answer = re.sub('[ ]+', ' ', answer)
        answer = answer.strip()
        return answer