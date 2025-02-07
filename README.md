# Small-Language-QA-Model
import tkinter as tk
from tkinter import messagebox, scrolledtext
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np

class SmallLanguageModelQA:
    def _init_(self, model_name='distilbert-base-uncased-distilled-squad'):
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Freeze most layers for efficiency
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_answer(self, context, question):
        # Tokenize inputs
        inputs = self.tokenizer(
            question, 
            context, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True
        )
        
        # Get model outputs
        outputs = self.model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Find best answer span
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        
        # Convert tokens to answer
        answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
        answer = self.tokenizer.decode(answer_tokens)
        
        return answer.strip()

class QAApplication:
    def _init_(self, master):
        self.master = master
        master.title("Small Language Model Q&A")
        master.geometry("800x600")
        
        # Initialize QA Model
        self.qa_model = SmallLanguageModelQA()
        
        # Context Input
        tk.Label(master, text="Book Context:").pack(pady=5)
        self.context_text = scrolledtext.ScrolledText(
            master, 
            wrap=tk.WORD, 
            width=80, 
            height=10
        )
        self.context_text.pack(pady=10)
        
        # Question Input
        tk.Label(master, text="Ask a Question:").pack(pady=5)
        self.question_entry = tk.Entry(master, width=80)
        self.question_entry.pack(pady=5)
        
        # Answer Button
        self.ask_button = tk.Button(
            master, 
            text="Get Answer", 
            command=self.get_answer
        )
        self.ask_button.pack(pady=10)
        
        # Answer Display
        tk.Label(master, text="Answer:").pack(pady=5)
        self.answer_text = scrolledtext.ScrolledText(
            master, 
            wrap=tk.WORD, 
            width=80, 
            height=5
        )
        self.answer_text.pack(pady=10)
        
    def get_answer(self):
        context = self.context_text.get("1.0", tk.END).strip()
        question = self.question_entry.get().strip()
        
        if not context or not question:
            messagebox.showwarning("Warning", "Please enter both context and question.")
            return
        
        try:
            answer = self.qa_model.extract_answer(context, question)
            
            # Clear previous answer
            self.answer_text.delete("1.0", tk.END)
            
            # Display answer
            if answer:
                self.answer_text.insert(tk.END, answer)
            else:
                self.answer_text.insert(tk.END, "No answer found.")
        
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = QAApplication(root)
    root.mainloop()

if _name_ == "_main_":
    main()
