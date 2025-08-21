import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import bitsandbytes as bnb
from trl import SFTTrainer
from huggingface_hub import login
import os

class ResearchPaperModelFinetuner:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", use_auth_token: str = None):
        """
        Initialize the model with authentication handling.
        
        Args:
            model_name: Name of the model to use (default: Mistral-7B)
            use_auth_token: Hugging Face authentication token (set via env var HF_TOKEN)
        """
        self.model_name = model_name
        
        # Handle authentication if token is provided
        if use_auth_token:
            try:
                login(token=use_auth_token)
            except Exception as e:
                print(f"Authentication failed: {e}")
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token if use_auth_token else None
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize base model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                use_auth_token=use_auth_token if use_auth_token else None
            )
            
            # Prepare model for QLoRA
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Define LoRA configuration
            self.lora_config = LoraConfig(
                r=16,  # rank
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Get PEFT model
            self.model = get_peft_model(self.model, self.lora_config)
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            # Fallback to a smaller model if the main one fails
            print("Falling back to a smaller model...")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize a fallback model that doesn't require authentication."""
        try:
            self.model_name = "facebook/opt-1.3b"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self.model = prepare_model_for_kbit_training(self.model)
            
            self.lora_config = LoraConfig(
                r=8,  # Smaller rank for the fallback model
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, self.lora_config)
            
        except Exception as e:
            print(f"Error initializing fallback model: {e}")
            raise RuntimeError("Could not initialize any model. Please check your environment and dependencies.")
    
    def prepare_training_data(self, research_papers: list, queries: list, responses: list):
        """
        Prepare training data for fine-tuning.
        
        Args:
            research_papers: List of research paper texts
            queries: List of user queries
            responses: List of model responses
        """
        # Create prompt template
        def create_prompt(paper, query, response):
            return f"""Research Paper: {paper}
            
User Query: {query}

Model Response: {response}"""
        
        # Create dataset
        prompts = [create_prompt(p, q, r) for p, q, r in zip(research_papers, queries, responses)]
        dataset = Dataset.from_dict({"text": prompts})
        
        return dataset
    
    def train(self, 
              train_dataset: Dataset,
              output_dir: str = "./research_paper_model",
              num_train_epochs: int = 3,
              per_device_train_batch_size: int = 4,
              gradient_accumulation_steps: int = 4,
              learning_rate: float = 2e-4,
              max_steps: int = -1):
        """
        Train the model using LoRA/QLoRA.
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_steps=10,
            save_steps=100,
            fp16=True,
            optim="paged_adamw_8bit"
        )
        
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            packing=True
        )
        
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model(output_dir)
        
    def generate_response(self, research_paper: str, query: str, max_length: int = 512):
        """
        Generate response using the fine-tuned model.
        """
        prompt = f"""Research Paper: {research_paper}
        
User Query: {query}

Model Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Model Response:")[-1].strip()

if __name__ == "__main__":
    # Use Hugging Face token 
    # export HF_TOKEN="your_hf_token_here"
    finetuner = ResearchPaperModelFinetuner(use_auth_token=os.getenv("HF_TOKEN"))

    # Example training data
    research_papers = [
        "This paper discusses deep learning in computer vision...",
        "The study focuses on natural language processing techniques..."
    ]
    queries = [
        "What are the main contributions of this paper?",
        "How does this work compare to previous approaches?"
    ]
    responses = [
        "The main contributions include...",
        "This work improves upon previous approaches by..."
    ]
    
    # Prepare and train
    train_dataset = finetuner.prepare_training_data(research_papers, queries, responses)
    finetuner.train(train_dataset)
    
    # Generate response
    test_paper = "A new approach to machine learning..."
    test_query = "What is the novelty of this approach?"
    response = finetuner.generate_response(test_paper, test_query)
    print(response)
