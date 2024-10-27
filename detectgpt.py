import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DetectGPT:
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        batch_size: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        num_samples: int = 25  # Reduced from 100 for faster processing
    ):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.batch_size = batch_size
        self.num_samples = num_samples

    def get_log_probs(self, text: str) -> torch.Tensor:
        """Get log probabilities for each token in the text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            log_probs = torch.nn.functional.log_softmax(outputs.logits[0, :-1], dim=-1)
            token_log_probs = log_probs[range(len(log_probs)), inputs.input_ids[0, 1:]]
        return token_log_probs

    def generate_perturbations(self, text: str) -> List[str]:
        """Generate perturbations of the input text."""
        sentences = sent_tokenize(text)
        perturbed_texts = []
        
        for _ in range(self.num_samples):
            perturbed = []
            for sent in sentences:
                inputs = self.tokenizer(
                    sent,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=min(len(inputs.input_ids[0]) + 20, 512),
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                perturbed.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            perturbed_texts.append(" ".join(perturbed))
        
        return perturbed_texts

    def analyze_text(
        self,
        text: str,
        threshold: float = 0.5
    ) -> dict:
        """Analyze text and determine if it's likely AI-generated."""
        # Skip very short texts
        if len(text.split()) < 20:
            return {
                "score": 0.5,
                "is_ai_generated": None,
                "confidence": 0,
                "interpretation": "Text too short for reliable analysis",
                "metrics": {}
            }

        perturbed_texts = self.generate_perturbations(text)
        original_log_probs = self.get_log_probs(text).mean().item()
        
        perturbed_log_probs = []
        for i in range(0, len(perturbed_texts), self.batch_size):
            batch = perturbed_texts[i:i + self.batch_size]
            batch_log_probs = [self.get_log_probs(t).mean().item() for t in batch]
            perturbed_log_probs.extend(batch_log_probs)
        
        ratio = np.exp(original_log_probs) / np.mean([np.exp(p) for p in perturbed_log_probs])
        score = 1 / (1 + np.exp(-np.log(ratio)))
        
        result = {
            "score": float(score),
            "is_ai_generated": score > threshold,
            "confidence": abs(score - 0.5) * 2,
            "metrics": {
                "likelihood_ratio": float(ratio),
                "original_log_prob": float(original_log_probs),
                "perturbed_log_probs_mean": float(np.mean(perturbed_log_probs)),
                "perturbed_log_probs_std": float(np.std(perturbed_log_probs))
            }
        }
        
        # Add interpretation
        if score > 0.8:
            result["interpretation"] = "Very likely AI-generated"
        elif score > 0.6:
            result["interpretation"] = "Probably AI-generated"
        elif score > 0.4:
            result["interpretation"] = "Uncertain"
        elif score > 0.2:
            result["interpretation"] = "Probably human-written"
        else:
            result["interpretation"] = "Very likely human-written"
            
        return result

def process_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        import subprocess
        subprocess.run(["pdftotext", "-layout", file_path, "output.txt"], check=True)
        with open("output.txt", "r", encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect AI-generated text using DetectGPT")
    parser.add_argument("input_file", help="Path to input text file or PDF")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0-1)")
    args = parser.parse_args()

    # Initialize detector
    print("Initializing DetectGPT...")
    detector = DetectGPT()

    # Read input file
    print("Reading input file...")
    if args.input_file.endswith('.pdf'):
        text = process_pdf(args.input_file)
    else:
        with open(args.input_file, "r", encoding='utf-8') as f:
            text = f.read()

    if not text.strip():
        print("Error: Empty or unreadable input")
        return

    # Analyze text
    print("Analyzing text...")
    result = detector.analyze_text(text, args.threshold)

    # Print results
    print("\nResults:")
    print(f"AI Detection Score: {result['score']:.2f}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nDetailed Metrics:")
    for key, value in result['metrics'].items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()