import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict
import nltk
from tqdm import tqdm
import re
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DetectGPT:
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        batch_size: int = 8,
        num_samples: int = 15
    ):
        print("Loading model and tokenizer...")
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.max_length = 512
        self.temperature = 0.7
        self.top_k = 50
        self.top_p = 0.95

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs intelligently."""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        max_words = 150
        split_paragraphs = []
        for para in paragraphs:
            words = para.split()
            if len(words) > max_words:
                sentences = nltk.sent_tokenize(para)
                current_chunk = []
                current_length = 0
                
                for sent in sentences:
                    sent_words = sent.split()
                    if current_length + len(sent_words) > max_words and current_chunk:
                        split_paragraphs.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_length = len(sent_words)
                    else:
                        current_chunk.append(sent)
                        current_length += len(sent_words)
                
                if current_chunk:
                    split_paragraphs.append(' '.join(current_chunk))
            else:
                split_paragraphs.append(para)
        
        return split_paragraphs

    @torch.no_grad()
    def get_log_probs_batch(self, texts: List[str]) -> torch.Tensor:
        """Process multiple texts in a batch."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        outputs = self.model(**inputs, labels=inputs.input_ids)
        log_probs = torch.nn.functional.log_softmax(outputs.logits[:, :-1], dim=-1)
        
        batch_log_probs = []
        for i in range(len(texts)):
            log_prob = log_probs[i][range(len(log_probs[i])), inputs.input_ids[i, 1:]]
            batch_log_probs.append(log_prob.mean().item())
            
        return torch.tensor(batch_log_probs)

    def analyze_paragraph(self, paragraph: str, pbar=None) -> Dict:
        """Analyze a single paragraph."""
        if pbar:
            pbar.set_description(f"Analyzing paragraph ({len(paragraph.split())} words)")
            
        if len(paragraph.split()) < 10:
            if pbar:
                pbar.update(1)
            return {
                "text": paragraph,
                "score": 0.5,
                "is_ai_generated": None,
                "confidence": 0,
                "interpretation": "Too short for analysis",
                "word_count": len(paragraph.split())
            }

        # Get original probability
        original_log_prob = self.get_log_probs_batch([paragraph])[0].item()
        
        # Generate perturbations with progress
        inputs = self.tokenizer(
            paragraph,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        perturbed_paragraphs = []
        for i in range(self.num_samples):
            if pbar:
                pbar.set_description(f"Generating perturbation {i+1}/{self.num_samples}")
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.max_length,
                do_sample=True,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            perturbed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            perturbed_paragraphs.append(perturbed)
        
        if pbar:
            pbar.set_description("Computing probabilities")
        
        # Get perturbed probabilities
        perturbed_log_probs = self.get_log_probs_batch(perturbed_paragraphs)
        
        # Calculate metrics
        ratio = np.exp(original_log_prob) / np.mean([np.exp(p) for p in perturbed_log_probs])
        score = 1 / (1 + np.exp(-np.log(ratio)))
        
        if pbar:
            pbar.update(1)
        
        # Determine interpretation
        if score > 0.8:
            interpretation = "Very likely AI-generated"
            is_ai = True
        elif score > 0.6:
            interpretation = "Probably AI-generated"
            is_ai = True
        elif score > 0.4:
            interpretation = "Uncertain"
            is_ai = None
        elif score > 0.2:
            interpretation = "Probably human-written"
            is_ai = False
        else:
            interpretation = "Very likely human-written"
            is_ai = False
            
        return {
            "text": paragraph,
            "score": float(score),
            "is_ai_generated": is_ai,
            "confidence": abs(score - 0.5) * 2,
            "interpretation": interpretation,
            "word_count": len(paragraph.split())
        }

    def analyze_text(self, text: str) -> Dict:
        """Analyze text with paragraph-level breakdown and progress bars."""
        print("Splitting text into paragraphs...")
        paragraphs = self.split_into_paragraphs(text)
        print(f"Found {len(paragraphs)} paragraphs")
        
        print("\nAnalyzing paragraphs...")
        paragraph_analyses = []
        
        # Create progress bar for paragraphs
        with tqdm(total=len(paragraphs), desc="Overall progress") as pbar:
            for paragraph in paragraphs:
                analysis = self.analyze_paragraph(paragraph, pbar)
                paragraph_analyses.append(analysis)
        
        print("\nCalculating final results...")
        # Calculate weighted overall score based on paragraph length
        total_words = sum(a["word_count"] for a in paragraph_analyses)
        if total_words > 0:
            overall_score = sum(a["score"] * a["word_count"] for a in paragraph_analyses) / total_words
        else:
            overall_score = 0.5
            
        return {
            "overall_score": float(overall_score),
            "paragraph_analyses": paragraph_analyses,
            "summary": {
                "total_paragraphs": len(paragraph_analyses),
                "total_words": total_words,
                "ai_likely": sum(1 for a in paragraph_analyses if a["is_ai_generated"] is True),
                "human_likely": sum(1 for a in paragraph_analyses if a["is_ai_generated"] is False),
                "uncertain": sum(1 for a in paragraph_analyses if a["is_ai_generated"] is None)
            }
        }

def format_results(results: Dict) -> str:
    """Format results for display."""
    output = []
    output.append("\nOverall Analysis:")
    output.append(f"Overall AI Score: {results['overall_score']:.2f}")
    
    # Interpret overall score
    if results['overall_score'] > 0.8:
        overall_interp = "Very likely AI-generated"
    elif results['overall_score'] > 0.6:
        overall_interp = "Probably AI-generated"
    elif results['overall_score'] > 0.4:
        overall_interp = "Uncertain or Mixed"
    elif results['overall_score'] > 0.2:
        overall_interp = "Probably human-written"
    else:
        overall_interp = "Very likely human-written"
    output.append(f"Overall Interpretation: {overall_interp}")
    
    output.append("\nSummary:")
    output.append(f"Total Paragraphs: {results['summary']['total_paragraphs']}")
    output.append(f"Total Words: {results['summary']['total_words']}")
    output.append(f"AI-generated Paragraphs: {results['summary']['ai_likely']}")
    output.append(f"Human-written Paragraphs: {results['summary']['human_likely']}")
    output.append(f"Uncertain Paragraphs: {results['summary']['uncertain']}")
    
    output.append("\nDetailed Paragraph Analysis:")
    for i, analysis in enumerate(results['paragraph_analyses'], 1):
        output.append(f"\n{i}. AI Score: {analysis['score']:.2f} - {analysis['interpretation']}")
        output.append(f"   Confidence: {analysis['confidence']:.2f}")
        output.append(f"   Word Count: {analysis['word_count']}")
        # Truncate very long paragraphs in display
        text = analysis['text']
        if len(text) > 100:
            text = text[:100] + "..."
        output.append(f"   Text: {text}")
    
    return "\n".join(output)

def main():
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Detect AI-generated text using DetectGPT")
    parser.add_argument("input_file", help="Path to input text file or PDF")
    args = parser.parse_args()

    start_time = time.time()
    
    print("\n=== Initializing DetectGPT ===")
    detector = DetectGPT()

    print("\n=== Reading Input File ===")
    if args.input_file.endswith('.pdf'):
        try:
            import subprocess
            print("Converting PDF to text...")
            subprocess.run(["pdftotext", "-layout", args.input_file, "output.txt"], check=True)
            with open("output.txt", "r", encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return
    else:
        with open(args.input_file, "r", encoding='utf-8') as f:
            text = f.read()

    if not text.strip():
        print("Error: Empty or unreadable input")
        return

    print("\n=== Starting Analysis ===")
    results = detector.analyze_text(text)
    
    print("\n=== Analysis Complete ===")
    # Print formatted results
    print(format_results(results))
    print(f"\nTotal time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()