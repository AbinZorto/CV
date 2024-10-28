# local_detection.py
import sys
import subprocess
from model import GPT2PPL
import json
from typing import Dict, Any
import torch

class TextDetector:
    def __init__(self):
        """Initialize the GPT2PPL model for text detection"""
        self.model = GPT2PPL(device=self.get_device())

    @staticmethod
    def get_device():
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file"""
        try:
            subprocess.run(["pdftotext", "-layout", pdf_path, "output.txt"], check=True)
            with open("output.txt", "r", encoding='utf-8') as file:
                return file.read()
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to extract text from PDF: {e}")
        except FileNotFoundError:
            raise Exception("pdftotext not found. Please install poppler-utils.")
        except Exception as e:
            raise Exception(f"Error reading PDF: {e}")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text using the GPT2PPL model"""
        try:
            results, explanation = self.model(text)
            
            if isinstance(results, dict) and "status" in results:
                return {
                    "status": "error",
                    "message": results["status"]
                }

            analysis = {
                "status": "finished",
                "results": {
                    "metrics": {
                        "perplexity": results.get("Perplexity", 0),
                        "perplexity_per_line": results.get("Perplexity per line", 0),
                        "burstiness": results.get("Burstiness", 0)
                    },
                    "classification": {
                        "is_ai_generated": results.get("label", -1) == 0,
                        "explanation": explanation
                    }
                },
                "raw_results": results
            }
            return analysis
        except Exception as e:
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }

def main(pdf_path: str):
    try:
        print("Initializing AI text detector...")
        detector = TextDetector()

        print("Extracting text from PDF...")
        text = detector.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print("Warning: Extracted text is empty")
            return

        print("Analyzing text...")
        results = detector.analyze_text(text)

        if results["status"] == "error":
            print(f"Error: {results['message']}")
            return

        print("\nDetection Results:")
        print(json.dumps(results, indent=2))

        classification = results["results"]["classification"]
        print("\nSummary:")
        print(f"AI Generated: {classification['is_ai_generated']}")
        print(f"Explanation: {classification['explanation']}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python local_detection.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    main(pdf_path)