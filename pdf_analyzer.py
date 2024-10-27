import sys
from pathlib import Path
import re
from collections import defaultdict
import math
import zlib

class SensitiveDetector:
    def __init__(self):
        self.baselines = {
            'formal': "I am writing to express my interest in the position. My background includes relevant experience.",
            'technical': "Implemented solutions using Python and SQL. Developed efficient algorithms.",
            'casual': "I enjoy working on challenging problems. Learning new skills is important to me."
        }
        
        self.ai_patterns = [
            r'\b(therefore|thus|hence)\b',
            r'\b(clearly|obviously|evidently)\b',
            r'\b(specifically|particularly|especially)\b',
            r'\b(firstly|secondly|finally)\b',
            r'\b(demonstrate|showcase|highlight)\b',
            r'\b(expertise|proficiency|capability)\b',
            r'\b(passionate|enthusiastic|eager)\b'
        ]
    
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        return text.strip()
    
    def split_into_sentences(self, text):
        text = self.clean_text(text)
        # Improved sentence splitting
        sentences = []
        current = []
        
        # Split into potential sentences
        parts = re.split(r'([.!?])\s+', text)
        
        for i in range(0, len(parts)-1, 2):
            if i+1 < len(parts):
                current.append(parts[i] + parts[i+1])
            else:
                current.append(parts[i])
                
        for sent in current:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Ignore very short segments
                sentences.append(sent)
                
        return sentences if sentences else [text] if text else []
    
    def get_compression_ratio(self, text):
        if not text or len(text) < 10:
            return 0
        try:
            text_bytes = text.encode('utf-8')
            compressed = zlib.compress(text_bytes, level=9)
            return len(compressed) / len(text_bytes)
        except:
            return 0
    
    def get_entropy(self, text):
        if not text:
            return 0
        freq = defaultdict(int)
        for char in text.lower():
            freq[char] += 1
        
        length = len(text)
        entropy = 0
        for count in freq.values():
            prob = count / length
            entropy -= prob * math.log2(prob)
        return entropy
    
    def analyze_segment(self, text):
        if not text or len(text.strip()) < 10:
            return None
            
        text = self.clean_text(text)
        
        compression_ratio = self.get_compression_ratio(text)
        entropy = self.get_entropy(text)
        
        pattern_matches = sum(bool(re.search(pattern, text.lower())) 
                            for pattern in self.ai_patterns)
        pattern_score = pattern_matches / len(self.ai_patterns)
        
        similarities = []
        for baseline in self.baselines.values():
            base_comp = self.get_compression_ratio(baseline)
            base_entropy = self.get_entropy(baseline)
            similarity = 1 - (abs(compression_ratio - base_comp) + 
                            abs(entropy - base_entropy)) / 2
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        regularity_score = 1 - (entropy / 5.0)
        
        score = (
            compression_ratio * 0.3 +
            regularity_score * 0.3 +
            pattern_score * 0.2 +
            max_similarity * 0.2
        )
        
        flags = {
            'repetitive': compression_ratio > 0.5,
            'formal_patterns': pattern_score > 0.2,
            'low_entropy': entropy < 3.8,
            'high_similarity': max_similarity > 0.7
        }
        
        return {
            'text': text,
            'classification': 'AI' if score > 0.45 else 'Human',
            'confidence': score,
            'metrics': {
                'entropy': entropy,
                'compression_ratio': compression_ratio,
                'pattern_score': pattern_score,
                'baseline_similarity': max_similarity
            },
            'flags': flags
        }

def analyze_file(file_path):
    """Analyze either a PDF or text file and optionally modify tex files."""
    try:
        detector = SensitiveDetector()
        results = []
        original_text = ""
        
        # Read file content based on extension
        if file_path.lower().endswith('.tex'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                original_text = file.read()
                # Remove existing LaTeX markup for analysis
                text_for_analysis = re.sub(r'\\AIHighlight{([^}]*)}', r'\1', original_text)
        elif file_path.lower().endswith('.pdf'):
            # Read file content based on extension
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    try:
                        reader = PyPDF2.PdfReader(file)
                        text = ''
                        for page in reader.pages:
                            text += page.extract_text() + '\n'
                    except Exception as e:
                        print(f"Error reading PDF, trying alternate method: {str(e)}")
                        # If PDF reading fails, try reading as text
                        file.seek(0)
                        text = file.read().decode('utf-8', errors='ignore')
            except ImportError:
                print("PyPDF2 not installed, treating as text file")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
        else:
            # Read as text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
        
        # Analyze each sentence and track positions for tex files
        if file_path.lower().endswith('.tex'):
            modified_text = original_text
            sentences = detector.split_into_sentences(text_for_analysis)
            
            # Process sentences in reverse to maintain string positions
            for i, sentence in enumerate(sentences[::-1], 1):
                analysis = detector.analyze_segment(sentence)
                if analysis:
                    results.append({
                        'segment': len(sentences) - i + 1,
                        **analysis
                    })
                    
                    if analysis['classification'] == 'AI':
                        # Escape special LaTeX characters in the sentence
                        escaped_sentence = re.escape(sentence)
                        # Add highlighting command around the sentence
                        modified_text = re.sub(
                            f"(?<!\\\\AIHighlight{{){escaped_sentence}(?!}})",
                            f"\\\\AIHighlight{{{sentence}}}",
                            modified_text
                        )
            
            # Write modified tex file
            backup_path = file_path + '.backup'
            import shutil
            shutil.copy2(file_path, backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(modified_text)
            
            print(f"\nOriginal file backed up to: {backup_path}")
            print("LaTeX file updated with AI highlighting markup")
            
        else:
            # Original analysis for non-tex files
            sentences = detector.split_into_sentences(text)
            for i, sentence in enumerate(sentences, 1):
                analysis = detector.analyze_segment(sentence)
                if analysis:
                    results.append({
                        'segment': i,
                        **analysis
                    })
        
        return results
    
    except Exception as e:
        raise Exception(f"Error analyzing file: {str(e)}")

def print_results(results):
    """Print analysis results with improved formatting."""
    if not results:
        print("No text segments were successfully analyzed.")
        return
        
    print("\nDetailed Analysis Results:")
    print("-" * 120)
    print(f"{'Seg':<4} {'Class':<6} {'Conf':<6} {'Entr':<6} {'Comp':<6} {'Pat':<6} {'Flags':<20} {'Text':<50}")
    print("-" * 120)
    
    for result in results:
        flags = []
        if result['flags']['repetitive']: flags.append('REP')
        if result['flags']['formal_patterns']: flags.append('FORM')
        if result['flags']['low_entropy']: flags.append('LOW-E')
        if result['flags']['high_similarity']: flags.append('SIM')
        
        metrics = result['metrics']
        text_preview = result['text'][:47] + '...' if len(result['text']) > 47 else result['text']
        
        print(f"{result['segment']:<4} "
              f"{result['classification']:<6} "
              f"{result['confidence']:.2f}{'':2} "
              f"{metrics['entropy']:.2f}{'':2} "
              f"{metrics['compression_ratio']:.2f}{'':2} "
              f"{metrics['pattern_score']:.2f}{'':2} "
              f"{','.join(flags):<20} "
              f"{text_preview}")
    
    ai_count = sum(1 for r in results if r['classification'] == 'AI')
    print(f"\nSummary:")
    print(f"Total segments analyzed: {len(results)}")
    print(f"AI-like segments: {ai_count} ({ai_count/len(results)*100:.1f}%)")
    
    print("\nFlag Legend:")
    print("REP - Repetitive patterns")
    print("FORM - Formal/templated language")
    print("LOW-E - Low entropy (high predictability)")
    print("SIM - High similarity to baseline")
    
    print("\nMetrics Guide:")
    print("Conf - Confidence score (higher = more certain)")
    print("Entr - Entropy (higher = more random/natural)")
    print("Comp - Compression ratio (higher = more repetitive)")
    print("Pat - Pattern match score (higher = more formal/templated)")

def add_latex_preamble(file_path):
    """Add required LaTeX commands to the preamble if they don't exist."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    highlight_command = r"\newcommand{\AIHighlight}[1]{\colorbox{yellow}{#1}}"
    package_required = r"\usepackage{xcolor}"
    
    # Check if commands already exist
    if highlight_command not in content:
        # Find document class declaration
        doc_class_match = re.search(r'\\documentclass.*?\n', content)
        if doc_class_match:
            insert_pos = doc_class_match.end()
            
            # Add package if needed
            if package_required not in content:
                content = (content[:insert_pos] + package_required + '\n' +
                          highlight_command + '\n' + content[insert_pos:])
            else:
                content = (content[:insert_pos] + highlight_command + '\n' +
                          content[insert_pos:])
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyzer.py <path_to_file>")
        sys.exit(1)
        
    try:
        file_path = sys.argv[1]
        if file_path.lower().endswith('.tex'):
            add_latex_preamble(file_path)
        results = analyze_file(file_path)
        print_results(results)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
