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
        # First remove common LaTeX commands and environments
        latex_patterns = [
            r'\\begin{[^}]*}',
            r'\\end{[^}]*}',
            r'\\item\s*',
            r'\\[a-zA-Z]+{[^}]*}',  # Generic command with one argument
            r'\\[a-zA-Z]+\[[^]]*\]{[^}]*}',  # Command with optional argument
            r'\\vspace{[^}]*}',
            r'\\hspace{[^}]*}',
            r'%.*?\n',  # LaTeX comments
            r'\\\\',  # Line breaks
            r'\[[^]]*\]',  # Optional arguments
            r'\\[a-zA-Z]+',  # Single commands without arguments
        ]
        
        # Remove LaTeX commands one by one
        for pattern in latex_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Clean up whitespace and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        return text.strip()
    
    def split_into_sentences(self, text):
        # First split text into regular content and itemize blocks
        parts = re.split(r'(\\begin{itemize}.*?\\end{itemize})', text, flags=re.DOTALL)
        sentences = []
        
        for part in parts:
            if '\\begin{itemize}' in part:
                # Handle itemize environment
                items = re.findall(r'\\item\s*(.*?)(?=\\item|\n\\end{itemize})', part, re.DOTALL)
                for item in items:
                    clean_item = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', item)  # Keep text inside commands
                    clean_item = clean_item.strip()
                    if len(clean_item) > 10:  # Keep minimum length check to avoid empty items
                        sentences.append(clean_item)
            else:
                # Handle regular text
                clean_text = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', part)  # Keep text inside commands
                
                current_sentence = ""
                for char in clean_text:
                    current_sentence += char
                    if char in '.!?' and len(current_sentence.strip()) > 10:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
        
        # Only filter out empty strings and pure LaTeX commands
        return [s for s in sentences]
    
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
            
        # Skip list introductions and incomplete sentences
        if (text.strip().endswith(':') or 
            text.strip().endswith(',') or 
            text.strip().endswith(';') or
            text.strip().endswith('and') or
            text.strip().endswith('or')):
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
        
        # Lower this threshold if you want more highlights
        is_ai = score > 0.4 # Changed from 0.45 to 0.40
        
        return {
            'text': text,
            'classification': 'AI' if is_ai else 'Human',
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
    try:
        detector = SensitiveDetector()
        results = []
        
        if file_path.lower().endswith('.tex'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                original_text = file.read()
            
            # Remove existing highlights
            text_without_highlights = re.sub(r'\\AIHighlight{([^}]*)}', r'\1', original_text)
            modified_text = text_without_highlights
            
            # Process text between markers
            start_marker = "Dear Hiring Manager"
            end_marker = "Best regards"
            
            start_idx = text_without_highlights.find(start_marker)
            end_idx = text_without_highlights.find(end_marker)
            
            if start_idx != -1:
                if end_idx == -1:
                    content_to_analyze = text_without_highlights[start_idx:]
                else:
                    content_to_analyze = text_without_highlights[start_idx:end_idx]
                
                # Split into sentences while preserving structure
                sentences = detector.split_into_sentences(content_to_analyze)
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    # Skip LaTeX commands and environments
                    if sentence.strip().startswith('\\') or 'begin{' in sentence or 'end{' in sentence:
                        continue
                        
                    analysis = detector.analyze_segment(sentence)
                    if analysis:
                        results.append({
                            'segment': 1,
                            **analysis
                        })
                        
                        if analysis['classification'] == 'AI':
                            clean_sentence = sentence.strip()
                            print(f"\nTrying to highlight: {clean_sentence}")
                            
                            try:
                                # Find and highlight the sentence
                                idx = modified_text.find(clean_sentence)
                                if idx >= 0:
                                    modified_text = (
                                        modified_text[:idx] +
                                        f"\\AIHighlight{{{clean_sentence}}}" +
                                        modified_text[idx + len(clean_sentence):]
                                    )
                                    print(f"Highlighted successfully")
                            except Exception as e:
                                print(f"Warning: Could not highlight: {clean_sentence[:50]}...")
                                print(f"Error: {str(e)}")
            
            # Write modified tex file
            backup_path = file_path + '.backup'
            import shutil
            shutil.copy2(file_path, backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(modified_text)
            
            print(f"\nOriginal file backed up to: {backup_path}")
            print("LaTeX file updated with AI highlighting markup")
            
        else:
            # Original analysis for non-tex files...
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
        print(f"Error in analyze_file: {str(e)}")  # Add debug output
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
    
    # Update to use soul package instead of colorbox
    highlight_command = r"\usepackage{soul}" + '\n' + r"\sethlcolor{yellow!30}" + '\n' + r"\newcommand{\AIHighlight}[1]{\hl{#1}}"
    package_required = r"\usepackage{xcolor}"
    
    # Check if commands already exist
    if r"\usepackage{soul}" not in content:
        # Find document class declaration
        doc_class_match = re.search(r'\\documentclass.*?\n', content)
        if doc_class_match:
            insert_pos = doc_class_match.end()
            
            # Add packages if needed
            if package_required not in content:
                content = (content[:insert_pos] + package_required + '\n' +
                          highlight_command + '\n' +
                          content[insert_pos:])
            else:
                content = (content[:insert_pos] + highlight_command + '\n' +
                          content[insert_pos:])
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)

def cleanup_latex_files(file_path):
    """Clean up auxiliary LaTeX files."""
    base_path = file_path.rsplit('.', 1)[0]
    extensions = [
        '.aux', '.log', '.out', '.synctex.gz', 
        '.fls', '.fdb_latexmk', '.bbl', '.blg'
    ]
    
    for ext in extensions:
        try:
            aux_file = base_path + ext
            if Path(aux_file).exists():
                Path(aux_file).unlink()
        except Exception as e:
            print(f"Warning: Could not remove {ext} file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyzer.py <path_to_file>")
        sys.exit(1)
        
    try:
        file_path = sys.argv[1]
        if file_path.lower().endswith('.tex'):
            # Clean up auxiliary files first
            cleanup_latex_files(file_path)
            
            # Add required packages
            add_latex_preamble(file_path)
            
            # Run analysis
            results = analyze_file(file_path)
            print_results(results)
            
            # Clean up again after analysis
            cleanup_latex_files(file_path)
            
            print("\nNote: You may need to recompile the LaTeX document multiple times")
            print("to resolve all references correctly.")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
