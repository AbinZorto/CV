import re
import shutil

# Test API response data
TEST_RESPONSE = {
    "text__ai_detection": {
        "results": [{
            "ai_score": 0.6153,
            "items": [
                {
                    "text": "ABIN ZORTOMACHINE LEARNING ENGINEERFlat 6, 60 West-ferry Road, London, E148LN — +447939412710 —abinzorto@outlook.com — ¯linkedin.com/in/abin-zortoOctober 27, 2024Hiring ManagerComplexioFoundational AIDear Hiring Manager,I am writing to express my strong interest in the DevOps Engineer position at Complexio.",
                    "prediction": "original",
                    "ai_score": 0.00019999999999997797
                },
                {
                    "text": "With my experience in deploying andmanaging ML pipelines across multiple cloud environments and my extensive work with Kubernetes, I believe I can contributesignificantly to Complexio's mission of automating business activities through AI solutions.Currently pursuing my PhD in Computer Science at the",
                    "prediction": "original",
                    "ai_score": 0.04180000000000006
                },
                {
                    "text": "University of East London, I have gained valuable hands-on experience inmanaging distributed systems and implementing ML solutions at scale.",
                    "prediction": "ai-generated",
                    "ai_score": 0.6224000000000001
                },
                {
                    "text": "As a Senior Research Assistant, I have:• Architected and managed multi-cloud ML pipelines across AWS, Azure, and Google Cloud, processing 100,000+ datapoints daily• Successfully deployed and managed Kubernetes clusters for ML workloads, improving system reliability by 35%• Implemented robust CI/CD pipelines using GitHub Actions and Jenkins, reducing deployment time by 50%• Worked extensively with Neo4j and vector databases for efficient data management and similarity",
                    "prediction": "ai-generated",
                    "ai_score": 0.9916
                },
                {
                    "text": "search• Developed and optimized data integration pipelines connecting multiple sources to centralized data lakesI am particularly excited about Complexio's focus on leveraging multiple LLMs and graph databases to understand and automatebusiness processes.",
                    "prediction": "original",
                    "ai_score": 0.02510000000000001
                },
                {
                    "text": "My experience in managing GPU-accelerated ML workflows and optimizing distributed systems aligns perfectlywith your technical requirements.Additionally, my background in implementing secure data integration workflows compliant withhealthcare regulations demonstrates my understanding of handling sensitive data at scale.While I may not meet the exact exp",
                    "prediction": "ai-generated",
                    "ai_score": 0.8852
                },
                {
                    "text": "erience requirements of 7+ years in cloud infrastructure, my intensive hands-on experiencewith cutting-edge technologies and proven track record of successfully implementing complex ML systems demonstrates mycapability to handle the responsibilities of this role effectively.",
                    "prediction": "ai-generated",
                    "ai_score": 0.998
                },
                {
                    "text": "I am confident that my technical expertise, coupled with my research background in AI and distributed systems, would make me avaluable addition to your team.I would welcome the opportunity to discuss how my skills and experience align with Complexio'sneeds in more detail.Thank you for considering my application.Best regards,Abin Zorto",
                    "prediction": "ai-generated",
                    "ai_score": 0.9999
                }
            ],
            "status": "success",
            "provider": "winstonai",
            "cost": 0.034202
        }],
        "errors": []
    }
}

def clean_latex_text(text):
    """Remove LaTeX commands and normalize text for comparison."""
    cleaned = text
    cleaned = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', cleaned)
    cleaned = re.sub(r'\\[a-zA-Z]+', ' ', cleaned)
    cleaned = re.sub(r'[\\{}$%&_#~^\']', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip().lower()

def find_text_in_latex(search_text, latex_content):
    """Find text in LaTeX content and return the original LaTeX version."""
    # Split content into paragraphs
    paragraphs = re.split(r'\n\s*\n', latex_content)
    
    clean_search = clean_latex_text(search_text)
    print(f"Looking for: {clean_search[:100]}")
    
    for para in paragraphs:
        clean_para = clean_latex_text(para)
        if clean_search in clean_para:
            print(f"Found match in: {clean_para[:100]}")
            return para
    return None

def highlight_latex_file(file_path):
    """Test function to highlight AI-detected text in LaTeX file."""
    try:
        # Read the original file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"\nProcessing file: {file_path}")
        
        # Create backup
        backup_path = file_path + '.backup'
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at: {backup_path}")

        # Add required packages
        if r"\usepackage{soul}" not in content:
            packages = (
                r"\usepackage{soul}" + '\n' +
                r"\sethlcolor{yellow!30}" + '\n' +
                r"\newcommand{\AIHighlight}[1]{\hl{#1}}" + '\n'
            )
            match = re.search(r'\\documentclass.*?\n', content)
            if match:
                insert_pos = match.end()
                content = content[:insert_pos] + packages + content[insert_pos:]
                print("Added required LaTeX packages")

        modified_content = content
        highlight_count = 0

        # Process AI-detected segments
        items = TEST_RESPONSE['text__ai_detection']['results'][0]['items']
        
        print("\nProcessing AI-detected segments:")
        for idx, item in enumerate(items, 1):
            if item['prediction'] == 'ai-generated' and item['ai_score'] > 0.6:
                text = item['text'].strip()
                score = item['ai_score']
                
                print(f"\nSegment {idx}:")
                print(f"AI Score: {score}")
                print(f"Original text: {text[:100]}...")
                
                # Find the text in the LaTeX content
                latex_text = find_text_in_latex(text, modified_content)
                if latex_text:
                    # Don't highlight if already highlighted
                    if not r'\AIHighlight{' in latex_text:
                        # Create highlighted version
                        highlighted_text = f"\\AIHighlight{{{latex_text}}}"
                        # Replace in content
                        modified_content = modified_content.replace(latex_text, highlighted_text)
                        highlight_count += 1
                        print(f"Successfully highlighted text")
                else:
                    print("Text not found in LaTeX content")

        # Write modified content back to file
        if highlight_count > 0:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)
            print(f"\nHighlighted {highlight_count} AI-detected segments")
        else:
            print("\nNo changes were made to the file")

        # Print sample of content structure
        print("\nContent structure (first few paragraphs):")
        paragraphs = re.split(r'\n\s*\n', content)[:3]
        for i, para in enumerate(paragraphs):
            clean_para = clean_latex_text(para)
            print(f"\nParagraph {i+1}:")
            print(clean_para[:100])

    except Exception as e:
        print(f"Error in highlight_latex_file: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_highlight.py <path_to_tex_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        highlight_latex_file(file_path)
        print("\nProcess completed!")
        print("Please compile your LaTeX document to see the highlights.")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
