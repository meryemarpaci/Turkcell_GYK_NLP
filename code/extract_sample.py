import json
import os

def extract_sample(input_file, output_file, sample_size=1000):
    """
    Extract a sample from a large JSON dataset
    """
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            data.append(json.loads(line))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the sample data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Successfully extracted {len(data)} samples to {output_file}")

if __name__ == "__main__":
    input_file = "archive/News_Category_Dataset_v3.json"
    output_file = "data/news_sample.json"
    extract_sample(input_file, output_file, sample_size=1000) 