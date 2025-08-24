from src.ml.xai_comparator import XAIComparator
import os
from datetime import datetime
import json

# Create output directory
output_dir = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(output_dir, exist_ok=True)

# Load your synthetic data
print("Loading synthetic UK financial data...")
with open('data/synthetic_uk_financial.json', 'r') as f:
    data = json.load(f)
    
test_texts = data['texts']

print(f"Running XAI comparison on {len(test_texts)} texts\n")

# Initialize comparator
comparator = XAIComparator("ProsusAI/finbert")

# Run evaluation
try:
    results = comparator.run_comprehensive_evaluation(test_texts)
    
    # Generate visualizations
    comparator.generate_comparison_visualizations(results, output_dir)
    
    # Save results
    comparator.save_evaluation_results(results, f"{output_dir}/results.json")
    
    # Generate attention heatmaps for a few examples
    for i in range(3):  # First 3 texts
        text = test_texts[i]
        attention_data = comparator.attention_extractor.extract_attention(text)
        comparator.attention_extractor.generate_attention_visualization(
            attention_data,
            save_path=f"{output_dir}/attention_heatmap_{i}.png"
        )
        print(f"Generated heatmap for text {i+1}")
    
    # Print summary report
    summary = comparator.generate_summary_report(results)
    print("\n" + summary)
    
    print(f"\n✅ Success! Results saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - {output_dir}/attention_lime_speed_comparison.png")
    print(f"  - {output_dir}/attention_lime_agreement.png")
    print(f"  - {output_dir}/attention_lime_faithfulness_stability.png")
    print(f"  - {output_dir}/results.json")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()