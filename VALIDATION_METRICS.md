# Validation and Metrics Documentation

## Validation Frequency

Validation runs **every 500 steps** (configurable via `eval_steps` in config).

During each validation run:
1. Standard loss computation on validation set
2. Text generation on 50 random samples
3. Semantic metrics computation
4. Results logged to WandB automatically

## Available Metrics

### 1. **Standard Metrics** (Always Available)
- `eval_loss`: Cross-entropy loss on validation set
- `avg_pred_length`: Average word count in predictions
- `avg_ref_length`: Average word count in references
- `length_ratio`: Ratio of prediction to reference length

### 2. **BERT Score** (Requires: `bert-score`)
Measures semantic similarity using BERT embeddings:
- `bert_score_precision`: How much of the prediction is relevant
- `bert_score_recall`: How much of the reference is captured
- `bert_score_f1`: Harmonic mean (best overall metric)

**Range:** 0.0 to 1.0 (higher is better)

### 3. **ROUGE Scores** (Requires: `rouge-score`)
Measures n-gram overlap between prediction and reference:
- `rouge1`: Unigram overlap (word-level similarity)
- `rouge2`: Bigram overlap (phrase-level similarity)
- `rougeL`: Longest common subsequence

**Range:** 0.0 to 1.0 (higher is better)

### 4. **Semantic Similarity** (Requires: `sentence-transformers`)
Cosine similarity between sentence embeddings:
- `semantic_similarity`: Overall semantic closeness

**Range:** -1.0 to 1.0 (higher is better, typically 0.5-1.0)

## Installation

### Install All Metrics (Recommended)
```bash
pip install bert-score rouge-score sentence-transformers scikit-learn
```

### Individual Packages
```bash
# BERT Score only
pip install bert-score

# ROUGE only
pip install rouge-score

# Semantic Similarity only
pip install sentence-transformers scikit-learn
```

## Configuration

In your config file (`configs/qwen25vl_prompt_gen.yaml`):

```yaml
training:
  # Validation frequency
  eval_steps: 500                # Run validation every 500 steps
  logging_steps: 10              # Log training loss every 10 steps
  
  # Generation settings for metrics
  eval_generation:
    max_new_tokens: 256          # Max tokens to generate during eval
    do_sample: false             # Use greedy decoding for consistency
```

## WandB Integration

All metrics are automatically logged to WandB under the following names:
- `eval/loss`
- `eval/bert_score_f1`
- `eval/rouge1`, `eval/rouge2`, `eval/rougeL`
- `eval/semantic_similarity`
- `eval/avg_pred_length`, etc.

You can track them in real-time at: https://wandb.ai/your-project

## Sample Output

During validation, you'll see logs like:

```
================================================================================
Generating predictions for semantic metrics...
================================================================================
SAMPLE PREDICTIONS:
[Sample 1]
Reference: Adjust the exposure by +1 stop to brighten the image. Apply color correction to...
Generated: Increase the exposure by one stop to make the image brighter. Correct the colors...

[Sample 2]
Reference: Remove the distracting object in the upper left corner using content-aware fill...
Generated: Use content-aware fill to remove the distraction in the top left area...

[Sample 3]
Reference: Apply slight vignetting to draw attention to the center. Enhance contrast by +15%...
Generated: Add subtle vignetting around edges to focus on center. Boost contrast by 15 percent...
================================================================================

Evaluation metrics:
  eval_loss: 1.234
  bert_score_f1: 0.876
  rouge1: 0.654
  rouge2: 0.432
  rougeL: 0.598
  semantic_similarity: 0.823
  avg_pred_length: 47.3
  avg_ref_length: 51.2
  length_ratio: 0.924
```

## Performance Impact

- **Standard loss evaluation:** ~10-30 seconds (depends on eval set size)
- **Text generation (50 samples):** ~30-60 seconds (depends on GPU)
- **Metrics computation:** ~5-10 seconds

**Total validation time:** ~1-2 minutes every 500 steps

To reduce validation time:
- Decrease the number of samples in `QwenTrainerWithGeneration._generate_predictions()` (default: 50)
- Increase `eval_steps` (e.g., 1000 instead of 500)

## Interpreting Metrics

### Good Scores
- `bert_score_f1` > 0.85: Excellent semantic similarity
- `rouge1` > 0.6: Strong word overlap
- `semantic_similarity` > 0.8: Very similar semantics

### Training Progress
Watch for:
- Decreasing `eval_loss` (model learning)
- Increasing BERT Score (better semantic quality)
- Stable `length_ratio` near 1.0 (appropriate length)
- Increasing ROUGE scores (better text overlap)

### Red Flags
- `eval_loss` increasing: Overfitting or learning rate too high
- `length_ratio` << 1.0: Generating too short outputs
- `length_ratio` >> 1.0: Generating too verbose outputs
- Low BERT Score but high ROUGE: Copying rather than understanding

## Customization

To modify validation behavior, edit `script/train_qwen25vl_prompt_generation.py`:

```python
# Change number of samples for generation
def _generate_predictions(self, dataloader, max_samples=100):  # default: 50
    ...

# Change generation parameters
eval_generation_config = {
    "max_new_tokens": 512,      # Generate longer text
    "do_sample": True,          # Use sampling instead of greedy
    "temperature": 0.7,         # Add some randomness
    "top_p": 0.9,              # Nucleus sampling
}
```

