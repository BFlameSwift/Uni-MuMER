"""
Evaluation Metrics Calculator

This module provides comprehensive evaluation metrics for text generation tasks,
including Edit Score, BLEU-4, Character Error Rate (CER), and various error
threshold statistics.

"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import editdistance
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEvaluator:
    """
    A comprehensive text evaluation toolkit for computing various metrics
    including Edit Score, BLEU-4, CER, and error threshold statistics.
    """
    
    def __init__(self, tokenizer_func: Optional[callable] = None):
        """
        Initialize the TextEvaluator.
        
        Args:
            tokenizer_func (callable, optional): Custom tokenization function.
                                               Defaults to whitespace splitting.
        """
        self.tokenizer = tokenizer_func or self._default_tokenizer
        self.smoothing_function = SmoothingFunction().method4
    
    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """
        Default tokenization function using whitespace splitting.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        return text.split()
    
    def _compute_edit_score(self, gt_tokens: List[str], pred_tokens: List[str]) -> float:
        """
        Compute edit score as a percentage (higher is better).
        
        Args:
            gt_tokens (List[str]): Ground truth tokens
            pred_tokens (List[str]): Predicted tokens
            
        Returns:
            float: Edit score percentage (0-100)
        """
        if not gt_tokens and not pred_tokens:
            return 100.0  # Perfect match for empty sequences
        
        if not gt_tokens or not pred_tokens:
            return 0.0  # Complete mismatch if one sequence is empty
        
        edit_distance = editdistance.eval(gt_tokens, pred_tokens)
        normalizer = max(len(gt_tokens), len(pred_tokens))
        
        return (1 - edit_distance / normalizer) * 100
    
    def _compute_cer(self, data: List[Dict[str, str]], gt_key: str, pred_key: str) -> float:
        """
        Compute Character Error Rate (CER) across the entire dataset.
        
        Args:
            data (List[Dict]): List of data samples
            gt_key (str): Key for ground truth in data dict
            pred_key (str): Key for predictions in data dict
            
        Returns:
            float: Character Error Rate (lower is better)
        """
        total_distance = 0
        total_length = 0
        
        for sample in data:
            pred = sample[pred_key]
            gt = sample[gt_key]
            
            distance = editdistance.eval(pred.split(), gt.split())
            total_distance += distance
            total_length += len(gt)
        
        return total_distance / total_length if total_length > 0 else 0.0
    
    def _compute_error_thresholds(self, data: List[Dict[str, str]], 
                                gt_key: str, pred_key: str) -> Dict[int, int]:
        """
        Compute error threshold statistics for edit distances.
        
        Args:
            data (List[Dict]): List of data samples
            gt_key (str): Key for ground truth in data dict
            pred_key (str): Key for predictions in data dict
            
        Returns:
            Dict[int, int]: Count of samples within each error threshold
        """
        error_counts = {k: 0 for k in range(4)}  # 0, 1, 2, 3 errors
        
        for sample in data:
            gt_tokens = self.tokenizer(sample[gt_key])
            pred_tokens = self.tokenizer(sample[pred_key])
            
            edit_distance = editdistance.eval(gt_tokens, pred_tokens)
            
            for threshold in range(4):
                if edit_distance <= threshold:
                    error_counts[threshold] += 1
        
        return error_counts
    
    def _compute_bleu4(self, references: List[List[List[str]]], 
                      hypotheses: List[List[str]]) -> float:
        """
        Compute BLEU-4 score for the corpus.
        
        Args:
            references (List[List[List[str]]]): Reference sentences (tokenized)
            hypotheses (List[List[str]]): Hypothesis sentences (tokenized)
            
        Returns:
            float: BLEU-4 score as percentage
        """
        bleu_score = corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing_function
        )
        
        return bleu_score * 100
    
    def evaluate(self, json_path: str, output_path: Optional[str] = None,
                gt_key: str = "gt", pred_key: str = "pred") -> Dict[str, Any]:
        """
        Comprehensive evaluation of predictions against ground truth.
        
        Args:
            json_path (str): Path to JSON file containing predictions and ground truth
            output_path (str, optional): Path to save formatted results as text file
            gt_key (str): Key for ground truth in JSON data
            pred_key (str): Key for predictions in JSON data
            
        Returns:
            Dict[str, Any]: Dictionary containing all computed metrics
            
        Raises:
            FileNotFoundError: If the input JSON file doesn't exist
            KeyError: If required keys are missing from the data
            ValueError: If the data format is invalid
        """
        try:
            logger.info(f"Loading data from {json_path}")
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            if not isinstance(data, list) or not data:
                raise ValueError("Data must be a non-empty list of dictionaries")
            
            # Validate required keys
            sample = data[0]
            if gt_key not in sample or pred_key not in sample:
                raise KeyError(f"Required keys '{gt_key}' and/or '{pred_key}' not found in data")
            
            total_samples = len(data)
            logger.info(f"Processing {total_samples} samples")
            
            # Prepare data for various computations
            references = []
            hypotheses = []
            edit_scores = []
            
            for sample in data:
                gt_tokens = self.tokenizer(sample[gt_key])
                pred_tokens = self.tokenizer(sample[pred_key])
                
                # Compute edit score for this sample
                edit_score = self._compute_edit_score(gt_tokens, pred_tokens)
                edit_scores.append(edit_score)
                
                # Prepare for BLEU computation
                references.append([gt_tokens])
                hypotheses.append(pred_tokens)
            
            # Compute all metrics
            mean_edit_score = sum(edit_scores) / total_samples
            bleu4_score = self._compute_bleu4(references, hypotheses)
            cer = self._compute_cer(data, gt_key, pred_key)
            error_counts = self._compute_error_thresholds(data, gt_key, pred_key)
            
            # Calculate error rates as percentages
            error_rates = {
                threshold: (count / total_samples) * 100
                for threshold, count in error_counts.items()
            }
            
            # Organize results
            results = {
                "total_samples": total_samples,
                "mean_edit_score": round(mean_edit_score, 4),
                "bleu4_score": round(bleu4_score, 4),
                "character_error_rate": round(cer, 4),
                "exact_match_rate": round(error_rates[0], 4),  # 0 errors (perfect match)
                "error_le_1_rate": round(error_rates[1], 4),   # ≤ 1 error
                "error_le_2_rate": round(error_rates[2], 4),   # ≤ 2 errors
                "error_le_3_rate": round(error_rates[3], 4),   # ≤ 3 errors
                "error_threshold_stats": {
                    f"errors_le_{k}": {
                        "count": error_counts[k],
                        "percentage": round(error_rates[k], 4)
                    }
                    for k in range(4)
                }
            }
            
            # Generate formatted output
            formatted_output = self._format_results(results)
            
            # Print to console
            print(formatted_output)
            
            # Save to file if requested
            if output_path:
                logger.info(f"Saving results to {output_path}")
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(formatted_output)
            
            logger.info("Evaluation completed successfully")
            return results
            
        except FileNotFoundError:
            logger.error(f"File not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """
        Format evaluation results into a readable string.
        
        Args:
            results (Dict[str, Any]): Results dictionary from evaluation
            
        Returns:
            str: Formatted results string
        """
        output = f"""
{'='*60}
                    EVALUATION RESULTS
{'='*60}

Dataset Statistics:
  Total Samples: {results['total_samples']:,}

Core Metrics:
  Mean Edit Score:       {results['mean_edit_score']:>8.4f}%
  BLEU-4 Score:          {results['bleu4_score']:>8.4f}%
  Character Error Rate:  {results['character_error_rate']:>8.4f}

Error Threshold Analysis:
  Exact Match Rate:      {results['exact_match_rate']:>8.4f}% (0 errors)
  Error ≤ 1:             {results['error_le_1_rate']:>8.4f}%
  Error ≤ 2:             {results['error_le_2_rate']:>8.4f}%
  Error ≤ 3:             {results['error_le_3_rate']:>8.4f}%

Detailed Error Distribution:
"""
        
        for k in range(4):
            stats = results['error_threshold_stats'][f'errors_le_{k}']
            output += f"  Errors ≤ {k}: {stats['count']:>6,} samples ({stats['percentage']:>6.2f}%)\n"
        
        output += f"\n{'='*60}\n"
        
        return output


def evaluate_text_generation(json_path: str, output_path: Optional[str] = None,
                           gt_key: str = "gt", pred_key: str = "pred") -> Dict[str, Any]:
    """
    Convenience function for evaluating text generation results.
    
    Args:
        json_path (str): Path to JSON file containing predictions and ground truth
        output_path (str, optional): Path to save formatted results
        gt_key (str): Key for ground truth in JSON data
        pred_key (str): Key for predictions in JSON data
        
    Returns:
        Dict[str, Any]: Dictionary containing all computed metrics
    """
    evaluator = TextEvaluator()
    return evaluator.evaluate(json_path, output_path, gt_key, pred_key)


# Example usage
def evaluate_all_results(root_folder: str, output_dir: Optional[str] = None):
    """
    Traverse all subdirectories and evaluate JSON files in 'results' folders.
    
    Args:
        root_folder (str): Root directory to start traversal
        output_dir (str, optional): Directory to save evaluation results
    """
    evaluator = TextEvaluator()
    
    for root, dirs, files in os.walk(root_folder):
        # Check if current directory is named 'results'
        if os.path.basename(root) == 'results':
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    print(f"\nProcessing: {json_path}")
                    
                    try:
                        # Generate output path if output_dir is specified
                        output_path = None
                        if output_dir:
                            # Create relative path structure in output directory
                            rel_path = os.path.relpath(json_path, root_folder)
                            output_path = os.path.join(output_dir, rel_path.replace('.json', '_results.txt'))
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Evaluate the JSON file
                        metrics = evaluator.evaluate(
                            json_path=json_path,
                            output_path=output_path,
                            gt_key="gt",
                            pred_key="pred"
                        )
                        
                        # Print summary
                        print(f"Evaluation completed for: {json_path}")
                        print(f"Exact Match Rate: {metrics['exact_match_rate']:.2f}%")
                        print(f"BLEU-4 Score: {metrics['bleu4_score']:.2f}%")
                        print(f"Character Error Rate: {metrics['character_error_rate']:.4f}")
                        
                    except Exception as e:
                        print(f"Error processing {json_path}: {e}")

if __name__ == "__main__":
    # Example usage: traverse all results folders and evaluate JSON files
    # root_directory = "./data"  # Replace with your root folder path
    # output_directory = "./data"   # Optional: where to save evaluation results
    
    root_directory = "./example_data"  # Replace with your root folder path
    output_directory = "./example_data"   # Optional: where to save evaluation results
    
    evaluate_all_results(root_directory, output_directory)
    