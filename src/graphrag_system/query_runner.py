"""
GraphRAG Query Runner for Phase 5 Experiments

Provides batch query execution for GraphRAG experiments with proper
result formatting for evaluation.
"""
import time
from pathlib import Path
from typing import List

from src.evaluation.schemas import RetrievedContext, SystemPrediction
from src.graphrag_system.runner import run_graphrag_query


def run_graphrag_batch_queries(
    questions: list[dict],
    workspace_dir: Path,
    method: str = "local",
    response_type: str = "Single sentence",
    verbose: bool = True,
) -> list[SystemPrediction]:
    """
    Run GraphRAG queries for multiple questions.
    
    Args:
        questions: List of question dicts (must have "question_id", "question")
        workspace_dir: Path to GraphRAG workspace with indexed data
        method: Query method ("local" or "global")
        response_type: Response length directive
        verbose: Print progress
    
    Returns:
        List of SystemPrediction objects
    """
    predictions = []
    
    if verbose:
        print(f"\nRunning {len(questions)} GraphRAG queries...")
        print(f"  Method: {method}")
        print(f"  Response type: {response_type}")
        print(f"  Workspace: {workspace_dir}")
    
    start_time = time.time()
    
    for i, question_dict in enumerate(questions, 1):
        question_id = question_dict.get("question_id", f"question_{i}")
        question_text = question_dict.get("question", "")
        
        if verbose and i % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(questions) - i)
            print(f"  Progress: {i}/{len(questions)} "
                  f"(~{remaining:.0f}s remaining)")
        
        # Run GraphRAG query
        try:
            response = run_graphrag_query(
                project_dir=workspace_dir,
                question=question_text,
                method=method,
                response_type=response_type,
            )
            
            # Check for errors
            if response.startswith("ERROR:"):
                print(f"  ⚠ Query {question_id} failed: {response}")
                predicted_answer = ""
            else:
                predicted_answer = response
            
        except Exception as e:
            print(f"  ⚠ Query {question_id} exception: {e}")
            predicted_answer = ""
        
        # Create SystemPrediction
        # Note: GraphRAG CLI doesn't expose retrieved context directly,
        # so we create a placeholder context
        prediction = SystemPrediction(
            question_id=question_id,
            system_name="graphrag",
            predicted_answer=predicted_answer,
            retrieved_contexts=[
                RetrievedContext(
                    id="graphrag_community_context",
                    text=f"[GraphRAG {method} search result from workspace]",
                    score=1.0,
                    rank=1,
                    source_type="graphrag_community",
                )
            ],
            metadata={
                "question": question_text,
                "method": method,
                "response_type": response_type,
                "workspace_dir": str(workspace_dir),
            },
        )
        
        predictions.append(prediction)
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Completed {len(predictions)} queries in {total_time:.1f}s")
        print(f"  Average: {total_time / len(questions):.2f}s per query")
    
    return predictions


def run_graphrag_experiment(
    questions: list[dict],
    workspace_dir: Path,
    output_path: Path = None,
    method: str = "local",
    response_type: str = "Single sentence",
    verbose: bool = True,
) -> tuple[list[SystemPrediction], dict]:
    """
    Run full GraphRAG experiment: queries + timing stats.
    
    Args:
        questions: List of question dicts
        workspace_dir: GraphRAG workspace directory
        output_path: Optional path to save predictions JSON
        method: Query method
        response_type: Response length directive
        verbose: Print progress
    
    Returns:
        (predictions, stats) tuple
    """
    start_time = time.time()
    
    # Run batch queries
    predictions = run_graphrag_batch_queries(
        questions=questions,
        workspace_dir=workspace_dir,
        method=method,
        response_type=response_type,
        verbose=verbose,
    )
    
    total_time = time.time() - start_time
    
    # Compute stats
    stats = {
        "num_questions": len(questions),
        "num_predictions": len(predictions),
        "total_time_seconds": total_time,
        "avg_time_per_query_seconds": total_time / len(questions) if questions else 0,
        "method": method,
        "response_type": response_type,
    }
    
    # Save if requested
    if output_path:
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "predictions": [p.to_dict() for p in predictions],
                    "stats": stats,
                },
                f,
                indent=2,
            )
        
        if verbose:
            print(f"\n✓ Saved predictions to {output_path}")
    
    return predictions, stats
