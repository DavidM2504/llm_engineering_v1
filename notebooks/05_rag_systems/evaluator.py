"""
This script launches a Gradio-based dashboard to evaluate a RAG (Retrieval-Augmented Generation) system.

FEATURES:
- Runs retrieval evaluation (MRR, nDCG, keyword coverage)
- Runs answer quality evaluation (accuracy, completeness, relevance)
- Displays results with color-coded metrics
- Shows category-based bar charts
- Includes progress tracking during evaluation

NOTES FOR EXTENDING:

- Logging:
  Replace print/debugging with logging:
    import logging
    logging.basicConfig(level=logging.INFO)

- Adding new metrics:
  Extend:
    * get_color()
    * format_metric_html()
    * evaluation aggregation logic

- UI customization:
  Modify Gradio components (themes, layout, charts)

- Data export:
  You can save results to CSV using pandas (df.to_csv(...))
"""

import gradio as gr
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

# Import evaluation pipelines (external modules)
from evaluation.eval import evaluate_all_retrieval, evaluate_all_answers

# Load environment variables (API keys, config, etc.)
load_dotenv(override=True)


# ------------------------------------------------------------------
# THRESHOLDS FOR COLOR CODING
# ------------------------------------------------------------------

# Retrieval metrics thresholds
MRR_GREEN = 0.9
MRR_AMBER = 0.75

NDCG_GREEN = 0.9
NDCG_AMBER = 0.75

COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Answer quality thresholds (scale 1–5)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


# ------------------------------------------------------------------
# COLOR SELECTION LOGIC
# ------------------------------------------------------------------
def get_color(value: float, metric_type: str) -> str:
    """
    Determine color (green/orange/red) based on metric value.

    Used for visual feedback in the dashboard:
    - Green = good
    - Orange = acceptable
    - Red = poor

    metric_type determines which thresholds to apply.
    """
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"

    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"

    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"

    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"

    return "black"


# ------------------------------------------------------------------
# METRIC FORMATTING (HTML UI)
# ------------------------------------------------------------------
def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    """
    Format a metric into styled HTML for display in Gradio.

    Features:
    - Color-coded border and value
    - Flexible formatting:
        * percentage (e.g., 85.2%)
        * score (e.g., 4.32/5)
        * default float (e.g., 0.9231)
    """
    color = get_color(value, metric_type)

    # Format value based on type
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"

    # Return styled HTML block
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


# ------------------------------------------------------------------
# RETRIEVAL EVALUATION PIPELINE
# ------------------------------------------------------------------
def run_retrieval_evaluation(progress=gr.Progress()):
    """
    Run retrieval evaluation across all test cases.

    Steps:
    1. Iterate over evaluation dataset
    2. Accumulate metrics (MRR, nDCG, coverage)
    3. Track per-category performance
    4. Update progress bar in UI
    5. Return summary HTML + chart data
    """
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0

    # Track metrics per category
    category_mrr = defaultdict(list)

    count = 0

    # Iterate through evaluation generator
    for test, result, prog_value in evaluate_all_retrieval():
        count += 1

        # Aggregate metrics
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage

        # Store category-level metrics
        category_mrr[test.category].append(result.mrr)

        # Update progress bar in UI
        progress(prog_value, desc=f"Evaluating test {count}...")

    # Compute averages
    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    # Build HTML summary
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    # Prepare data for bar chart
    category_data = []
    for category, mrr_scores in category_mrr.items():
        avg_cat_mrr = sum(mrr_scores) / len(mrr_scores)
        category_data.append({"Category": category, "Average MRR": avg_cat_mrr})

    df = pd.DataFrame(category_data)

    return final_html, df


# ------------------------------------------------------------------
# ANSWER EVALUATION PIPELINE
# ------------------------------------------------------------------
def run_answer_evaluation(progress=gr.Progress()):
    """
    Run answer quality evaluation.

    Metrics:
    - Accuracy
    - Completeness
    - Relevance

    Similar structure to retrieval evaluation.
    """
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0

    category_accuracy = defaultdict(list)

    count = 0

    for test, result, prog_value in evaluate_all_answers():
        count += 1

        # Aggregate metrics
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance

        # Track per category
        category_accuracy[test.category].append(result.accuracy)

        # Update progress bar
        progress(prog_value, desc=f"Evaluating test {count}...")

    # Compute averages
    avg_accuracy = total_accuracy / count
    avg_completeness = total_completeness / count
    avg_relevance = total_relevance / count

    # Build HTML summary
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    # Prepare chart data
    category_data = []
    for category, scores in category_accuracy.items():
        avg_cat_accuracy = sum(scores) / len(scores)
        category_data.append({"Category": category, "Average Accuracy": avg_cat_accuracy})

    df = pd.DataFrame(category_data)

    return final_html, df


# ------------------------------------------------------------------
# GRADIO APP
# ------------------------------------------------------------------
def main():
    """
    Launch the Gradio dashboard.

    UI Structure:
    - Title + description
    - Retrieval evaluation section
    - Answer evaluation section
    - Each section has:
        * Run button
        * Metrics panel (HTML)
        * Bar chart
    """
    # Define UI theme
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for the Insurellm RAG system")

        # ---------------- RETRIEVAL SECTION ----------------
        gr.Markdown("## 🔍 Retrieval Evaluation")

        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            # Metrics display (HTML)
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )

            # Bar chart
            with gr.Column(scale=1):
                retrieval_chart = gr.BarPlot(
                    x="Category",
                    y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1],
                    height=400,
                )

        # ---------------- ANSWER SECTION ----------------
        gr.Markdown("## 💬 Answer Evaluation")

        answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                answer_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )

            with gr.Column(scale=1):
                answer_chart = gr.BarPlot(
                    x="Category",
                    y="Average Accuracy",
                    title="Average Accuracy by Category",
                    y_lim=[1, 5],
                    height=400,
                )

        # ---------------- EVENT WIRING ----------------
        # Connect buttons to evaluation functions
        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart],
        )

        answer_button.click(
            fn=run_answer_evaluation,
            outputs=[answer_metrics, answer_chart],
        )

    # Launch app in browser
    app.launch(inbrowser=True)


# Entry point
if __name__ == "__main__":
    main()