# =============================================================================
# Program: evaluator.py
#
# PURPOSE:
#   Provides a unified evaluation harness for any price-prediction function
#   (model or heuristic). Given a "predictor" callable and a list of Items,
#   it runs predictions in parallel, computes error metrics, and renders two
#   interactive Plotly charts:
#     1. A running-average error trend chart with a 95% confidence interval.
#     2. A scatter plot of predicted vs actual prices, colour-coded by error.
#
# CONNECTIONS TO OTHER PROGRAMS:
#   - Accepts any predictor that takes an Item (items.py) and returns a price.
#     Typically this is DeepNeuralNetworkRunner.inference from
#     deep_neural_network.py, but it can wrap any callable (e.g. a GPT API call).
#   - Item objects come from items.py / loaders.py. The evaluator only reads
#     .title and .price; it does NOT require .summary or .full, so it can also
#     be used to evaluate a model that bypasses the summarisation step.
# =============================================================================

import re
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import accumulate
import math
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor

# ANSI colour codes for colouring per-prediction output in the terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

# Default number of parallel worker threads for concurrent prediction calls
WORKERS = 5

# Default number of items to evaluate (subset of full dataset for speed)
DEFAULT_SIZE = 200


class Tester:
    """
    Orchestrates the full evaluation of one predictor function.

    Workflow:
      1. Call run()  — runs predictions in parallel, prints per-item errors
      2. run() calls report() — computes aggregate metrics and renders charts
    """

    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE, workers=WORKERS):
        """
        predictor : any callable that accepts an Item and returns a numeric price
                    (e.g. DeepNeuralNetworkRunner.inference from
                    deep_neural_network.py, or a lambda wrapping a GPT call)
        data      : list of Item objects (items.py) to evaluate against
        title     : display name for the predictor; auto-derived if None
        size      : how many items from data to evaluate
        workers   : number of threads for parallel prediction
        """
        self.predictor = predictor
        self.data = data
        self.title = title or self.make_title(predictor)
        self.size = size

        # Accumulated per-datapoint results (populated by run())
        self.titles = []   # Truncated product titles (for hover text in charts)
        self.guesses = []  # Model predictions in USD
        self.truths = []   # Ground-truth prices from item.price (items.py)
        self.errors = []   # Absolute errors |guess - truth|
        self.colors = []   # "green" / "orange" / "red" based on error magnitude

        self.workers = workers

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def make_title(predictor) -> str:
        """
        Derive a human-readable title from the predictor's __name__ attribute.
        Handles common naming patterns:
          double underscore → "." (e.g. module__method → module.method)
          single underscore → " " (snake_case → Title Case)
          "Gpt" → "GPT" (cosmetic fix)
        """
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    @staticmethod
    def post_process(value):
        """
        Normalise a raw predictor return value to a plain float.
        Handles two cases:
          - String: strips "$" and "," then extracts the first number found
            (useful when the predictor is a language model that returns text
            like "$1,299.00")
          - Numeric: passes through directly
        Returns 0 if no number can be extracted from a string.
        """
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", value)
            return float(match.group()) if match else 0
        else:
            return value

    def color_for(self, error, truth):
        """
        Assign a traffic-light colour based on how bad the prediction error is,
        using both absolute ($) and relative (%) thresholds so very cheap items
        aren't unfairly penalised:
          green  : error < $40  OR  < 20% of truth
          orange : error < $80  OR  < 40% of truth
          red    : otherwise
        """
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    # -------------------------------------------------------------------------
    # Per-datapoint prediction (called in parallel by run())
    # -------------------------------------------------------------------------

    def run_datapoint(self, i):
        """
        Run the predictor on a single item and compute its error.
        Called concurrently by ThreadPoolExecutor inside run().

        Returns a tuple: (title, guess, truth, error, color)
        """
        datapoint = self.data[i]
        value = self.predictor(datapoint)          # Call the model / API
        guess = self.post_process(value)           # Normalise to float
        truth = datapoint.price                    # Ground truth from items.py
        error = abs(guess - truth)
        color = self.color_for(error, truth)
        # Truncate long titles so they fit in chart hover text
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."
        return title, guess, truth, error, color

    # -------------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------------

    def chart(self, title):
        """
        Render an interactive Plotly scatter plot of predicted vs actual prices.
        Each dot is colour-coded (green/orange/red) by error severity.
        The dashed diagonal line represents perfect predictions (guess == truth).
        Hover text shows the product title, predicted price, and actual price.
        """
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "error": self.errors,
                "color": self.colors,
            }
        )

        # Pre-format hover text for each point
        df["hover"] = [
            f"{t}\nGuess=${g:,.2f} Actual=${y:,.2f}"
            for t, g, y in zip(df["title"], df["guess"], df["truth"])
        ]

        max_val = float(max(df["truth"].max(), df["guess"].max()))

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Actual Price", "guess": "Predicted Price"},
            width=1000,
            height=800,
        )

        # Attach hover text to each colour trace individually
        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6)

        # Ideal prediction line: if the model were perfect, all dots would lie here
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(width=2, dash="dash", color="deepskyblue"),
                name="y = x",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_xaxes(range=[0, max_val])
        fig.update_yaxes(range=[0, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def error_trend_chart(self):
        """
        Render an interactive Plotly line chart showing how the cumulative
        average absolute error evolves as more datapoints are evaluated.
        A shaded 95% confidence interval band indicates statistical uncertainty.
        The chart title shows the final average error ± CI.

        This chart is useful for detecting whether the model's performance is
        consistent across the dataset or degrades on particular item types.
        """
        n = len(self.errors)

        # --- Running statistics computed in pure Python (no numpy) ---
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n + 1))
        running_means = [s / i for s, i in zip(running_sums, x)]

        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]

        # 95% CI for the mean: 1.96 * (std / sqrt(n))
        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [m - c for m, c in zip(running_means, ci)]

        fig = go.Figure()

        # Shaded confidence-interval band
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],        # Forward then reversed for closed polygon
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="95% CI",
            )
        )

        # Main running-average line with hover details
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                name="Cumulative Avg Error",
                customdata=list(zip(ci,)),
                hovertemplate=(
                    "n=%{x}<br>"
                    "Avg Error=$%{y:,.2f}<br>"
                    "±95% CI=$%{customdata[0]:,.2f}<extra></extra>"
                ),
            )
        )

        final_mean = running_means[-1]
        final_ci = ci[-1]
        title = f"{self.title} Error: ${final_mean:,.2f} ± ${final_ci:,.2f}"

        fig.update_layout(
            title=title,
            xaxis_title="Number of Datapoints",
            yaxis_title="Average Absolute Error ($)",
            width=1000,
            height=360,
            template="plotly_white",
            showlegend=False,
        )

        fig.show()

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def report(self):
        """
        Compute aggregate metrics over all evaluated items and render both charts.
        Metrics reported:
          - Average absolute error (MAE in dollars)
          - Mean Squared Error (MSE)
          - R² score (percentage of price variance explained by the model)
        """
        average_error = sum(self.errors) / self.size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100
        title = f"{self.title} results<br><b>Error:</b> ${average_error:,.2f} <b>MSE:</b> {mse:,.0f} <b>r²:</b> {r2:.1f}%"
        self.error_trend_chart()  # Render error trend chart first
        self.chart(title)         # Then render scatter plot

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def run(self):
        """
        Run all predictions in parallel using a ThreadPoolExecutor, accumulate
        results, print a colour-coded dollar error for each prediction, then
        call report() to display the charts.

        Parallelism (WORKERS threads) is important because predictors that call
        remote APIs (e.g. GPT via preprocessor.py) would be very slow sequentially.
        """
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for title, guess, truth, error, color in tqdm(
                ex.map(self.run_datapoint, range(self.size)), total=self.size
            ):
                self.titles.append(title)
                self.guesses.append(guess)
                self.truths.append(truth)
                self.errors.append(error)
                self.colors.append(color)
                # Print a colour-coded error token inline for quick visual feedback
                print(f"{COLOR_MAP[color]}${error:.0f} ", end="")
        self.report()


# =============================================================================
# Convenience wrapper
# =============================================================================

def evaluate(function, data, size=DEFAULT_SIZE, workers=WORKERS):
    """
    Shorthand to evaluate a predictor in one line:
        evaluate(runner.inference, test_items)

    Internally creates a Tester and calls run(). This is the typical entry
    point used from a Jupyter notebook after training in deep_neural_network.py.
    """
    Tester(function, data, size=size, workers=workers).run()
