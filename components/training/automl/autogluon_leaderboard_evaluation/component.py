from typing import List, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
)
def leaderboard_evaluation(
    models: List[dsl.Model],
    eval_metric: str,
    html_artifact: dsl.Output[dsl.HTML],
) -> NamedTuple("outputs", best_model=str):
    """Evaluate multiple AutoGluon models and generate a leaderboard.

    This component aggregates evaluation results from a list of Model artifacts
    (reading pre-computed metrics from JSON) and generates an HTML-formatted
    leaderboard ranking the models by their performance metrics. Each model
    artifact is expected to contain metrics at
    model.path / model.metadata["display_name"] / metrics / metrics.json.

    Args:
        models: List of Model artifacts with "display_name" in metadata and metrics at model.path/model_name/metrics/metrics.json.
        eval_metric: Metric name for ranking (e.g. "accuracy", "root_mean_squared_error"); leaderboard sorted by it descending.
        html_artifact: Output artifact for the HTML-formatted leaderboard (model names and metrics).

    Raises:
        FileNotFoundError: If any model metrics path cannot be found.
        KeyError: If metadata lacks "display_name" or metrics JSON lacks the eval_metric key.

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_leaderboard_evaluation import (
            leaderboard_evaluation
        )

        @dsl.pipeline(name="model-evaluation-pipeline")
        def evaluation_pipeline(trained_models):
            leaderboard = leaderboard_evaluation(
                models=trained_models,
                eval_metric="root_mean_squared_error",
            )
            return leaderboard
    """  # noqa: E501
    import html as html_module
    import json
    from pathlib import Path

    import pandas as pd

    def _build_leaderboard_table(df: pd.DataFrame) -> str:
        """Build table HTML with Notebook and Predictor as separate columns (raw URI as link text)."""
        display_cols = [c for c in df.columns if c not in ("notebook", "predictor")]
        rows = []
        rows.append(
            "<thead><tr>"
            + "".join(f"<th>{html_module.escape(str(c))}</th>" for c in [df.index.name or "rank"] + display_cols)
            + "<th>Notebook</th><th>Predictor</th></tr></thead><tbody>"
        )
        for idx, row in df.iterrows():
            cells = [f"<td>{html_module.escape(str(idx))}</td>"]
            for col in display_cols:
                val = row[col]
                cells.append(f"<td>{html_module.escape(str(val))}</td>")
            notebook_uri = html_module.escape(str(row["notebook"]))
            predictor_uri = html_module.escape(str(row["predictor"]))
            cells.append(
                f'<td class="uri-cell">'
                f'<a href="{notebook_uri}" class="uri-link" data-uri="{notebook_uri}" target="_blank" rel="noopener">URI</a>'  # noqa: E501
                f'<div class="uri-popover" role="dialog" aria-label="URI" hidden>'
                f'<pre class="uri-popover-text"></pre>'
                f'<button type="button" class="uri-popover-close" aria-label="Close">×</button>'
                f"</div></td>"
            )
            cells.append(
                f'<td class="uri-cell">'
                f'<a href="{predictor_uri}" class="uri-link" data-uri="{predictor_uri}" target="_blank" rel="noopener">URI</a>'  # noqa: E501
                f'<div class="uri-popover" role="dialog" aria-label="URI" hidden>'
                f'<pre class="uri-popover-text"></pre>'
                f'<button type="button" class="uri-popover-close" aria-label="Close">×</button>'
                f"</div></td>"
            )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        return "<table>" + "".join(rows) + "</tbody></table>"

    # Modified theme colors for a lighter look. Only :root CSS vars are changed.
    def _build_leaderboard_html(table_html: str, eval_metric: str, best_model_name: str, num_models: int) -> str:
        """Build a styled HTML document for the leaderboard (lighter theme)."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoML Leaderboard</title>
  <style>
    :root {{
      --bg: #f7fafc;
      --surface: #ffffff;
      --surface-hover: #f1f5f9;
      --border: #dde3eb;
      --text: #23282e;
      --text-muted: #5c6975;
      --accent: #2977ff;
      --radius: 12px;
      --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 2rem;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
    }}
    .container {{
      max-width: 100%;
      width: 100%;
      margin: 0 auto;
      padding: 0 1rem;
    }}
    header {{
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    h1 {{
      margin: 0 0 0.25rem 0;
      font-size: 1.75rem;
      font-weight: 600;
    }}
    h2 {{
      margin: 0 0 0.5rem 0;
      font-size: 1.15rem;
      font-weight: 400;
      color: var(--text-muted);
    }}
    .table-scroll {{
      overflow-x: auto;
      overflow-y: visible;
      width: 100%;
      max-width: 100%;
      min-width: 0;
      -webkit-overflow-scrolling: touch;
      margin-bottom: 2rem;
      border-radius: var(--radius);
      box-shadow: 0 4px 14px 0 #e6eaf1;
    }}
    .table-scroll::-webkit-scrollbar {{
      height: 8px;
    }}
    .table-scroll::-webkit-scrollbar-track {{
      background: var(--surface-hover);
      border-radius: 4px;
    }}
    .table-scroll::-webkit-scrollbar-thumb {{
      background: var(--border);
      border-radius: 4px;
    }}
    .table-scroll::-webkit-scrollbar-thumb:hover {{
      background: var(--text-muted);
    }}
    table {{
      width: 100%;
      max-width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      table-layout: auto;
    }}
    th, td {{
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--surface-hover);
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 0.95rem;
      letter-spacing: 0.03em;
      font-weight: 600;
      border-bottom: 2px solid var(--border);
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    tr:hover {{
      background: var(--surface-hover);
      transition: background 0.08s;
    }}
    .caption {{
      color: var(--text-muted);
      font-size: 0.96em;
      margin-top: -0.8em;
      margin-bottom: 2em;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .uri-cell {{
      position: relative;
      max-width: 20rem;
    }}
    .uri-cell .uri-link {{
      font-size: 0.875rem;
    }}
    .uri-popover {{
      display: none;
      position: fixed;
      min-width: 16rem;
      max-width: 28rem;
      padding: 0.75rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.12);
      z-index: 9999;
    }}
    .uri-popover:not([hidden]) {{
      display: block;
    }}
    .uri-popover-text {{
      margin: 0 0 0.5rem 0;
      padding: 0.5rem;
      font-size: 0.8rem;
      word-break: break-all;
      overflow-wrap: break-word;
      background: var(--surface-hover);
      border-radius: 4px;
      white-space: pre-wrap;
      max-height: 8rem;
      overflow-y: auto;
    }}
    .uri-popover-close {{
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: none;
      border: none;
      font-size: 1.25rem;
      line-height: 1;
      cursor: pointer;
      color: var(--text-muted);
    }}
    .uri-popover-close:hover {{
      color: var(--text);
    }}
    .main-content {{
      width: 100%;
      max-width: 75%;
      margin-left: auto;
      margin-right: auto;
    }}
    .table-wrapper {{
      display: flex;
      justify-content: center;
      width: 100%;
    }}
    .table-wrapper .table-scroll {{
      flex: 1 1 auto;
      min-width: 0;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="main-content">
      <header>
        <h1>AutoML Model Leaderboard</h1>
        <h2>Ranking top {num_models} models by <b>{eval_metric}</b></h2>
        <div class="caption">
          <span>Best model: <b>{best_model_name}</b></span>
        </div>
      </header>
      <div class="table-wrapper">
      <div class="table-scroll">
      {table_html}
      </div>
      </div>
    </div>
  </div>
  <script>
    (function() {{
      function closeAllPopovers() {{
        document.querySelectorAll('.uri-popover').forEach(function(p) {{ p.hidden = true; }});
      }}
      function positionPopover(popover, link) {{
        var rect = link.getBoundingClientRect();
        var gap = 8;
        var popoverRect = popover.getBoundingClientRect();
        var viewH = window.innerHeight;
        var viewW = window.innerWidth;
        var top = rect.bottom + gap;
        var left = rect.left;
        if (top + popoverRect.height > viewH - 10) {{
          top = rect.top - popoverRect.height - gap;
        }}
        if (top < 10) top = 10;
        if (left + popoverRect.width > viewW - 10) left = viewW - popoverRect.width - 10;
        if (left < 10) left = 10;
        popover.style.top = top + 'px';
        popover.style.left = left + 'px';
      }}
      document.querySelectorAll('.uri-link').forEach(function(link) {{
        link.addEventListener('click', function(e) {{
          e.preventDefault();
          var uri = this.getAttribute('data-uri');
          var popover = this.nextElementSibling;
          if (!uri || !popover || !popover.classList.contains('uri-popover')) return;
          closeAllPopovers();
          popover.querySelector('.uri-popover-text').textContent = uri;
          popover.hidden = false;
          requestAnimationFrame(function() {{ positionPopover(popover, link); }});
        }});
      }});
      document.querySelectorAll('.uri-popover-close').forEach(function(btn) {{
        btn.addEventListener('click', function() {{ this.closest('.uri-popover').hidden = true; }});
      }});
      document.addEventListener('click', function(e) {{
        if (!e.target.closest('.uri-cell')) closeAllPopovers();
      }});
    }})();
  </script>
</body>
</html>
"""

    def _round_metrics(metrics: dict, decimals: int = 4) -> dict:
        """Round numeric values in a metrics dict to the given number of decimals."""
        return {k: round(v, decimals) if isinstance(v, (int, float)) else v for k, v in metrics.items()}

    if not models:
        raise ValueError("At least one model is required")

    results = []
    for model in models:
        metrics_path = Path(model.path) / model.metadata["display_name"] / "metrics" / "metrics.json"
        with metrics_path.open("r") as f:
            eval_results = json.load(f)
        display_name = model.metadata["display_name"]
        model_uri = f"{model.uri.rstrip('/')}/{display_name}"
        predictor_uri = f"{model_uri}/predictor/predictor.pkl"
        notebook_uri = f"{model_uri}/notebooks/automl_predictor_notebook.ipynb"
        results.append(
            {
                "model": display_name,
                **_round_metrics(eval_results),
                "notebook": notebook_uri,
                "predictor": predictor_uri,
            }
        )

    # Notice: AutoGluon follows the "higher is better" strategy for all metrics.
    # This means that some metrics—like log_loss and root_mean_squared_error—will have their signs FLIPPED and are shown as negative. # noqa: E501
    # This is to ensure that a higher value always means a better model, so users do not need to know about the metric's normal directionality when interpreting the leaderboard. # noqa: E501
    leaderboard_df = pd.DataFrame(results).sort_values(by=eval_metric, ascending=False)
    n = len(leaderboard_df)
    leaderboard_df.index = pd.RangeIndex(start=1, stop=n + 1, name="rank")

    html_table = _build_leaderboard_table(leaderboard_df)

    best_model_name = leaderboard_df.iloc[0]["model"]
    html_content = _build_leaderboard_html(
        table_html=html_table,
        eval_metric=eval_metric,
        best_model_name=best_model_name,
        num_models=len(leaderboard_df),
    )
    with open(html_artifact.path, "w", encoding="utf-8") as f:
        f.write(html_content)

    html_artifact.metadata["data"] = leaderboard_df.to_dict()
    html_artifact.metadata["display_name"] = "automl_leaderboard"
    return NamedTuple("outputs", best_model=str)(best_model=best_model_name)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
