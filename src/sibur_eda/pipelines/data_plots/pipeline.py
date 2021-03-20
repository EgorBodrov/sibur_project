from kedro.pipeline import Pipeline, node
from .nodes import make_plot, make_piechart, make_correlation

def create_pipeline(**kwargs):
	return Pipeline(
        [
            node(
                func=make_plot,
                inputs="sibur",
                outputs="plot",
                name="plot_node",
            ),
            node(
                func=make_piechart,
                inputs="sibur",
                outputs="piechart",
                name="piechart_node",
            ),
            node(
                func=make_correlation,
                inputs="sibur",
                outputs="correlation",
                name="correlation_node",
            ),
        ]
    )