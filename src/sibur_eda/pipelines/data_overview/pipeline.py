from kedro.pipeline import Pipeline, node
from .nodes import head_view, tail_view, statistics_view

def create_pipeline(**kwargs):
	return Pipeline(
        [
            node(
                func=head_view,
                inputs="sibur",
                outputs="head_values",
                name="head_values_node",
            ),
            node(
                func=tail_view,
                inputs="sibur",
                outputs="tail_values",
                name="tail_values_node",
            ),
            node(
                func=statistics_view,
                inputs="sibur",
                outputs="statistics_values",
                name="statistics_values_node",
            )
        ]
    )