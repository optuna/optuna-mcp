import argparse
from dataclasses import dataclass
import json
import sys
import threading
import typing

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Image
import optuna
import optuna_dashboard
import plotly


class OptunaMCP(FastMCP):
    def __init__(
        self, name: str, storage: str | None = None, *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        kwargs["name"] = name
        super().__init__(*args, **kwargs)
        self.storage: str | optuna.storages.BaseStorage | None = storage
        self.study: optuna.study.Study | None = None
        self.dashboard_thread_port: tuple[threading.Thread, int] | None = None


@dataclass
class TrialToAdd:
    """
    A trial to be added to an Optuna study.

    Attributes:
        params: The parameter values for the trial.
        distributions: The distributions used for the parameters.
            A key is the parameter name and a value is a distribution.
            The distribution is a dictionary that can be converted to a JSON string, e.g.,
            {
                "name": "IntDistribution",
                "attributes": {"step": null, "low": 1, "high": 9, "log": false}
            }.
            The name of the distribution must be one of the following:
            - FloatDistribution
            - IntDistribution
            - CategoricalDistribution
        values: The objective values for the trial, or None if not set.
            If the state is "COMPLETE", this must be set.
        state: The state of the trial.
            - "COMPLETE": The trial completed successfully.
            - "PRUNED": The trial was pruned.
            - "FAIL": The trial failed.
        user_attrs: User-defined attributes for the trial, or None if not set.
        system_attrs: System-defined attributes for the trial, or None if not set.
    """

    params: dict[str, typing.Any]
    distributions: dict[str, typing.Any]
    values: list[float] | None
    state: typing.Literal["COMPLETE", "PRUNED", "FAIL"]
    user_attrs: dict[str, typing.Any] | None
    system_attrs: dict[str, typing.Any] | None


def register_tools(mcp: OptunaMCP) -> OptunaMCP:
    @mcp.tool()
    def create_study(
        study_name: str,
        directions: list[typing.Literal["minimize", "maximize"]] | None = None,
    ) -> str:
        """Create a new Optuna study with the given study_name and directions.

        If the study already exists, it will be simply loaded.
        """

        mcp.study = optuna.create_study(
            study_name=study_name,
            storage=mcp.storage,
            load_if_exists=True,
            directions=directions,
        )
        if mcp.storage is None:
            mcp.storage = mcp.study._storage

        return f"Optuna study {study_name} has been prepared"

    @mcp.tool()
    def get_all_study_names() -> str:
        """Get all study names from the storage."""
        storage: str | optuna.storages.BaseStorage | None = None
        if mcp.study is not None:
            storage = mcp.study._storage
        elif mcp.storage is not None:
            storage = mcp.storage
        else:
            return "No storage specified."

        study_names = optuna.get_all_study_names(storage)
        return f"All study names: {study_names}"

    @mcp.tool()
    def ask(search_space: dict) -> str:
        """Suggest new parameters using Optuna

        search_space must be a string that can be evaluated to a dictionary to specify Optuna's distributions.

        Example:
            {"x": {"name": "FloatDistribution", "attributes": {"step": null, "low": -10.0, "high": 10.0, "log": false}}}
        """
        try:
            distributions = {
                name: optuna.distributions.json_to_distribution(json.dumps(dist))
                for name, dist in search_space.items()
            }
        except Exception as e:
            return f"Error: {e}"

        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")

        trial = mcp.study.ask(fixed_distributions=distributions)

        return f"Trial {trial.number} suggested: {json.dumps(trial.params)}"

    @mcp.tool()
    def tell(trial_number: int, values: float | list[float]) -> str:
        """Report the result of a trial"""
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")

        mcp.study.tell(
            trial=trial_number,
            values=values,
            state=optuna.trial.TrialState.COMPLETE,
            skip_if_finished=True,
        )
        return f"Trial {trial_number} reported with values {json.dumps(values)}"

    @mcp.tool()
    def set_sampler(
        name: typing.Literal["TPESampler", "NSGAIISampler", "RandomSampler", "GPSampler"],
    ) -> str:
        """Set the sampler for the study.
        The sampler must be one of the following:
        - TPESampler
        - NSGAIISampler
        - RandomSampler
        - GPSampler

        The default sampler for single-objective optimization is TPESampler.
        The default sampler for multi-objective optimization is NSGAIISampler.
        GPSampler is a Gaussian process-based sampler suitable for low-dimensional numerical optimization problems.
        """
        sampler = getattr(optuna.samplers, name)()
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        mcp.study.sampler = sampler
        return f"Sampler set to {name}"

    @mcp.tool()
    def set_trial_user_attr(trial_number: int, key: str, value: typing.Any) -> str:
        """Set user attributes for a trial"""
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")

        storage = mcp.study._storage
        trial_id = storage.get_trial_id_from_study_id_trial_number(
            mcp.study._study_id, trial_number
        )
        storage.set_trial_user_attr(trial_id, key, value)
        return f"User attribute {key} set to {json.dumps(value)} for trial {trial_number}"

    @mcp.tool()
    def get_trial_user_attrs(trial_number: int) -> str:
        """Get user attributes in a trial"""
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        storage = mcp.study._storage
        trial_id = storage.get_trial_id_from_study_id_trial_number(
            mcp.study._study_id, trial_number
        )
        trial = storage.get_trial(trial_id)
        return f"User attributes in trial {trial_number}: {json.dumps(trial.user_attrs)}"

    @mcp.tool()
    def set_metric_names(metric_names: list[str]) -> str:
        """Set metric_names. metric_names are labels used to distinguish what each objective value is.

        Args:
            metric_names:
                The list of metric name for each objective value.
                The length of metric_names list must be the same with the number of objectives.
        """
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        mcp.study.set_metric_names(metric_names)
        return f"metric_names set to {json.dumps(metric_names)}"

    @mcp.tool()
    def get_metric_names() -> str:
        """Get metric_names"""
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        return f"Metric names: {json.dumps(mcp.study.metric_names)}"

    @mcp.tool()
    def get_directions() -> str:
        """Get the directions of the study."""
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        directions = [d.name.lower() for d in mcp.study.directions]
        return f"Directions: {json.dumps(directions)}"

    @mcp.tool()
    def get_trials() -> str:
        """Get all trials in a CSV format"""
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        csv_string = mcp.study.trials_dataframe().to_csv()
        return f"Trials: \n{csv_string}"

    @mcp.tool()
    def best_trial() -> str:
        """Get the best trial

        This feature can only be used for single-objective optimization. If your study is multi-objective, use best_trials instead.
        """
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")

        trial = mcp.study.best_trial
        return f"Best trial: {trial.number} with params {json.dumps(trial.params)} and value {json.dumps(trial.value)}"

    @mcp.tool()
    def best_trials() -> str:
        """Return trials located at the Pareto front in the study."""
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        return f"Best trials: {mcp.study.best_trials}"

    def _create_trial(trial: TrialToAdd) -> optuna.trial.FrozenTrial:
        """Create a trial from the given parameters."""
        return optuna.trial.create_trial(
            params=trial.params,
            distributions={
                k: optuna.distributions.json_to_distribution(json.dumps(d))
                for k, d in trial.distributions.items()
            },
            values=trial.values,
            state=optuna.trial.TrialState[trial.state],
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs,
        )

    @mcp.tool()
    def add_trial(trial: TrialToAdd) -> str:
        """Add a trial to the study."""
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        mcp.study.add_trial(_create_trial(trial))
        return "Trial was added."

    @mcp.tool()
    def add_trials(trials: list[TrialToAdd]) -> str:
        """Add multiple trials to the study."""
        frozen_trials = [_create_trial(trial) for trial in trials]
        if mcp.study is None:
            raise ValueError("No study has been created. Please create a study first.")
        mcp.study.add_trials(frozen_trials)
        return f"{len(trials)} trials were added."

    @mcp.tool()
    def plot_optimization_history(
        target: int | None = None,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the optimization history plot as an image.

        Args:
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
                For single-objective optimization, None (auto) is recommended.
                For multi-objective optimization, this must be specified.
            target_name:
                Target's name to display on the axis label and the legend.
        """
        fig = optuna.visualization.plot_optimization_history(
            mcp.study,
            target=(lambda t: t.values[target]) if target is not None else None,
            target_name=target_name,
        )
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_hypervolume_history(
        reference_point: list[float],
    ) -> Image:
        """Return the hypervolume history plot as an image.

        Args:
            reference_point:
                A list of reference points to calculate the hypervolume.
        """
        fig = optuna.visualization.plot_hypervolume_history(
            mcp.study,
            reference_point=reference_point,
        )
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_pareto_front(
        target_names: list[str] | None = None,
        include_dominated_trials: bool = True,
        targets: list[int] | None = None,
    ) -> Image:
        """Return the Pareto front plot as an image for multi-objective optimization.

        Args:
            target_names:
                Objective name list used as the axis titles. If :obj:`None` is specified,
                "Objective {objective_index}" is used instead. If ``targets`` is specified
                for a study that does not contain any completed trial,
                ``target_name`` must be specified.
            include_dominated_trials:
                A flag to include all dominated trial's objective values.
            targets:
                A list of indices to specify the objective values to display.
                Note that this is 0-indexed, i.e., to plot the first and second objective value, set this to [0, 1].
                If the number of objectives is neither 2 nor 3, ``targets`` must be specified.
                By default, all objectives are displayed.
        """
        fig = optuna.visualization.plot_pareto_front(
            mcp.study,
            target_names=target_names,
            include_dominated_trials=include_dominated_trials,
            targets=targets,
        )
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_contour(
        params: list[str] | None = None,
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the contour plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the color bar.
        """
        fig = optuna.visualization.plot_contour(
            mcp.study, params=params, target=lambda t: t.values[target], target_name=target_name
        )
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_parallel_coordinate(
        params: list[str] | None = None,
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the parallel coordinate plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the axis label and the legend.
        """
        fig = optuna.visualization.plot_parallel_coordinate(
            mcp.study,
            params=params,
            target=lambda t: t.values[target],
            target_name=target_name,
        )
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_slice(
        params: list[str] | None = None,
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the slice plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the axis label.
        """
        fig = optuna.visualization.plot_slice(
            mcp.study,
            params=params,
            target=lambda t: t.values[target],
            target_name=target_name,
        )
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_param_importances(
        params: list[str] | None = None,
        target: int | None = None,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the parameter importances plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
                By default, all objective will be plotted by setting target to None.
            target_name:
                Target’s name to display on the legend.
        """
        evaluator = optuna.importance.PedAnovaImportanceEvaluator()
        fig = optuna.visualization.plot_param_importances(
            mcp.study,
            evaluator=evaluator,
            params=params,
            target=(lambda t: t.values[target]) if target is not None else None,
            target_name=target_name,
        )
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_edf(
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the EDF plot as an image.

        Args:
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the axis label.
        """
        fig = optuna.visualization.plot_edf(
            mcp.study,
            target=lambda t: t.values[target],
            target_name=target_name,
        )
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_timeline() -> Image:
        """Return the timeline plot as an image."""
        fig = optuna.visualization.plot_timeline(mcp.study)
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def plot_rank(
        params: list[str] | None = None,
        target: int = 0,
        target_name: str = "Objective Value",
    ) -> Image:
        """Return the rank plot as an image.

        Args:
            params:
                Parameter list to visualize. The default is all parameters.
            target:
                An index to specify the value to display. To plot nth objective value, set this to n.
                Note that this is 0-indexed, i.e., to plot the first objective value, set this to 0.
            target_name:
                Target’s name to display on the color bar.
        """
        fig = optuna.visualization.plot_rank(mcp.study)
        return Image(data=plotly.io.to_image(fig), format="png")

    @mcp.tool()
    def launch_optuna_dashboard(port: int = 58080) -> str:
        """Launch the Optuna dashboard"""
        storage: str | optuna.storages.BaseStorage | None = None
        if mcp.dashboard_thread_port is not None:
            return f"Optuna dashboard is already running. Open http://127.0.0.1:{mcp.dashboard_thread_port[1]}."

        if mcp.study is not None:
            storage = mcp.study._storage
        elif mcp.storage is not None:
            storage = mcp.storage
        else:
            raise ValueError(
                "No study has been created or no storage URL has been provided."
                "Please create a study or provide a storage URL directly."
            )

        def runner(storage: optuna.storages.BaseStorage | str, port: int) -> None:
            try:
                optuna_dashboard.run_server(storage=storage, host="127.0.0.1", port=port)
            except Exception as e:
                print(f"Error starting the dashboard: {e}", file=sys.stderr)
                sys.exit(1)

        # TODO(y0z): Consider better implementation
        thread = threading.Thread(
            target=runner,
            args=(storage, port),
            daemon=True,
        )
        thread.start()
        mcp.dashboard_thread_port = (thread, port)

        return f"Optuna dashboard is running at http://127.0.0.1:{port}"

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="optuna-mcp",
        description="A server for managing Optuna studies using the MCP.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="The storage URL for Optuna studies. If not specified, it will use the in-memory storage.",
    )
    args = parser.parse_args()

    mcp = OptunaMCP("Optuna", storage=args.storage)
    mcp = register_tools(mcp)
    mcp.run()


if __name__ == "__main__":
    main()
