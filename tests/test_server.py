from collections.abc import Generator
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mcp.types import EmbeddedResource
from mcp.types import ImageContent
from mcp.types import TextContent
import optuna
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
import optuna_dashboard
import pytest

from optuna_mcp.server import OptunaMCP
from optuna_mcp.server import register_tools
from optuna_mcp.server import TrialToAdd


STORAGE_MODES: list[str] = ["inmemory", "sqlite"]


@pytest.fixture(params=STORAGE_MODES)
def storage(tmp_path: Path, request: Any) -> Generator[str | None, None, None]:
    if request.param == "inmemory":
        yield None
        return
    elif request.param == "sqlite":
        file_path = tmp_path / "test.db"
        yield f"sqlite:///{str(file_path)}"

        if file_path.exists():
            file_path.unlink()


@pytest.fixture
def mcp(storage: str | None) -> OptunaMCP:
    """Fixture to create an instance of OptunaMCP."""
    m = OptunaMCP(name="Optuna", storage=storage)
    return register_tools(m)


@pytest.mark.anyio
async def test_list_tools(mcp: OptunaMCP) -> None:
    tools = await mcp.list_tools()

    assert len(tools) == 26
    assert {tool.name for tool in tools} == {
        "create_study",
        "get_all_study_names",
        "ask",
        "tell",
        "set_sampler",
        "set_trial_user_attr",
        "get_trial_user_attrs",
        "set_metric_names",
        "get_metric_names",
        "get_directions",
        "get_trials",
        "best_trial",
        "best_trials",
        "add_trial",
        "add_trials",
        "plot_optimization_history",
        "plot_hypervolume_history",
        "plot_pareto_front",
        "plot_contour",
        "plot_parallel_coordinate",
        "plot_slice",
        "plot_param_importances",
        "plot_edf",
        "plot_timeline",
        "plot_rank",
        "launch_optuna_dashboard",
    }


@pytest.mark.anyio
async def test_create_study(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    assert mcp.study.study_name == "test_study"


@pytest.mark.anyio
async def test_get_all_study_names(mcp: OptunaMCP) -> None:
    result: Sequence[TextContent | ImageContent | EmbeddedResource] = await mcp.call_tool(
        "get_all_study_names", arguments={}
    )
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    if mcp.storage is None:
        assert result[0].text == "No storage specified."
    else:
        assert "[]" in result[0].text


@pytest.mark.parametrize(
    "search_space",
    [
        {},
        {"x": json.loads(distribution_to_json(optuna.distributions.FloatDistribution(0.0, 1.0)))},
    ],
)
@pytest.mark.anyio
async def test_ask(mcp: OptunaMCP, search_space: dict) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    result = await mcp.call_tool("ask", arguments={"search_space": search_space})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert len(mcp.study.trials) == 1
    assert len(mcp.study.trials[0].params) == len(search_space)
    for key in search_space:
        assert key in mcp.study.trials[0].params


@pytest.mark.parametrize(
    "directions, values",
    [
        (["minimize"], [0.0]),
        (["minimize", "maximize"], [0.0, 1.0]),
    ],
)
@pytest.mark.anyio
async def test_tell(mcp: OptunaMCP, directions: list[str], values: list[float]) -> None:
    await mcp.call_tool(
        "create_study", arguments={"study_name": "test_study", "directions": directions}
    )
    assert mcp.study is not None
    t = mcp.study.ask()
    result = await mcp.call_tool("tell", arguments={"trial_number": t.number, "values": values})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert len(mcp.study.trials) == 1
    assert mcp.study.trials[0].values == values


@pytest.mark.parametrize(
    "sampler_name", ["TPESampler", "NSGAIISampler", "RandomSampler", "GPSampler"]
)
@pytest.mark.anyio
async def test_set_sampler(mcp: OptunaMCP, sampler_name: str) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    result = await mcp.call_tool("set_sampler", arguments={"name": sampler_name})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert sampler_name in result[0].text
    assert isinstance(mcp.study.sampler, getattr(optuna.samplers, sampler_name))


@pytest.mark.anyio
async def test_trial_user_attrs(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    t = mcp.study.ask()
    await mcp.call_tool(
        "set_trial_user_attr", arguments={"trial_number": t.number, "key": "abc", "value": "def"}
    )
    result = await mcp.call_tool("get_trial_user_attrs", arguments={"trial_number": t.number})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    r = json.loads(":".join(result[0].text.split(":")[1:]).strip())
    assert isinstance(r, dict)
    assert r == {"abc": "def"}


@pytest.mark.parametrize(
    "metric_names",
    [
        ["accuracy"],
        ["recall", "precision"],
    ],
)
@pytest.mark.anyio
async def test_metric_names(mcp: OptunaMCP, metric_names: list[str]) -> None:
    directions = ["maximize"] * len(metric_names)
    await mcp.call_tool(
        "create_study", arguments={"study_name": "test_study", "directions": directions}
    )
    assert mcp.study is not None
    await mcp.call_tool("set_metric_names", arguments={"metric_names": metric_names})
    result = await mcp.call_tool("get_metric_names", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    r = json.loads(":".join(result[0].text.split(":")[1:]).strip())
    assert isinstance(r, list)
    assert r == metric_names


@pytest.mark.parametrize(
    "directions",
    [
        ["minimize"],
        ["minimize", "maximize"],
        ["maximize", "maximize", "minimize"],
    ],
)
@pytest.mark.anyio
async def test_get_directions(mcp: OptunaMCP, directions: list[str]) -> None:
    await mcp.call_tool(
        "create_study", arguments={"study_name": "test_study", "directions": directions}
    )
    result = await mcp.call_tool("get_directions", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    r = json.loads(":".join(result[0].text.split(":")[1:]).strip())
    assert isinstance(r, list)
    assert r == directions


@pytest.mark.anyio
async def test_get_trials(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.ask()
    result = await mcp.call_tool("get_trials", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    lines = result[0].text.strip().split("\n")
    assert len(lines) == 3
    assert lines[0] == ("Trials: ")
    assert lines[1].startswith(",number")
    assert lines[2].startswith("0,0")


@pytest.mark.anyio
async def test_best_trial(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    t = mcp.study.ask()
    mcp.study.tell(t, [0.0])
    result = await mcp.call_tool("best_trial", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text.strip().split(" ")[-1] == "0.0"


@pytest.mark.anyio
async def test_best_trials(mcp: OptunaMCP) -> None:
    await mcp.call_tool(
        "create_study",
        arguments={"study_name": "test_study", "directions": ["minimize", "minimize"]},
    )
    assert mcp.study is not None
    t = mcp.study.ask()
    mcp.study.tell(t, [0.0, 1.1])
    result = await mcp.call_tool("best_trials", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text.strip().endswith("]")


@pytest.mark.anyio
async def test_add_trial(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    trial = TrialToAdd(
        params={"x": 0.5},
        distributions={
            "x": json.loads(distribution_to_json(optuna.distributions.FloatDistribution(0.0, 1.0)))
        },
        values=[0.0],
        state="COMPLETE",
        user_attrs=None,
        system_attrs=None,
    )
    await mcp.call_tool("add_trial", arguments={"trial": trial})
    actual = mcp.study.get_trials()
    assert len(actual) == 1
    assert actual[0].params == trial.params
    assert actual[0].distributions == {
        k: json_to_distribution(json.dumps(v)) for k, v in trial.distributions.items()
    }
    assert actual[0].values == trial.values
    assert actual[0].state.name == trial.state


@pytest.mark.anyio
async def test_add_trials(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    trial = TrialToAdd(
        params={"x": 0.5},
        distributions={
            "x": json.loads(distribution_to_json(optuna.distributions.FloatDistribution(0.0, 1.0)))
        },
        values=[0.0],
        state="COMPLETE",
        user_attrs=None,
        system_attrs=None,
    )
    trial_fail = TrialToAdd(
        params={"y": -0.5},
        distributions={
            "y": json.loads(
                distribution_to_json(optuna.distributions.FloatDistribution(-1.0, 1.0))
            )
        },
        values=None,
        state="FAIL",
        user_attrs=None,
        system_attrs=None,
    )
    trials = [trial, trial_fail]
    await mcp.call_tool("add_trials", arguments={"trials": trials})
    actual = mcp.study.get_trials()
    assert len(actual) == len(trials)
    for i, trial in enumerate(trials):
        assert actual[i].params == trial.params
        assert actual[i].distributions == {
            k: json_to_distribution(json.dumps(v)) for k, v in trial.distributions.items()
        }
        assert actual[i].values == trial.values
        assert actual[i].state.name == trial.state


@pytest.mark.anyio
async def test_plot_optimization_history(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=10)

    result = await mcp.call_tool("plot_optimization_history", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_hypervolume_history(mcp: OptunaMCP) -> None:
    await mcp.call_tool(
        "create_study",
        arguments={"study_name": "test_study", "directions": ["minimize", "minimize"]},
    )
    assert mcp.study is not None
    mcp.study.optimize(
        lambda trial: (trial.suggest_float("x", 0.0, 1.0), trial.suggest_float("y", 0.0, 1.0)),
        n_trials=10,
    )

    result = await mcp.call_tool(
        "plot_hypervolume_history", arguments={"reference_point": [1.0, 1.0]}
    )
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_pareto_front(mcp: OptunaMCP) -> None:
    await mcp.call_tool(
        "create_study",
        arguments={"study_name": "test_study", "directions": ["minimize", "minimize"]},
    )
    assert mcp.study is not None
    mcp.study.optimize(
        lambda trial: (trial.suggest_float("x", 0.0, 1.0), trial.suggest_float("y", 0.0, 1.0)),
        n_trials=10,
    )

    result = await mcp.call_tool("plot_pareto_front", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_contour(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0) + trial.suggest_float("y", 0.0, 1.0),
        n_trials=10,
    )

    result = await mcp.call_tool("plot_contour", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_parallel_coordinate(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=10)

    result = await mcp.call_tool("plot_parallel_coordinate", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_slice(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=10)

    result = await mcp.call_tool("plot_slice", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_param_importances(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0) + trial.suggest_float("y", 0.0, 1.0),
        n_trials=10,
    )

    result = await mcp.call_tool("plot_param_importances", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_edf(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=10)

    result = await mcp.call_tool("plot_edf", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_timeline(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=10)

    result = await mcp.call_tool("plot_timeline", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.anyio
async def test_plot_rank(mcp: OptunaMCP) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0) + trial.suggest_float("y", 0.0, 1.0),
        n_trials=10,
    )

    result = await mcp.call_tool("plot_rank", arguments={})
    assert isinstance(result, Sequence)
    assert len(result) == 1
    assert isinstance(result[0], ImageContent)


@pytest.mark.parametrize(
    "port, expected_port",
    [
        (None, 58080),
        (58081, 58081),
    ],
)
@pytest.mark.anyio
async def test_launch_optuna_dashboard(
    mcp: OptunaMCP, port: int | None, expected_port: int
) -> None:
    await mcp.call_tool("create_study", arguments={"study_name": "test_study"})
    assert mcp.study is not None
    mcp.study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0) + trial.suggest_float("y", 0.0, 1.0),
        n_trials=10,
    )

    with patch.object(optuna_dashboard, "run_server", return_value=None):
        arguments = {} if port is None else {"port": port}
        result = await mcp.call_tool("launch_optuna_dashboard", arguments=arguments)
        assert isinstance(result, Sequence)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text.endswith(f":{expected_port}")

        assert mcp.dashboard_thread_port is not None
        assert mcp.dashboard_thread_port[0] is not None
        assert mcp.dashboard_thread_port[1] == expected_port
