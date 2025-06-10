# Contribution Guidelines

If you are interested in contributing to Optuna MCP Server, please read the following guidelines.
If you are new to GitHub, please refer to [our blog](https://medium.com/optuna/optuna-wants-your-pull-request-ff619572302c) for more information.


## Coding Standards and Guidelines

Please adhere to the following coding standards and guidelines:

- Your code comments and documentation must be written in English
- All files must pass linter and formetter checks to be merged to the repository.

To run Linter and Formatter, pre-commit can be used as follows:

```console
$ pip install pre-commit
$ pre-commit install
$ pre-commit run --all-files
```

## Creating a Pull Request

When you are ready to create a pull request, please try to keep the following in mind.

First, the **title** of your pull request should:

- briefly describe and reflect the changes
- wrap any code with backticks
- not end with a period

Second, the **description** of your pull request should:

- describe the motivation
- describe the changes
- if still work-in-progress, describe remaining tasks


## How to Debug Optuna MCP Server

[MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) is helpful.

```sh
mcp dev optuna_mcp/server.py
```

Open http://localhost:5173/.

