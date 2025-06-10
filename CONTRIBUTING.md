# Contribution Guidelines

If you are interested in contributing to Optuna MCP Server, please read the following guidelines.
If you are new to GitHub, please refer to [our blog](https://medium.com/optuna/optuna-wants-your-pull-request-ff619572302c) for more information.


## Setup Development Environment

First of all, fork Optuna MCP on GitHub. You can learn about fork in the [official documentation](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo).

After forking, download and install Optuna MCP on your computer as follows:

```
$ git clone git@github.com:YOUR_NAME/optuna-mcp.git
$ cd optuna-mcp
$ uv sync   # Install dependencies
$ pwd       # Check the path to your optuna-mcp directory.
/PATH/TO/optuna-mcp
$ which uv  # Check the path to your uv binary.
/PATH/TO/uv
```

After the installation, configure the MCP client (e.g., Claude Desktop) as explained in the [README.md](README.md#installation).
Please fill the placeholders of path to your uv binary and optuna-mcp directory:

```json
{
  "mcpServers": {
    "Optuna": {
      "command": "/PATH/TO/uv",
      "args": [
        "--directory",
        "/PATH/TO/optuna-mcp",
        "run",
        "optuna-mcp"
        "--storage",
        "sqlite:///optuna.db"
      ],
    }
  }
}
```


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

