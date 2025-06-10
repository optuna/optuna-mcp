## FFmpeg Encoding Parameter Optimization with Optuna MCP

### Overview

This demo showcases how to use Optuna MCP server to automatically find optimal FFmpeg encoding parameters. It optimizes x264 encoding options to maximize video quality (measured by SSIM score) while keeping encoding time reasonable.

### Demo
Using the Optuna MCP and FFmpeg MCP servers, you can try executing this demo with the following prompt. For demonstration purposes, I use the [Big Buck Bunny (320p) video](https://peach.blender.org/) file, which is freely available, but you can use any video of your choice to run this demo.

```
Please use Optuna to search for FFmpeg encoding parameters, such as x264 options, that improve the SSIM score while keeping the execution time relatively short.
You can use the following input file to run FFmpeg:
/path/to/BigBuckBunny_320x180.mp4
```

After submitting the above prompt, the LLM sets up the Optuna study, samples parameters for FFmpeg, executes FFmpeg, and repeats this process to find better encoding parameters, as shown below:


![ffmpeg-1](../images/demo-ffmpeg-1.png)

When I executed this example, the LLM sampled 8 different encoding parameters and reported results as shown below:

![ffmpeg-2](../images/demo-ffmpeg-2.png)

### How to Set Up the MCP Server for FFmpeg

I attempted to find an MCP server for FFmpeg but couldn't identify one suitable for this use case. Therefore, I created a very simple MCP server to run FFmpeg. Below is the full source code:

```python
from mcp.server.fastmcp import FastMCP
import ffmpeg
import re
import time
import traceback


mcp = FastMCP("FFmpeg")


@mcp.tool()
def run_ffmpeg(input_file: str, refs: int, qcomp: float, qdiff: int, me_range: int, x264opts: str) -> str:
    """Run ffmpeg and returns the value of SSIM (Strucutural SIMilarity) and the elapsed time.

    Arguments:
    - input_file: Path to the input video file
    - refs: Number of reference frames (1-16)
    - qcomp: Quantizer compression (0.0-1.0)
    - qdiff: Max QP difference between frames (1-51)
    - me_range: Motion estimation range (4-64)
    - x264opts: FFmpeg x264 options string in the same format as FFmpeg command line
                (e.g. 'keyint=12:b-adapt=...')
    """
    start = time.time()
    try:
        enc = ffmpeg.input(input_file, t=50)  # Use only the first 50 seconds instead of the entire video for demo purposes.
        enc = enc.output(
            '-', 
            f='null', 
            tune='ssim', 
            ssim=1, 
            an=None, 
            vcodec='libx264', 
            s='320x240', 
            vb='100k',
            refs=refs, 
            qcomp=qcomp, 
            qdiff=qdiff, 
            me_range=me_range,
            x264opts=x264opts
        )
        _, stderr = enc.run(capture_stderr=True)

        elapsed = time.time() - start
        ssim_mean = re.search('SSIM Mean Y:([0-9.]+)', stderr.decode('utf-8')).group(1)

        return f"SSIM (mean): {ssim_mean}, Elapsed: {elapsed} seconds"
    except Exception as e:
        stacktrace = '\n'.join(traceback.format_tb(e.__traceback__))
        return f"Error: {e}, {traceback}: {traceback}"


if __name__ == "__main__":
    mcp.run()
```

<details>
<summary>pyproject.toml and Example Settings for Claude Desktop to Run the FFmpeg MCP</summary>

To use ffmpeg-mcp, please create `server.py` and `pyproject.toml` as follows:


```
$ tree ./ffmpeg-mcp
ffmpeg-mcp/
├── server.py
└── pyproject.toml

$ cat pyproject.toml
[project]
name = "ffmpeg-mcp-for-optuna-mcp-demo"
version = "0.1.0"
description = "A simple MCP server for the Optuna MCP demo application"
requires-python = ">=3.12"
dependencies = [
    "mcp[cli]>=1.5.0",
    "ffmpeg-python",
]
```

To include it in Claude Desktop, go to Claude > Settings > Developer > Edit Config > `claude_desktop_config.json` and add the following:


```json
    "FFMpeg": {
      "command": "/path/to/uv",
      "args": [
        "--directory",
        "/path/to/ffmpeg-mcp",
        "run",
        "server.py"
      ]
    }
```

</details>

### Acknowledgements

This demo is heavily inspired by [@sile's excellent work](https://gist.github.com/sile/8aa1ff7808dd55298f51dd70c8b83092) using Optuna to find optimal FFmpeg encoding parameters. His work demonstrated that automated parameter tuning can discover configurations that outperform FFmpeg's built-in presets.

