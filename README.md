# diffinator

<p align="center">
  <img src="docs/diffinator.jpg" width="500" alt="diffinator">
</p>

A tool that analyzes and categorizes changes between GitHub commits or releases. It groups commits by type, tracks important file changes, and generates structured reports in console or markdown format. The tool is configurable through YAML files, allowing you to define custom categorization rules and important files to monitor for a specific project on github.

Currently it includes the configuration (`configs/llamacpp.yaml`) for analyzing changes between two releases of [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Key Features

- Supports both commit and release (tag) comparisons
- Customizable commit categorization through YAML config
- Important file tracking and diff analysis
- Console output with color and pagination
- Markdown output for documentation
- Configurable through repository-specific YAML files
- Smart commit message categorization based on prefixes

## Installation

```bash
git clone git@github.com:jnoller/diffinator.git
cd diffinator
pip install .
```

## Options

```bash
diffinator --help

usage: diffinator [-h] [-t TOKEN] [-c CONFIG] [-v {tag,commit}] [-o {console,markdown}] [-f OUTPUT_FILE] [-n] [--list-configs] [version_a] [version_b]

Analyze changes between GitHub repository releases or commits

positional arguments:
  version_a             First version/commit to compare
  version_b             Second version/commit to compare

options:
  -h, --help            show this help message and exit
  -t, --token TOKEN     GitHub API token (optional - only needed to avoid rate limits)
  -c, --config CONFIG   Name of bundled config (e.g. 'llamacpp') or path to custom YAML config file
  -v, --version-type {tag,commit}
                        Specify whether versions are tags or commits (overrides config file)
  -o, --output {console,markdown}
                        Output format (default: console)
  -f, --output-file OUTPUT_FILE
                        Output file (default: stdout)
  -n, --nodiff          Omit diff output from file changes
  --list-configs        List available bundled configurations

Example: diffinator -c llamacpp v1.0 v2.0
```

## Usage

List available configurations:

```bash
diffinator --list-configs

Bundled configurations:
  • defaults.yaml
    Default configuration for repository analysis
  • llamacpp.yaml
    Configuration for llama.cpp repository
```

Compare two releases with key file diffs inline:

```bash
diffinator -c llamacpp b4273 b4418


Summary:
  • Total commits: 39
  • Files changed: 133
  • Important files modified: 2

Commits by Type:

LLAMA-RUN:
  • llama-run: fix context size (#11094)

... truncated ...

Important File Changes:

• ggml/src/ggml-metal/ggml-metal.m (modified | +2/-2)
  Changes:
                     GGML_ASSERT(ne12 % ne02 == 0);
                     GGML_ASSERT(ne13 % ne03 == 0);

    -                const uint r2 = ne12/ne02;
    -                const uint r3 = ne13/ne03;
    +                const uint32_t r2 = ne12/ne02;
    +                const uint32_t r3 = ne13/ne03;
```

Compare two releases with inline diffs disabled:

```bash
diffinator -c llamacpp --nodiff b4273 b4418
```

Generate a markdown report:

```bash
diffinator -c llamacpp --output markdown b4273 b4418
```

Full example report see [example-report.md](docs/example-report.md)
