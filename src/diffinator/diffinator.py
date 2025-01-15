#!/usr/bin/env python3

from typing import Dict, List, Optional, TextIO
import sys
import argparse
from dataclasses import dataclass
from githubkit import GitHub, Response
from githubkit.versions.latest.models import (
    Release,
    CommitComparison
)
from githubkit.utils import Unset
from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text
from rich.live import Live
from io import StringIO
import yaml
from pathlib import Path
from abc import ABC, abstractmethod

@dataclass
class ChangeGroup:
    name: str
    keywords: List[str]
    changes: List[str] = None

    def __post_init__(self):
        self.changes = []

class OutputFormatter(ABC):
    """Base class for output formatters"""
    
    def __init__(self, output: TextIO = sys.stdout, show_diff: bool = True):
        self.output = output
        self.show_diff = show_diff
    
    @abstractmethod
    def print_summary(self, total_commits: int, total_files: int, important_files: int):
        pass
    
    @abstractmethod
    def print_commits_by_type(self, commit_groups: Dict[str, List[str]]):
        pass
    
    @abstractmethod
    def print_changes_by_category(self, categories: Dict[str, List[str]]):
        pass
    
    @abstractmethod
    def print_file_changes(self, file_changes: List[dict]):
        pass

class ConsoleFormatter(OutputFormatter):
    """Rich console output formatter with pagination support"""
    
    def __init__(self, console: Console, output: TextIO = sys.stdout, show_diff: bool = True):
        super().__init__(output, show_diff)
        self.console = console
        self.buffer = StringIO()
        self.capture_console = Console(
            file=self.buffer,
            force_terminal=True,
            width=120
        )
    
    def _print(self, *args, **kwargs):
        """Helper method to print to the capture console"""
        self.capture_console.print(*args, **kwargs)
    
    def print_summary(self, total_commits: int, total_files: int, important_files: int):
        self._print("\n[bold]Summary:[/bold]")
        self._print(f"  • Total commits: [cyan]{total_commits}[/cyan]")
        self._print(f"  • Files changed: [cyan]{total_files}[/cyan]")
        self._print(f"  • Important files modified: [cyan]{important_files}[/cyan]")
    
    def print_commits_by_type(self, commit_groups: Dict[str, List[str]]):
        self._print("\n[bold]Commits by Type:[/bold]")
        for group, commits in commit_groups.items():
            if commits:  # Only show groups with commits
                self._print(f"\n[bold blue]{group.upper()}:[/bold blue]")
                for commit in commits:
                    self._print(f"  • [yellow]{commit}[/yellow]")
    
    def print_changes_by_category(self, categories: Dict[str, List[str]]):
        if any(changes for changes in categories.values()):
            self._print("\n[bold]Changes by Category:[/bold]")
            for category, changes in categories.items():
                if changes:  # Only print categories with changes
                    self._print(f"\n[bold blue]{category}:[/bold blue]")
                    for change in changes:
                        self._print(f"  • {change}")
    
    def print_file_changes(self, file_changes: List[dict]):
        self._print("\n[bold]Important File Changes:[/bold]")
        for file in file_changes:
            self._print(f"\n[red]• {file['name']} ({file['status']} | +{file['additions']}/-{file['deletions']})[/red]")
            if file.get('changes') and self.show_diff:  # Only show diff if enabled
                self._print("  Changes:")
                if isinstance(file['changes'], str) and file['changes'].startswith('@@ '):
                    changes = file['changes'].splitlines()
                    current_chunk = []
                    for line in changes:
                        if line.startswith('@@'):
                            if current_chunk:
                                self._print("    ...")
                            current_chunk = []
                        elif line.startswith('+'):
                            self._print(f"    {line}", style="green")
                        elif line.startswith('-'):
                            self._print(f"    {line}", style="red")
                        else:
                            self._print(f"    {line}")
                else:
                    self._print(f"    {file['changes']}")
    
    def render_output(self):
        """Render the buffered output through the pager"""
        content = self.buffer.getvalue()
        with self.console.pager():
            self.console.print(content)

class MarkdownFormatter(OutputFormatter):
    """Markdown output formatter"""
    
    def __init__(self, output: TextIO = sys.stdout, show_diff: bool = True):
        super().__init__(output, show_diff)
    
    def print_summary(self, total_commits: int, total_files: int, important_files: int):
        print("\n## Summary", file=self.output)
        print(f"- Total commits: {total_commits}", file=self.output)
        print(f"- Files changed: {total_files}", file=self.output)
        print(f"- Important files modified: {important_files}", file=self.output)
    
    def print_commits_by_type(self, commit_groups: Dict[str, List[str]]):
        print("\n## Commits by Type", file=self.output)
        for group, commits in commit_groups.items():
            if commits:
                print(f"\n### {group.upper()}", file=self.output)
                for commit in commits:
                    print(f"- {commit}", file=self.output)
    
    def print_changes_by_category(self, categories: Dict[str, List[str]]):
        if any(changes for changes in categories.values()):
            print("\n## Changes by Category", file=self.output)
            for category, changes in categories.items():
                if changes:
                    print(f"\n### {category}", file=self.output)
                    for change in changes:
                        print(f"- {change}", file=self.output)
    
    def print_file_changes(self, file_changes: List[dict]):
        print("\n## Important File Changes", file=self.output)
        for file in file_changes:
            print(f"\n### {file['name']}", file=self.output)
            print(f"Status: {file['status']} | +{file['additions']}/-{file['deletions']}", file=self.output)
            if file.get('changes') and self.show_diff:  # Only show diff if enabled
                # Always use diff format for changes
                print("\n```diff", file=self.output)
                if isinstance(file['changes'], str) and file['changes'].startswith('@@ '):
                    changes = file['changes'].splitlines()
                    for line in changes:
                        if not line.startswith('\\'):
                            print(line, file=self.output)
                else:
                    print(file['changes'], file=self.output)
                print("```", file=self.output)
                print("", file=self.output)

class Diffinator:
    def __init__(self, token: Optional[str] = None, owner: str = None, 
                 repo: str = None, config_path: Optional[str] = None,
                 formatter: Optional[OutputFormatter] = None):
        """
        Initialize Diffinator using GitHub's REST API
        
        Args:
            token: GitHub API token (optional - only needed to avoid rate limits)
            owner: GitHub repository owner
            repo: GitHub repository name
            config_path: Path to repository configuration YAML file
            formatter: Output formatter instance
        """
        if not owner or not repo:
            raise ValueError("Both owner and repo must be specified")

        self.gh = GitHub(token) if token else GitHub()
        self.console = Console()
        self.owner = owner
        self.repo = repo
        self.config_path = Path(config_path) if config_path else None
        
        # Load configuration
        self.config = self._load_config()
        
        # Get repository-specific configuration or fall back to default
        repo_key = f"{self.owner}/{self.repo}"
        repo_config = self.config['repositories'].get(repo_key)
        
        if not repo_config:
            # If no specific config found, use defaults
            repo_config = self.config['repositories']['default']
            self.console.print(f"[yellow]Warning: No specific configuration found for {repo_key}, using defaults[/yellow]")
        
        # Set up groups from config
        self.groups = repo_config['groups']
        
        # Set up change groups for categorization
        self.change_groups = [
            ChangeGroup(
                name=group_config['name'],
                keywords=group_config['keywords']
            )
            for group_config in self.groups.values()
        ]
        
        # Set up commit groups using the same groups, plus 'other'
        self.commit_groups = {
            **{group_id: [] for group_id in self.groups.keys()},
            'other': []  # Add 'other' group for uncategorized commits
        }
        
        # Set up important files from config
        self.important_files = repo_config['important_files']
        
        # Set up formatter
        self.formatter = formatter or ConsoleFormatter(self.console)

    def _load_config(self) -> dict:
        """Load configuration from YAML files"""
        # Load defaults first
        default_paths = [
            Path(__file__).parent / "configs" / "defaults.yaml",        # Package configs directory
            Path.home() / ".config" / "diffinator" / "defaults.yaml",     # User's config directory
            Path("configs/defaults.yaml"),                              # Local configs directory
        ]
        
        defaults = None
        for path in default_paths:
            if path.exists():
                with open(path, 'r') as f:
                    defaults = yaml.safe_load(f)
                break
        
        if not defaults:
            raise FileNotFoundError("Could not find defaults.yaml configuration file")
        
        # Load repository-specific config
        repo_config = {'repositories': {}}
        
        if self.config_path:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Could not find configuration file: {self.config_path}")
            with open(self.config_path, 'r') as f:
                repo_config = yaml.safe_load(f)
        
        # Create the final config structure
        return {
            'repositories': {
                **repo_config.get('repositories', {}),
                'default': defaults
            }
        }

    def get_release(self, tag: str) -> Release:
        """
        Get release information for a specific tag
        Uses: GET /repos/{owner}/{repo}/releases/tags/{tag}
        """
        response: Response[Release] = self.gh.rest.repos.get_release_by_tag(
            owner=self.owner,
            repo=self.repo,
            tag=tag
        )
        return response.parsed_data
    
    def compare_releases(self, base: str, head: str) -> CommitComparison:
        """
        Compare two releases
        Uses: GET /repos/{owner}/{repo}/compare/{basehead}
        Note: basehead format is 'base...head'
        """
        response: Response[CommitComparison] = self.gh.rest.repos.compare_commits(
            owner=self.owner,
            repo=self.repo,
            basehead=f"{base}...{head}"
        )
        return response.parsed_data

    def categorize_changes(self, release_notes: str) -> Dict[str, List[str]]:
        """Categorize changes from release notes into groups"""
        # Split release notes into lines and clean them
        lines = [line.strip() for line in release_notes.split('\n') if line.strip()]
        
        # Initialize groups
        for group in self.change_groups:
            group.changes = []

        # Categorize each line
        uncategorized = []
        for line in lines:
            line_lower = line.lower()
            categorized = False
            
            for group in self.change_groups:
                # Check if line starts with any of the keywords
                if any(line_lower.startswith(keyword.lower()) for keyword in group.keywords):
                    group.changes.append(line)
                    categorized = True
                    break
            
            if not categorized:
                uncategorized.append(line)

        return {
            group.name: group.changes for group in self.change_groups
        } | {"Other": uncategorized}

    def analyze_file_changes(self, comparison: CommitComparison) -> Dict[str, List[str]]:
        """Analyze which important files were changed"""
        results = {
            "important": [],
            "other": []
        }
        
        # Look through all files in the comparison
        for file in comparison.files:
            # Normalize path separators to forward slashes
            filename = file.filename.replace('\\', '/')
            if filename in self.important_files:
                patch_content = None
                if hasattr(file, 'patch') and file.patch and not isinstance(file.patch, type(Unset)):
                    patch_content = file.patch
                elif hasattr(file, 'contents_url'):
                    # Try to get the diff by comparing before/after content
                    try:
                        # Get the file at both commits
                        contents_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{filename}"
                        base_response = self.gh.request('GET', contents_url, params={'ref': comparison.merge_base_commit.sha})
                        head_response = self.gh.request('GET', contents_url, params={'ref': comparison.commits[-1].sha})
                        
                        if base_response.status_code == 200 and head_response.status_code == 200:
                            import base64, difflib
                            # Decode base64 content
                            base_content = base64.b64decode(base_response.json()['content']).decode('utf-8')
                            head_content = base64.b64decode(head_response.json()['content']).decode('utf-8')
                            
                            # Create a unified diff
                            diff = difflib.unified_diff(
                                base_content.splitlines(keepends=True),
                                head_content.splitlines(keepends=True),
                                fromfile=f'a/{filename}',
                                tofile=f'b/{filename}'
                            )
                            patch_content = ''.join(diff)
                    except Exception as e:
                        patch_content = f"Could not fetch diff: {str(e)}"
                
                if not patch_content and file.status == "modified":
                    patch_content = (
                        f"Changes: +{file.additions} additions, -{file.deletions} deletions" 
                        if hasattr(file, 'additions') and hasattr(file, 'deletions') 
                        else "File was modified but diff is not available"
                    )
                
                results["important"].append({
                    "name": filename,
                    "changes": patch_content,
                    "status": file.status,
                    "additions": getattr(file, 'additions', 0),
                    "deletions": getattr(file, 'deletions', 0)
                })
            else:
                results["other"].append(filename)
        return results

    def print_results(self, categories: Dict[str, List[str]], comparison: CommitComparison, file_changes: Dict[str, List[str]]):
        """Print analysis results using the configured formatter"""
        # Print summary
        self.formatter.print_summary(
            len(comparison.commits),
            len(comparison.files),
            len(file_changes['important'])
        )
        
        # If raw mode, just print commits without categorization
        if hasattr(self.config, 'output_mode') and self.config['output_mode'] == 'raw':
            self.formatter.print_commits_by_type({'Raw Changes': [
                commit.commit.message.splitlines()[0] 
                for commit in comparison.commits
            ]})
            return
        
        # Otherwise, only show the Changes by Category (removing Commits by Type)
        self.formatter.print_changes_by_category(categories)
        self.formatter.print_file_changes(file_changes['important'])
        
        # Render the output if it's a console formatter
        if isinstance(self.formatter, ConsoleFormatter):
            self.formatter.render_output()

    def compare_commits(self, base: str, head: str) -> CommitComparison:
        """
        Compare two commits directly
        Uses: GET /repos/{owner}/{repo}/compare/{basehead}
        Note: basehead format is 'base...head'
        """
        response: Response[CommitComparison] = self.gh.rest.repos.compare_commits(
            owner=self.owner,
            repo=self.repo,
            basehead=f"{base}...{head}"
        )
        return response.parsed_data

    def get_commit_notes(self, comparison: CommitComparison) -> str:
        """Generate release-note style text from commit messages"""
        notes = []
        for commit in comparison.commits:
            # Get the first line and clean up spaces around colons
            message = commit.commit.message.splitlines()[0]
            if ':' in message:
                prefix, rest = message.split(':', 1)
                message = f"{prefix.strip()}: {rest.strip()}"
            notes.append(message)
        
        return "\n".join(notes)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate changelog between two git refs')
    parser.add_argument("version_a", help="First version/commit to compare", nargs='?')
    parser.add_argument("version_b", help="Second version/commit to compare", nargs='?')
    parser.add_argument("-t", "--token", help="GitHub API token (optional - only needed to avoid rate limits)")
    parser.add_argument("-c", "--config", dest="config",
                       help="Name of bundled config (e.g. 'llamacpp') or path to custom YAML config file")
    parser.add_argument("-r", "--repo", help="GitHub repository in the format 'owner/repo' (e.g. 'tensorflow/tensorflow')")
    parser.add_argument("-v", "--version-type", choices=['tag', 'commit'], 
                       help="Specify whether versions are tags or commits (overrides config file)")
    parser.add_argument("-o", "--output", choices=['console', 'markdown'], default='console',
                       help="Output format (default: console)")
    parser.add_argument("-f", "--output-file", help="Output file (default: stdout)")
    parser.add_argument("-n", "--nodiff", action="store_true", 
                       help="Omit diff output from file changes")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available bundled configurations")
    parser.add_argument('--raw', action='store_true', help='Output raw uncategorized list of changes')
    return parser.parse_args()

def generate_report(config, old_ref, new_ref, args):
    # ... existing code ...

    if args.raw:
        print("\nRaw Changes:")
        print("============")
        for commit in commits:
            print(f"  • {commit.title} ({commit.sha})")
        return

    # ... rest of the existing report generation code ...

def main():
    args = parse_args()
    
    # Handle listing configs
    if args.list_configs:
        configs_dir = Path(__file__).parent / "configs"
        if configs_dir.exists():
            print("\nBundled configurations:")
            for config_file in configs_dir.glob("*.yaml"):
                print(f"  • {config_file.name}")
                # Optionally show some details from the config
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        if 'description' in config:
                            print(f"    {config.get('description', '')}")
                except Exception as e:
                    print(f"    Error reading config: {e}")
            return 0
        else:
            print("No bundled configurations found.")
            return 1

    # For comparison operations, require version arguments and either config or repo
    if not args.config and not args.repo:
        parser.error("either -c/--config or -r/--repo is required")
    
    if not args.version_a or not args.version_b:
        parser.error("version_a and version_b are required for comparison")

    try:
        # Load config first to get owner and repo
        # First check if it's a bundled config
        if args.repo and not args.config:
            bundled_path = Path(__file__).parent / "configs" / "defaults.yaml"
        else:
            bundled_path = Path(__file__).parent / "configs" / f"{args.config}.yaml"
        
        if bundled_path.exists():
            config_path = bundled_path
        else:
            # If bundled path with .yaml didn't work, try without it
            bundled_path = Path(__file__).parent / "configs" / args.config
            if bundled_path.exists():
                config_path = bundled_path
            else:
                # If not a bundled config, try as a direct path
                config_path = Path(args.config)
                if not config_path.exists() and not str(config_path).endswith('.yaml'):
                    config_path = Path(f"{args.config}.yaml")
        
        if not config_path.exists():
            configs_dir = Path(__file__).parent / "configs"
            raise FileNotFoundError(
                f"Could not find configuration file: {args.config}\n"
                f"Use --list-configs to see available bundled configurations"
            )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.repo and not args.config:
            owner, repo = args.repo.split('/')
            # Add repository-specific information
            config['repository'] = {
                'owner': owner,
                'name': repo,
                'version_type': args.version_type or 'tag'
            }
        else:
            if 'repository' not in config:
                raise ValueError("Configuration file must contain 'repository' section with 'owner' and 'name'")
                
            owner = config['repository']['owner']
            repo = config['repository']['name']
        
        # Get version type from args or config
        version_type = args.version_type or config['repository'].get('version_type', 'tag')
        
        # Set up output
        output = open(args.output_file, 'w') if args.output_file else sys.stdout
        formatter = (MarkdownFormatter(output, show_diff=not args.nodiff) if args.output == 'markdown' 
                    else ConsoleFormatter(Console(width=None), output, show_diff=not args.nodiff))
        
        diffinator = Diffinator(
            token=args.token,
            owner=owner,
            repo=repo,
            config_path=str(config_path),
            formatter=formatter
        )
        
        # Get version information based on type
        if version_type == 'tag':
            comparison = diffinator.compare_releases(args.version_a, args.version_b)
        else:
            comparison = diffinator.compare_commits(args.version_a, args.version_b)

        if args.raw:
            print("\nRaw Changes:")
            print("============")
            for commit in comparison.commits:
                message = commit.commit.message.splitlines()[0]
                print(f"  • {message}")
            return
        
        # If not raw output, continue with normal processing
        release_notes = diffinator.get_commit_notes(comparison)
        categories = diffinator.categorize_changes(release_notes)
        file_changes = diffinator.analyze_file_changes(comparison)
        
        # Print results
        print(f"\nAnalyzing changes between {args.version_a} and {args.version_b}:")
        diffinator.print_results(categories, comparison, file_changes)
        
        if args.output_file:
            output.close()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
