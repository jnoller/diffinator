#!/usr/bin/env python3
from typing import Dict, List, Optional
import sys
import argparse
from dataclasses import dataclass
from githubkit import GitHub, Response
from githubkit.versions.latest.models import (
    Release,  # For release info
    CommitComparison  # This should be the correct model name for the comparison endpoint
)
from githubkit.utils import Unset  # Add this import for Unset type
from rich.console import Console
from rich import print as rprint

@dataclass
class ChangeGroup:
    name: str
    keywords: List[str]
    changes: List[str] = None

    def __post_init__(self):
        self.changes = []

class LlamaLog:
    def __init__(self, token: Optional[str] = None, owner: str = "ggerganov", repo: str = "llama.cpp"):
        """
        Initialize LlamaLog using GitHub's REST API
        
        Args:
            token: GitHub API token (optional - only needed to avoid rate limits)
            owner: GitHub repository owner (default: ggerganov)
            repo: GitHub repository name (default: llama.cpp)
        """
        # Token is optional for public repos
        self.gh = GitHub(token) if token else GitHub()
        self.console = Console()
        self.owner = owner
        self.repo = repo
        
        # Set repository-specific configurations
        if owner == "ggerganov" and repo == "llama.cpp":
            # llama.cpp specific groups
            self.change_groups = [
                ChangeGroup("Server", ["server", "api", "endpoint", "http"]),
                ChangeGroup("GGML", ["ggml", "tensor", "quantize", "matrix"]),
                ChangeGroup("Models", ["model", "weight", "checkpoint", "gguf"]),
                ChangeGroup("Performance", ["performance", "speed", "faster", "optimize"]),
                ChangeGroup("Bug Fixes", ["fix", "bug", "issue", "crash"]),
            ]
            
            # Important files with normalized path separators
            self.important_files = [
                "gguf-py/pyproject.toml",  # Using forward slashes consistently
                "CMakeLists.txt",
                "requirements.txt",
                "server/server.cpp",
                "ggml.h",
                "ggml.c",
            ]
        else:
            # Default groups for any repository
            self.change_groups = [
                ChangeGroup("Features", ["feature", "add", "new", "support"]),
                ChangeGroup("Bug Fixes", ["fix", "bug", "issue", "crash"]),
                ChangeGroup("Documentation", ["doc", "readme", "example"]),
                ChangeGroup("Dependencies", ["dependency", "upgrade", "bump", "requirement"]),
                ChangeGroup("Performance", ["performance", "optimize", "speed", "faster"]),
            ]
            
            # Default important files for any repository
            self.important_files = [
                "CMakeLists.txt",
                "requirements.txt",
                "setup.py",
                "pyproject.toml",
                "Cargo.toml",
                "package.json",
            ]
        
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
                if any(keyword in line_lower for keyword in group.keywords):
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
        """Print analysis results"""
        # Print summary
        self.console.print("\n[bold]Summary:[/bold]")
        self.console.print(f"  • Total commits: [cyan]{len(comparison.commits)}[/cyan]")
        self.console.print(f"  • Files changed: [cyan]{len(comparison.files)}[/cyan]")
        self.console.print(f"  • Important files modified: [cyan]{len(file_changes['important'])}[/cyan]")

        # Group and print commits by type
        self.console.print("\n[bold]Commits by Type:[/bold]")
        commit_groups = {
            "server": [],
            "ggml": [],
            "vulkan": [],
            "cuda": [],
            "metal": [],
            "llama": [],     # llama model/architecture related changes
            "convert": [],   # model conversion related
            "cmake": [],     # build system changes
            "tests": [],     # test related changes
            "gguf-py": [],   # Python GGUF tool changes
            "tts": [],       # text-to-speech related
            "common": [],    # common utilities/features
            "devops": [],    # CI/CD, docker, infrastructure
            "docs": [],      # documentation
            "other": []      # truly miscellaneous
        }
        
        for commit in comparison.commits:
            message = commit.commit.message.splitlines()[0].lower()
            categorized = False
            for group in commit_groups.keys():
                if group in message and group != "other":
                    commit_groups[group].append(commit.commit.message.splitlines()[0])
                    categorized = True
                    break
            if not categorized:
                commit_groups["other"].append(commit.commit.message.splitlines()[0])

        for group, commits in commit_groups.items():
            if commits:  # Only show groups with commits
                self.console.print(f"\n[bold blue]{group.upper()}:[/bold blue]")
                for commit in commits:
                    self.console.print(f"  • [yellow]{commit}[/yellow]")

        # Print categorized changes
        if any(changes for changes in categories.values()):
            self.console.print("\n[bold]Changes by Category:[/bold]")
            for category, changes in categories.items():
                if changes:  # Only print categories with changes
                    self.console.print(f"\n[bold blue]{category}:[/bold blue]")
                    for change in changes:
                        self.console.print(f"  • {change}")

        # Print file changes with details
        self.console.print("\n[bold]Important File Changes:[/bold]")
        for file in file_changes["important"]:
            self.console.print(f"\n[red]• {file['name']} ({file['status']} | +{file['additions']}/-{file['deletions']})[/red]")
            if file.get('changes'):
                self.console.print("  Changes:")
                if isinstance(file['changes'], str) and file['changes'].startswith('@@ '):
                    # This is a git diff patch
                    changes = file['changes'].splitlines()
                    current_chunk = []
                    for line in changes:
                        if line.startswith('@@'):  # New diff chunk
                            if current_chunk:
                                self.console.print("    ...")
                            current_chunk = []
                        elif line.startswith('+'):
                            self.console.print(f"    [green]{line}[/green]")
                        elif line.startswith('-'):
                            self.console.print(f"    [red]{line}[/red]")
                        else:
                            self.console.print(f"    {line}")
                else:
                    # This is a message about the changes
                    self.console.print(f"    {file['changes']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze changes between llama.cpp releases")
    parser.add_argument("version_a", help="First version to compare")
    parser.add_argument("version_b", help="Second version to compare")
    parser.add_argument("--token", help="GitHub API token (optional - only needed to avoid rate limits)")
    parser.add_argument("--owner", default="ggerganov", help="GitHub repository owner (default: ggerganov)")
    parser.add_argument("--repo", default="llama.cpp", help="GitHub repository name (default: llama.cpp)")
    
    args = parser.parse_args()
    
    try:
        llamalog = LlamaLog(
            token=args.token,
            owner=args.owner,
            repo=args.repo
        )
        
        # Get release information
        release_a = llamalog.get_release(args.version_a)
        release_b = llamalog.get_release(args.version_b)
        
        # Compare releases
        comparison = llamalog.compare_releases(args.version_a, args.version_b)
        
        # Analyze changes
        categories = llamalog.categorize_changes(release_b.body if release_b.body else "")
        file_changes = llamalog.analyze_file_changes(comparison)
        
        # Print results
        print(f"\nAnalyzing changes between {args.version_a} and {args.version_b}:")
        llamalog.print_results(categories, comparison, file_changes)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
