import datetime
import logging
import os
import subprocess

from github import Github

# We will use the agent proxy LLMClient to communicate with the LLM.
from etft.llm import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_codebase_state():
    """Returns a string representation of the codebase tree structure."""
    try:
        # Get a simple tree structure using find or tree, ignoring some common irrelevant folders
        result = subprocess.run(
            ['find', '.', '-not', '-path', r'*/\.*', '-not', '-path', '*/__pycache__/*', '-not', '-path', '*/.venv/*', '-type', 'f'],
            capture_output=True,
            text=True
        )
        files = result.stdout.splitlines()
        return "\n".join(sorted(files))
    except Exception as e:
        logger.error(f"Error getting codebase state: {e}")
        return "Error getting codebase state."


def main():
    logger.info("Starting daily automation...")

    # Load roadmap
    roadmap_path = "strategy_docs/development_roadmap.md"
    try:
        with open(roadmap_path) as f:
            roadmap_content = f.read()
    except FileNotFoundError:
        logger.error(f"Roadmap not found at {roadmap_path}")
        return

    # Get codebase state
    codebase_state = get_codebase_state()

    prompt = f"""You are the lead developer of the AutoScientist-ETFT project.
Review the following development roadmap and the current codebase state (list of files).
Formulate a precise gap analysis between the current state and the goal described in the document in production quality.
Then, create a user-testable (functional or non-functional, testable by end-user or human tester via CLI scripts) use-case slice fitting the timebox of one day implementation.

Development Roadmap:
{roadmap_content}

Current Codebase State (List of Files):
{codebase_state}

Format your response exactly as follows:
===GAP ANALYSIS===
<Markdown for Gap Analysis>
===DAILY TASK===
<Markdown for Daily Task>
"""

    llm = LLMClient()
    logger.info("Querying LLM for gap analysis and daily task...")
    try:
        response_text = llm.complete(prompt, system="You are an expert lead developer.")
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return

    if "===GAP ANALYSIS===" not in response_text or "===DAILY TASK===" not in response_text:
        logger.error("LLM response did not contain the expected markers.")
        return

    parts = response_text.split("===DAILY TASK===")
    gap_analysis = parts[0].replace("===GAP ANALYSIS===", "").strip()
    daily_task = parts[1].strip()

    # Add the restriction to the daily task
    daily_task += (
        "\n\n**Note for Dev Agent:** You may only open and mark progress on this issue. You must never close it."
    )

    # Create directories
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join("daily_reports", date_str)
    os.makedirs(out_dir, exist_ok=True)

    # Write files
    gap_file = os.path.join(out_dir, "gap_analysis.md")
    with open(gap_file, "w") as f:
        f.write(gap_analysis)

    task_file = os.path.join(out_dir, "daily_task.md")
    with open(task_file, "w") as f:
        f.write(daily_task)

    logger.info(f"Saved gap analysis to {gap_file}")
    logger.info(f"Saved daily task to {task_file}")

    # Create GitHub Issue
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.warning("GITHUB_TOKEN not found. Skipping issue creation.")
        return

    repo_name = os.environ.get("GITHUB_REPOSITORY")
    if not repo_name:
        logger.warning("GITHUB_REPOSITORY not found. Skipping issue creation.")
        return

    try:
        g = Github(github_token)
        repo = g.get_repo(repo_name)

        issue_title = f"Daily Task for {date_str}"
        issue = repo.create_issue(title=issue_title, body=daily_task)
        logger.info(f"Created GitHub issue: {issue.html_url}")
    except Exception as e:
        logger.error(f"Error creating GitHub issue: {e}")


if __name__ == "__main__":
    main()
