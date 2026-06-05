import os
import warnings
import logging

# Suppress all warnings
warnings.simplefilter("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("pydantic").setLevel(logging.ERROR)

from typing import Any

import dotenv
import github.Auth
from github import Github
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.workflow import Workflow, step, Context, StartEvent, StopEvent, Event
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent, AgentOutput, ToolCall, ToolCallResult, AgentWorkflow, AgentStream
import asyncio
# from llama_index.core.agent.workflow.workflow_events import Event, StartEvent
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
import sys
import github.Auth

dotenv.load_dotenv()

DEBUG = os.getenv("DEBUG")

def debug(*args):
    if DEBUG:
        print("[DEBUG]", *args)
# Set logging level for llama_index to see if it helps reveal issues
logging.getLogger("llama_index").setLevel(logging.DEBUG)

github_token = github.Auth.Token(sys.argv[1])
git = Github(auth=github_token)
repo_url = sys.argv[2]
pr_number = sys.argv[3]
openai_api_key = sys.argv[4]
openai_base_url = sys.argv[5]

# github_token = github.Auth.Token(os.getenv('GITHUB_TOKEN'))
git = Github(auth=github_token)
# repo_url = os.getenv("REPO_URL")
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"
repo = git.get_repo(full_repo_name)

def get_pr_details(pr_number):
    pull_request = repo.get_pull(pr_number)
    return {
        "author": pull_request.user.login,
        "Title": pull_request.title,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": f"{pull_request.html_url}.diff",
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_shas": [commit.sha for commit in pull_request.get_commits()],
    }

def pr_commit_details(commit_sha):
    commit = repo.get_commit(commit_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files

def get_file_content(file_path):
    file_content = repo.get_contents(file_path)
    return file_content.decoded_content.decode()

async def add_review_to_state(ctx: Context, review):
    current_state = await ctx.get("state")
    current_state["final_review"] = review
    await ctx.set("state", current_state)

async def add_comment_to_state(ctx: Context, draft_comment):
    current_state = await ctx.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.set("state", current_state)

def post_review_to_pr(pr_number: int, comment: str):
    debug("POSTING REVIEW")
    debug("PR =", pr_number)
    debug("COMMENT =", comment[:500] if comment else "<EMPTY>")
    pull_request = repo.get_pull(pr_number)
    try:
        pull_request.create_review(body=comment, event="COMMENT")
        debug("REVIEW CREATED")
        debug("REVIEW ID =", getattr(result, "id", None))
    except Exception as e:
        if "one pending review" in str(e):
            # Try to find the pending review and submit it
            for review in pull_request.get_reviews():
                if review.user.login == git.get_user().login and review.state == "PENDING":
                    # In PyGithub, to submit a pending review, we use the method submit()
                    # If it's not present, we can try using the requester
                    if hasattr(review, "submit"):
                        review.submit(event="COMMENT", body=comment)
                        return
                    else:
                        # Fallback to direct API call via requester
                        review._requester.requestJsonAndCheck(
                            "POST",
                            f"{pull_request.url}/reviews/{review.id}/events",
                            input={"event": "COMMENT", "body": comment}
                        )
                        return
            # If no pending review found, just raise
            raise e
        raise e

get_pr_details_tool = FunctionTool.from_defaults(
    get_pr_details,
)

pr_commit_details_tool = FunctionTool.from_defaults(
    pr_commit_details,
)

get_file_content_tool = FunctionTool.from_defaults(
    get_file_content,
)

add_comment_to_state_tool = FunctionTool.from_defaults(
    add_comment_to_state,
)

add_review_to_state_tool = FunctionTool.from_defaults(
    add_review_to_state,
)

post_review_to_pr_tool = FunctionTool.from_defaults(
    post_review_to_pr,
)

llm = OpenAILike(
    model=os.getenv("OPENAI_MODEL"),
    api_key=openai_api_key,
    api_base=openai_base_url,
    is_function_calling_model=True,
    timeout=120
)

async def mark_context_as_gathered(ctx: Context):
    current_state = await ctx.get("state")
    current_state["context_gathered"] = True
    await ctx.set("state", current_state)

mark_context_as_gathered_tool = FunctionTool.from_defaults(
    mark_context_as_gathered,
)

async def main():
    query = "Write a review for PR: " + pr_number

    commenter_agent = ReActAgent(
        llm=llm,
        name="CommentorAgent",
        instruction=
        """You are the CommentorAgent.
Your mission is to draft a review comment.
STRICT RULES:
1. You MUST call 'get_file_content' with 'file_path="README.md"'.
2. You MUST call 'add_comment_to_state_tool' with a detailed markdown review based on the PR context.
3. You MUST call 'handoff' to 'ReviewAndPostingAgent'.
Do NOT provide an Answer.
""",
        description="Drafts a pull request review comment.",
        tools=[add_comment_to_state_tool, get_file_content_tool],
        can_handoff_to=["ReviewAndPostingAgent"]
    )

    review_and_posting_agent = ReActAgent(
        llm=llm,
        name="ReviewAndPostingAgent",
        instruction=
        f"""You are the ReviewAndPostingAgent.
Your mission is to post the review to GitHub.
STRICT RULES:
1. If 'draft_comment' is NOT in state, you MUST call 'handoff' to 'ContextAgent'.
2. If 'draft_comment' IS in state, you MUST call 'post_review_to_pr' with pr_number=#{pr_number} and the drafted comment.
3. ONLY AFTER 'post_review_to_pr' returns, you MUST provide a final 'Answer' starting with "SUCCESS: Review posted to PR #1".
""",
        description="Finalizes and posts the pull request review.",
        tools=[add_review_to_state_tool, post_review_to_pr_tool],
        can_handoff_to = ["ContextAgent", "CommentorAgent"]
    )

    context_agent = ReActAgent(
        name="ContextAgent",
        description="Gathers context for the pull request.",
        instruction=
        f"""You are the ContextAgent.
Your mission is to gather context for PR #1.
STRICT RULES:
1. You MUST call 'get_pr_details' with pr_number={pr_number}.
2. You MUST call 'pr_commit_details' with the head_sha from the output of get_pr_details.
3. You MUST call 'get_file_content' with file_path="app/models.py".
4. You MUST call 'handoff' to 'CommentorAgent'.
Do NOT provide an Answer.
""",
        tools=[get_pr_details_tool, pr_commit_details_tool, get_file_content_tool],
        llm=llm,
        can_handoff_to = ["CommentorAgent"]
    )
    orchestrator = AgentWorkflow(
        agents=[context_agent, commenter_agent, review_and_posting_agent],
        root_agent=review_and_posting_agent.name,
        initial_state={
            "gathered_contexts": "",
            "review_comment": "",
            "final_review": "",
        },
    )

    # 3. Run and Stream Events
    try:
        handler = orchestrator.run(user_msg=query)
    except Exception as e:
        print(f"Error starting workflow: {e}")
        return

    current_agent = None
    try:
        async for event in handler.stream_events():
            if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
                current_agent = event.current_agent_name
                print(f"Current agent: {current_agent}")

            if isinstance(event, ToolCall):
                print(f"Selected tools: ['{event.tool_name}']")
                print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")
            elif isinstance(event, ToolCallResult):
                print(f"Output from tool: {event.tool_output}")
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    print(event.response.content)
    except Exception as e:
        print(f"Error during workflow execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    git.close()
