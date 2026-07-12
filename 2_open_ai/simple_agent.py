import os
import subprocess
import requests
import gradio as gr
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, trace, function_tool, SQLiteSession
load_dotenv(override=True)


@function_tool
def show_notification(title: str, message: str) -> str:
    """Show a macOS desktop notification pop-up.

    Args:
        title: The bold title line of the notification.
        message: The body text shown under the title.
    """
    # Escape backslashes and quotes so they don't break the AppleScript string.
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    safe_message = message.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(
        ["osascript", "-e", f'display notification "{safe_message}" with title "{safe_title}"'],
        check=True,
    )
    return "Notification displayed."


agent = Agent(
    name="Jokesteller",
    instructions=(
        "You are an ironic joke teller about Epam Systems. After you tell a joke, always call the "
        "show_notification tool to pop it up on the screen."
    ),
    model="gpt-4o-mini",
    tools=[show_notification],
)


# Persistent conversation memory, stored in jokes.db so it survives restarts.
# The session_id groups turns together; the file path is where they're saved.
session = SQLiteSession("jokes_conversation", "jokes.db")


async def respond(message, history):
    """Called by Gradio on every user message.

    The session remembers previous turns automatically, so we only pass the
    new message — no manual to_input_list() bookkeeping needed.
    """
    with trace("Telling a joke"):
        result = await Runner.run(agent, message, session=session)
    return result.final_output


if __name__ == "__main__":
    gr.ChatInterface(
        respond,
        title="🤖 Epam Jokesteller",
        description="Ask for a joke. Every turn is saved to jokes.db and pops up as a notification.",
    ).launch()
    
