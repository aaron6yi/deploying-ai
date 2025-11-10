from assignment_chat.main import assignment_chat
import gradio as gr
from dotenv import load_dotenv

from utils.logger import get_logger

_logs = get_logger(__name__)

load_dotenv('.secrets')


def travel_lite_chat(message: str, history: list[dict]) -> str:
    """
    Thin wrapper so Gradio passes history straight into the core router.
    """
    return assignment_chat(message, history)


chat = gr.ChatInterface(
    fn=travel_lite_chat,
    type="messages",
    title="Travel Lite Guide",
    description="Concise, local style tips. Try /weather, /ask, /web, or /prefs.",
)


if __name__ == "__main__":
    _logs.info('Starting Assignment Chat App (Travel Lite Guide)...')
    chat.launch()



