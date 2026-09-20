from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ScriptedModel(FakeMessagesListChatModel):
    """Plays back scripted AIMessages; ignores bound tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def ai_tool_call(name: str, args: dict, call_id: str = "1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
