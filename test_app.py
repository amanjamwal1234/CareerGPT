import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

fake_gradio = types.ModuleType("gradio")
fake_gradio.ChatInterface = Mock()
original_gradio = sys.modules.get("gradio")
sys.modules["gradio"] = fake_gradio
import app
if original_gradio is None:
    sys.modules.pop("gradio", None)
else:
    sys.modules["gradio"] = original_gradio


def _tool_call(name, arguments, tool_call_id="tool-1"):
    return SimpleNamespace(
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(finish_reason, message):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)]
    )


class TestHelperFunctions(unittest.TestCase):
    @patch("app.requests.post")
    def test_push_posts_expected_payload(self, mock_post):
        with patch.dict(
            os.environ,
            {"PUSHOVER_TOKEN": "token-value", "PUSHOVER_USER": "user-value"},
            clear=False,
        ):
            app.push("hello")

        mock_post.assert_called_once_with(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": "token-value",
                "user": "user-value",
                "message": "hello",
            },
        )

    @patch("app.push")
    def test_record_user_details_calls_push_and_returns_ok(self, mock_push):
        result = app.record_user_details(
            email="test@example.com", name="Alex", notes="Interested in backend roles"
        )

        self.assertEqual(result, {"recorded": "ok"})
        mock_push.assert_called_once_with(
            "Recording Alex with email test@example.com and notes Interested in backend roles"
        )

    @patch("app.push")
    def test_record_unknown_question_calls_push_and_returns_ok(self, mock_push):
        result = app.record_unknown_question("What is your expected salary?")

        self.assertEqual(result, {"recorded": "ok"})
        mock_push.assert_called_once_with("Recording What is your expected salary?")


class TestMeConstructor(unittest.TestCase):
    @patch("app.OpenAI")
    @patch("app.PdfReader")
    def test_init_loads_profile_summary_and_openai_client(self, mock_pdf_reader, mock_openai):
        mock_pdf_reader.return_value.pages = [
            SimpleNamespace(extract_text=Mock(return_value=" First page text")),
            SimpleNamespace(extract_text=Mock(return_value=None)),
            SimpleNamespace(extract_text=Mock(return_value=" Second page text")),
        ]
        mocked_file = mock_open(read_data="Summary content")

        with patch("builtins.open", mocked_file), patch.dict(
            os.environ, {"GOOGLE_API_KEY": "api-key"}, clear=False
        ):
            me = app.Me()

        mock_openai.assert_called_once_with(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="api-key",
        )
        self.assertEqual(me.name, "Aman Jamwal")
        self.assertIn("linkedin.com/in/", me.linkedin)
        self.assertIn("First page text", me.linkedin)
        self.assertIn("Second page text", me.linkedin)
        self.assertEqual(me.summary, "Summary content")


class TestMeMethods(unittest.TestCase):
    def test_handle_tool_call_executes_known_tool(self):
        me = app.Me.__new__(app.Me)
        tool_calls = [_tool_call("record_user_details", {"email": "x@y.com"}, "call-123")]

        with patch("app.record_user_details", return_value={"recorded": "ok"}) as mock_tool:
            results = me.handle_tool_call(tool_calls)

        mock_tool.assert_called_once_with(email="x@y.com")
        self.assertEqual(
            results,
            [
                {
                    "role": "tool",
                    "content": json.dumps({"recorded": "ok"}),
                    "tool_call_id": "call-123",
                }
            ],
        )

    def test_handle_tool_call_handles_unknown_tool(self):
        me = app.Me.__new__(app.Me)
        results = me.handle_tool_call([_tool_call("does_not_exist", {"k": "v"}, "call-404")])

        self.assertEqual(
            results,
            [{"role": "tool", "content": json.dumps({}), "tool_call_id": "call-404"}],
        )

    def test_system_prompt_includes_required_context(self):
        me = app.Me.__new__(app.Me)
        me.name = "Aman Jamwal"
        me.summary = "Seasoned engineer."
        me.linkedin = "LinkedIn profile text."

        prompt = me.system_prompt()

        self.assertIn("You are acting as Aman Jamwal.", prompt)
        self.assertIn("record_unknown_question tool", prompt)
        self.assertIn("record_user_details tool", prompt)
        self.assertIn("## Summary:\nSeasoned engineer.", prompt)
        self.assertIn("## LinkedIn Profile:\nLinkedIn profile text.", prompt)

    def test_chat_returns_content_when_model_responds_without_tool_calls(self):
        me = app.Me.__new__(app.Me)
        me.system_prompt = Mock(return_value="SYSTEM PROMPT")
        me.openai = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        return_value=_response(
                            "stop", SimpleNamespace(content="Final answer", tool_calls=[])
                        )
                    )
                )
            )
        )

        history = [{"role": "assistant", "content": "Hi"}]
        result = me.chat("Tell me about your background", history)

        self.assertEqual(result, "Final answer")
        create_call = me.openai.chat.completions.create.call_args
        self.assertEqual(create_call.kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(
            create_call.kwargs["messages"],
            [
                {"role": "system", "content": "SYSTEM PROMPT"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Tell me about your background"},
            ],
        )
        self.assertEqual(create_call.kwargs["tools"], app.tools)

    def test_chat_processes_tool_calls_then_returns_final_content(self):
        me = app.Me.__new__(app.Me)
        me.system_prompt = Mock(return_value="SYSTEM PROMPT")

        tool_call = _tool_call(
            "record_user_details", {"email": "user@example.com"}, "call-777"
        )
        tool_message = SimpleNamespace(tool_calls=[tool_call], content=None)
        final_message = SimpleNamespace(content="Thanks, recorded that!", tool_calls=[])

        create_mock = Mock(
            side_effect=[
                _response("tool_calls", tool_message),
                _response("stop", final_message),
            ]
        )
        me.openai = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        me.handle_tool_call = Mock(
            return_value=[
                {
                    "role": "tool",
                    "content": json.dumps({"recorded": "ok"}),
                    "tool_call_id": "call-777",
                }
            ]
        )

        result = me.chat("Here is my email", [])

        self.assertEqual(result, "Thanks, recorded that!")
        self.assertEqual(create_mock.call_count, 2)
        me.handle_tool_call.assert_called_once_with([tool_call])

        second_call_messages = create_mock.call_args_list[1].kwargs["messages"]
        self.assertEqual(second_call_messages[0], {"role": "system", "content": "SYSTEM PROMPT"})
        self.assertEqual(second_call_messages[1], {"role": "user", "content": "Here is my email"})
        self.assertEqual(second_call_messages[2], tool_message)
        self.assertEqual(second_call_messages[3]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
