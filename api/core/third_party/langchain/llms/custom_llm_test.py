import unittest
from unittest.mock import Mock
from custom_llm import CustomChatLLM
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema.output import ChatResult, ChatGenerationChunk, ChatGeneration


class TestCustomChatLLM(unittest.TestCase):

    def setUp(self):
        # Set up any necessary objects for testing
        self.custom_chat = CustomChatLLM(base_url="http://43.156.175.194:8000")

    def test_generate(self):
        # Mock the post method of CustomClient to simulate API response
        mock_response = {"choices": [{"role": "assistant", "content": "Generated text"}]}

        # Call the generate method with some test messages
        messages = [HumanMessage(content="Hello")]
        result = self.custom_chat._generate(messages)

        # Assert that the result is of the expected type
        self.assertIsInstance(result, ChatResult)

        # Add more assertions based on the expected behavior of the generate method

    def test_stream(self):
        # Call the stream method with some test messages
        messages = [HumanMessage(content="Hello")]
        chunks = []
        for chunk in self.custom_chat._stream(messages=messages):
            chunks.append(chunk)
        # chunks = list(self.custom_chat._stream(messages))

        self.assertTrue(len(chunks)>0)
        self.assertTrue(all(isinstance(chunk, ChatGenerationChunk) for chunk in chunks))

        # Add more assertions based on the expected behavior of the stream method

if __name__ == '__main__':
    unittest.main()
