import unittest
from unittest.mock import Mock
from custom_llm import CustomChatLLM
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema.output import ChatResult, ChatGenerationChunk, ChatGeneration


class TestCustomChatLLM(unittest.TestCase):

    def setUp(self):
        # Set up any necessary objects for testing
        self.server_url = "http://43.156.130.37:8000"

    def test_generate(self):
        client = CustomChatLLM(server_url=self.server_url)
        # Call the generate method with some test messages
        messages = [HumanMessage(content="Hello")]
        result = client.generate([messages])

        # Assert that the result is of the expected type
        self.assertTrue(len(result.generations)>0)

        # Add more assertions based on the expected behavior of the generate method

    def test_stream(self):
        client = CustomChatLLM(server_url=self.server_url,streaming=True)
        # Call the stream method with some test messages
        messages = [HumanMessage(content="Hello")]
        chunks = list(client._stream(messages))

        self.assertTrue(len(chunks)>0)
        self.assertTrue(all(isinstance(chunk, ChatGenerationChunk) for chunk in chunks))

        # Add more assertions based on the expected behavior of the stream method

if __name__ == '__main__':
    unittest.main()
