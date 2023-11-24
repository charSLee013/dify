"""Wrapper around curstom embedding models."""
from typing import Any, List, Optional, Union, Dict
import requests
from pydantic import BaseModel, Field, Extra
from langchain.embeddings.base import Embeddings

# This class defines the request structure for creating embeddings.
class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = None  
    input: Union[str, List[str]] = Field(description="The input to embed.")  # The text or list of texts to embed.
    user: Optional[str] = Field(default=None)  # Optional user identifier.


class CustomEmbeddings(BaseModel, Embeddings):
    """Wrapper around Custom API for embedding models, compatible with OpenAI's embedding API."""

    client: Any  

    server_url: Optional[str] = None  
    model_name: Optional[str] = None  
    headers: Optional[Dict[str, str]] = None  # Additional HTTP headers to send with the request.
    params: Optional[Dict[str, str]] = None  # Query parameters to append to the URL.


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents by sending a single HTTP request with the specified headers and parameters.

        Args:
            texts (List[str]): texts for list[str]

        Raises:
            ValueError: http rasie 

        Returns:
            List[List[float]]: List of embeddings
        """""

        default_headers = {"Content-Type": "application/json"}
        # Update with any additional headers provided
        if self.headers:
            default_headers.update(self.headers)

        # Append query parameters to the URL if provided
        if self.params:
            params_str = '&'.join([f'{k}={v}' for k, v in self.params.items()])
            request_url = f"{self.server_url}/v1/embeddings?{params_str}"
        else:
            request_url = f"{self.server_url}/v1/embeddings"

        request_data = CreateEmbeddingRequest(
            model=self.model_name,
            input=texts
        )

        response = requests.post(
            request_url,
            headers=default_headers,
            json=request_data.dict(exclude_none=True)
        )

        if not response.ok:
            raise ValueError(f"Custom API HTTP {response.status_code} error: {response.text}")

        json_response = response.json()
        # Extract and return the embeddings from the response
        return [e['embedding'] for e in json_response['data']]

    def invoke_embedding(self, text: str) -> List[float]:
        """
        Invokes the embedding process for a single text by leveraging the embed_documents method.
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    # A convenience method that directly returns the embedding for a single query text.
    def embed_query(self, text: str) -> List[float]:
        # Uses invoke_embedding to process a single text.
        return self.invoke_embedding(text)