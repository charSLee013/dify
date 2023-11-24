"""Wrapper around Custom OpenAI-compatible webserver."""
from __future__ import annotations

import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional, Iterator,
)

import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import AIMessageChunk
from langchain.schema.output import ChatResult, ChatGenerationChunk, ChatGeneration
from pydantic import Extra, BaseModel,PrivateAttr

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)

logger = logging.getLogger(__name__)


class CustomClient(BaseModel):
    server_url: str
    # Additional HTTP headers to send with the request.
    headers: Optional[Dict[str, str]] = None
    # Query parameters to append to the URL.
    params: Optional[Dict[str, str]] = None
    timeout: Optional[int] = 360  # Request time out

    def post(self, **request: Any) -> Any:
        """Send a POST request to the custom API server.
        """
        request_url = self.server_url + "/v1/chat/completions"
        if self.params:
            params_str = '&'.join([f'{k}={v}' for k, v in self.params.items()])
            request_url = f"{request_url}?{params_str}"

        default_headers = {"Content-Type": "application/json"}
        # Update with any additional headers provided
        if self.headers:
            default_headers.update(self.headers)
        stream = 'stream' in request and request['stream']
        response = requests.post(request_url,
                                 headers=default_headers,
                                 json=request,
                                 stream=stream,
                                 timeout=self.timeout
                                 )
        if response.status_code == 200:
            if stream:
                return response
            else:
                return response.json()
        elif response.status_code == 422:
            json_response = response.json()

            error_message = "\n".join(
                [f"{error['loc']} : {error['msg']}" for error in json_response["detail"]])

            raise ValueError(f"Parameter Error:\n{error_message}")
        else:
            raise ValueError(
                f"HTTP {response.status_code} error: {response.text}")


class CustomChatLLM(BaseChatModel):

    _client: CustomClient = PrivateAttr()
    model: str = "any-model"
    """Model name to use."""
    max_tokens: int = 4096
    """Denotes the number of tokens to predict per generation."""
    temperature: float = 0.7
    """A non-negative float that tunes the degree of randomness in generation."""
    top_p: float = 0.95
    """Total probability mass of tokens to consider at each step."""
    streaming: bool = False
    """Whether to stream the response or return it all at once."""
    completion_type:Optional[str] = None
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        
    """copy from CustomClient"""
    server_url: str
    # Additional HTTP headers to send with the request.
    headers: Optional[Dict[str, str]] = None
    # Query parameters to append to the URL.
    params: Optional[Dict[str, str]] = None
    timeout: Optional[int] = 360  # Request time out

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling API."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [{"role": "user", "content": "ping"}]
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "custsomLLM"
    
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = CustomClient(
            server_url=self.server_url,
            headers=self.headers,
            params=self.params,
            timeout=self.timeout,
        )

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        """Convert a BaseMessage object to a dictionary format suitable for sending to the API server.
        """
        content = message.content
        if isinstance(message, ChatMessage) or isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Got unknown type {message}")
        return {"content": content, "role": role}

    def _convert_dict_to_message(self, _dict: Dict[str, Any]) -> BaseMessage:
        """Convert a dictionary format received from the API server back to a BaseMessage object.
        """
        role = str(_dict["role"])
        if role.lower() == "user":
            return HumanMessage(content=_dict["content"])
        elif role.lower() == "assistant":
            return AIMessage(content=_dict["content"])
        elif role.lower() == "system":
            return SystemMessage(content=_dict["content"])
        else:
            return ChatMessage(content=_dict["content"], role=role)

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Convert a list of BaseMessage objects to a list of dictionary formats suitable for sending to the API server.
        """
        dict_messages = []
        sys_messages = []   # may sure system message only be top
        for m in messages:
            single_message = self._convert_message_to_dict(m)
            if isinstance(m, SystemMessage):
                sys_messages.append(single_message)
            else:
                dict_messages.append(single_message)
        return sys_messages+dict_messages

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate text using the custom language model.

        This method converts input messages into dictionary format and sends them as part of a POST request to generate text. It then processes the response and returns it as a `ChatResult`.
        """
        if self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            llm_output: Optional[Dict] = None
            for chunk in self._stream(
                    messages=messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk

                if chunk.generation_info is not None \
                        and 'token_usage' in chunk.generation_info:
                    llm_output = {
                        "token_usage": chunk.generation_info['token_usage'], "model_name": self.model}

            assert generation is not None
            return ChatResult(generations=[generation], llm_output=llm_output)
        else:
            message_dicts = self._create_message_dicts(messages)
            params = self._default_params
            params["messages"] = message_dicts
            params.update(kwargs)
            print(f"params: {params}\t")
            response = self._client.post(**params)
            return self._create_chat_result(response, stop)

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Streams the text generation process and returns an iterator of ChatGenerationChunk objects.
        """
        message_dicts = self._create_message_dicts(messages)
        params = self._default_params
        params["messages"] = message_dicts
        params['stream'] = True
        params.update(kwargs)

        for event in self._client.post(**params).iter_lines():
            if event:
                chunk = event.decode("utf-8")   # decode to 'data: {"id": "", "model": "any-model"}'
                if chunk.startswith('data:'):
                    try:
                        data = json.loads(chunk[5:])
                    except:
                        # it done
                        break
                    if not (data['choices'] and 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']):
                        continue
                    content = data['choices'][0]['delta']['content']
                    chunk_kwargs = {
                        'message': AIMessageChunk(content=content),
                    }
                    if 'usage' in data:
                        token_usage = data['usage']
                        overall_token_usage = {
                            'prompt_tokens': token_usage.get('prompt_tokens', 0),
                            'completion_tokens': token_usage.get('total_tokens', 0),
                            'total_tokens': token_usage.get('total_tokens', 0)
                        }
                        chunk_kwargs['generation_info'] = {
                            'token_usage': overall_token_usage}

                    yield ChatGenerationChunk(**chunk_kwargs)
                    if run_manager:
                        run_manager.on_llm_new_token(content)

    def _create_chat_result(self, response: Dict[str, Any], stop: Optional[List[str]] = None) -> ChatResult:
        """
         creates a ChatResult object from the API response and stop criteria. It processes the API response and combines it with the stop criteria to create a ChatResult object.
         """
        generations = []
        for res in response['choices']:
            if 'message' in res:
                res = res['message']
            message = self._convert_dict_to_message(res)
            gen = ChatGeneration(
                message=message
            )
            generations.append(gen)
        usage = response.get("usage")
        token_usage = {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the messages.
        """
        return sum([self.get_num_tokens(m.content) for m in messages])

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]

        return {"token_usage": token_usage, "model_name": self.model}
