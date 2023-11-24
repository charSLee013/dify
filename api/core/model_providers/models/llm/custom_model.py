import decimal
import logging
from typing import List, Optional, Any

from langchain.callbacks.manager import Callbacks
from langchain.schema import LLMResult

from core.model_providers.error import LLMBadRequestError
from core.model_providers.models.llm.base import BaseLLM
from core.model_providers.models.entity.message import PromptMessage
from core.model_providers.models.entity.model_params import ModelMode, ModelKwargs
from core.third_party.langchain.llms.custom_llm import CustomChatLLM
from core.model_providers.providers.base import BaseModelProvider


class CustomLLM(BaseLLM):
    def __init__(self,model_provider:BaseModelProvider,
                 name:str,
                 model_kwargs:ModelKwargs,
                 streaming:bool = False,
                 callbacks:Callbacks = None):
        credentials = model_provider.get_model_credentials(
            model_name=name,
            model_type=self.type
        )
        if credentials['completion_type'] == 'chat_completion':
            self.model_mode = ModelMode.CHAT
        else:
            self.model_mode = ModelMode.COMPLETION

        super().__init__(model_provider, name, model_kwargs, streaming, callbacks)
        
    def _init_client(self) -> Any:
        provider_model_kwargs = self._to_model_kwargs_input(self.model_rules, self.model_kwargs)
        
        client = CustomChatLLM(model=self.name,
                               **provider_model_kwargs,
                               **self.credentials)
        return client
        
    def _run(self,messages:List[PromptMessage],
             stop:Optional[List[str]] = None,
             callbacks:Callbacks = None,
             **kwargs) -> LLMResult:
        prompts = self._get_prompt_from_messages(messages)
        if not isinstance(prompts,list):
            prompts = [prompts]
        return self._client.generate([prompts], stop, callbacks)
    
    def get_num_tokens(self, messages: List[PromptMessage]) -> int:
        """
        get num tokens of prompt messages.

        :param messages:
        :return:
        """
        prompts = self._get_prompt_from_messages(messages)
        return max(self._client.get_num_tokens_from_messages(prompts), 0)
    
    def _set_model_kwargs(self, model_kwargs: ModelKwargs):
        """
        Allow for dynamically setting or updating key parameters of the model, such as temperature, maximum token count, etc.
        """
        provider_model_kwargs = self._to_model_kwargs_input(self.model_rules, model_kwargs)
        for k, v in provider_model_kwargs.items():
            if hasattr(self.client, k):
                setattr(self.client, k, v)

    def handle_exceptions(self, ex: Exception) -> Exception:
        return LLMBadRequestError(f"CustomLLM: {str(ex)}")
    
    @classmethod
    def support_streaming(cls):
        return True