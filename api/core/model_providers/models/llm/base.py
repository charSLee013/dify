from abc import abstractmethod
from typing import List, Optional, Any, Union
import decimal
import logging

from langchain.callbacks.manager import Callbacks
from langchain.schema import LLMResult, BaseMessage, ChatGeneration

from core.callback_handler.std_out_callback_handler import DifyStreamingStdOutCallbackHandler, DifyStdOutCallbackHandler
from core.helper import moderation
from core.model_providers.models.base import BaseProviderModel
from core.model_providers.models.entity.message import PromptMessage, MessageType, LLMRunResult, to_lc_messages
from core.model_providers.models.entity.model_params import ModelType, ModelKwargs, ModelMode, ModelKwargsRules
from core.model_providers.providers.base import BaseModelProvider
from core.third_party.langchain.llms.fake import FakeLLM

logger = logging.getLogger(__name__)


class BaseLLM(BaseProviderModel):
    model_mode: ModelMode = ModelMode.COMPLETION
    name: str
    model_kwargs: ModelKwargs
    credentials: dict
    streaming: bool = False
    type: ModelType = ModelType.TEXT_GENERATION
    deduct_quota: bool = True

    # 负责初始化BaseLLM对象。它设置了模型名称、模型参数、是否流式处理等,并且根据是否流式处理初始化默认的回调处理器
    def __init__(self, model_provider: BaseModelProvider,
                 name: str,
                 model_kwargs: ModelKwargs,
                 streaming: bool = False,
                 callbacks: Callbacks = None):
        self.name = name
        self.model_rules = model_provider.get_model_parameter_rules(name, self.type)
        self.model_kwargs = model_kwargs if model_kwargs else ModelKwargs(
            max_tokens=None,
            temperature=None,
            top_p=None,
            presence_penalty=None,
            frequency_penalty=None
        )
        self.credentials = model_provider.get_model_credentials(
            model_name=name,
            model_type=self.type
        )
        self.streaming = streaming

        if streaming:
            default_callback = DifyStreamingStdOutCallbackHandler()
        else:
            default_callback = DifyStdOutCallbackHandler()

        if not callbacks:
            callbacks = [default_callback]
        else:
            callbacks.append(default_callback)

        self.callbacks = callbacks

        client = self._init_client()
        super().__init__(model_provider, client)

    @abstractmethod
    def _init_client(self) -> Any:
        raise NotImplementedError

    @property
    def base_model_name(self) -> str:
        """
        get llm base model name

        :return: str
        """
        return self.name

    @property
    def price_config(self) -> dict:
        """
        获取模型价格配置,如每个令牌的成本等,并默认为美元计价
        """
        def get_or_default():
            default_price_config = {
                'prompt': decimal.Decimal('0'),
                'completion': decimal.Decimal('0'),
                'unit': decimal.Decimal('0'),
                'currency': 'USD'
            }
            rules = self.model_provider.get_rules()
            price_config = rules['price_config'][
                self.base_model_name] if 'price_config' in rules else default_price_config
            price_config = {
                'prompt': decimal.Decimal(price_config['prompt']),
                'completion': decimal.Decimal(price_config['completion']),
                'unit': decimal.Decimal(price_config['unit']),
                'currency': price_config['currency']
            }
            return price_config

        # 如果没有定义则初始化(懒汉)
        if not hasattr(self, '_price_config'):
            self._price_config = get_or_default()

        logger.debug(f"model: {self.name} price_config: {self._price_config}")
        return self._price_config

    def run(self, messages: List[PromptMessage],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs) -> LLMRunResult:
        """
        根据输入的提示消息和停止词来执行预测,并处理模型运行过程中的逻辑,如内容审查、配额检查、回调处理等,并返回LLMRunResult对象
        run predict by prompt messages and stop words.

        :param messages:
        :param stop:
        :param callbacks:
        :return:
        """
        # 将所有输入的messages合并成一个字符串，并对其进行内容审查
        moderation_result = moderation.check_moderation(
            self.model_provider,
            "\n".join([message.content for message in messages])
        )

        # 如果内容审查发现问题（moderation_result为None）
        # 则在kwargs中设置fake_response，这是一个预设的安全回应
        if not moderation_result:
            kwargs['fake_response'] = "I apologize for any confusion, " \
                                      "but I'm an AI assistant to be helpful, harmless, and honest."

        # 配额检查，确保未超出使用配额
        if self.deduct_quota:
            self.model_provider.check_quota_over_limit()

        # 设置回调 
        if not callbacks:
            callbacks = self.callbacks
        else:
            callbacks.extend(self.callbacks)

        # 如果kwargs中存在fake_response，则不会进行实际的预测调用，而是创建一个FakeLLM实例来生成假的预测结果
        if 'fake_response' in kwargs:
            prompts = self._get_prompt_from_messages(messages, ModelMode.CHAT)
            fake_llm = FakeLLM(
                response=kwargs['fake_response'],
                num_token_func=self.get_num_tokens,
                streaming=self.streaming,
                callbacks=callbacks
            )
            result = fake_llm.generate([prompts])
        else:
            try:
                # 调用LLM自身的实际运行逻辑 _run 来运行
                result = self._run(
                    messages=messages,
                    stop=stop,
                    callbacks=callbacks if not (self.streaming and not self.support_streaming) else None,
                    **kwargs
                )
            except Exception as ex:
                raise self.handle_exceptions(ex)
        
        # 解析_run方法返回的LLMResult对象，获取生成的内容
        function_call = None
        if isinstance(result.generations[0][0], ChatGeneration):
            completion_content = result.generations[0][0].message.content
            if 'function_call' in result.generations[0][0].message.additional_kwargs:
                function_call = result.generations[0][0].message.additional_kwargs.get('function_call')
        else:
            completion_content = result.generations[0][0].text

        # 如果设置为流式处理，但模型不支持流式处理，则会使用FakeLLM来模拟流式处理。
        if self.streaming and not self.support_streaming:
            # use FakeLLM to simulate streaming when current model not support streaming but streaming is True
            prompts = self._get_prompt_from_messages(messages, ModelMode.CHAT)
            fake_llm = FakeLLM(
                response=completion_content,
                num_token_func=self.get_num_tokens,
                streaming=self.streaming,
                callbacks=callbacks
            )
            fake_llm.generate([prompts])

        # 计算令牌使用情况
        # 如果结果中包含令牌使用信息，就使用这些信息来计算提示令牌数和完成令牌数。
        if result.llm_output and result.llm_output['token_usage']:
            prompt_tokens = result.llm_output['token_usage']['prompt_tokens']
            completion_tokens = result.llm_output['token_usage']['completion_tokens']
            total_tokens = result.llm_output['token_usage']['total_tokens']
        else:
            # 如果没有这些信息，就调用get_num_tokens方法来计算提示和完成的令牌数。
            prompt_tokens = self.get_num_tokens(messages)
            completion_tokens = self.get_num_tokens(
                [PromptMessage(content=completion_content, type=MessageType.ASSISTANT)])
            total_tokens = prompt_tokens + completion_tokens

        # 更新模型的最后使用时间
        self.model_provider.update_last_used()
        
        # 如果有设置扣减配额，则更新扣减后的配额
        if self.deduct_quota:
            self.model_provider.deduct_quota(total_tokens)

        # 返回一个LLMRunResult对象，包含生成内容、提示令牌数、完成令牌数和可能的函数调用信息
        return LLMRunResult(
            content=completion_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            function_call=function_call
        )

    @abstractmethod
    def _run(self, messages: List[PromptMessage],
             stop: Optional[List[str]] = None,
             callbacks: Callbacks = None,
             **kwargs) -> LLMResult:
        """
        run predict by prompt messages and stop words.

        :param messages:
        :param stop:
        :param callbacks:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_tokens(self, messages: List[PromptMessage]) -> int:
        """
        get num tokens of prompt messages.

        :param messages:
        :return:
        """
        raise NotImplementedError

    def calc_tokens_price(self, tokens: int, message_type: MessageType) -> decimal.Decimal:
        """
        calc tokens total price.

        :param tokens:
        :param message_type:
        :return:
        """
        if message_type == MessageType.USER or message_type == MessageType.SYSTEM:
            unit_price = self.price_config['prompt']
        else:
            unit_price = self.price_config['completion']
        unit = self.get_price_unit(message_type)

        total_price = tokens * unit_price * unit
        total_price = total_price.quantize(decimal.Decimal('0.0000001'), rounding=decimal.ROUND_HALF_UP)
        logging.debug(f"tokens={tokens}, unit_price={unit_price}, unit={unit}, total_price:{total_price}")
        return total_price

    def get_tokens_unit_price(self, message_type: MessageType) -> decimal.Decimal:
        """
        get token price.

        :param message_type:
        :return: decimal.Decimal('0.0001')
        """
        if message_type == MessageType.USER or message_type == MessageType.SYSTEM:
            unit_price = self.price_config['prompt']
        else:
            unit_price = self.price_config['completion']
        unit_price = unit_price.quantize(decimal.Decimal('0.0001'), rounding=decimal.ROUND_HALF_UP)
        logging.debug(f"unit_price={unit_price}")
        return unit_price

    def get_price_unit(self, message_type: MessageType) -> decimal.Decimal:
        """
        get price unit.

        :param message_type:
        :return: decimal.Decimal('0.000001')
        """
        if message_type == MessageType.USER or message_type == MessageType.SYSTEM:
            price_unit = self.price_config['unit']
        else:
            price_unit = self.price_config['unit']

        price_unit = price_unit.quantize(decimal.Decimal('0.000001'), rounding=decimal.ROUND_HALF_UP)
        logging.debug(f"price_unit={price_unit}")
        return price_unit

    def get_currency(self) -> str:
        """
        get token currency.

        :return: get from price config, default 'USD'
        """
        currency = self.price_config['currency']
        return currency

    def get_model_kwargs(self):
        return self.model_kwargs

    def set_model_kwargs(self, model_kwargs: ModelKwargs):
        self.model_kwargs = model_kwargs
        self._set_model_kwargs(model_kwargs)

    @abstractmethod
    def _set_model_kwargs(self, model_kwargs: ModelKwargs):
        raise NotImplementedError

    @abstractmethod
    def handle_exceptions(self, ex: Exception) -> Exception:
        """
        Handle llm run exceptions.

        :param ex:
        :return:
        """
        raise NotImplementedError

    def add_callbacks(self, callbacks: Callbacks):
        """
        Add callbacks to client.

        :param callbacks:
        :return:
        """
        if not self.client.callbacks:
            self.client.callbacks = callbacks
        else:
            self.client.callbacks.extend(callbacks)

    @property
    def support_streaming(self):
        return False

    @property
    def support_function_call(self):
        return False

    def _get_prompt_from_messages(self, messages: List[PromptMessage],
                                  model_mode: Optional[ModelMode] = None) -> Union[str , List[BaseMessage]]:
        if not model_mode:
            model_mode = self.model_mode

        if model_mode == ModelMode.COMPLETION:
            if len(messages) == 0:
                return ''

            return messages[0].content
        else:
            if len(messages) == 0:
                return []

            return to_lc_messages(messages)

    def _to_model_kwargs_input(self, model_rules: ModelKwargsRules, model_kwargs: ModelKwargs) -> dict:
        """
        将模型参数转换为提供者模型的参数,考虑了规则中的别名、默认值、最小值和最大值等设置
        convert model kwargs to provider model kwargs.

        :param model_rules:
        :param model_kwargs:
        :return:
        """
        model_kwargs_input = {}
        for key, value in model_kwargs.dict().items():
            rule = getattr(model_rules, key)
            if not rule.enabled:
                continue

            if rule.alias:
                key = rule.alias

            if rule.default is not None and value is None:
                value = rule.default

            if rule.min is not None:
                value = max(value, rule.min)

            if rule.max is not None:
                value = min(value, rule.max)

            model_kwargs_input[key] = value

        return model_kwargs_input
