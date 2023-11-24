from abc import ABC, abstractmethod
from datetime import datetime
from typing import Type, Optional

from flask import current_app
from pydantic import BaseModel

from core.model_providers.error import QuotaExceededError, LLMBadRequestError
from extensions.ext_database import db
from core.model_providers.models.entity.model_params import ModelType, ModelKwargsRules
from core.model_providers.models.entity.provider import ProviderQuotaUnit
from core.model_providers.rules import provider_rules
from models.provider import Provider, ProviderType, ProviderModel


class BaseModelProvider(BaseModel, ABC):
    """用于处理不同机器学习模型提供商的基础类。该类提供了一组抽象方法和属性，用于定义处理模型提供商相关任务的接口
    主要功能分为五种:
    1. 提供商信息获取
    2. 模型信息获取
    3. 凭证处理
    4. 配额和扣除
    5. 其他功能
    """
    provider: Provider

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def provider_name(self):
        """
        Returns the name of a provider.
        1.获取提供商的名称。
        """
        raise NotImplementedError

    def get_rules(self):
        """
        Returns the rules of a provider.
        1.获取提供商的规则，例如支持的提供商类型等。
        """
        return provider_rules[self.provider_name]

    def get_supported_model_list(self, model_type: ModelType) -> list[dict]:
        """
        get supported model object list for use.
        2.获取支持的模型列表，考虑了提供商规则和模型类型。

        :param model_type:
        :return:
        """
        rules = self.get_rules()
        if 'custom' not in rules['support_provider_types']:
            return self._get_fixed_model_list(model_type)

        if 'model_flexibility' not in rules:
            return self._get_fixed_model_list(model_type)

        if rules['model_flexibility'] == 'fixed':
            return self._get_fixed_model_list(model_type)

        # get configurable provider models
        provider_models = db.session.query(ProviderModel).filter(
            ProviderModel.tenant_id == self.provider.tenant_id,
            ProviderModel.provider_name == self.provider.provider_name,
            ProviderModel.model_type == model_type.value,
            ProviderModel.is_valid == True
        ).order_by(ProviderModel.created_at.asc()).all()

        provider_model_list = []
        for provider_model in provider_models:
            provider_model_dict = {
                'id': provider_model.model_name,
                'name': provider_model.model_name
            }

            if model_type == ModelType.TEXT_GENERATION:
                provider_model_dict['mode'] = self._get_text_generation_model_mode(provider_model.model_name)

            provider_model_list.append(provider_model_dict)

        return provider_model_list

    @abstractmethod
    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        """
        get supported model object list for use.
        2.获取固定模型列表的抽象方法，由子类实现。

        :param model_type:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _get_text_generation_model_mode(self, model_name) -> str:
        """
        get text generation model mode.
        2.获取文本生成模型的模式，由子类实现。

        :param model_name:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_class(self, model_type: ModelType) -> Type:
        """
        get specific model class.
        2.获取特定模型类型的模型类，由子类实现

        :param model_type:
        :return:
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_provider_credentials_valid_or_raise(cls, credentials: dict):
        """
        check provider credentials valid.
        3.检查提供商凭证是否有效。

        :param credentials:
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def encrypt_provider_credentials(cls, tenant_id: str, credentials: dict) -> dict:
        """
        encrypt provider credentials for save.
        3.加密提供商凭证以便保存。

        :param tenant_id:
        :param credentials:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_provider_credentials(self, obfuscated: bool = False) -> dict:
        """
        get credentials for llm use.
        3.获取用于LLM（语言模型）使用的提供商凭证。

        :param obfuscated:
        :return:
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        """
        check model credentials valid.
        3.检查模型凭证是否有效。

        :param model_name:
        :param model_type:
        :param credentials:
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType,
                                  credentials: dict) -> dict:
        """
        encrypt model credentials for save.
        3.加密模型凭证以便保存。

        :param tenant_id:
        :param model_name:
        :param model_type:
        :param credentials:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        """
        get model parameter rules.
        3.获取模型参数的规则

        :param model_name:
        :param model_type:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_credentials(self, model_name: str, model_type: ModelType, obfuscated: bool = False) -> dict:
        """
        get credentials for llm use.
        3.获取用于LLM使用的模型凭证。

        :param model_name:
        :param model_type:
        :param obfuscated:
        :return:
        """
        raise NotImplementedError

    @classmethod
    def is_provider_type_system_supported(cls) -> bool:
        return current_app.config['EDITION'] == 'CLOUD'

    def check_quota_over_limit(self):
        """
        check provider quota over limit.
        4.检查提供商配额是否超过限制。

        :return:
        """
        if self.provider.provider_type != ProviderType.SYSTEM.value:
            return

        rules = self.get_rules()
        if 'system' not in rules['support_provider_types']:
            return

        provider = db.session.query(Provider).filter(
            db.and_(
                Provider.id == self.provider.id,
                Provider.is_valid == True,
                Provider.quota_limit > Provider.quota_used
            )
        ).first()

        if not provider:
            raise QuotaExceededError()

    def deduct_quota(self, used_tokens: int = 0) -> None:
        """
        deduct available quota when provider type is system or paid.
        4.在系统或付费提供商的情况下扣减可用配额。

        :return:
        """
        if self.provider.provider_type != ProviderType.SYSTEM.value:
            return

        rules = self.get_rules()
        if 'system' not in rules['support_provider_types']:
            return

        if not self.should_deduct_quota():
            return

        if 'system_config' not in rules:
            quota_unit = ProviderQuotaUnit.TIMES.value
        elif 'quota_unit' not in rules['system_config']:
            quota_unit = ProviderQuotaUnit.TIMES.value
        else:
            quota_unit = rules['system_config']['quota_unit']

        if quota_unit == ProviderQuotaUnit.TOKENS.value:
            used_quota = used_tokens
        else:
            used_quota = 1

        db.session.query(Provider).filter(
            Provider.tenant_id == self.provider.tenant_id,
            Provider.provider_name == self.provider.provider_name,
            Provider.provider_type == self.provider.provider_type,
            Provider.quota_type == self.provider.quota_type,
            Provider.quota_limit > Provider.quota_used
        ).update({'quota_used': Provider.quota_used + used_quota})
        db.session.commit()

    def should_deduct_quota(self):
        """
        检查是否应该扣除配额。
        """
        return False

    def update_last_used(self) -> None:
        """
        update last used time.
        5.更新提供商的最后使用时间。

        :return:
        """
        db.session.query(Provider).filter(
            Provider.tenant_id == self.provider.tenant_id,
            Provider.provider_name == self.provider.provider_name
        ).update({'last_used': datetime.utcnow()})
        db.session.commit()

    def get_payment_info(self) -> Optional[dict]:
        """
        get product info if it payable.
        5.获取产品信息（如果提供商是可付费的）。

        :return:
        """
        return None

    def _get_provider_model(self, model_name: str, model_type: ModelType) -> ProviderModel:
        """
        get provider model.

        :param model_name:
        :param model_type:
        :return:
        """
        provider_model = db.session.query(ProviderModel).filter(
            ProviderModel.tenant_id == self.provider.tenant_id,
            ProviderModel.provider_name == self.provider.provider_name,
            ProviderModel.model_name == model_name,
            ProviderModel.model_type == model_type.value,
            ProviderModel.is_valid == True
        ).first()

        if not provider_model:
            raise LLMBadRequestError(f"The model {model_name} does not exist. "
                                     f"Please check the configuration.")

        return provider_model


class CredentialsValidateFailedError(Exception):
    pass
