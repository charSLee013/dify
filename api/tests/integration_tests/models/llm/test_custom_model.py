import json
import os
from unittest.mock import patch, MagicMock

from core.model_providers.error import LLMBadRequestError
from core.model_providers.models.entity.message import PromptMessage, MessageType
from core.model_providers.models.entity.model_params import ModelKwargs, ModelType
from core.model_providers.models.llm.custom_model import CustomLLM
from core.model_providers.providers.custom_provider import CustomProvider
from models.provider import Provider, ProviderType, ProviderModel
from langchain.schema import HumanMessage


def get_mock_provider():
    return Provider(
        id='provider_id',
        tenant_id='tenant_id',
        provider_name='custom',
        provider_type=ProviderType.CUSTOM.value,
        encrypted_config='',
        is_valid=True,
    )


def get_mock_model(model_name, mocker)->CustomLLM:
    model_kwargs = ModelKwargs(
        max_tokens=20,
        temperature=0.01
    )
    server_url = os.environ['CUSTOM_SERVER_URL']
    model_provider = CustomProvider(provider=get_mock_provider())

    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = ProviderModel(
        provider_name='custom',
        model_name=model_name,
        model_type=ModelType.TEXT_GENERATION.value,
        encrypted_config=json.dumps({
            'server_url':server_url,
            'completion_type': 'chat_completion'
        }),
        is_valid=True,
    )
    mocker.patch('extensions.ext_database.db.session.query', return_value=mock_query)

    return CustomLLM(
        model_provider=model_provider,
        name=model_name,
        model_kwargs=model_kwargs
    )


def decrypt_side_effect(tenant_id, encrypted_api_key):
    return encrypted_api_key


@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_get_num_tokens(mock_decrypt, mocker):
    model = get_mock_model('custom/model-123', mocker)
    rst = model.get_num_tokens([
        PromptMessage(type=MessageType.USER, content='Who is your manufacturer?')
    ])
    assert rst > 0


@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_run(mock_decrypt, mocker):
    mocker.patch('core.model_providers.providers.base.BaseModelProvider.update_last_used', return_value=None)

    model = get_mock_model('custom/model-123', mocker)
    messages = [
        PromptMessage(type=MessageType.USER, content='ping')
    ]
    rst = model.run(
        messages,
    )
    assert len(rst.content) > 0


@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_set_model_kwargs(mock_decrypt, mocker):
    model = get_mock_model('custom/model-123', mocker)
    model._client = MagicMock()  # Mocking the client

    model_kwargs = ModelKwargs(
        max_tokens=20,
        temperature=0.02
    )
    model._set_model_kwargs(model_kwargs)
    assert model.client.max_tokens == 20
    assert model.client.temperature == 0.02


@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_handle_exceptions(mock_decrypt, mocker):
    model = get_mock_model('custom/model-123', mocker)
    ex = Exception("Test exception")
    result = model.handle_exceptions(ex)
    assert isinstance(result, LLMBadRequestError)
    assert str(ex) in str(result)
