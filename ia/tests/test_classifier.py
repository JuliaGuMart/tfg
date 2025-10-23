import sys
import os
from unittest.mock import MagicMock, patch
import pytest

# Aseguramos que la carpeta 'ia' esté en el path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from classes import State
import BotCompleto


@pytest.fixture
def fake_message():
    """Crea un objeto simulado que se comporte como un mensaje con .content."""
    msg = MagicMock()
    msg.role = "user"
    msg.content = "mensaje de prueba"
    return msg


@pytest.fixture
def fake_state(fake_message):
    """Estado básico simulado para las pruebas."""
    # State se comporta como un dict, así que podemos asignar directamente
    state = State()
    state["messages"] = [fake_message]
    return state


@patch.object(BotCompleto, "llm")
def test_classifier_train(mock_llm, fake_state):
    """Debe clasificar como 'train' cuando el modelo devuelve 'train'."""
    mock_llm.invoke = MagicMock(return_value=MagicMock(content="train"))

    fake_state["messages"][-1].content = "Quiero viajar a Valencia"
    output = BotCompleto.classifier(fake_state)

    assert isinstance(output, dict)
    assert output["mssg_type"] == "train"


@patch.object(BotCompleto, "llm")
def test_classifier_question(mock_llm, fake_state):
    """Debe clasificar como 'question' cuando el modelo devuelve 'question'."""
    mock_llm.invoke = MagicMock(return_value=MagicMock(content="question"))

    fake_state["messages"][-1].content = "¿Puedo llevar equipaje extra?"
    output = BotCompleto.classifier(fake_state)

    assert output["mssg_type"] == "question"


@patch.object(BotCompleto, "llm")
def test_classifier_conversation(mock_llm, fake_state):
    """Debe clasificar como 'conversation' cuando el modelo devuelve 'conversation'."""
    mock_llm.invoke = MagicMock(return_value=MagicMock(content="conversation"))

    fake_state["messages"][-1].content = "hola, ¿qué tal?"
    output = BotCompleto.classifier(fake_state)

    assert output["mssg_type"] == "conversation"
