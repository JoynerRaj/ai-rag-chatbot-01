import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# need to add the project root so these imports actually work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# set up dummy django settings since we're testing this outside the normal flow
import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        INSTALLED_APPS=[
            'chat',
        ]
    )
    django.setup()

from chat.services.ai_agent import AIAgentService

class TestAIAgentErrorHandling(unittest.TestCase):
    @patch('chat.services.ai_agent.genai.Client')
    @patch('chat.services.ai_agent.semantic_cache_get', return_value=None)
    @patch('chat.services.ai_agent.Document.objects')
    def test_503_error_handling(self, mock_documents, mock_cache, mock_client):
        # act like this user already uploaded some documents
        mock_documents.filter.return_value.exists.return_value = True
        mock_documents.exists.return_value = True
        
        # simulate the api crashing with a 503 error
        mock_client_instance = mock_client.return_value
        mock_client_instance.models.generate_content.side_effect = Exception("503 UNAVAILABLE: This model is currently experiencing high demand.")
        
        # try asking a query and see if the service catches our fake exception properly
        result = AIAgentService.process_query("what is push pop 20ct?", document_id=None, user=None, chat_id=1)
        
        self.assertEqual(
            result, 
            "The AI model is currently experiencing high demand. Spikes in demand are usually temporary. Please wait a few moments and try again."
        )

    @patch('chat.services.ai_agent.genai.Client')
    @patch('chat.services.ai_agent.semantic_cache_get', return_value=None)
    @patch('chat.services.ai_agent.Document.objects')
    def test_429_quota_error(self, mock_documents, mock_cache, mock_client):
        # pretend they have documents uploaded
        mock_documents.filter.return_value.exists.return_value = True
        mock_documents.exists.return_value = True
        
        # throw a fake quota error
        mock_client_instance = mock_client.return_value
        mock_client_instance.models.generate_content.side_effect = Exception("429 Quota Exceeded for AI service.")
        
        # run the agent with a greeting and check the response string
        result = AIAgentService.process_query("hello", document_id=None, user=None, chat_id=1)
        self.assertEqual(
            result, 
            "The AI service quota has been exceeded. Please try again later."
        )
        
    @patch('chat.services.ai_agent.genai.Client')
    @patch('chat.services.ai_agent.semantic_cache_get', return_value=None)
    @patch('chat.services.ai_agent.Document.objects')
    def test_generic_error(self, mock_documents, mock_cache, mock_client):
        mock_documents.filter.return_value.exists.return_value = True
        mock_documents.exists.return_value = True
        
        mock_client_instance = mock_client.return_value
        mock_client_instance.models.generate_content.side_effect = Exception("Some weird unknown error")
        
        result = AIAgentService.process_query("test", document_id=None, user=None, chat_id=1)
        self.assertEqual(
            result, 
            "An unexpected error occurred while generating a response. Please try again later."
        )

if __name__ == '__main__':
    unittest.main()
