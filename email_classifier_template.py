# Configuration and imports
import uuid
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample email dataset
sample_emails = [
    {
        "id": "001",
        "from": "angry.customer@example.com",
        "subject": "Broken product received",
        "body": "I received my order #12345 yesterday but it arrived completely damaged. This is unacceptable and I demand a refund immediately. This is the worst customer service I've experienced.",
        "timestamp": "2024-03-15T10:30:00Z"
    },
    {
        "id": "002",
        "from": "curious.shopper@example.com",
        "subject": "Question about product specifications",
        "body": "Hi, I'm interested in buying your premium package but I couldn't find information about whether it's compatible with Mac OS. Could you please clarify this? Thanks!",
        "timestamp": "2024-03-15T11:45:00Z"
    },
    {
        "id": "003",
        "from": "happy.user@example.com",
        "subject": "Amazing customer support",
        "body": "I just wanted to say thank you for the excellent support I received from Sarah on your team. She went above and beyond to help resolve my issue. Keep up the great work!",
        "timestamp": "2024-03-15T13:15:00Z"
    },
    {
        "id": "004",
        "from": "tech.user@example.com",
        "subject": "Need help with installation",
        "body": "I've been trying to install the software for the past hour but keep getting error code 5123. I've already tried restarting my computer and clearing the cache. Please help!",
        "timestamp": "2024-03-15T14:20:00Z"
    },
    {
        "id": "005",
        "from": "business.client@example.com",
        "subject": "Partnership opportunity",
        "body": "Our company is interested in exploring potential partnership opportunities with your organization. Would it be possible to schedule a call next week to discuss this further?",
        "timestamp": "2024-03-15T15:00:00Z"
    }
]


class EmailProcessor:
    def __init__(self):
        """Initialize the email processor with OpenAI API key."""
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_API_ORGANIZATION"),
            project=os.getenv("OPENAI_API_PROJECT")
        )
        # Define valid categories
        self.valid_categories = {
            "complaint", "inquiry", "feedback",
            "support_request", "other"
        }

    def classify_email(self, email: Dict) -> Optional[str]:
        """
        Classify an email using LLM.
        Returns the classification category or None if classification fails.

        TODO:
        1. Design and implement the classification prompt
        2. Make the API call with appropriate error handling
        3. Validate and return the classification
        """
        chat_classify = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
            {
                "role": "system",
                "content": "Classify incoming e-mail."
            }, {

                "role": "user",
                "content": json.dumps(email)
            }],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "email_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                        "category": {
                            "type": "string",
                            "enum": list(self.valid_categories) # JSON parseable
                        }
                        },
                        "required": ["category"],
                        "additionalProperties": False
                    }
                }
            }
        )
        result = chat_classify.choices[0].message
        if result.refusal: # handle refusal
            return "other"

        parsed = json.loads(result.content)
        return parsed["category"]

    def generate_response(self, email: Dict, classification: str) -> Optional[str]:
        """
        Generate an automated response based on email classification.

        TODO:
        1. Design the response generation prompt
        2. Implement appropriate response templates
        3. Add error handling
        """

        chat_response = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
            {
                "role": "system",
                "content": f"Respond the {classification} e-mail."
            }, {
                "role": "user",
                "content": json.dumps(email)
            }],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "email_extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                        "subject": {
                            "type": "string",
                            "description": "The subject line of the email."
                        },
                        "body": {
                            "type": "string",
                            "description": "The main content or body of the email."
                        }
                        },
                        "required": ["subject", "body"],
                        "additionalProperties": False
                    }
                }
            }
        )
        result = chat_response.choices[0].message
        if result.refusal:
            raise Exception("OpenAI refused to structure a response.")

        return json.loads(result.content)


class EmailAutomationSystem:
    def __init__(self, processor: EmailProcessor):
        """Initialize the automation system with an EmailProcessor."""
        self.processor = processor
        self.response_handlers = {
            "complaint": self._handle_complaint,
            "inquiry": self._handle_inquiry,
            "feedback": self._handle_feedback,
            "support_request": self._handle_support_request,
            "other": self._handle_other
        }

    def process_email(self, email: Dict) -> Dict:
        """
        Process a single email through the complete pipeline.
        Returns a dictionary with the processing results.

        TODO:
        1. Implement the complete processing pipeline
        2. Add appropriate error handling
        3. Return processing results
        """
        # 1. classify e-mail
        category = self.processor.classify_email(email)
        # 2. generate a structured response
        mail_response = self.processor.generate_response(email, category)
        mail_response["id"] = str(uuid.uuid4())
        mail_response["from"] = "assistant@example.com"
        mail_response["timestamp"] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        mail_response["category"] = category
        # 3. get the handler and pass the response
        handler = self.response_handlers.get(category)
        handler(mail_response)
        return {
            "email_id": mail_response["id"],
            "success": True,
            "classification": mail_response["category"],
            "response_sent": mail_response["body"],
        }

    def _handle_complaint(self, email: Dict):
        """
        Handle complaint emails.
        TODO: Implement complaint handling logic
        """
        send_complaint_response(
            email_id=email["id"],
            response=email["body"]
        )

    def _handle_inquiry(self, email: Dict):
        """
        Handle inquiry emails.
        TODO: Implement inquiry handling logic
        """
        create_urgent_ticket(
            email_id=email["id"],
            category=email["category"],
            context=email["body"],
        )

    def _handle_feedback(self, email: Dict):
        """
        Handle feedback emails.
        TODO: Implement feedback handling logic
        """
        log_customer_feedback(
            email_id=email["id"],
            feedback=email["body"]
        )

    def _handle_support_request(self, email: Dict):
        """
        Handle support request emails.
        TODO: Implement support request handling logic
        """
        create_support_ticket(
            email_id=email["id"],
            context=email["body"],
        )

    def _handle_other(self, email: Dict):
        """
        Handle other category emails.
        TODO: Implement handling logic for other categories
        """
        send_standard_response(
            email_id=email["id"],
            response=email["body"]
        )

# Mock service functions
def send_complaint_response(email_id: str, response: str):
    """Mock function to simulate sending a response to a complaint"""
    logger.info(f"Sending complaint response for email {email_id}")
    # In real implementation: integrate with email service


def send_standard_response(email_id: str, response: str):
    """Mock function to simulate sending a standard response"""
    logger.info(f"Sending standard response for email {email_id}")
    # In real implementation: integrate with email service


def create_urgent_ticket(email_id: str, category: str, context: str):
    """Mock function to simulate creating an urgent ticket"""
    logger.info(f"Creating urgent ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def create_support_ticket(email_id: str, context: str):
    """Mock function to simulate creating a support ticket"""
    logger.info(f"Creating support ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def log_customer_feedback(email_id: str, feedback: str):
    """Mock function to simulate logging customer feedback"""
    logger.info(f"Logging feedback for email {email_id}")
    # In real implementation: integrate with feedback system


def run_demonstration():
    """Run a demonstration of the complete system."""
    # Initialize the system
    processor = EmailProcessor()
    automation_system = EmailAutomationSystem(processor)

    # Process all sample emails
    results = []
    for email in sample_emails:
        logger.info(f"\nProcessing email {email['id']}...")
        result = automation_system.process_email(email)
        results.append(result)

    # Create a summary DataFrame
    df = pd.DataFrame(results)
    print("\nProcessing Summary:")
    print(df[["email_id", "success", "classification", "response_sent"]])

    return df


# Example usage:
if __name__ == "__main__":
    results_df = run_demonstration()
