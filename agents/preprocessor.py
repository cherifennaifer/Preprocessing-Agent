from crewai import Agent, LLM
import os
from dotenv import load_dotenv

load_dotenv()

class CacheSafeLLM(LLM):
    def _format_messages_for_provider(self, messages):
        messages_without_cache_markers = [
            {
                key: value
                for key, value in message.items()
                if key != "cache_breakpoint"
            }
            for message in messages
        ]
        return super()._format_messages_for_provider(messages_without_cache_markers)


default_llm = CacheSafeLLM(
    model="groq/openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    provider="groq",
)

preprocessor = Agent(
    role="Data Preprocessing Specialist",
    goal="Analyze datasets, explain issues, ask user for decisions, and apply preprocessing automatically.",
    backstory="An AI agent that guides users through data cleaning and preprocessing with smart suggestions.",
    verbose=True,
    llm=default_llm,
)
