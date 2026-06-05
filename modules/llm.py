from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from .env
api_key = os.getenv("GROQ_API_KEY")

# Create client only if key exists
client = None

if api_key:
    client = Groq(
        api_key=api_key
    )


def generate_answer(question, papers):

    if not api_key:

        return """
❌ GROQ_API_KEY not found.

Please create a .env file and add:

GROQ_API_KEY=your_groq_api_key
"""

    # Build context from retrieved papers
    context = ""

    for paper in papers:

        context += f"""
TITLE:
{paper['title']}

ABSTRACT:
{paper['abstract']}

----------------------------------------
"""

    prompt = f"""
You are an AI Research Assistant.

Use ONLY the research papers provided below.

{context}

Question:
{question}

Provide:

1. Detailed Answer
2. Key Findings
3. Current Trends
4. Conclusion

Be clear, concise, and research-focused.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"""
❌ GROQ ERROR

Exception:
{str(e)}

Possible causes:

1. Invalid API key
2. Expired API key
3. Internet connection issue
4. Groq service unavailable
"""