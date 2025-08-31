# functions.py

from markdown import markdown
import markdown2
import pdfkit
from PyPDF2 import PdfReader
from openai import OpenAI, RateLimitError
from openai import OpenAI
from dotenv import load_dotenv
import os
import concurrent.futures

# --- CONFIG ---
USE_OLLAMA = True  # set to False to go back to OpenAI

if USE_OLLAMA:
    # Points client at local Ollama server
    client = OpenAI(
        api_key="ollama",  # dummy key, not used
        base_url="http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
    )
    DEFAULT_MODEL = "mistral"   # model thats pulled with `ollama pull` in powershell
else:
    # Standard OpenAI API setup
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    DEFAULT_MODEL = "gpt-4o-mini"


def create_prompt(resume_string: str, jd_string: str) -> str:
    """
    Creates a shorter prompt for resume optimization,
    better suited for smaller/faster local models.
    """
    return f"""
You are a resume optimization assistant. 
Your task is to tailor my resume to fit the job description.

### Instructions:
- Emphasize only the most relevant skills and experiences.
- Use strong action verbs and short bullet points.
- Integrate keywords from the job description naturally.
- Keep the resume to **one page** in Markdown format.
- At the end, add a section: "Additional Suggestions" with:
  - Extra skills or tools I should learn
  - Certifications or courses to improve alignment

---

### My Resume:
{resume_string}

### Job Description:
{jd_string}

---

### Output:
1. Optimized Resume (Markdown, max one page)
2. "Additional Suggestions"
"""



def get_resume_response(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.7) -> str:
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Expert resume writer"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
            )
            return future.result(timeout=300).choices[0].message.content  # timeout 300s
    except concurrent.futures.TimeoutError:
        return "⚠️ Took too long (>5 minutes). Try a smaller model or shorter resume/job description."
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"


def load_resume(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        # Read PDF and extract text
        text = ""
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    elif ext in [".md", ".txt"]:
        # Read plain text or markdown
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError("Unsupported file type. Please upload a .pdf, .md, or .txt resume.")

def optimize_resume(resume_text: str, jd_text: str) -> tuple[str, str]:
    prompt = create_prompt(resume_text, jd_text)
    response_string = get_resume_response(prompt)

    if "## Additional Suggestions" in response_string:
        new_resume, suggestions = response_string.split("## Additional Suggestions", 1)
        suggestions = "## Additional Suggestions\n\n" + suggestions.strip()
    else:
        new_resume = response_string
        suggestions = "## Additional Suggestions\n\n(No extra suggestions provided.)"

    return new_resume.strip(), suggestions.strip()

def process_resume(resume_file, jd_text):
    if resume_file is None or jd_text.strip() == "":
        return "⚠️ Please upload a resume and paste a job description.", "", ""
    resume_text = load_resume(resume_file.name)
    new_resume, suggestions = optimize_resume(resume_text, jd_text)
    return new_resume, new_resume, suggestions


def export_resume(new_resume: str, output_path: str = "resumes/resume_new.pdf") -> str:
    try:
        html_content = markdown2.markdown(new_resume)
        pdfkit.from_string(html_content, output_path)
        return f"✅ Successfully exported resume to {output_path}"
    except Exception as e:
        return f"❌ Failed to export resume: {str(e)}"