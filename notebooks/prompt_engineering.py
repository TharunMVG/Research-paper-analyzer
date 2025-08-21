import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from model_finetuning import ResearchPaperModelFinetuner

# Initialize LLM with API key from environment 
# Set before running: export GOOGLE_API_KEY="your_api_key_here"
API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
    google_api_key=API_KEY
)

memory = ConversationBufferMemory(memory_key="chat_history")

# Hugging Face token 
# export HF_TOKEN="your_hf_token_here"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

# Initialize fine-tuned model with authentication
try:
    finetuned_model = ResearchPaperModelFinetuner(use_auth_token=os.getenv("HF_TOKEN"))
except Exception as e:
    print(f"Could not load fine-tuned model: {e}")
    print("Falling back to base model only...")
    finetuned_model = None


def generate_rag_prompt(focus_area: str, context: str, question: str) -> str:
    """
    Generate a RAG-enhanced prompt with Chain of Thought reasoning.
    Uses fine-tuned model if available, otherwise falls back to base model.
    """
    if finetuned_model is not None:
        try:
            response = finetuned_model.generate_response(context, question)
            return response
        except Exception as e:
            print(f"Error using fine-tuned model: {e}")

    rag_template = """
    You are an expert research analyst specializing in {focus_area}. 
    Below is the context from research papers:

    Context: {context}

    Question: {question}

    Please perform a detailed analysis following these steps:
    1. Extract and identify key information related to {focus_area} from the context
    2. Analyze the relationships between different pieces of information
    3. Identify patterns, trends, and connections
    4. Consider potential implications and applications
    5. Synthesize findings into clear, actionable insights

    Provide your analysis with step-by-step reasoning and evidence from the context.
    """

    prompt = PromptTemplate(
        input_variables=["focus_area", "context", "question"],
        template=rag_template
    )

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return chain.run(focus_area=focus_area, context=context, question=question)


def generate_agent_prompt(focus_area: str) -> str:
    """
    Generate a Chain-of-Thought prompt tailored for a specific focus area.
    """
    focus_descriptions = {
        "research gaps": "Identify areas where current research is incomplete or missing, suggesting potential research questions.",
        "challenges": "Highlight technical, practical, or theoretical obstacles discussed in the papers.",
        "future work": "Summarize proposed directions for future research and suggest novel extensions.",
        "limitations": "Detail the constraints or weaknesses of the studies, including methodological or data-related issues."
    }

    description = focus_descriptions.get(focus_area.lower(), "Analyze the specified aspect of the research.")

    prompt = (
        f"You are an expert research analyst specializing in {focus_area}. Below is the context from research papers:\n\n"
        "Context: {context}\n\n"
        f"Question: {{question}}\n\n"
        f"Please perform a chain-of-thought analysis by:\n"
        f"1. Extracting relevant sections or information related to {focus_area}.\n"
        f"2. Analyzing these sections to provide a clear and concise summary.\n"
        f"3. Suggesting actionable insights or research directions based on {description}.\n"
        f"4. Providing specific examples and evidence from the context.\n"
        f"5. Drawing connections between different aspects of the research.\n\n"
        "Provide a structured response with step-by-step reasoning and supporting evidence."
    )
    return prompt


def generate_summary_prompt(topic: str, agent_outputs: str) -> str:
    """
    Generate a prompt for summarizing overall gaps, challenges, future work, and limitations.
    """
    prompt = (
        f"You are an expert research analyst tasked with summarizing the state of research on {topic}. "
        "Below are the analyses of research gaps, challenges, future work, and limitations extracted from relevant papers:\n\n"
        f"{agent_outputs}\n\n"
        "Please perform a comprehensive analysis to:\n"
        "1. Synthesize the key points from each category (gaps, challenges, future work, limitations).\n"
        "2. Identify common themes or patterns across the analyses.\n"
        "3. Analyze the relationships between different aspects of the research.\n"
        "4. Consider the broader implications for the field.\n"
        "5. Provide a structured summary of the overall state of research on {topic}, including:\n"
        "   - Overall Research Gaps\n"
        "   - Key Challenges\n"
        "   - Future Research Directions\n"
        "   - Major Limitations\n"
        "   - Interconnections between different aspects\n"
        "Ensure the summary is clear, cohesive, and directly addresses the topic. Use step-by-step reasoning and provide specific examples."
    )
    return prompt


def get_research_suggestions(paper_text: str, problem_statement: str) -> str:
    """
    Generate research suggestions using the LLM with an enhanced prompt.
    """
    prompt = (
        "You are an expert research analyst. Based on the following paper excerpt:\n\n"
        f"'{paper_text}'\n\n"
        f"And the problem statement: '{problem_statement}'\n\n"
        "Please provide research suggestions by:\n"
        "1. Analyzing the current state of research in the paper\n"
        "2. Identifying potential gaps and opportunities\n"
        "3. Suggesting three novel research directions with clear reasoning\n"
        "4. Explaining how each suggestion addresses the problem statement\n"
        "5. Providing specific implementation considerations\n"
        "Ensure your suggestions are innovative, feasible, and well-supported by the context."
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content if hasattr(response, 'content') else str(response)


if __name__ == "__main__":
    # Testing 
    sample_paper = (
        "The paper discusses lip-reading using deep learning. "
        "Limitations include low accuracy in noisy environments. "
        "Future work suggests exploring multimodal inputs."
    )
    sample_problem = "Improving automated lip-reading."

    for focus in ["research gaps", "challenges", "future work", "limitations"]:
        prompt_text = generate_agent_prompt(focus)
        print(f"Generated Prompt for {focus}:\n", prompt_text, "\n")

    # Test RAG prompt
    rag_output = generate_rag_prompt(
        "research gaps",
        sample_paper,
        "What are the main research gaps in automated lip-reading?"
    )
    print("RAG Prompt Output:\n", rag_output)
