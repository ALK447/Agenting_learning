#Creating 3 agents: searcher, planner; writer
#to lauch the agent use python 2_open_ai/deep_research.py

from agents import Agent, WebSearchTool, trace, Runner, function_tool
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import gradio as gr

#Agent #1 : The Search Agent: This agent will search the web for the given task and return a summary of the results. It will use the WebSearchTool to perform the searches.
load_dotenv(override=True)
MODEL_NAME = "gpt-4o-mini"
HOW_MANY_SEARCHES = 3

INSTRUCTIONS = """You are a research assistant. Given a search term, you search the web for that term and 
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary."""

task = "Most popular AI Agent frameworks in 2026"

settings = ModelSettings(tool_choice="required")

tools = [WebSearchTool()]

search_agent = Agent(name="Search Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, tools=tools, model_settings=settings)

# async def main():
#     result = await Runner.run(search_agent, task)
#     print(result.final_output)

# asyncio.run(main())


#Agent #2: The Planner Agent: This agent will take the summary produced by the Search Agent and create a detailed plan for a research paper based on that summary. It will outline the structure of the paper, including sections, key points to cover, and any additional research that may be needed.

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query")
    query: str = Field(description="The search term to use for the web search")
    
class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web search items to perform")

INSTRUCTIONS = f"""
You are a research assistant. Given a user query, come up with a set of web searches
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.
"""

planner_agent = Agent(name="Planner Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=WebSearchPlan)

# async def main():
#     result = await Runner.run(planner_agent, task)
#     print(result.final_output)
    
# asyncio.run(main())


#Agent #3: The Writer Agent: This agent will take the summary produced by the Search Agent and the plan produced by the Planner Agent and write a research paper based on that information. It will follow the structure outlined in the plan and incorporate the key points from the summary.

INSTRUCTIONS = """
You are a senior researcher tasked with writing a cohesive report for a research query.
You will be provided with the original query, and some research.
Generate a comprehensive report based on the research and the query.
The final output should be in markdown format, and it should be lengthy and detailed. Aim 
for 5-10 pages of content, at least 1000 words.
"""

class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the research findings")
    mark_down_report: str = Field(description="The final report")
    follow_up_questions: str= Field(description="Suggested topics to research further")

writer_agent = Agent(name="Writer Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=ReportData)


# Clarifying Agent: asks 3 questions to understand the topic better before research begins.

class ClarifyingQuestions(BaseModel):
    questions: list[str] = Field(description="Exactly 3 short clarifying questions")

CLARIFY_INSTRUCTIONS = """
You are a research assistant helping to scope a research request.
Given a research topic, generate exactly 3 concise clarifying questions that will
help you produce a more focused and useful report. Ask about scope, audience,
depth, time period, or specific angles the user cares about.
"""

clarifying_agent = Agent(name="Clarifying Agent", instructions=CLARIFY_INSTRUCTIONS, model=MODEL_NAME, output_type=ClarifyingQuestions)


#Orchestration: The following code will orchestrate the three agents to perform a complete research task. It will first use the Planner Agent to generate a set of web searches, then use the Search Agent to perform those searches and summarize the results, and finally use the Writer Agent to produce a comprehensive report based on the summary and the original query.

#perform preparation for the research task
async def run_researches(query: str):
    print("Planning Searches...")
    result = await Runner.run(planner_agent, f"Query: {query}")
    searches = result.final_output.searches
    print(f"Will perform {len(searches)} searches...")
    tasks = [search(item) for item in searches]
    results = await asyncio.gather(*tasks)
    print("Finished searching")
    return results

#perform the web search for a given search item
async def search(item: WebSearchItem):
    input_message = f"Search term:{item.query}\nReason: {item.reason}"
    result = await Runner.run(search_agent, input_message)
    return result.final_output

async def write_report(query:str, research_results:list):
    print("Thinking about report...")
    input_message = f"Original Query: {query}\nSummarized search results : {research_results}"
    result = await Runner.run(writer_agent, input_message)
    print("Finished writing report")
    return result.final_output

async def get_clarifying_questions(query: str):
    yield gr.update(value="💭 Thinking about your topic...", visible=True), "", "", "", gr.update(visible=False)
    result = await Runner.run(clarifying_agent, f"Research topic: {query}")
    qs = result.final_output.questions
    yield gr.update(visible=False), qs[0], qs[1], qs[2], gr.update(visible=True)

async def run_all(query: str, q1: str, q2: str, q3: str, a1: str, a2: str, a3: str):
    context = (
        f"Research topic: {query}\n\n"
        f"Clarifying context provided by the user:\n"
        f"Q: {q1}\nA: {a1}\n\n"
        f"Q: {q2}\nA: {a2}\n\n"
        f"Q: {q3}\nA: {a3}"
    )
    with trace("Research Task"):
        yield gr.update(value="📋 Planning searches...", visible=True), "", "", "", gr.update(visible=False)
        plan_result = await Runner.run(planner_agent, f"Query: {context}")
        searches = plan_result.final_output.searches

        yield gr.update(value=f"🔍 Running {len(searches)} web searches...", visible=True), "", "", "", gr.update(visible=False)
        tasks = [search(item) for item in searches]
        research_results = await asyncio.gather(*tasks)

        yield gr.update(value="✍️ Writing your research report...", visible=True), "", "", "", gr.update(visible=False)
        report = await write_report(query, research_results)

        yield gr.update(visible=False), report.short_summary, report.mark_down_report, report.follow_up_questions, gr.update(visible=True)

CSS = """
body { background: #f5f5f7 !important; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.gradio-container { background: #f5f5f7 !important; max-width: 860px !important; margin: 0 auto !important; padding: 40px 24px !important; }
.hero { text-align: center; padding: 48px 0 32px; }
.hero h1 { font-size: 2.6rem; font-weight: 700; background: linear-gradient(135deg, #4f46e5, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 8px; }
.hero p { color: #6b7280; font-size: 1rem; margin: 0; }
.card { background: #ffffff; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); padding: 28px 28px 20px; margin-bottom: 20px; }
.card-title { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 16px; }
.question-label { font-size: 0.95rem; font-weight: 600; color: #374151; margin: 16px 0 4px; }
footer { display: none !important; }
#query-box textarea { border: 1.5px solid #e5e7eb; border-radius: 10px; font-size: 1rem; padding: 14px; resize: none; transition: border 0.2s; }
#query-box textarea:focus { border-color: #4f46e5; outline: none; box-shadow: 0 0 0 3px rgba(79,70,229,0.1); }
#ask-btn, #research-btn { background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; border: none !important; border-radius: 10px !important; color: white !important; font-weight: 600 !important; font-size: 1rem !important; padding: 14px 28px !important; cursor: pointer !important; transition: opacity 0.2s !important; height: 52px !important; }
#ask-btn:hover, #research-btn:hover { opacity: 0.88 !important; }
#a1 textarea, #a2 textarea, #a3 textarea { border: 1.5px solid #e5e7eb !important; border-radius: 10px !important; font-size: 0.95rem !important; background: #fafafa !important; }
.tab-nav button { font-weight: 500 !important; border-radius: 8px 8px 0 0 !important; }
.tab-nav button.selected { color: #4f46e5 !important; border-bottom: 2px solid #4f46e5 !important; }
#summary-box textarea, #followup-box textarea { background: #fafafa !important; border: 1.5px solid #e5e7eb !important; border-radius: 10px !important; color: #1f2937 !important; font-size: 0.95rem !important; line-height: 1.6 !important; }
#status-bar { background: linear-gradient(135deg, #ede9fe, #e0e7ff); border-radius: 12px; padding: 14px 20px; font-size: 0.95rem; font-weight: 500; color: #4f46e5; text-align: center; margin-bottom: 16px; }
"""

with gr.Blocks(title="Deep Research Agent") as ui:
    gr.HTML("""
        <div class="hero">
            <h1>Deep Research Agent</h1>
            <p>Powered by GPT-4o-mini · Plans, searches, and writes a full research report</p>
        </div>
    """)

    # Step 1: query input
    with gr.Group(elem_classes="card"):
        gr.HTML('<div class="card-title">Step 1 — Your Topic</div>')
        query_input = gr.Textbox(
            label="",
            placeholder="What do you want to research? e.g. Most popular AI agent frameworks in 2026",
            lines=2,
            elem_id="query-box",
        )
        ask_btn = gr.Button("Ask Clarifying Questions →", elem_id="ask-btn")

    status = gr.Markdown("", elem_id="status-bar", visible=False)

    # Step 2: clarifying questions (hidden until agent responds)
    with gr.Group(elem_classes="card", visible=False) as questions_card:
        gr.HTML('<div class="card-title">Step 2 — Help us focus the research</div>')
        q1 = gr.Textbox(label="Question 1", interactive=False, lines=1)
        a1 = gr.Textbox(label="Your answer", lines=2, elem_id="a1")
        q2 = gr.Textbox(label="Question 2", interactive=False, lines=1)
        a2 = gr.Textbox(label="Your answer", lines=2, elem_id="a2")
        q3 = gr.Textbox(label="Question 3", interactive=False, lines=1)
        a3 = gr.Textbox(label="Your answer", lines=2, elem_id="a3")
        research_btn = gr.Button("Start Research →", elem_id="research-btn")

    # Step 3: results (hidden until research completes)
    with gr.Group(elem_classes="card", visible=False) as results_card:
        gr.HTML('<div class="card-title">Step 3 — Results</div>')
        with gr.Tabs():
            with gr.Tab("Summary"):
                summary_output = gr.Textbox(label="", lines=5, elem_id="summary-box")
            with gr.Tab("Full Report"):
                report_output = gr.Markdown()
            with gr.Tab("Follow-up Questions"):
                followup_output = gr.Textbox(label="", lines=6, elem_id="followup-box")

    ask_btn.click(
        fn=get_clarifying_questions,
        inputs=query_input,
        outputs=[status, q1, q2, q3, questions_card],
    )
    research_btn.click(
        fn=run_all,
        inputs=[query_input, q1, q2, q3, a1, a2, a3],
        outputs=[status, summary_output, report_output, followup_output, results_card],
    )

ui.launch(css=CSS)