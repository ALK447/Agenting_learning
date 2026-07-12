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

task = "Most popular I Agent frameworks in 2026"

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

async def run_all(query: str):
    with trace("Research Task"):
        research_results = await run_researches(query)
        report = await write_report(query, research_results)
        return report.short_summary, report.mark_down_report, report.follow_up_questions

with gr.Blocks(title="Deep Research") as ui:
    gr.Markdown("# Deep Research Agent")
    with gr.Row():
        query_input = gr.Textbox(label="Research Query", placeholder="Enter your research question...", scale=4)
        submit_btn = gr.Button("Research", variant="primary", scale=1)
    with gr.Row():
        summary_output = gr.Textbox(label="Short Summary", lines=3)
    with gr.Row():
        report_output = gr.Markdown(label="Full Report")
    with gr.Row():
        followup_output = gr.Textbox(label="Follow-up Questions", lines=3)

    submit_btn.click(fn=run_all, inputs=query_input, outputs=[summary_output, report_output, followup_output])

ui.launch()