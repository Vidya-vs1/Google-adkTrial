# --- Step 0: Setup and Installation (These commands are for Colab/Jupyter, run manually in terminal if needed) ---
# !pip install google-adk -q
# !pip install litellm -q
# print("Installation complete.") # This print is from the original notebook cell

# --- Step 1: Import necessary libraries ---
import os
import asyncio
import time # Used in mock tools
from typing import Optional, Dict, Any, List
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from google.adk.tools.base_tool import BaseTool # Needed for callbacks
from google.adk.tools.tool_context import ToolContext # Needed for state access in tools/callbacks
from google.adk.agents.callback_context import CallbackContext # Needed for callbacks
from google.adk.models.llm_request import LlmRequest # Needed for callbacks
from google.adk.models.llm_response import LlmResponse # Needed for callbacks

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")

# --- Step 2: Configure API Keys (Adjust for local environment) ---
# @title Configure API Keys (Replace with your actual keys!)

# --- IMPORTANT: Replace placeholders or use environment variables ---
# If running locally, set these in your terminal or a .env file
# Example for local environment (replace with your actual keys or use environment variables):
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # Only needed if using Google models

# Assuming LiteLLM is configured for Ollama, the API key might not be strictly necessary
# unless you are also using Google models for certain agents.

# Configure ADK to use API keys directly (not Vertex AI for this multi-model setup)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

print("API Key configuration setup.")

# --- Step 3: Define Model Constants ---
# @title Define Model Constants for easier use

# Define a model constant for Gemini (if needed)
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

# Define a model constant for your Ollama model
# The format is "ollama/<model_name>"
MODEL_OLLAMA_MISTRAL = "ollama/qwen3:1.7b" # Make sure this model is pulled in Ollama
print(f"Defined model constants: {MODEL_GEMINI_2_0_FLASH}, {MODEL_OLLAMA_MISTRAL}")

print("\nEnvironment configured.")

# --- Step 4: Define Academic Recommendation Tools ---
# @title Define Academic Recommendation Tools

def say_hello(name: Optional[str] = None) -> str:
    """Provides a simple greeting. If a name is provided, it will be used."""
    # Check if name is None or an empty string before using it
    if name is not None and name.strip() != "":
        greeting = f"Hello, {name.strip()}!"
        print(f"--- Tool: say_hello called with name: {name.strip()} ---")
    else:
        greeting = "Hello there!"
        print(f"--- Tool: say_hello called without a specific name ---")
    return greeting

def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation."""
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."

# Updated ask_user_info to use ToolContext and store info in state
def ask_user_info(user_info_type: str, tool_context: ToolContext) -> str:
    """Simulates asking the user for a specific piece of information and stores it in session state."""
    print(f"--- Tool: ask_user_info called for type: {user_info_type} ---")

    # Initialize user_info in state if it doesn't exist
    if "user_info" not in tool_context.state:
        tool_context.state["user_info"] = {}
        print("--- Tool: Initialized 'user_info' in state. ---")

    # Simulate getting info and storing it
    simulated_response = ""
    # In a real system, this tool would pause and wait for actual user input.
    # Here, we simulate responses for the example conversation flow.
    if user_info_type == "Degree Level Sought":
        simulated_response = "Master's"
        tool_context.state["user_info"]["degree_level"] = simulated_response
    elif user_info_type == "Field of Study":
        simulated_response = "Computer Science"
        tool_context.state["user_info"]["field_of_study"] = simulated_response
    elif user_info_type == "Academic Background":
        simulated_response = "Bachelor's in Software Engineering, GPA 3.8/4.0"
        tool_context.state["user_info"]["academic_background"] = simulated_response
    elif user_info_type == "Budget":
        simulated_response = "$30,000 - $40,000 per year"
        tool_context.state["user_info"]["budget"] = simulated_response
    elif user_info_type == "Location Preferences":
        simulated_response = "Canada or Germany"
        tool_context.state["user_info"]["location_preferences"] = simulated_response
    else:
        simulated_response = f"Information needed: {user_info_type}" # Fallback


    print(f"--- Tool: Stored '{user_info_type}' as '{simulated_response}' in state['user_info']. ---")
    # Return a confirmation or the simulated response that the agent can use
    return f"Okay, I have noted your {user_info_type}: {simulated_response}."


def query_course_dataset(user_criteria: Dict[str, Any], ranking_preference: Optional[str] = None) -> List[Dict[str, Any]]:
    """Queries a mock dataset based on user criteria including academic background, budget, and eligibility."""
    print(f"--- Tool: query_course_dataset called with criteria: {user_criteria}, ranking preference: {ranking_preference} ---")

    # Mock dataset (re-defined for self-containment in this step)
    mock_dataset = [
        {"university": "University of Toronto", "location": "Canada", "field": "Computer Science", "degree": "Master's", "ranking_qs": 25, "cost_usd_yr": 35000, "scholarships": True, "eligibility": "GPA > 3.5, Relevant Bachelor's"},
        {"university": "Technical University of Munich", "location": "Germany", "field": "Computer Science", "degree": "Master's", "ranking_qs": 30, "cost_usd_yr": 1500, "scholarships": False, "eligibility": "GPA > 3.3, Relevant Bachelor's"},
        {"university": "University of British Columbia", "location": "Canada", "field": "Computer Science", "degree": "Master's", "ranking_qs": 45, "cost_usd_yr": 30000, "scholarships": True, "eligibility": "GPA > 3.4, Relevant Bachelor's"},
        {"university": "Ludwig Maximilian University of Munich", "location": "Germany", "field": "Biology", "degree": "Master's", "ranking_qs": 32, "cost_usd_yr": 1500, "scholarships": True, "eligibility": "GPA > 3.0, Relevant Bachelor's"},
         {"university": "University of Waterloo", "location": "Canada", "field": "Computer Science", "degree": "Bachelor's", "ranking_qs": 150, "cost_usd_yr": 25000, "scholarships": False, "eligibility": "High School Average > 90%"},
         {"university": "Technical University of Berlin", "location": "Germany", "field": "Computer Science", "degree": "Master's", "ranking_qs": 40, "cost_usd_yr": 1800, "scholarships": True, "eligibility": "GPA > 3.2, Relevant Bachelor's"},
         {"university": "McGill University", "location": "Canada", "field": "Biology", "degree": "PhD", "ranking_qs": 50, "cost_usd_yr": 20000, "scholarships": True, "eligibility": "Master's degree, Research Proposal"},
    ]


    filtered_results = []
    user_field = user_criteria.get("Field of Study", "").lower()
    user_locations = [loc.strip().lower() for loc in user_criteria.get("Location Preferences", "").split(" or ") if loc.strip()]
    user_degree = user_criteria.get("Degree Level Sought", "").lower().replace("'s", "") # Handle Master's -> Master
    user_academic_background = user_criteria.get("Academic Background", "").lower()

    # Basic budget parsing (e.g., "$30,000 - $40,000")
    budget_str = user_criteria.get("Budget", "")
    try:
        min_budget, max_budget = map(int, budget_str.replace("$", "").replace(",", "").split("-"))
    except:
        min_budget, max_budget = 0, float('inf') # No budget specified or invalid format

    for entry in mock_dataset:
        match = True
        entry_field = entry["field"].lower()
        entry_location = entry["location"].lower()
        entry_degree = entry["degree"].lower().replace("'s", "")
        entry_cost = entry.get("cost_usd_yr", float('inf'))
        entry_eligibility = entry.get("eligibility", "").lower()


        # Filter by Field of Study
        if user_field and user_field not in entry_field:
            match = False

        # Filter by Location Preferences
        if user_locations and not any(loc in entry_location for loc in user_locations):
             match = False

        # Filter by Degree Level
        if user_degree and entry_degree not in user_degree: # Check if entry degree is in user's desired degree (handles broader match)
             match = False

        # Filter by Budget (using max budget here, min budget handled in process)
        if entry_cost > max_budget:
             match = False

        # Basic Eligibility Check (simplified: check if user background mentions key terms in eligibility)
        # This is a very rudimentary simulation. Real eligibility is complex.
        if entry_eligibility and user_academic_background:
             # Example: Check if eligibility requires 'bachelor's' and user background mentions it
             if 'bachelor' in entry_eligibility and 'bachelor' not in user_academic_background:
                 match = False
             # Add more specific eligibility checks if needed

        if match:
            filtered_results.append(entry)

    # Apply ranking preference (simplified sort)
    if ranking_preference and ranking_preference.lower() == "qs":
        filtered_results.sort(key=lambda x: x.get("ranking_qs", float('inf'))) # Sort by QS ranking

    print(f"--- Tool: query_course_dataset returning {len(filtered_results)} results. ---")
    return filtered_results

# 2. Improve the `process_recommendations` tool
def process_recommendations(query_results: List[Dict[str, Any]], user_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Processes query results to prioritize/filter recommendations based on more nuanced criteria."""
    print(f"--- Tool: process_recommendations called with {len(query_results)} results and criteria: {user_criteria} ---")

    if not query_results:
        print("--- Tool: process_recommendations - No query results to process. ---")
        return [] # Return empty if no results

    # Simple prioritization logic: Assign a score based on criteria
    # Higher score is better
    def calculate_score(rec, user_criteria):
        score = 0
        # Example Scoring (can be made more complex)
        # Prioritize lower QS ranking (lower number is better, so subtract from a high number)
        score += (200 - rec.get("ranking_qs", 200))

        # Prioritize lower cost (subtract from a high number based on expected range)
        max_expected_cost = 50000 # Assume max cost to scale
        cost = rec.get("cost_usd_yr", max_expected_cost)
        score += (max_expected_cost - cost) / 1000 # Scale cost score

        # Prioritize scholarships
        if rec.get("scholarships", False):
            score += 1000 # Significant bonus for scholarships

        # Could add scoring for location match precision, eligibility ease, etc.

        return score

    # Calculate scores and sort by score (descending)
    scored_results = [(rec, calculate_score(rec, user_criteria)) for rec in query_results]
    scored_results.sort(key=lambda item: item[1], reverse=True) # Sort by score

    # Return just the recommendation dictionaries, in prioritized order
    processed_list = [rec for rec, score in scored_results]

    # Ensure eligibility check here if not fully done in query
    # Simplified check: For simulation, assuming basic checks were enough in query or can be refined here.
    # In a real scenario, this might involve more detailed parsing of eligibility strings
    # and matching against structured user academic background data.

    print(f"--- Tool: process_recommendations returning {len(processed_list)} processed recommendations (prioritized). ---")
    return processed_list

# 3. Update the `format_recommendations` tool
def format_recommendations(recommendations: List[Dict[str, Any]]) -> str:
    """Formats the list of processed recommendations into a user-friendly string, including details."""
    print(f"--- Tool: format_recommendations called with {len(recommendations)} recommendations. ---")
    if not recommendations:
        return "I couldn't find any recommendations based on your criteria. Please try adjusting your preferences."

    formatted_output = "Based on your criteria, here are some potential options:\n\n"
    for i, rec in enumerate(recommendations):
        formatted_output += f"**{i+1}. {rec.get('university', 'Unknown University')}** ({rec.get('location', 'Unknown Location')})\n"
        formatted_output += f"   - Degree: {rec.get('degree', 'Unknown')}\n"
        formatted_output += f"   - Field: {rec.get('field', 'Unknown')}\n"
        formatted_output += f"   - Estimated Annual Cost (USD): ${rec.get('cost_usd_yr', 'Unknown'):,}\n"
        formatted_output += f"   - QS Ranking: {rec.get('ranking_qs', 'Unknown')}\n"
        formatted_output += f"   - Scholarships Available: {'Yes' if rec.get('scholarships', False) else 'No'}\n"
        formatted_output += f"   - Eligibility (Simplified): {rec.get('eligibility', 'Check University Site')}\n\n"

    print(f"--- Tool: format_recommendations returning formatted string. ---")
    return formatted_output


print("Academic Recommendation Tools defined.")


# --- Safety Guardrail Callbacks ---
# @title Define Safety Guardrail Callbacks

def validate_user_input(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Validates the latest user message for inappropriate content.
    If inappropriate content is found, blocks the LLM call and returns a polite refusal.
    Otherwise, returns None to proceed.
    """
    print(f"--- Callback: validate_user_input running for agent: {callback_context.agent_name} ---")

    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts:
                if content.parts[0].text:
                    last_user_message_text = content.parts[0].text
                    break

    print(f"--- Callback: Inspecting last user message: '{last_user_message_text[:100]}...' ---")

    # Simple inappropriate content check (replace with more sophisticated logic in production)
    inappropriate_keywords = ["profanity", "offensive", "inappropriate"]
    if any(keyword in last_user_message_text.lower() for keyword in inappropriate_keywords):
        print(f"--- Callback: Found inappropriate content. Blocking LLM call! ---")
        callback_context.state["input_validation_blocked"] = True
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="I'm sorry, I cannot process that request. Please keep our conversation polite and appropriate.")],
            )
        )
    else:
        print(f"--- Callback: Input seems appropriate. Allowing LLM call. ---")
        return None

print("✅ validate_user_input function defined.")


def validate_query_args(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """
    Validates arguments for the 'query_course_dataset' tool.
    Checks for presence of required keys and potentially value validity.
    If arguments are invalid, blocks the tool execution and returns an error dictionary.
    Otherwise, allows the tool call to proceed by returning None.
    """
    tool_name = tool.name
    agent_name = tool_context.agent_name
    print(f"--- Callback: validate_query_args running for tool '{tool_name}' in agent '{agent_name}' ---")
    print(f"--- Callback: Inspecting args: {args} ---")

    target_tool_name = "query_course_dataset"

    if tool_name == target_tool_name:
        required_keys = ["Field of Study", "Location Preferences"] # Keys expected by query_course_dataset
        missing_keys = [key for key in required_keys if key not in args or not args[key]]

        # Also check if user_info in state has these keys, indicating info was gathered
        # This is a more robust check than just looking at the tool args provided by LLM directly
        user_info_state = tool_context.state.get("user_info", {})
        missing_from_state = [key for key in ["degree_level", "field_of_study", "academic_background", "budget", "location_preferences"] if key not in user_info_state or not user_info_state[key]]


        # If the LLM didn't provide args OR the info is missing from state, block.
        # This ensures the tool is only called AFTER info gathering completed and stored.
        if missing_keys or missing_from_state:
            print(f"--- Callback: Blocking '{target_tool_name}'. Missing info from LLM args ({missing_keys}) or state ({missing_from_state}). ---")
            tool_context.state["tool_validation_blocked"] = True
            return {
                "status": "error",
                "error_message": f"Tool validation failed: Cannot run recommendation query. Missing required information. Please provide your Degree Level, Field of Study, Academic Background, Budget, and Location Preferences first."
            }


        # Add more specific value validation if needed
        # Example: Check if budget format is plausible, if degree level is recognized, etc.

        print(f"--- Callback: Arguments for '{target_tool_name}' seem valid (or info available in state). Allowing tool. ---")
    else:
        print(f"--- Callback: Tool '{tool_name}' is not the target tool for this validation. Allowing. ---")

    return None

print("✅ validate_query_args function defined.")


# --- Agent Definitions ---
# @title Define Agents and Root Agent

# Redefine the sub-agents to ensure they are not linked to a previous parent
# 1. Define the greeting_agent
greeting_agent = Agent(
    name="greeting_agent",
    model=LiteLlm(model=MODEL_OLLAMA_MISTRAL), # Use LiteLlm with Ollama model
    description="Handles initial user greetings.",
    instruction="You are a friendly Greeting Agent. Your sole purpose is to welcome the user warmly using the 'say_hello' tool. If the user provides a name in their greeting, pass it to the 'say_hello' tool using the 'name' argument. Otherwise, call 'say_hello' without any arguments to trigger its default greeting. Do not attempt to gather information or provide recommendations.",
    tools=[say_hello],
)
print(f"✅ Agent '{greeting_agent.name}' defined.")

# 2. Define the information_gathering_agent
information_gathering_agent = Agent(
    name="information_gathering_agent",
    model=LiteLlm(model=MODEL_OLLAMA_MISTRAL), # Use LiteLlm with Ollama model
    description="Gathers academic requirements, budget, and location preferences from the user.",
    instruction="You are the Information Gathering Agent. Your primary task is to systematically gather the following information from the user, one piece at a time, using the 'ask_user_info' tool: Degree Level Sought, Field of Study, Academic Background, Budget, and Location Preferences. Start by asking for the 'Degree Level Sought'. Once you ask a question, wait for the user's response (which will be returned by the tool). After receiving a response, ask for the next piece of information in the list until you have used the 'ask_user_info' tool for all five types. Do not provide recommendations or handle greetings. Your task is complete ONLY after you have successfully called the 'ask_user_info' tool for ALL five types of information.", # Significantly UPDATED INSTRUCTION for sequential asking and completion signal
    tools=[ask_user_info], # Assign the information gathering tool
)
print(f"✅ Agent '{information_gathering_agent.name}' defined.")


# 3. Define the recommendation_agent
recommendation_agent = Agent(
    name="recommendation_agent",
    model=LiteLlm(model=MODEL_OLLAMA_MISTRAL), # Use LiteLlm with Ollama model
    description="Recommends universities and courses based on user criteria.",
    instruction="You are the Recommendation Agent. Your task is to provide academic course and university recommendations. Retrieve the user criteria from the session state's 'user_info' dictionary (degree_level, field_of_study, academic_background, budget, location_preferences). Compile these into a dictionary. Use the 'query_course_dataset' tool with this compiled criteria dictionary. Then, use 'process_recommendations' with the query results and the user criteria. Finally, use 'format_recommendations' to present the results clearly to the user. If no recommendations are found after processing, inform the user. Do not attempt to gather information.", # UPDATED INSTRUCTION
    tools=[query_course_dataset, process_recommendations, format_recommendations], # Assign recommendation tools
    output_key="final_recommendation_report", # Save the final response to state
)
print(f"✅ Agent '{recommendation_agent.name}' defined.")


# Define the root_agent for orchestration and delegation
root_agent = Agent(
    name="academic_recommendation_root_agent_v4_ollama", # New version name
    model=LiteLlm(model=MODEL_OLLAMA_MISTRAL), # Use LiteLlm with Ollama model for orchestration
    description="Orchestrates the academic course recommendation process using Ollama, delegating to specialized agents, with input and tool argument guardrails.",
    instruction="You are the main Academic Recommendation Bot powered by Ollama. Your role is to guide the user through the recommendation process. "
                "Delegate initial greetings to the 'greeting_agent'. "
                "If the user expresses any interest in academic recommendations or provides *any* information about their academic goals (e.g., mentioning degree level, field of study, background, budget, or location preferences), *immediately* delegate to the 'information_gathering_agent' to ensure all necessary details are collected systematically. " # Explicitly prioritize info gathering
                "Once the 'information_gathering_agent' has completed its task (which you can infer when it has used the 'ask_user_info' tool for all required information types, or the user explicitly indicates they are done providing info), and the user asks for recommendations, *then* delegate to the 'recommendation_agent'. " # Clarified transition trigger
                "Manage the flow between these agents based on the user's intent and ensuring information gathering is completed BEFORE recommendations are attempted.", # Emphasize sequence
    tools=[], # Root agent might not need specific tools itself initially
    sub_agents=[greeting_agent, information_gathering_agent, recommendation_agent], # Link the sub-agents
    before_model_callback=validate_user_input, # Add the input validation callback
    before_tool_callback=validate_query_args # Add the tool argument validation callback
)
print(f"✅ Root Agent '{root_agent.name}' defined with input and tool argument guardrails.")


# --- Step 5: Interaction Flow ---
# @title Develop Interaction Flow and Run Conversation

# Assume root_agent (latest defined version), greeting_agent, information_gathering_agent, recommendation_agent
# and all tools and callbacks are defined and available.

# 1. Instantiate an InMemorySessionService
session_service_rec = InMemorySessionService()
print("✅ InMemorySessionService instantiated for recommendation system.")

# 2. Define constants for APP_NAME, USER_ID, and SESSION_ID.
APP_NAME_REC = "academic_recommendation_app"
USER_ID_REC = "student_user_001"
SESSION_ID_REC = "recommendation_session_001"


async def run_recommendation_conversation():
    print("\n--- Starting Academic Recommendation Conversation ---")

    # Move session creation and runner instantiation here:
    session_rec = await session_service_rec.create_session(
        app_name=APP_NAME_REC,
        user_id=USER_ID_REC,
        session_id=SESSION_ID_REC,
        # initial state can be set here, e.g., state={'user_info': {}}
    )
    print(f"✅ Session '{SESSION_ID_REC}' created for user '{USER_ID_REC}' in app '{APP_NAME_REC}'.")

    runner_rec = Runner(
        agent=root_agent, # Pass the latest defined root_agent
        app_name=APP_NAME_REC,
        session_service=session_service_rec # Use the session service for this app/user
    )
    print(f"✅ Runner instantiated for root agent '{runner_rec.agent.name}'.")

    # Update interact_with_recommendation_bot to accept runner_rec as an argument:
    async def interact_with_recommendation_bot(query: str):
        print(f"\n>>> User Query: {query}")
        content = types.Content(role='user', parts=[types.Part(text=query)])
        final_response_text = "Agent did not produce a final response."
        async for event in runner_rec.run_async(user_id=USER_ID_REC, session_id=SESSION_ID_REC, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        print(f"<<< Agent Response: {final_response_text}")

    # Call the interact_with_recommendation_bot function multiple times
    await interact_with_recommendation_bot("Hello bot!") # Expect delegation to greeting_agent

    # Simulate providing information - should trigger delegation to information_gathering_agent
    # The information_gathering_agent's instruction is to ask for ALL info sequentially once delegated to
    await interact_with_recommendation_bot("I'm looking for a Master's in Computer Science.")
    # The agent will continue asking for the remaining info in subsequent turns via ask_user_info tool calls

    # Simulating the agent asking and user responding for the remaining info.
    # In a real interactive system, these would be separate user inputs after the agent asks.
    # For this simulation, the ask_user_info tool PROVIDES the simulated response directly.
    # We still need to trigger the *turns* that would allow the agent to make the subsequent tool calls.
    # Each call to interact_with_recommendation_bot simulates one user turn.

    # The information_gathering_agent is instructed to call ask_user_info 5 times sequentially.
    # Each of the next 4 user inputs below simulates the user's response *after* the agent asks.
    # The agent will process the user input, realize it needs more info, and call ask_user_info again.

    # Simulate user responding after agent asks for Academic Background
    await interact_with_recommendation_bot("My background is a Bachelor's in Software Engineering with a 3.8 GPA.")

    # Simulate user responding after agent asks for Budget
    await interact_with_recommendation_bot("My budget is around $30,000 to $40,000 per year.")

    # Simulate user responding after agent asks for Location Preferences
    await interact_with_recommendation_bot("I prefer locations in Canada or Germany.")

    # At this point, the information_gathering_agent should have called ask_user_info 5 times and completed its task.
    # The next user input explicitly asks for recommendations.

    # Simulate asking for recommendations - should trigger delegation to recommendation_agent
    # The recommendation_agent should ideally use the info gathered and stored in state via the ask_user_info calls
    await interact_with_recommendation_bot("Can you recommend some universities based on this?")


    # Simulate a blocked input (if guardrail is active)
    # await interact_with_recommendation_bot("This is inappropriate content.") # Uncomment to test input guardrail

    # Simulate a query that might trigger tool validation failure (if guardrail is active)
    # This might require crafting a query that the LLM interprets as needing the tool
    # but without providing necessary info, which can be tricky to force consistently.
    # A more reliable test would involve mocking the LLM response to trigger the tool call with bad args.
    # For simplicity, we'll rely on the agent's natural flow.

    print("\n--- Conversation Ended ---")

    # Optional: Inspect final session state
    print("\n--- Inspecting Final Session State ---")
    final_session = await session_service_rec.get_session(app_name=APP_NAME_REC,
                                                         user_id=USER_ID_REC,
                                                         session_id=SESSION_ID_REC)
    if final_session:
        print("Final State:")
        # Use .get() for safer access to potentially missing keys
        print(f"  User Info: {final_session.state.get('user_info', 'Not Collected')}") # Should now be collected
        print(f"  Raw Query Results: {final_session.state.get('raw_query_results', 'Not Available')}") # Assuming query tool saves here if successful
        print(f"  Processed Recommendations: {final_session.state.get('processed_recommendations_list', 'Not Available')}") # Assuming process tool saves here if successful
        print(f"  Final Recommendation Report: {final_session.state.get('final_recommendation_report', 'Not Available')}") # From output_key on recommendation_agent
        print(f"  Input Validation Blocked: {final_session.state.get('input_validation_blocked', 'False')}") # From input guardrail
        print(f"  Tool Validation Blocked: {final_session.state.get('tool_validation_blocked', 'False')}") # From tool guardrail
    else:
        print("\n❌ Error: Could not retrieve final session state.")


# --- Execute the async function ---
# @title Execute the Conversation (for .py script)

# Use asyncio.run() to execute the main async function
# This is needed when running as a standard Python script
if __name__ == "__main__":
    print("Executing conversation using asyncio.run()...")
    try:
        asyncio.run(run_recommendation_conversation())
    except Exception as e:
        print(f"An error occurred during conversation execution: {e}")