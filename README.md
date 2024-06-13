# AI School Architect

## Overview
In this project, we developed AI School Architect to orchestrate AI agents performing various tasks such as research, code review, testing, and coding. This setup enhances productivity by delegating specific tasks to specialized agents, streamlining workflows, and ensuring accuracy and efficiency.

## Now It's Your Turn!
Embrace your creativity and personalize this project to craft a solution that uniquely addresses the challenges and inefficiencies you face in your own educational or professional environment. After seeing what our AI agents can do, it’s time for you to take the reins. Use the foundation we’ve built and apply it to a challenge you face in your own context. Here’s how you can get started:

## Minimum Requirements
- **Custom Agent Creation:** Develop new custom agents to match your specific workflow needs. This could include automating repetitive tasks, integrating with specific APIs, or creating specialized commands that you frequently use.

## Stretch Goals
- **Advanced Task Automation:** Enhance the agents to perform more complex tasks such as data analysis, report generation, and project management.
- **Context-Aware Assistance:** Develop features that enable the agents to understand the context of tasks better, offering more accurate suggestions and task executions based on the current project structure and standards.
- **Collaboration Features:** Implement tools that facilitate better collaboration among team members, such as task assignment automation, integration with project management tools, and real-time collaboration features.
- **Continuous Improvement:** Integrate the agents with feedback loops to learn from their actions and improve their performance over time, providing more personalized and relevant assistance as you continue to use them.

## Privacy and Submission Guidelines
- **Submission Requirements:** Please submit a link to your public repo with your implementation or a loom video showcasing your work on the BloomTech AI Platform.
- **Sensitive Information:** If your implementation involves sensitive information, you are not required to submit a public repository. Instead, a detailed review of your project through a Loom video is acceptable, where you can demonstrate the functionality and discuss the technologies used without exposing confidential data.

## Project Structure

- `agent_supervisor.py`: Contains the configuration and setup for AI agents and the workflow.
- `server.py`: Wraps the workflow graph in a FastAPI server using LangServe.

### `agent_supervisor.py`

This script includes the following key components:

- **Agent Configuration**: Defines the tools and prompts for various agents, including researchers, coders, reviewers, and QA testers.
- **Workflow Definition**: Uses `StateGraph` to define the workflow and how agents interact with each other and the supervisor.

### `server.py`

This script sets up a FastAPI server to expose the agent workflow as an API. It utilizes LangServe to create and run the server.

```python
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.agent_supervisor import graph

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Edit this to add the chain you want to add
add_routes(app, graph, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
