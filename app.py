from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
import os
import asyncio

async def run_memory_chat():
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    config_file="browser_mcp.json"

    client = MCPClient(config_file)
    
    llm = ChatGroq(model="openai/gpt-oss-20b")

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,
    )

    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear the memory")

    try:
        while True:
            user_input=input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Bye!")
                break
            elif user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Memory Cleared")
                continue
            
            print("\n Assistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"\nError: {e}")
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())
