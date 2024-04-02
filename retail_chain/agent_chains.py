from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.llms.ollama import Ollama
from langchain.memory import VectorStoreRetrieverMemory

import tools as t
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.base import StructuredTool
from langchain.globals import set_verbose
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss

set_verbose(True)

# # Creating a VectorStore powered memory
# index = faiss.IndexFlatL2(384)
# embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
# retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
# memory = VectorStoreRetrieverMemory(retriever=retriever)

llm = Ollama(model="mistral")

tools = [StructuredTool.from_function(t.product_purchase), StructuredTool.from_function(t.schedule_purchase),
         StructuredTool.from_function(t.price_tracking), StructuredTool.from_function(t.order_tracking)]

prompt = ChatPromptTemplate.from_messages(
    [("system",
      """Respond to the human as helpfully and accurately as possible. You have access to the following tools:

    {tools}
    
    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
    
    Valid "action" values: "Final Answer" or {tool_names}
    
    Provide only ONE action per $JSON_BLOB, as shown:
    
    ```
    {{
      "action": $TOOL_NAME,
      "action_input": $INPUT
    }}
    ```
    
    Follow this format:
    
    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
    Action:
    ```
    {{
      "action": "Final Answer",
      "action_input": "Final response to human"
    }}
    
    Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation""",
      ),
     ("placeholder", "{chat_history}"),
     (
         "human",
         """{input}
    
    {agent_scratchpad}
     (reminder to respond in a JSON blob no matter what)""",
     ), ]
)

agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    # memory=memory,
    max_iterations=100,
)

while True:
    user_input = input("User: ")
    response = agent_executor.invoke({
        "input": user_input
    })
    print("Agent:", response['output'])
