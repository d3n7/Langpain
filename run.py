import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import DuckDuckGoSearchTool
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

#authentication
with open('auth.txt', 'r') as f:
    auth = f.readlines()[0].strip()
os.environ['OPENAI_API_KEY'] = auth
cost = 0

#set up GPT-3.5 Chat Bot
chat = ChatOpenAI(temperature=0)
history = []

#set up GPT-3.5 agent tools
llm = OpenAI()
agentTools = [DuckDuckGoSearchTool()]
agentTools.extend(load_tools(['llm-math', 'terminal'], llm=llm))

#set up GPT-3.5 director
dllm = ChatOpenAI(temperature=0)
directorSys = 'Respond only with "yes" or "no".'
directorPrompt = 'Does the following request require searching the internet, interacting with the filesystem, ' \
                 'executing code, or doing math calculations?\n"{}"\nRespond only with yes or no.'
def direct(query):
    yn = dllm([SystemMessage(content=directorSys), HumanMessage(content=directorPrompt.format(query))]).content.lower()
    return 'yes' in yn

#Main loop
print('Type exit to quit')
while 1:
    prompt = input('> ')
    if prompt.lower() == 'exit':
        print('Total cost: $'+str(cost))
        print('currently missing non-agent costs')
        exit()

    with get_openai_callback() as cb:
        history.append(HumanMessage(content=prompt))
        answer = ''
        if direct(prompt):
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            #give the agent 1 message of context if there is any
            if len(history) > 1:
                memory.chat_memory.add_ai_message(history[-2].content)
            agent = initialize_agent(tools=agentTools, llm=chat, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
            try:
                answer = agent.run(prompt)
            except Exception as e:
                print('ERROR: '+str(e))
        else:
            answer = chat(history).content
        history.append(AIMessage(content=answer))
        cost += cb.total_cost
        print(answer)