# Import necessary libraries
from huggingface_hub import snapshot_download
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

# Download model snapshot
snapshot_download(repo_id="TheBloke/Llama-2-7B-GGUF", allow_patterns="llama-2-7b.Q5_0.gguf", local_dir="/content/")

# Initialize the callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Create LlamaCpp instance
llm = LlamaCpp(
    model_path="/content/llama-2-7b.Q5_0.gguf",
    temperature=0.7,
    max_tokens=256,
    max_new_tokens=32,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
    stop=['User:', 'Human:']
)

# Customize the chatbot's template
template = """Instruction: You are a friendly English AI chatbot. Respond to the user in a polite manner and be informative.

{chat_history}
User: {query}
AI: 
"""

prompt = PromptTemplate(input_variables=['chat_history', 'query'], template=template)

# Use a more advanced conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", max_history_length=3)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

def chat_with_bot(user_query):
    response = llm_chain.predict(query=user_query)
    return response
