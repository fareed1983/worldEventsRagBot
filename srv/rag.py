import threading
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from pprint import pprint
import uuid

llm = ChatOllama(model="llama3.2:3b")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

jobs = {}

lock = threading.Lock()


events_vector_store = Chroma(
    persist_directory="../tests/chroma_db",
    collection_name="events",
    embedding_function=embedding_model
)

sections_vector_store = Chroma(
    persist_directory="../tests/chroma_db",
    collection_name="sections",
    embedding_function=embedding_model
)

def start_chat_job(mode, category, query):

    if len(jobs) == 2:
        print("Max jobs running!")
        return None
    
    print("Starting job...")
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "Startng...",
        "jobId": job_id,
        "result": None
    } 

    if mode == "raw":
        run_func = run_raw_job
    else:
        run_func = run_rag_job

    thread = threading.Thread(
        target=run_func,
        args=(job_id, query, category)
    )
    
    thread.start()

    return job_id

def fetch_chat_job(job_id):
    job = jobs.get(job_id)
    
    if not job:
        return None
    
    job_copy = job.copy()

    if job["status"] == "Done":
        jobs.pop(job_id, None)

    return job_copy

def run_raw_job(job_id, query, category):
    print(f"Query: {query}\n")

    print("Generating without using RAG methods...\n")

    raw_prompt = ChatPromptTemplate.from_template( 
        """
        Answer this query given all the knowledge you have:
        {query} 
        """
    )

    chain = raw_prompt | llm | StrOutputParser()
    raw_output = chain.invoke({
        "query": query
    })

    print("Raw Output:\n\n")
    print(raw_output)

    jobs[job_id]["status"] = "Done"
    jobs[job_id]["result"] = raw_output


def run_rag_job(job_id, query, category):

    print("\n\n\nRunning RAG pipeline...")

    print(f"Query: {query}\nCategory: {category}")

    keywords_prompt = ChatPromptTemplate.from_template( 
        """
        Give me the top 3 very strong aditional single-word keywords I can use in my search to find the answer to this query:
        Query: {query}
        Just output the comma separated keywords and nothing else. I am not asking for future events. Only keywords.
        The keywords should not include what is already in the query except if strongly required by the query.
        Dont provide keywords already present in the query.
        Dont say anything other than the keywords. Don't output anything else and don't complain.
        """
    )

    chain = keywords_prompt | llm | StrOutputParser()

    jobs[job_id]["status"] = "Getting additional keywords..."

    keywords_output = chain.invoke({
        "query": query
    })

    print("Keywords: " + keywords_output)
    jobs[job_id]["status"] = f"Got additional keywords to use with LLM: {keywords_output}.<br>Now searching for top events in database..." 

    query_vector = embedding_model.embed_query(f"{query}\nKeywords: {keywords_output}")

    search_args = {"k": 5}

    if category is not None:
        search_args["filter"] = {"category": category}
    
    top_events = events_vector_store.similarity_search_by_vector(
        query_vector, 
        **search_args
    )

    event_index = {}
    section_index = {}

    
    jobs[job_id]["status"] = f"Got {len(top_events)} events that may match.<br>Now filtering relevant sections..."

    #pprint(top_events)

    formatted_events = ""

    for event in top_events:
        #print(event)
        event_index[event.id] = event
        # don't use the question and reinsert the date
        event.page_content = event.page_content.split('?\n\n', 1)[1]  
        event.page_content = f"{event.metadata['day']} {event.metadata['month']}, {event.metadata['year']}: {event.page_content}"
        formatted_events += f"\n\n\nEvent {event.id}: {event.page_content}"
        top_sections = sections_vector_store.similarity_search_by_vector(
            query_vector,
            k=5,
            ids=event.metadata["sections"]
        )
        for section in top_sections:
            formatted_events += f"\n\t{event.id}.{section.id}: {section.metadata["title"]}"
            section_index[section.id]=section
        

    select_events_prompt = ChatPromptTemplate.from_template( 
        """
        I was this query by the user: {query}

        These events and their sections could be relevant but some of them are not relevant:
            {formatted_events}

        Filter them and select only the events and sections most relevant to the user's query using a maximum of 5 events and 20 sections. Skip events not directly relevant to the user's query.

        If an event is not answering the query, just ignore it and don't include it in the output.
        
        Give the output in a JSON format strictly like this example: {{"e115": ["s899", "s655", ...], "e782": ["s311", ...], ...}}
        The JSON schema is {{"<eventId>": [""<sectionId>"", ""<sectionId>"", ...], ""<eventId>"": [""<sectionId>"", ""<sectionId>"", ...], ...}}
        Only use the correct event IDs in keys and nothing else.
        Just output one object in the JSON and nothing else! 
        """
    )

    filled_select_events_prompt = select_events_prompt.format(
        query = query,
        formatted_events = formatted_events
    )

    # print("\nPrompt to select events:")
    # print(filled_select_events_prompt)

    print(f"\nSelect events prompt length: {len(filled_select_events_prompt)}\n\n")
    jobs[job_id]["status"] = f"Discarding irrelevant events and sections with LLM..."

    chain = select_events_prompt | llm | StrOutputParser()
    json_str = chain.invoke({
        "query": query,
        "formatted_events": formatted_events
    })

    clean_json = json_str[json_str.find("{"):json_str.rfind("}") + 1]

    print(f"JSON: {clean_json}")

    import json

    retrieved_content = ""
    
    tot_sections = 0

    try:
        data = json.loads(clean_json)
        for event_id, sections in data.items():
            event = event_index.get(event_id)
            if event is None: 
                continue
                
            retrieved_content += f"{event.page_content} "
            
            tot_sections += len(sections)

            for section_id in sections:
                section_id = section_id.split(".", 1)[-1] # sometimes the format returned is e12313.s2343
                section = section_index.get(section_id)
                if section is None:
                    continue
                retrieved_content += f"{section.metadata['summary']} "
            retrieved_content += "\n\n"
        
        print("\nRetrieved content to use: \n")
        print(retrieved_content)
        jobs[job_id]["status"] = f"Using {len(data.items())} events and {tot_sections} sections.<br>Performaing final generation with LLM..."

    except Exception:
        data = None
        retrieved_content = "No articles found"

    # Add today's date to the prompt so that LLM does not infer events in the future
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    final_prompt = ChatPromptTemplate.from_template( 
        """
        Todays date: {today}
        
        I was asked this query by the user: {query}

        Answer the user's query using only relevant parts of the below events:
        {retrieved_content}
        Ignore events and points that are not directly relevant to the query and don't mention them.
        Add factual information from your own knowledge-base to enhance the provided events but do not speculate.
        If events provided are not answering the query fully, use only true facts your own knowledge-base to add to the answer. 
        Before closing, provide an overall summary.
        Do not infer or assume specifications or dates.
        Do not complain in any way about the events provided to you.

        """
    )

    filled_final_prompt = final_prompt.format(
        query =  query,
        retrieved_content = retrieved_content,
        today = today
    )

    # print("\nFinal prompt:")
    # print(filled_final_prompt)

    print(f"\nFinal prompt length: {len(filled_final_prompt)}\n\n")

    chain = final_prompt | llm | StrOutputParser()

    output = chain.invoke({
        "query": query,
        "retrieved_content": retrieved_content,
        "today": today
    })

    print("\n\nFinal output:\n")
    print(output)

    jobs[job_id]["status"] = "Done"
    jobs[job_id]["result"] = output


