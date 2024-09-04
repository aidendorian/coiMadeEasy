import IPython
import time
import google.generativeai as genai
import sys
import vertexai
from rich import print as rich_print
from rich.markdown import Markdown as rich_Markdown
from IPython.display import Markdown, display
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Part,
)
from multimodal_qa_with_rag_utils import (
    get_similar_text_from_query,
    print_text_to_text_citation,
    get_gemini_response,
    get_answer_from_qa_system,
)
import sys
PROJECT_ID = "vanity-434412"
LOCATION = "asia-south1"



from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel
GOOGLE_API_KEY="AIzaSyDQoJahM227W4N9ms5XWkyyx0TXlJcE50o"
genai.configure(api_key=GOOGLE_API_KEY)

text_model = GenerativeModel("gemini-1.0-pro")
multimodal_model_15 = GenerativeModel("gemini-1.5-pro-001")
multimodal_model_15_flash = GenerativeModel("gemini-1.5-flash-001")
multimodal_model_10 = GenerativeModel("gemini-1.0-pro-vision-001")
text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

from multimodal_qa_with_rag_utils import (
    get_document_metadata,
    set_global_variable,
)

set_global_variable("text_embedding_model", text_embedding_model)
set_global_variable("multimodal_embedding_model", multimodal_embedding_model)
pdf_folder_path = "C:\\VS Codex\\Python\\data\\"

text_metadata_df = get_document_metadata(
    multimodal_model_15,
    pdf_folder_path,
    embedding_size=1408,
    add_sleep_after_page = True,
    sleep_time_after_page = 5,
    add_sleep_after_document=True,  
    sleep_time_after_document=5,
)



print("\n\n --- Completed processing. ---")




query = "What was in article 32A before it was repealed"






matching_results_text = get_similar_text_from_query(
    query,
    text_metadata_df,
    column_name="text_embedding_chunk",
    top_n=3,
    chunk_text=True,
)
print_text_to_text_citation(matching_results_text, print_top=True, chunk_text=True)
context = "\n".join(
    [value["chunk_text"] for key, value in matching_results_text.items()]
)

prompt = f"""Answer the question with the given context. If the specific answer is not in the context, please answer "I don't know".
Question: {query}
Context: {context}
Answer:
"""
print("\n **** Result: ***** \n")
safety_settings = {
}

rich_Markdown(
    get_gemini_response(
        multimodal_model_15,
        model_input=prompt,
        stream=True,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(temperature=0.5, max_output_tokens=8192),
    )
)








