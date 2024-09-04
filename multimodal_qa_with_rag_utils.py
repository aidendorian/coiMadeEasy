import glob
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from IPython.display import display
import PIL
import fitz
import numpy as np
import pandas as pd
import requests
from vertexai.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
)
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
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel





#here



# function to set embeddings as global variable

def set_global_variable(variable_name: str, value: any) -> None:
    """
    Sets the value of a global variable.

    Args:
        variable_name: The name of the global variable (as a string).
        value: The value to assign to the global variable. This can be of any type.
    """
    global_vars = globals()  # Get a dictionary of global variables
    global_vars[variable_name] = value 




text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
def get_text_embedding_from_text_embedding_model(
    text: str,
    return_array: Optional[bool] = False,
) -> list:
    """
    Generates a numerical text embedding from a provided text input using a text embedding model.

    Args:
        text: The input text string to be embedded.
        return_array: If True, returns the embedding as a NumPy array.
                      If False, returns the embedding as a list. (Default: False)

    Returns:
        list or numpy.ndarray: A 768-dimensional vector representation of the input text.
                               The format (list or NumPy array) depends on the
                               value of the 'return_array' parameter.
    """
    embeddings = text_embedding_model.get_embeddings([text])
    text_embedding = [embedding.values for embedding in embeddings][0]

    if return_array:
        text_embedding = np.fromiter(text_embedding, dtype=float)

    # returns 768 dimensional array
    return text_embedding






def get_pdf_doc_object(pdf_path: str) -> tuple[fitz.Document, int]:
    """
    Opens a PDF file using fitz.open() and returns the PDF document object and the number of pages.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A tuple containing the `fitz.Document` object and the number of pages in the PDF.

    Raises:
        FileNotFoundError: If the provided PDF path is invalid.

    """

    # Open the PDF file
    doc: fitz.Document = fitz.open(pdf_path)

    # Get the number of pages in the PDF file
    num_pages: int = len(doc)

    return doc, num_pages


# Add colors to the print
class Color:
    """
    This class defines a set of color codes that can be used to print text in different colors.
    This will be used later to print citations and results to make outputs more readable.
    """

    PURPLE: str = "\033[95m"
    CYAN: str = "\033[96m"
    DARKCYAN: str = "\033[36m"
    BLUE: str = "\033[94m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"
    RED: str = "\033[91m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    END: str = "\033[0m"


def get_text_overlapping_chunk(
    text: str, character_limit: int = 1000, overlap: int = 100
) -> dict:
    """
    * Breaks a text document into chunks of a specified size, with an overlap between chunks to preserve context.
    * Takes a text document, character limit per chunk, and overlap between chunks as input.
    * Returns a dictionary where the keys are chunk numbers and the values are the corresponding text chunks.

    Args:
        text: The text document to be chunked.
        character_limit: Maximum characters per chunk (defaults to 1000).
        overlap: Number of overlapping characters between chunks (defaults to 100).

    Returns:
        A dictionary where keys are chunk numbers and values are the corresponding text chunks.

    Raises:
        ValueError: If `overlap` is greater than `character_limit`.

    """

    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Initialize variables
    chunk_number = 1
    chunked_text_dict = {}

    # Iterate over text with the given limit and overlap
    for i in range(0, len(text), character_limit - overlap):
        end_index = min(i + character_limit, len(text))
        chunk = text[i:end_index]

        # Encode and decode for consistent encoding
        chunked_text_dict[chunk_number] = chunk.encode("ascii", "ignore").decode(
            "utf-8", "ignore"
        )

        # Increment chunk number
        chunk_number += 1

    return chunked_text_dict


def get_page_text_embedding(text_data: Union[dict, str]) -> dict:
    """
    * Generates embeddings for each text chunk using a specified embedding model.
    * Takes a dictionary of text chunks and an embedding size as input.
    * Returns a dictionary where the keys are chunk numbers and the values are the corresponding embeddings.

    Args:
        text_data: Either a dictionary of pre-chunked text or the entire page text.
        embedding_size: Size of the embedding vector (defaults to 128).

    Returns:
        A dictionary where keys are chunk numbers or "text_embedding" and values are the corresponding embeddings.

    """

    embeddings_dict = {}

    if not text_data:
        return embeddings_dict

    if isinstance(text_data, dict):
        # Process each chunk
        # print(text_data)
        for chunk_number, chunk_value in text_data.items():
            text_embd = get_text_embedding_from_text_embedding_model(text=chunk_value)
            embeddings_dict[chunk_number] = text_embd
    else:
        # Process the first 1000 characters of the page text
        text_embd = get_text_embedding_from_text_embedding_model(text=text_data)
        embeddings_dict["text_embedding"] = text_embd

    return embeddings_dict


def get_chunk_text_metadata(
    page: fitz.Page,
    character_limit: int = 1000,
    overlap: int = 100,
    embedding_size: int = 128,
) -> tuple[str, dict, dict, dict]:
    """
    * Extracts text from a given page object, chunks it, and generates embeddings for each chunk.
    * Takes a page object, character limit per chunk, overlap between chunks, and embedding size as input.
    * Returns the extracted text, the chunked text dictionary, and the chunk embeddings dictionary.

    Args:
        page: The fitz.Page object to process.
        character_limit: Maximum characters per chunk (defaults to 1000).
        overlap: Number of overlapping characters between chunks (defaults to 100).
        embedding_size: Size of the embedding vector (defaults to 128).

    Returns:
        A tuple containing:
            - Extracted page text as a string.
            - Dictionary of embeddings for the entire page text (key="text_embedding").
            - Dictionary of chunked text (key=chunk number, value=text chunk).
            - Dictionary of embeddings for each chunk (key=chunk number, value=embedding).

    Raises:
        ValueError: If `overlap` is greater than `character_limit`.

    """

    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Extract text from the page
    text: str = page.get_text().encode("ascii", "ignore").decode("utf-8", "ignore")

    # Get whole-page text embeddings
    page_text_embeddings_dict: dict = get_page_text_embedding(text)

    # Chunk the text with the given limit and overlap
    chunked_text_dict: dict = get_text_overlapping_chunk(text, character_limit, overlap)
    # print(chunked_text_dict)

    # Get embeddings for the chunks
    chunk_embeddings_dict: dict = get_page_text_embedding(chunked_text_dict)
    # print(chunk_embeddings_dict)

    # Return all extracted data
    return text, page_text_embeddings_dict, chunked_text_dict, chunk_embeddings_dict





def get_gemini_response(
    generative_multimodal_model,
    model_input: List[str],
    stream: bool = True,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
    },
    print_exception: bool = False,
) -> str:
    """
    This function generates text in response to a list of model inputs.

    Args:
        model_input: A list of strings representing the inputs to the model.
        stream: Whether to generate the response in a streaming fashion (returning chunks of text at a time) or all at once. Defaults to False.

    Returns:
        The generated text as a string.
    """
    response = generative_multimodal_model.generate_content(
        model_input,
        generation_config=generation_config,
        stream=stream,
        safety_settings=safety_settings,
    )
    response_list = []

    for chunk in response:
        try:
            response_list.append(chunk.text)
        except Exception as e:
            if print_exception:
              print(
                  "Exception occurred while calling gemini. Something is blocked. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----",
                  e,
              )
            else:
              print("Exception occurred while calling gemini. Something is blocked. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----")
            response_list.append("**Something blocked.**")
            continue
    response = "".join(response_list)
    print(response)
    return response


def get_text_metadata_df(
    filename: str, text_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and a text metadata dictionary as input,
    iterates over the text metadata dictionary and extracts the text, chunk text,
    and chunk embeddings for each page, creates a Pandas DataFrame with the
    extracted data, and returns it.

    Args:
        filename: The filename of the document.
        text_metadata: A dictionary containing the text metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted text, chunk text, and chunk embeddings for each page.
    """

    final_data_text: List[Dict] = []

    for key, values in text_metadata.items():
        for chunk_number, chunk_text in values["chunked_text_dict"].items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = int(key) + 1
            data["text"] = values["text"]
            data["text_embedding_page"] = values["page_text_embeddings"][
                "text_embedding"
            ]
            data["chunk_number"] = chunk_number
            data["chunk_text"] = chunk_text
            data["text_embedding_chunk"] = values["chunk_embeddings_dict"][chunk_number]

            final_data_text.append(data)

    return_df = pd.DataFrame(final_data_text)
    return_df = return_df.reset_index(drop=True)
    return return_df





def get_document_metadata(
    generative_multimodal_model,
    pdf_folder_path: str,
    embedding_size: int = 128,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
    },
    add_sleep_after_page: bool = False,
    sleep_time_after_page: int = 2,
    add_sleep_after_document: bool = False,
    sleep_time_after_document: int = 2,
) -> Tuple[pd.DataFrame]:


    text_metadata_df_final= pd.DataFrame()

    for pdf_path in glob.glob(pdf_folder_path + "/*.pdf"):
        print(
            "\n\n",
            "Processing the file: ---------------------------------",
            pdf_path,
            "\n\n",
        )

        doc, num_pages = get_pdf_doc_object(pdf_path)

        file_name = pdf_path.split("/")[-1]

        text_metadata: Dict[Union[int, str], Dict] = {}

        for page_num in range(num_pages):
            print(f"Processing page: {page_num + 1}")

            page = doc[page_num]

            text = page.get_text()
            (
                text,
                page_text_embeddings_dict,
                chunked_text_dict,
                chunk_embeddings_dict,
            ) = get_chunk_text_metadata(page, embedding_size=embedding_size)

            text_metadata[page_num] = {
                "text": text,
                "page_text_embeddings": page_text_embeddings_dict,
                "chunked_text_dict": chunked_text_dict,
                "chunk_embeddings_dict": chunk_embeddings_dict,
            }
            # Add sleep to reduce issues with Quota error on API
            if add_sleep_after_page:
                time.sleep(sleep_time_after_page)
                print(
                    "Sleeping for ",
                    sleep_time_after_page,
                    """ sec before processing the next page to avoid quota issues. You can disable it: "add_sleep_after_page = False"  """,
                )
        # Add sleep to reduce issues with Quota error on API
        if add_sleep_after_document:
            time.sleep(sleep_time_after_document)
            print(
                "\n \n Sleeping for ",
                sleep_time_after_document,
                """ sec before processing the next document to avoid quota issues. You can disable it: "add_sleep_after_document = False"  """,
            )

        text_metadata_df = get_text_metadata_df(file_name, text_metadata)

        text_metadata_df_final = pd.concat(
            [text_metadata_df_final, text_metadata_df], axis=0
        )

        text_metadata_df_final = text_metadata_df_final.reset_index(drop=True)
    return text_metadata_df_final


# Helper Functions


def get_user_query_text_embeddings(user_query: str) -> np.ndarray:
    """
    Extracts text embeddings for the user query using a text embedding model.

    Args:
        user_query: The user query text.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query text embedding.
    """

    return get_text_embedding_from_text_embedding_model(user_query)


def get_cosine_score(
    dataframe: pd.DataFrame, column_name: str, input_text_embd: np.ndarray
) -> float:
    """
    Calculates the cosine similarity between the user query embedding and the dataframe embedding for a specific column.

    Args:
        dataframe: The pandas DataFrame containing the data to compare against.
        column_name: The name of the column containing the embeddings to compare with.
        input_text_embd: The NumPy array representing the user query embedding.

    Returns:
        The cosine similarity score (rounded to two decimal places) between the user query embedding and the dataframe embedding.
    """

    text_cosine_score = round(np.dot(dataframe[column_name], input_text_embd), 2)
    return text_cosine_score



def print_text_to_text_citation(
    final_text: Dict[int, Dict[str, Any]],
    print_top: bool = True,
    chunk_text: bool = True,
) -> None:
    """
    Prints a formatted citation for each matched text in a dictionary.

    Args:
        final_text: A dictionary containing information about matched text passages,
                    with keys as text number and values as dictionaries containing
                    page number, cosine similarity score, chunk number (optional),
                    chunk text (optional), and page text (optional).
        print_top: A boolean flag indicating whether to only print the first citation (True) or all citations (False).
        chunk_text: A boolean flag indicating whether to print individual text chunks (True) or the entire page text (False).

    Returns:
        None (prints formatted citations to the console).
    """

    color = Color()

    # Iterate through the matched text citations
    for textno, text_dict in final_text.items():
        # Print the citation header
        print(color.RED + f"Citation {textno + 1}:", "Matched text: \n" + color.END)

        # Print the cosine similarity score
        print(color.BLUE + "score: " + color.END, text_dict["cosine_score"])

        # Print the file_name
        print(color.BLUE + "file_name: " + color.END, text_dict["file_name"])

        # Print the page number
        print(color.BLUE + "page_number: " + color.END, text_dict["page_num"])

        # Print the matched text based on the chunk_text argument
        if chunk_text:
            # Print chunk number and chunk text
            print(color.BLUE + "chunk_number: " + color.END, text_dict["chunk_number"])
            print(color.BLUE + "chunk_text: " + color.END, text_dict["chunk_text"])
        else:
            # Print page text
            print(color.BLUE + "page text: " + color.END, text_dict["page_text"])

        # Only print the first citation if print_top is True
        if print_top and textno == 0:
            break


def get_similar_text_from_query(
    query: str,
    text_metadata_df: pd.DataFrame,
    column_name: str = "",
    top_n: int = 3,
    chunk_text: bool = True,
    print_citation: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar text passages from a metadata DataFrame based on a text query.

    Args:
        query: The text query used for finding similar passages.
        text_metadata_df: A Pandas DataFrame containing the text metadata to search.
        column_name: The column name in the text_metadata_df containing the text embeddings or text itself.
        top_n: The number of most similar text passages to return.
        embedding_size: The dimensionality of the text embeddings (only used if text embeddings are stored in the column specified by `column_name`).
        chunk_text: Whether to return individual text chunks (True) or the entire page text (False).
        print_citation: Whether to immediately print formatted citations for the matched text passages (True) or just return the dictionary (False).

    Returns:
        A dictionary containing information about the top N most similar text passages, including cosine scores, page numbers, chunk numbers (optional), and chunk text or page text (depending on `chunk_text`).

    Raises:
        KeyError: If the specified `column_name` is not present in the `text_metadata_df`.
    """

    if column_name not in text_metadata_df.columns:
        raise KeyError(f"Column '{column_name}' not found in the 'text_metadata_df'")

    query_vector = get_user_query_text_embeddings(query)

    # Calculate cosine similarity between query text and metadata text
    cosine_scores = text_metadata_df.apply(
        lambda row: get_cosine_score(
            row,
            column_name,
            query_vector,
        ),
        axis=1,
    )

    # Get top N cosine scores and their indices
    top_n_indices = cosine_scores.nlargest(top_n).index.tolist()
    top_n_scores = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched text and their information
    final_text: Dict[int, Dict[str, Any]] = {}

    for matched_textno, index in enumerate(top_n_indices):
        # Create a sub-dictionary for each matched text
        final_text[matched_textno] = {}

        # Store page number
        final_text[matched_textno]["file_name"] = text_metadata_df.iloc[index][
            "file_name"
        ]

        # Store page number
        final_text[matched_textno]["page_num"] = text_metadata_df.iloc[index][
            "page_num"
        ]

        # Store cosine score
        final_text[matched_textno]["cosine_score"] = top_n_scores[matched_textno]

        if chunk_text:
            # Store chunk number
            final_text[matched_textno]["chunk_number"] = text_metadata_df.iloc[index][
                "chunk_number"
            ]

            # Store chunk text
            final_text[matched_textno]["chunk_text"] = text_metadata_df["chunk_text"][
                index
            ]
        else:
            # Store page text
            final_text[matched_textno]["text"] = text_metadata_df["text"][index]

    # Optionally print citations immediately
    if print_citation:
        print_text_to_text_citation(final_text, chunk_text=chunk_text)

    return final_text

def get_answer_from_qa_system(
    query: str,
    text_metadata_df,
    top_n_text: int = 10,
    instruction: Optional[str] = None,
    model=None,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=1, max_output_tokens=8192
    ),
    safety_settings: Optional[dict] = {
    },
) -> Union[str, None]:
    """Fetches answers from a combined text QA system.

    Args:
        query (str): The user's question.
        text_metadata_df: DataFrame containing text embeddings, file names, and page numbers.
        top_n_text (int, optional): Number of top text chunks to consider. Defaults to 10.
        instruction (str, optional): Customized instruction for the model. Defaults to a generic one.
        model: Model to use for QA.
        safety_settings: Safety settings for the model.
        generation_config: Generation configuration for the model.

    Returns:
        Union[str, None]: The generated answer or None if an error occurs.
    """
    # Build Gemini content
    if instruction is None:  # Use default instruction if not provided
        instruction = """Task: Answer the following questions in detail, providing clear reasoning and evidence from the text in bullet points.
                      Instructions:

                      1. **Analyze:** Carefully examine the provided text context.
                      2. **Synthesize:** Integrate information from both the visual and textual elements.
                      3. **Reason:**  Deduce logical connections and inferences to address the question.
                      4. **Respond:** Provide a concise, accurate answer in the following format:

                        * **Question:** [Question]
                        * **Answer:** [Direct response to the question]
                        * **Explanation:** [Bullet-point reasoning steps if applicable]
                        * **Source** [name of the file, page from where the information is citied]

                      5. **Ambiguity:** If the context is insufficient to answer, respond "Not enough context to answer."

                      """

    # Retrieve relevant chunks of text based on the query
    matching_results_chunks_data = get_similar_text_from_query(
        query,
        text_metadata_df,
        column_name="text_embedding_chunk",
        top_n=top_n_text,
        chunk_text=True,
    )
    # combine all the selected relevant text chunks
    context_text = ["Text Context: "]
    for key, value in matching_results_chunks_data.items():
        context_text.extend(
            [
                "Text Source: ",
                f"""file_name: "{value["file_name"]}" Page: "{value["page_num"]}""",
                "Text",
                value["chunk_text"],
            ]
        )
    # Get Gemini response with streaming (if supported)
    response = get_gemini_response(
        model,
        stream=True,
        safety_settings=safety_settings,
        generation_config=generation_config,
    )
    print(type(response)," 3")
    return (
        response,
        matching_results_chunks_data,
    )
