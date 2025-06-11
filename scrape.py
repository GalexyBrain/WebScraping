import asyncio
import datetime
import json
import os
from typing import List, Dict, Set, Optional, Tuple
import uuid
import argparse
import time
import math

from dotenv import load_dotenv
from google import genai
from google.generative_ai import types, client as genai_client  # Use specific import for client
from google.generative_ai.types import GenerateContentResponse  # Import for type hinting
from pydantic import BaseModel, Field, ValidationError

# --- Research Progress Tracking ---
class ResearchProgress:
    """Tracks the progress of the deep research process."""

    def __init__(self, depth: int, breadth: int):
        """
        Initializes the progress tracker.

        Args:
            depth: The maximum depth of the research.
            breadth: The maximum breadth (number of parallel queries) at each level.
        """
        self.total_depth = depth
        self.total_breadth = breadth
        self.current_depth = depth  # Start at the initial depth
        self.current_breadth = 0
        self.queries_by_depth: Dict[int, Dict[str, Dict]] = {} # Stores query details per depth level
        self.query_order: List[str] = []  # Tracks the order queries were added
        self.query_parents: Dict[str, Optional[str]] = {}  # Tracks parent query for each sub-query
        self.total_queries = 0  # Total number of unique queries planned/executed
        self.completed_queries = 0
        self.query_ids: Dict[str, str] = {}  # Stores persistent unique IDs for queries
        self.root_query: Optional[str] = None  # Stores the initial root query

    async def start_query(self, query: str, depth: int, parent_query: Optional[str] = None):
        """
        Records the start of a new query or sub-query.

        Args:
            query: The query string.
            depth: The current depth level of this query.
            parent_query: The query string of the parent, if this is a sub-query.
        """
        # Generate a unique ID if the query is new
        if query not in self.query_ids:
            query_id = str(uuid.uuid4())
            self.query_ids[query] = query_id
        else:
            query_id = self.query_ids[query] # Use existing ID if query is revisited

        # Set the root query if it's the first one
        if self.root_query is None and parent_query is None:
            self.root_query = query

        # Initialize the depth level if it doesn't exist
        if depth not in self.queries_by_depth:
            self.queries_by_depth[depth] = {}

        # Add the query if it's not already tracked at this depth
        if query not in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query] = {
                "completed": False,
                "learnings": [],
                "sources": [],  # Store source information {title: str, url: str}
                "id": query_id, # Use persistent ID
                "parent_id": self.query_ids.get(parent_query) if parent_query else None # Store parent ID
            }
            if query not in self.query_order: # Avoid adding duplicates to order
                self.query_order.append(query)
            if parent_query:
                 # Ensure parent query exists in the structure before assigning
                if parent_query in self.query_ids:
                    self.query_parents[query] = parent_query
                else:
                    print(f"Warning: Parent query '{parent_query}' not found for child '{query}'.") # Add warning

            self.total_queries += 1 # Increment total only for truly new queries

        # Update current progress state
        self.current_depth = depth
        # Count queries at the current depth accurately
        self.current_breadth = len(self.queries_by_depth.get(depth, {}))

        await self._report_progress(f"Started query: {query} at depth {depth}")

    async def add_learning(self, query: str, depth: int, learning: str):
        """Adds a learning/insight discovered for a specific query."""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            # Avoid adding duplicate learnings
            if learning not in self.queries_by_depth[depth][query]["learnings"]:
                self.queries_by_depth[depth][query]["learnings"].append(learning)
                await self._report_progress(f"Learning added for: {query}")
        else:
             print(f"Warning: Could not add learning. Query '{query}' at depth {depth} not found.")


    async def complete_query(self, query: str, depth: int):
        """Marks a query as completed and updates progress."""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            if not self.queries_by_depth[depth][query]["completed"]:
                self.queries_by_depth[depth][query]["completed"] = True
                self.completed_queries += 1
                await self._report_progress(f"Completed query: {query} at depth {depth}")

                # Check if parent needs completion (only if all its direct children are done)
                parent_query = self.query_parents.get(query)
                if parent_query:
                    await self._check_and_complete_parent(parent_query)
        else:
            print(f"Warning: Could not complete. Query '{query}' at depth {depth} not found.")


    async def add_sources(self, query: str, depth: int, sources: List[Dict[str, str]]):
        """
        Records web sources found for a specific query.

        Args:
            query: The query string.
            depth: The depth level of the query.
            sources: A list of dictionaries, each with 'url' and 'title'.
        """
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            current_sources = self.queries_by_depth[depth][query]["sources"]
            current_urls = {source["url"] for source in current_sources}

            added_count = 0
            for source in sources:
                # Ensure source has both url and title before adding
                if "url" in source and "title" in source and source["url"] not in current_urls:
                    current_sources.append(source)
                    current_urls.add(source["url"])
                    added_count += 1

            if added_count > 0:
                 await self._report_progress(f"Added {added_count} sources for query: {query}")
        else:
            print(f"Warning: Could not add sources. Query '{query}' at depth {depth} not found.")


    async def _check_and_complete_parent(self, parent_query: str):
        """Checks if all children of a parent query are complete, and if so, completes the parent."""
        # Find the depth of the parent query
        parent_depth = None
        for d, queries in self.queries_by_depth.items():
            if parent_query in queries:
                parent_depth = d
                break

        if parent_depth is None:
            print(f"Warning: Parent query '{parent_query}' not found in any depth level.")
            return # Parent query doesn't exist in our tracking

        # Find all direct children of this parent
        children = [q for q, p in self.query_parents.items() if p == parent_query]

        if not children:
            # If a query has no children added *yet*, it shouldn't be completed prematurely.
            # Completion should happen after its own processing or after its children (if any) are done.
            # Let's check if the parent itself is already marked completed. If not, don't force completion here.
             if not self.queries_by_depth[parent_depth][parent_query]["completed"]:
                 # It might be completed later by its own process_query call finishing
                 pass
             return

        # Check if all *tracked* children are complete
        all_children_complete = True
        for child_query in children:
            child_found = False
            for d, queries_at_depth in self.queries_by_depth.items():
                 if child_query in queries_at_depth:
                     child_found = True
                     if not queries_at_depth[child_query]["completed"]:
                         all_children_complete = False
                         break # Found an incomplete child
            if not child_found:
                 # If a child was planned but never started/tracked, consider it incomplete for parent status
                 # Or handle as needed - maybe log a warning? For now, assume parent can't complete.
                 print(f"Warning: Child query '{child_query}' of '{parent_query}' not found in tracking.")
                 all_children_complete = False # Treat missing child as incomplete
            if not all_children_complete:
                 break # Exit outer loop if an incomplete child was found

        # Complete the parent only if all its children are complete
        if all_children_complete:
            await self.complete_query(parent_query, parent_depth)


    async def _report_progress(self, action: str):
        """Reports the current research progress."""
        progress_percentage = int((self.completed_queries / max(1, self.total_queries)) * 100)
        progress_data = {
            "type": "research_progress",
            "action": action,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), # Use timezone aware UTC time
            "completed_queries": self.completed_queries,
            "total_queries": self.total_queries,
            "progress_percentage": progress_percentage,
            "current_depth_level": self.current_depth, # Add current depth level
            "tree": self._build_research_tree() if self.root_query else None # Build tree on each report
        }

        # Print progress to console (or could stream/log elsewhere)
        print(f"[Progress] {action} ({self.completed_queries}/{self.total_queries}) - {progress_percentage}%")
        # Optional: Print JSON for detailed inspection
        # print(json.dumps(progress_data, indent=2))


    def _build_research_tree(self):
        """Builds a hierarchical tree structure of the research queries."""
        if not self.root_query:
            return {}

        # Create a lookup for query data by ID for easier access
        query_data_by_id = {}
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                 query_data_by_id[data['id']] = {
                     "query": query,
                     "id": data['id'],
                     "status": "completed" if data["completed"] else "in_progress",
                     "depth": depth,
                     "learnings": data["learnings"],
                     "sources": data["sources"],
                     "parent_id": data.get("parent_id"), # Get parent ID from data
                     "sub_queries": [] # Initialize sub_queries list
                 }

        # Build the tree structure using parent_id links
        tree = {}
        nodes_by_id = {} # Helper to quickly find nodes

        # First pass: create all nodes
        for q_id, data in query_data_by_id.items():
            nodes_by_id[q_id] = data

        # Second pass: link children to parents
        for q_id, data in nodes_by_id.items():
            parent_id = data.get("parent_id")
            if parent_id and parent_id in nodes_by_id:
                nodes_by_id[parent_id]["sub_queries"].append(data)
            elif parent_id is None and data["query"] == self.root_query: # Found the root
                 tree = data # Assign the root node to the tree

        # Ensure root is correctly identified if the loop didn't catch it (e.g., single node)
        if not tree and self.root_query and self.query_ids.get(self.root_query) in nodes_by_id:
             tree = nodes_by_id[self.query_ids[self.root_query]]

        return tree


    def get_all_learnings(self) -> List[str]:
        """Retrieves a flat list of all unique learnings gathered."""
        all_learnings = set()
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                for learning in data.get("learnings", []):
                    all_learnings.add(learning)
        return list(all_learnings)

    def get_all_sources(self) -> List[Dict[str, str]]:
        """Retrieves a flat list of all unique sources gathered."""
        all_sources = {} # Use URL as key for uniqueness
        for depth, queries in self.queries_by_depth.items():
             for query, data in queries.items():
                 for source in data.get("sources", []):
                     if "url" in source and source["url"] not in all_sources:
                         all_sources[source["url"]] = source
        return list(all_sources.values())


# --- Pydantic Models for API Responses ---

class ResearchParameters(BaseModel):
    """Schema for determining research breadth and depth."""
    breadth: int = Field(..., ge=1, le=10) # Add validation constraints
    depth: int = Field(..., ge=1, le=5)
    explanation: str

class FollowUpQuestions(BaseModel):
    """Schema for generating follow-up questions."""
    follow_up_queries: List[str]

class QueryResponse(BaseModel):
    """Schema for generating search queries."""
    queries: List[str]

class ProcessedResult(BaseModel):
    """Schema for processing search results."""
    learnings: List[str]
    follow_up_questions: List[str]

class SimilarityResult(BaseModel):
    """Schema for checking query similarity."""
    are_similar: bool


# --- Deep Search Core Logic ---

class DeepSearch:
    """
    Performs deep, recursive web research using Google's Gemini API.

    Modes:
    - "fast": Prioritizes speed (reduced breadth/depth).
    - "balanced": Default balance of speed and comprehensiveness.
    - "comprehensive": Maximum detail and coverage (deeper recursion).
    """
    def __init__(self, api_key: str, mode: str = "balanced"):
        """
        Initializes the DeepSearch instance.

        Args:
            api_key: Your Google API key for Gemini.
            mode: The research mode ("fast", "balanced", "comprehensive").
        """
        if mode not in ["fast", "balanced", "comprehensive"]:
            raise ValueError("Invalid mode. Choose 'fast', 'balanced', or 'comprehensive'.")

        self.api_key = api_key
        self.mode = mode
        # Use the recommended client initialization
        genai.configure(api_key=api_key)
        self.client = genai_client.GenerativeModel(model_name='gemini-1.5-flash-latest') # Use latest flash model
        self.query_history: Set[str] = set() # Track queries generated across the entire run
        self.progress: Optional[ResearchProgress] = None # Initialize progress tracker later

        # Define mode-specific parameters
        self.mode_params = {
            "fast": {"breadth_factor": 0.6, "depth_factor": 0.5, "max_sub_queries": 1, "max_learnings": 2},
            "balanced": {"breadth_factor": 1.0, "depth_factor": 1.0, "max_sub_queries": 2, "max_learnings": 3},
            "comprehensive": {"breadth_factor": 1.2, "depth_factor": 1.5, "max_sub_queries": 3, "max_learnings": 5}
        }
        self.current_mode_params = self.mode_params[self.mode]

        # Tool configuration for Google Search
        self.google_search_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name='google_search',
                    description='Performs a Google search and returns relevant results.',
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            'query': types.Schema(type=types.Type.STRING, description='The search query')
                        },
                        required=['query']
                    )
                )
            ]
        )


    async def _call_gemini_api(
        self,
        prompt: str,
        response_schema: Optional[type[BaseModel]] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        use_search: bool = False
    ) -> Tuple[Optional[str], Optional[BaseModel], Optional[Dict]]:
        """
        Helper function to call the Gemini API with error handling and optional JSON parsing.

        Args:
            prompt: The input prompt for the model.
            response_schema: Pydantic model for structured JSON output.
            temperature: Controls randomness (0.0-1.0).
            max_output_tokens: Maximum tokens in the response.
            use_search: Whether to enable the Google Search tool.

        Returns:
            A tuple containing:
            - Raw text response (str or None if error).
            - Parsed Pydantic model (BaseModel or None if not applicable or error).
            - Raw response dictionary (dict or None if error).
        """
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_output_tokens,
        }
        tools = [self.google_search_tool] if use_search else None

        if response_schema:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = response_schema # Pass the schema directly

        try:
            # Use the instance client
            response = await self.client.generate_content_async(
                contents=prompt,
                generation_config=types.GenerationConfig(**generation_config), # Pass config object
                tools=tools
            )

            raw_response_dict = GenerateContentResponse.to_dict(response) # Get raw dict if needed

            # Handle potential API denials or empty responses
            if not response.candidates:
                 print(f"Warning: API returned no candidates for prompt: {prompt[:100]}...")
                 return None, None, raw_response_dict

            # Extract text part safely
            text_response = ""
            if response.candidates[0].content.parts:
                 text_response = response.candidates[0].content.parts[0].text

            # Parse JSON if schema was provided
            parsed_model = None
            if response_schema:
                try:
                    # Gemini client might automatically parse if schema is provided and mime type is JSON
                    # Check if the response object has a direct way to access parsed data (might vary by SDK version)
                    # For now, attempt manual parsing from text, assuming it's valid JSON
                    if text_response:
                        json_data = json.loads(text_response)
                        parsed_model = response_schema(**json_data)
                    else:
                         print(f"Warning: JSON response expected but got empty text for prompt: {prompt[:100]}...")

                except (json.JSONDecodeError, ValidationError, AttributeError) as parse_error:
                    print(f"Error parsing JSON response: {parse_error}")
                    print(f"Raw text received: {text_response}")
                    # Return raw text even if parsing fails, might still be useful
                    return text_response, None, raw_response_dict
                except Exception as e: # Catch other potential parsing issues
                     print(f"Unexpected error during JSON parsing: {e}")
                     return text_response, None, raw_response_dict


            return text_response, parsed_model, raw_response_dict

        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            # Consider more specific error handling (e.g., rate limits, auth errors)
            return None, None, None


    def determine_research_breadth_and_depth(self, query: str) -> Dict[str, int | str]:
        """
        Determines initial research breadth and depth based on query complexity using Gemini.
        Uses synchronous call as it's a setup step.

        Args:
            query: The initial research query.

        Returns:
            A dictionary with 'breadth', 'depth', and 'explanation'.
        """
        user_prompt = f"""
        Analyze this research query and determine the appropriate initial breadth (number of parallel search queries)
        and depth (levels of follow-up questions) needed for thorough research, considering the current research mode '{self.mode}'.

        Query: {query}

        Mode '{self.mode}' implies:
        - fast: Focus on core aspects, less exploration.
        - balanced: Good coverage of main facets.
        - comprehensive: Aim for exhaustive detail and nuances.

        Consider:
        1. Complexity and ambiguity of the topic.
        2. Expected scope (broad overview vs. specific details).
        3. Potential for diverse sub-topics or perspectives.
        4. How the mode ('{self.mode}') should influence the scale.

        Return ONLY a JSON object matching this schema:
        {{
          "breadth": integer (1-10, adjusted for mode),
          "depth": integer (1-5, adjusted for mode),
          "explanation": "brief reasoning"
        }}
        """
        # Use a synchronous call for this setup step
        try:
            # Configure synchronous client for this specific call if needed, or use the main async client carefully
            # For simplicity, let's assume a sync call method exists or adapt the async one
            # NOTE: The provided SDK snippet primarily uses async. A dedicated sync call might look different.
            # Let's simulate the call structure using the async helper and running it synchronously.

             async def run_sync():
                 _, parsed_response, _ = await self._call_gemini_api(
                     prompt=user_prompt,
                     response_schema=ResearchParameters,
                     temperature=0.2, # Lower temp for deterministic analysis
                     max_output_tokens=512 # Smaller max tokens for this task
                 )
                 return parsed_response

             parsed_response = asyncio.run(run_sync()) # Run the async func synchronously

             if parsed_response and isinstance(parsed_response, ResearchParameters):
                 # Apply mode-based adjustments if needed (or let the model handle it via prompt)
                 # Example adjustment (optional, model might do this already):
                 # breadth = max(1, int(parsed_response.breadth * self.current_mode_params['breadth_factor']))
                 # depth = max(1, int(parsed_response.depth * self.current_mode_params['depth_factor']))

                 return {
                     "breadth": parsed_response.breadth, # Use model's suggestion directly
                     "depth": parsed_response.depth,
                     "explanation": parsed_response.explanation
                 }
             else:
                 print("Warning: Failed to get valid parameters from API.")
                 raise ValueError("Failed to determine research parameters.")


        except Exception as e:
            print(f"Error determining research parameters: {str(e)}. Using mode defaults.")
            # Default values based on mode
            defaults = {
                "fast": {"breadth": 3, "depth": 1},
                "balanced": {"breadth": 5, "depth": 2},
                "comprehensive": {"breadth": 7, "depth": 3}
            }
            params = defaults.get(self.mode, defaults["balanced"])
            params["explanation"] = f"Using default values for '{self.mode}' mode due to error."
            return params


    async def generate_queries(
        self,
        topic: str,
        num_queries: int,
        learnings: List[str] = [],
        parent_query: Optional[str] = None, # Context about the parent
        existing_queries: Optional[Set[str]] = None # Pass the global history
    ) -> Set[str]:
        """
        Generates diverse, specific search queries for a topic, considering context and history.

        Args:
            topic: The central topic or query to generate sub-queries for.
            num_queries: The desired number of queries.
            learnings: Key insights gathered so far related to the topic.
            parent_query: The query that led to this generation request (provides context).
            existing_queries: Set of all queries generated so far in the research process to avoid duplicates.

        Returns:
            A set of unique query strings.
        """
        if existing_queries is None:
            existing_queries = set()

        mode_guidance = {
            "fast": "Generate concise, highly relevant queries focusing on the core topic.",
            "balanced": "Generate a mix of queries covering key aspects and related sub-topics.",
            "comprehensive": "Generate diverse and in-depth queries exploring nuances, different angles, and potential gaps in knowledge."
        }

        learnings_text = "\n".join([f"- {l}" for l in learnings]) if learnings else "No specific learnings yet."
        # Include parent query context if available
        parent_context = f'These queries should follow up on the parent query: "{parent_query}"' if parent_query else ""

        # Filter existing queries relevant to the current topic/parent for the prompt context (optional optimization)
        # relevant_history = {q for q in existing_queries if topic.lower() in q.lower() or (parent_query and parent_query.lower() in q.lower())} # Simple relevance check
        # history_text = "\n".join(f"- {q}" for q in relevant_history) if relevant_history else "None relevant yet."
        # For simplicity, provide the full history size context
        history_info = f"We have already explored {len(existing_queries)} queries in total."


        user_prompt = f"""
        You are a research assistant exploring the topic: "{topic}"
        {parent_context}
        Current research mode: '{self.mode}'. {mode_guidance[self.mode]}

        Based on what we know:
        {learnings_text}

        {history_info} Avoid generating queries that are semantically identical or too similar to previous ones.

        Generate {num_queries} distinct, specific, and well-formed search engine queries to gather more information. Focus on uncovering new aspects or deeper details relevant to "{topic}".

        Return ONLY a JSON object matching this schema:
        {{
          "queries": ["query 1", "query 2", ...]
        }}
        """

        text_response, parsed_response, _ = await self._call_gemini_api(
            prompt=user_prompt,
            response_schema=QueryResponse,
            temperature=0.8, # Slightly higher temp for diversity
            max_output_tokens=1024
        )

        if parsed_response and isinstance(parsed_response, QueryResponse):
            generated_queries = set(parsed_response.queries)
            # Filter out queries that are identical to existing ones (case-insensitive)
            new_queries = {q for q in generated_queries if q.lower() not in {eq.lower() for eq in existing_queries}}

            # Optional: Add semantic similarity check if needed (can be slow)
            # final_queries = set()
            # for q in new_queries:
            #     is_similar = False
            #     for existing_q in existing_queries:
            #         if await self._are_queries_similar(q, existing_q):
            #             is_similar = True
            #             break
            #     if not is_similar:
            #         final_queries.add(q)
            # return final_queries

            # Limit to num_queries requested
            return set(list(new_queries)[:num_queries])

        elif text_response: # Fallback: Try to parse from raw text if JSON failed
             print("Warning: Failed to parse JSON for queries, attempting fallback parsing.")
             try:
                 # Basic line splitting and filtering
                 lines = text_response.strip().split('\n')
                 queries = set()
                 in_queries_list = False
                 for line in lines:
                     line = line.strip()
                     if '"queries": [' in line:
                         in_queries_list = True
                         continue
                     if in_queries_list:
                         if line.endswith(']'):
                             in_queries_list = False
                             line = line[:-1] # Remove trailing bracket
                         if line.startswith('"') and line.endswith('"'):
                              query = line[1:-1].replace('\\"', '"') # Handle escaped quotes
                              if query and query.lower() not in {eq.lower() for eq in existing_queries}:
                                   queries.add(query)
                         elif line.startswith('"') and line.endswith('",'):
                              query = line[1:-2].replace('\\"', '"')
                              if query and query.lower() not in {eq.lower() for eq in existing_queries}:
                                   queries.add(query)

                 if queries:
                      return set(list(queries)[:num_queries])
                 else:
                      print("Warning: Fallback query parsing failed.")

             except Exception as fallback_e:
                 print(f"Error during fallback query parsing: {fallback_e}")


        print(f"Warning: Query generation failed for '{topic}'. Returning empty set.")
        return set() # Return empty set on failure


    def _extract_sources_from_response(self, response_dict: Optional[Dict]) -> List[Dict[str, str]]:
        """Extracts source URLs and titles from Gemini API grounding metadata."""
        sources = []
        if not response_dict:
            return sources

        try:
            # Navigate through the dictionary structure safely
            candidates = response_dict.get('candidates', [])
            if not candidates: return sources

            # Check for grounding metadata (structure might vary slightly)
            grounding_metadata = candidates[0].get('grounding_metadata')
            if not grounding_metadata: return sources

            retrieval_queries = grounding_metadata.get('retrieval_queries', [])
            search_results = grounding_metadata.get('search_entry_point', {}).get('rendered_content') # Example path

            # Extract from web search results if available (preferred)
            web_search_queries = grounding_metadata.get('web_search_queries', [])
            citations = grounding_metadata.get('citations', []) # More direct source info

            if citations:
                 for citation in citations:
                      uri = citation.get('uri')
                      title = citation.get('title', 'Unknown Title') # Provide default title
                      if uri:
                           # Basic check to avoid duplicates by URL
                           if uri not in {s['url'] for s in sources}:
                                sources.append({"url": uri, "title": title})

            # Fallback: Try extracting from rendered content if citations are missing
            # This part is highly dependent on the exact API response structure and might need adjustment
            elif search_results:
                 # Example: Parse HTML if rendered_content is HTML
                 # This requires a robust HTML parser (e.g., BeautifulSoup) which is not imported here.
                 # For simplicity, we'll assume a simpler structure or skip this fallback.
                 print("Note: Citations metadata preferred but missing. Skipping extraction from rendered_content for now.")
                 pass # Add parsing logic here if needed


        except (AttributeError, KeyError, IndexError) as e:
            print(f"Error parsing sources from response: {e}. Response structure might have changed.")
        except Exception as e:
            print(f"Unexpected error extracting sources: {e}")

        return sources


    async def search(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Performs a search using the Gemini model with the Google Search tool enabled.

        Args:
            query: The search query string.

        Returns:
            A tuple containing:
            - The generated text answer from the model.
            - A list of source dictionaries [{'url': str, 'title': str}].
        """
        print(f"Performing search for: {query}")
        prompt = f"Please provide a comprehensive answer with sources for the query: {query}"

        # Use the search tool
        text_response, _, raw_response_dict = await self._call_gemini_api(
            prompt=prompt,
            use_search=True,
            temperature=0.5, # Lower temperature for factual search summarization
            max_output_tokens=4096 # Allow longer responses for search results
        )

        if text_response is None:
             text_response = f"Error performing search for: {query}" # Provide error message in text

        sources = self._extract_sources_from_response(raw_response_dict)

        # Basic formatting (optional): Add source list to the end of the text
        # formatted_text = text_response
        # if sources:
        #     formatted_text += "\n\n**Sources:**\n"
        #     for i, src in enumerate(sources):
        #         formatted_text += f"{i+1}. [{src.get('title', 'Source')}]({src.get('url', '#')})\n"

        # Return the raw text and the structured sources separately
        return text_response or "", sources


    async def process_result(
        self,
        query: str,
        result_text: str,
        num_learnings: int,
        num_follow_up: int,
        existing_queries: Set[str] # Pass history for context
    ) -> Dict[str, List[str]]:
        """
        Processes search result text to extract key learnings and generate relevant follow-up questions.

        Args:
            query: The original query for context.
            result_text: The text obtained from the search.
            num_learnings: Desired number of learnings.
            num_follow_up: Desired number of follow-up questions.
            existing_queries: Set of all queries generated so far.

        Returns:
            A dictionary with "learnings" and "follow_up_questions" lists.
        """
        user_prompt = f"""
        Analyze the following search result obtained for the query: "{query}"
        Research mode: '{self.mode}'.

        Search Result Text:
        ---
        {result_text[:6000]}
        ---
        (Result truncated if necessary)

        Based *only* on the text provided above:
        1. Extract the {num_learnings} most important and distinct learnings/insights directly supported by the text. Be concise.
        2. Generate {num_follow_up} specific and insightful follow-up questions that arise *from this text* and would logically extend the research on "{query}". Avoid generic questions. Ensure these questions are different from existing ones (total existing: {len(existing_queries)}).

        Return ONLY a JSON object matching this schema:
        {{
          "learnings": ["concise learning 1", ...],
          "follow_up_questions": ["specific follow-up question 1", ...]
        }}
        """

        _, parsed_response, _ = await self._call_gemini_api(
            prompt=user_prompt,
            response_schema=ProcessedResult,
            temperature=0.6, # Moderate temperature for extraction and generation
            max_output_tokens=1536 # Sufficient for learnings and questions
        )

        if parsed_response and isinstance(parsed_response, ProcessedResult):
            # Filter follow-up questions against history
            new_follow_ups = [
                 q for q in parsed_response.follow_up_questions
                 if q.lower() not in {eq.lower() for eq in existing_queries}
            ]
            return {
                "learnings": parsed_response.learnings[:num_learnings], # Ensure limit
                "follow_up_questions": new_follow_ups[:num_follow_up] # Ensure limit
            }
        else:
            print(f"Warning: Failed to process result for '{query}'. Generating basic follow-up.")
            # Fallback: Generate simple follow-up questions if processing fails
            fallback_follow_ups = await self.generate_queries(
                topic=query,
                num_queries=num_follow_up,
                learnings=[], # No learnings extracted
                parent_query=query,
                existing_queries=existing_queries
            )
            return {
                "learnings": [f"Could not automatically extract learnings for '{query}'."],
                "follow_up_questions": list(fallback_follow_ups)
            }


    async def _are_queries_similar(self, query1: str, query2: str) -> bool:
        """
        Checks if two queries are semantically similar using Gemini. (Use sparingly due to API calls)
        """
        # Simple checks first
        if query1.strip().lower() == query2.strip().lower():
            return True
        # Consider adding more efficient checks (e.g., Jaccard similarity on words) if needed

        # Use Gemini for semantic check
        user_prompt = f"""
        Are the following two search queries semantically similar? (i.e., would they likely retrieve very similar search results?)

        Query 1: "{query1}"
        Query 2: "{query2}"

        Return ONLY a JSON object matching this schema:
        {{
          "are_similar": boolean
        }}
        """
        _, parsed_response, _ = await self._call_gemini_api(
            prompt=user_prompt,
            response_schema=SimilarityResult,
            temperature=0.1, # Very low temp for deterministic comparison
            max_output_tokens=256
        )

        if parsed_response and isinstance(parsed_response, SimilarityResult):
            return parsed_response.are_similar
        else:
            print(f"Warning: Similarity check failed between '{query1}' and '{query2}'. Assuming not similar.")
            return False # Default to not similar on error to avoid discarding potentially unique queries


    async def _research_recursive(self, query: str, depth: int, breadth: int, parent_query: Optional[str] = None):
        """Recursive helper function to perform research at a given depth."""
        if depth <= 0:
            print(f"Reached max depth for query: {query}")
            return

        if not self.progress:
             print("Error: ResearchProgress tracker not initialized.")
             return

        # --- 1. Start and Search ---
        await self.progress.start_query(query, depth, parent_query)
        search_text, sources = await self.search(query)

        # Add sources found for this query
        if sources:
            await self.progress.add_sources(query, depth, sources)

        # --- 2. Process Results ---
        num_learnings_to_extract = self.current_mode_params['max_learnings']
        num_follow_ups_to_generate = breadth # Generate enough potential follow-ups for the breadth

        processed_result = await self.process_result(
            query=query,
            result_text=search_text,
            num_learnings=num_learnings_to_extract,
            num_follow_up=num_follow_ups_to_generate,
            existing_queries=self.query_history # Pass the global history
        )

        # Add extracted learnings
        for learning in processed_result.get("learnings", []):
            await self.progress.add_learning(query, depth, learning)

        # --- 3. Generate and Recurse on Follow-up Questions ---
        follow_up_questions = processed_result.get("follow_up_questions", [])

        # Filter follow-ups that haven't been explored yet
        new_unique_follow_ups = {
            q for q in follow_up_questions
            if q.lower() not in {qh.lower() for qh in self.query_history}
        }

        # Limit the number of follow-ups based on breadth for the *next* level
        next_level_queries = list(new_unique_follow_ups)[:breadth]

        # Update global history *before* recursing
        self.query_history.update(next_level_queries)

        if next_level_queries and depth > 1:
            print(f"Depth {depth} -> {depth-1}: Exploring {len(next_level_queries)} follow-up(s) for '{query}'...")

            # Calculate breadth for the next level (can decrease in deeper levels)
            # Example: Reduce breadth by 1 for each level deeper, minimum of 1
            next_breadth = max(1, breadth - 1) if self.mode != "comprehensive" else breadth # Keep breadth in comprehensive

            tasks = []
            for next_query in next_level_queries:
                 # Check again if query somehow got added to history concurrently (unlikely with current structure but safe)
                 if next_query.lower() not in {p_query.lower() for p_depth in self.progress.queries_by_depth for p_query in self.progress.queries_by_depth[p_depth]}:
                      tasks.append(
                          self._research_recursive(
                              query=next_query,
                              depth=depth - 1,
                              breadth=next_breadth,
                              parent_query=query # Pass current query as parent
                          )
                      )

            if tasks:
                 await asyncio.gather(*tasks)
            else:
                 print(f"No new, unique follow-up queries to explore for '{query}' at depth {depth-1}.")

        else:
             print(f"No further recursion needed for '{query}' at depth {depth}.")


        # --- 4. Mark Query as Complete ---
        # This query's processing is done (recursion for children handled above)
        await self.progress.complete_query(query, depth)


    async def run_research(self, initial_query: str) -> Dict:
        """
        Starts and manages the deep research process for the initial query.

        Args:
            initial_query: The starting query for the research.

        Returns:
            A dictionary containing the final research tree, all learnings, and all sources.
            Example: {"tree": {...}, "learnings": [...], "sources": [...]}
        """
        print(f"--- Starting Deep Research ---")
        print(f"Initial Query: {initial_query}")
        print(f"Mode: {self.mode}")

        # 1. Determine initial parameters
        params = self.determine_research_breadth_and_depth(initial_query)
        initial_breadth = params['breadth']
        initial_depth = params['depth']
        print(f"Determined Parameters: Breadth={initial_breadth}, Depth={initial_depth}")
        print(f"Reasoning: {params['explanation']}")

        # 2. Initialize Progress Tracker
        self.progress = ResearchProgress(depth=initial_depth, breadth=initial_breadth)
        self.query_history = {initial_query} # Start history with the root query

        # 3. Start Recursive Research
        print(f"\n--- Research Execution ---")
        start_time = time.time()
        try:
            await self._research_recursive(
                query=initial_query,
                depth=initial_depth,
                breadth=initial_breadth,
                parent_query=None # Root query has no parent
            )
        except Exception as e:
             print(f"\n--- Research Error ---")
             print(f"An error occurred during research: {e}")
             # Optionally log the full traceback
             import traceback
             traceback.print_exc()
        finally:
            end_time = time.time()
            print(f"\n--- Research Complete ---")
            print(f"Total execution time: {end_time - start_time:.2f} seconds")
            print(f"Total unique queries explored: {len(self.query_history)}")
            if self.progress:
                 print(f"Total learnings gathered: {len(self.progress.get_all_learnings())}")
                 print(f"Total sources found: {len(self.progress.get_all_sources())}")


        # 4. Consolidate and Return Results
        if self.progress:
            final_tree = self.progress._build_research_tree()
            all_learnings = self.progress.get_all_learnings()
            all_sources = self.progress.get_all_sources()

            # Save the final tree structure
            try:
                 tree_filename = f"research_tree_{initial_query[:20].replace(' ','_')}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
                 with open(tree_filename, "w", encoding='utf-8') as f:
                     json.dump(final_tree, f, indent=2, ensure_ascii=False)
                 print(f"Research tree saved to: {tree_filename}")
            except Exception as e:
                 print(f"Error saving research tree: {e}")


            return {
                "tree": final_tree,
                "learnings": all_learnings,
                "sources": all_sources
            }
        else:
            print("Error: Research progress tracker was not available.")
            return {"tree": {}, "learnings": [], "sources": []}


    async def generate_final_report(self, query: str, learnings: list[str], sources: list[dict[str, str]]) -> str:
        """
        Generates a final, creatively formatted report summarizing the research findings.

        Args:
            query: The initial research query.
            learnings: A list of all unique learnings gathered.
            sources: A list of all unique sources found.

        Returns:
            A formatted string containing the final report in Markdown.
        """
        print("\n--- Generating Final Report ---")
        if not learnings:
            return "No learnings were gathered during the research process to generate a report."

        # Format learnings and sources for the prompt
        learnings_text = "\n".join([f"- {learning}" for learning in learnings])
        sources_text = "\n".join([f"- [{src.get('title', 'Source')}]({src.get('url')})" for i, src in enumerate(sources)]) if sources else "No specific sources were cited."


        user_prompt = f"""
        You are a creative storyteller and research synthesizer. Your task is to transform the following research findings into an engaging and distinctive report about "{query}".

        Research Query: {query}

        Key Discoveries (Learnings):
        ---
        {learnings_text[:6000]}
        ---
        (Learnings truncated if necessary)

        Sources Consulted:
        ---
        {sources_text[:2000]}
        ---
        (Sources truncated if necessary)

        Craft a captivating report in Markdown format that:

        ## CREATIVE APPROACH & STRUCTURE
        1.  **Imaginative Opening:** Start with a hook that draws the reader into the world of "{query}". Avoid generic introductions.
        2.  **Narrative Flow:** Weave the key discoveries into a coherent narrative. Don't just list facts; connect them, show relationships, and build understanding. Use your unique voice.
        3.  **Insightful Synthesis:** Go beyond summarizing. Offer fresh perspectives, highlight surprising connections, or pose thought-provoking questions based on the learnings.
        4.  **Distinctive Style:** Experiment with tone and style (e.g., investigative, reflective, enthusiastic, critical) appropriate to the topic, but maintain clarity.
        5.  **Memorable Conclusion:** End with a strong concluding thought, a lingering question, or a call to further exploration that leaves an impact.

        ## MARKDOWN FORMATTING FOR IMPACT
        * **Evocative Headings:** Use `##` and `###` for sections with creative, descriptive titles.
        * **Emphasis:** Use **bold** and *italics* strategically to highlight key terms or ideas.
        * **Blockquotes:** Use `> ` for impactful quotes, contrasting points, or highlighting crucial learnings.
        * **Lists:** Use bullet (`* ` or `- `) or numbered lists for clarity when presenting multiple related points (e.g., steps, components, types).
        * **Horizontal Rules:** Use `---` to create clear separations between major sections or for dramatic effect.
        * **(Optional) Tables:** If appropriate, use Markdown tables to organize comparative data or structured information concisely.

        ## GUIDELINES
        * **Focus on Learnings:** Base the report primarily on the provided "Key Discoveries".
        * **Acknowledge Sources:** Briefly mention that the report is based on synthesized information from various sources (you don't need to cite inline unless specifically requested). Include the provided source list at the very end under a "Sources Consulted" heading.
        * **Be Creative, Be Clear:** Let your creativity shine, but ensure the report is easy to understand and logically structured. Avoid academic jargon unless the topic demands it.
        * **Length:** Aim for a comprehensive yet readable report. Don't be overly verbose or unnecessarily brief.

        Produce ONLY the Markdown report based on these instructions. Start directly with the report content (e.g., the first heading).
        """

        # Use the helper function for the API call
        report_text, _, _ = await self._call_gemini_api(
            prompt=user_prompt,
            temperature=0.8, # Higher temperature for creative writing
            max_output_tokens=8192 # Allow ample space for a detailed report
        )

        if report_text:
             # Append the source list manually if the model didn't include it reliably
             if "Sources Consulted" not in report_text[-500:]: # Check near the end
                  report_text += "\n\n## Sources Consulted\n" + sources_text
             print("Final report generated successfully.")
             return report_text
        else:
            print("Error: Failed to generate the final report.")
            # Fallback: Return a simple summary if generation fails
            return f"# Research Summary for: {query}\n\n" \
                   f"## Key Learnings:\n{learnings_text}\n\n" \
                   f"## Sources Consulted:\n{sources_text}\n\n" \
                   f"(Automated report generation failed)"


# --- Main Execution Block ---

async def main():
    """Main function to parse arguments and run the deep research."""
    parser = argparse.ArgumentParser(description="Perform deep research using Gemini API.")
    parser.add_argument("query", type=str, help="The initial research query.")
    parser.add_argument("-m", "--mode", type=str, default="balanced",
                        choices=["fast", "balanced", "comprehensive"],
                        help="Research mode (default: balanced).")
    parser.add_argument("-k", "--api-key", type=str, default=os.getenv("GOOGLE_API_KEY"),
                        help="Google API Key (reads from GOOGLE_API_KEY env var by default).")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Optional file path to save the final report (Markdown).")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: Google API Key is required. Set GOOGLE_API_KEY environment variable or use --api-key.")
        return

    # Load .env file if it exists (optional)
    load_dotenv()
    # Re-check API key from env var if not provided via argument after load_dotenv
    if not args.api_key:
         args.api_key = os.getenv("GOOGLE_API_KEY")
         if not args.api_key:
              print("Error: Google API Key is required. Set GOOGLE_API_KEY environment variable or use --api-key.")
              return


    # Initialize DeepSearch
    try:
        deep_search = DeepSearch(api_key=args.api_key, mode=args.mode)
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return
    except Exception as e:
         print(f"An unexpected error occurred during initialization: {e}")
         return


    # Run the research process
    research_results = await deep_search.run_research(args.query)

    # Generate the final report
    if research_results and research_results.get("learnings"):
        final_report = await deep_search.generate_final_report(
            query=args.query,
            learnings=research_results["learnings"],
            sources=research_results["sources"]
        )

        print("\n--- Final Report ---")
        print(final_report)

        # Save report to file if requested
        if args.output:
            try:
                with open(args.output, "w", encoding='utf-8') as f:
                    f.write(final_report)
                print(f"\nReport saved to: {args.output}")
            except IOError as e:
                print(f"\nError saving report to file: {e}")
    else:
        print("\nNo learnings gathered, skipping final report generation.")


if __name__ == "__main__":
    # Ensure an event loop is running for async operations
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nResearch interrupted by user.")
    except Exception as e:
         print(f"\nAn unexpected error occurred in the main execution: {e}")
         import traceback
         traceback.print_exc()import asyncio
import datetime
import json
import os
from typing import List, Dict, Set, Optional, Tuple
import uuid
import argparse
import time
import math

from dotenv import load_dotenv
from google import genai
from google.generative_ai import types, client as genai_client  # Use specific import for client
from google.generative_ai.types import GenerateContentResponse  # Import for type hinting
from pydantic import BaseModel, Field, ValidationError

# --- Research Progress Tracking ---
class ResearchProgress:
    """Tracks the progress of the deep research process."""

    def __init__(self, depth: int, breadth: int):
        """
        Initializes the progress tracker.

        Args:
            depth: The maximum depth of the research.
            breadth: The maximum breadth (number of parallel queries) at each level.
        """
        self.total_depth = depth
        self.total_breadth = breadth
        self.current_depth = depth  # Start at the initial depth
        self.current_breadth = 0
        self.queries_by_depth: Dict[int, Dict[str, Dict]] = {} # Stores query details per depth level
        self.query_order: List[str] = []  # Tracks the order queries were added
        self.query_parents: Dict[str, Optional[str]] = {}  # Tracks parent query for each sub-query
        self.total_queries = 0  # Total number of unique queries planned/executed
        self.completed_queries = 0
        self.query_ids: Dict[str, str] = {}  # Stores persistent unique IDs for queries
        self.root_query: Optional[str] = None  # Stores the initial root query

    async def start_query(self, query: str, depth: int, parent_query: Optional[str] = None):
        """
        Records the start of a new query or sub-query.

        Args:
            query: The query string.
            depth: The current depth level of this query.
            parent_query: The query string of the parent, if this is a sub-query.
        """
        # Generate a unique ID if the query is new
        if query not in self.query_ids:
            query_id = str(uuid.uuid4())
            self.query_ids[query] = query_id
        else:
            query_id = self.query_ids[query] # Use existing ID if query is revisited

        # Set the root query if it's the first one
        if self.root_query is None and parent_query is None:
            self.root_query = query

        # Initialize the depth level if it doesn't exist
        if depth not in self.queries_by_depth:
            self.queries_by_depth[depth] = {}

        # Add the query if it's not already tracked at this depth
        if query not in self.queries_by_depth[depth]:
            self.queries_by_depth[depth][query] = {
                "completed": False,
                "learnings": [],
                "sources": [],  # Store source information {title: str, url: str}
                "id": query_id, # Use persistent ID
                "parent_id": self.query_ids.get(parent_query) if parent_query else None # Store parent ID
            }
            if query not in self.query_order: # Avoid adding duplicates to order
                self.query_order.append(query)
            if parent_query:
                 # Ensure parent query exists in the structure before assigning
                if parent_query in self.query_ids:
                    self.query_parents[query] = parent_query
                else:
                    print(f"Warning: Parent query '{parent_query}' not found for child '{query}'.") # Add warning

            self.total_queries += 1 # Increment total only for truly new queries

        # Update current progress state
        self.current_depth = depth
        # Count queries at the current depth accurately
        self.current_breadth = len(self.queries_by_depth.get(depth, {}))

        await self._report_progress(f"Started query: {query} at depth {depth}")

    async def add_learning(self, query: str, depth: int, learning: str):
        """Adds a learning/insight discovered for a specific query."""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            # Avoid adding duplicate learnings
            if learning not in self.queries_by_depth[depth][query]["learnings"]:
                self.queries_by_depth[depth][query]["learnings"].append(learning)
                await self._report_progress(f"Learning added for: {query}")
        else:
             print(f"Warning: Could not add learning. Query '{query}' at depth {depth} not found.")


    async def complete_query(self, query: str, depth: int):
        """Marks a query as completed and updates progress."""
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            if not self.queries_by_depth[depth][query]["completed"]:
                self.queries_by_depth[depth][query]["completed"] = True
                self.completed_queries += 1
                await self._report_progress(f"Completed query: {query} at depth {depth}")

                # Check if parent needs completion (only if all its direct children are done)
                parent_query = self.query_parents.get(query)
                if parent_query:
                    await self._check_and_complete_parent(parent_query)
        else:
            print(f"Warning: Could not complete. Query '{query}' at depth {depth} not found.")


    async def add_sources(self, query: str, depth: int, sources: List[Dict[str, str]]):
        """
        Records web sources found for a specific query.

        Args:
            query: The query string.
            depth: The depth level of the query.
            sources: A list of dictionaries, each with 'url' and 'title'.
        """
        if depth in self.queries_by_depth and query in self.queries_by_depth[depth]:
            current_sources = self.queries_by_depth[depth][query]["sources"]
            current_urls = {source["url"] for source in current_sources}

            added_count = 0
            for source in sources:
                # Ensure source has both url and title before adding
                if "url" in source and "title" in source and source["url"] not in current_urls:
                    current_sources.append(source)
                    current_urls.add(source["url"])
                    added_count += 1

            if added_count > 0:
                 await self._report_progress(f"Added {added_count} sources for query: {query}")
        else:
            print(f"Warning: Could not add sources. Query '{query}' at depth {depth} not found.")


    async def _check_and_complete_parent(self, parent_query: str):
        """Checks if all children of a parent query are complete, and if so, completes the parent."""
        # Find the depth of the parent query
        parent_depth = None
        for d, queries in self.queries_by_depth.items():
            if parent_query in queries:
                parent_depth = d
                break

        if parent_depth is None:
            print(f"Warning: Parent query '{parent_query}' not found in any depth level.")
            return # Parent query doesn't exist in our tracking

        # Find all direct children of this parent
        children = [q for q, p in self.query_parents.items() if p == parent_query]

        if not children:
            # If a query has no children added *yet*, it shouldn't be completed prematurely.
            # Completion should happen after its own processing or after its children (if any) are done.
            # Let's check if the parent itself is already marked completed. If not, don't force completion here.
             if not self.queries_by_depth[parent_depth][parent_query]["completed"]:
                 # It might be completed later by its own process_query call finishing
                 pass
             return

        # Check if all *tracked* children are complete
        all_children_complete = True
        for child_query in children:
            child_found = False
            for d, queries_at_depth in self.queries_by_depth.items():
                 if child_query in queries_at_depth:
                     child_found = True
                     if not queries_at_depth[child_query]["completed"]:
                         all_children_complete = False
                         break # Found an incomplete child
            if not child_found:
                 # If a child was planned but never started/tracked, consider it incomplete for parent status
                 # Or handle as needed - maybe log a warning? For now, assume parent can't complete.
                 print(f"Warning: Child query '{child_query}' of '{parent_query}' not found in tracking.")
                 all_children_complete = False # Treat missing child as incomplete
            if not all_children_complete:
                 break # Exit outer loop if an incomplete child was found

        # Complete the parent only if all its children are complete
        if all_children_complete:
            await self.complete_query(parent_query, parent_depth)


    async def _report_progress(self, action: str):
        """Reports the current research progress."""
        progress_percentage = int((self.completed_queries / max(1, self.total_queries)) * 100)
        progress_data = {
            "type": "research_progress",
            "action": action,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), # Use timezone aware UTC time
            "completed_queries": self.completed_queries,
            "total_queries": self.total_queries,
            "progress_percentage": progress_percentage,
            "current_depth_level": self.current_depth, # Add current depth level
            "tree": self._build_research_tree() if self.root_query else None # Build tree on each report
        }

        # Print progress to console (or could stream/log elsewhere)
        print(f"[Progress] {action} ({self.completed_queries}/{self.total_queries}) - {progress_percentage}%")
        # Optional: Print JSON for detailed inspection
        # print(json.dumps(progress_data, indent=2))


    def _build_research_tree(self):
        """Builds a hierarchical tree structure of the research queries."""
        if not self.root_query:
            return {}

        # Create a lookup for query data by ID for easier access
        query_data_by_id = {}
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                 query_data_by_id[data['id']] = {
                     "query": query,
                     "id": data['id'],
                     "status": "completed" if data["completed"] else "in_progress",
                     "depth": depth,
                     "learnings": data["learnings"],
                     "sources": data["sources"],
                     "parent_id": data.get("parent_id"), # Get parent ID from data
                     "sub_queries": [] # Initialize sub_queries list
                 }

        # Build the tree structure using parent_id links
        tree = {}
        nodes_by_id = {} # Helper to quickly find nodes

        # First pass: create all nodes
        for q_id, data in query_data_by_id.items():
            nodes_by_id[q_id] = data

        # Second pass: link children to parents
        for q_id, data in nodes_by_id.items():
            parent_id = data.get("parent_id")
            if parent_id and parent_id in nodes_by_id:
                nodes_by_id[parent_id]["sub_queries"].append(data)
            elif parent_id is None and data["query"] == self.root_query: # Found the root
                 tree = data # Assign the root node to the tree

        # Ensure root is correctly identified if the loop didn't catch it (e.g., single node)
        if not tree and self.root_query and self.query_ids.get(self.root_query) in nodes_by_id:
             tree = nodes_by_id[self.query_ids[self.root_query]]

        return tree


    def get_all_learnings(self) -> List[str]:
        """Retrieves a flat list of all unique learnings gathered."""
        all_learnings = set()
        for depth, queries in self.queries_by_depth.items():
            for query, data in queries.items():
                for learning in data.get("learnings", []):
                    all_learnings.add(learning)
        return list(all_learnings)

    def get_all_sources(self) -> List[Dict[str, str]]:
        """Retrieves a flat list of all unique sources gathered."""
        all_sources = {} # Use URL as key for uniqueness
        for depth, queries in self.queries_by_depth.items():
             for query, data in queries.items():
                 for source in data.get("sources", []):
                     if "url" in source and source["url"] not in all_sources:
                         all_sources[source["url"]] = source
        return list(all_sources.values())


# --- Pydantic Models for API Responses ---

class ResearchParameters(BaseModel):
    """Schema for determining research breadth and depth."""
    breadth: int = Field(..., ge=1, le=10) # Add validation constraints
    depth: int = Field(..., ge=1, le=5)
    explanation: str

class FollowUpQuestions(BaseModel):
    """Schema for generating follow-up questions."""
    follow_up_queries: List[str]

class QueryResponse(BaseModel):
    """Schema for generating search queries."""
    queries: List[str]

class ProcessedResult(BaseModel):
    """Schema for processing search results."""
    learnings: List[str]
    follow_up_questions: List[str]

class SimilarityResult(BaseModel):
    """Schema for checking query similarity."""
    are_similar: bool


# --- Deep Search Core Logic ---

class DeepSearch:
    """
    Performs deep, recursive web research using Google's Gemini API.

    Modes:
    - "fast": Prioritizes speed (reduced breadth/depth).
    - "balanced": Default balance of speed and comprehensiveness.
    - "comprehensive": Maximum detail and coverage (deeper recursion).
    """
    def __init__(self, api_key: str, mode: str = "balanced"):
        """
        Initializes the DeepSearch instance.

        Args:
            api_key: Your Google API key for Gemini.
            mode: The research mode ("fast", "balanced", "comprehensive").
        """
        if mode not in ["fast", "balanced", "comprehensive"]:
            raise ValueError("Invalid mode. Choose 'fast', 'balanced', or 'comprehensive'.")

        self.api_key = api_key
        self.mode = mode
        # Use the recommended client initialization
        genai.configure(api_key=api_key)
        self.client = genai_client.GenerativeModel(model_name='gemini-1.5-flash-latest') # Use latest flash model
        self.query_history: Set[str] = set() # Track queries generated across the entire run
        self.progress: Optional[ResearchProgress] = None # Initialize progress tracker later

        # Define mode-specific parameters
        self.mode_params = {
            "fast": {"breadth_factor": 0.6, "depth_factor": 0.5, "max_sub_queries": 1, "max_learnings": 2},
            "balanced": {"breadth_factor": 1.0, "depth_factor": 1.0, "max_sub_queries": 2, "max_learnings": 3},
            "comprehensive": {"breadth_factor": 1.2, "depth_factor": 1.5, "max_sub_queries": 3, "max_learnings": 5}
        }
        self.current_mode_params = self.mode_params[self.mode]

        # Tool configuration for Google Search
        self.google_search_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name='google_search',
                    description='Performs a Google search and returns relevant results.',
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            'query': types.Schema(type=types.Type.STRING, description='The search query')
                        },
                        required=['query']
                    )
                )
            ]
        )


    async def _call_gemini_api(
        self,
        prompt: str,
        response_schema: Optional[type[BaseModel]] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        use_search: bool = False
    ) -> Tuple[Optional[str], Optional[BaseModel], Optional[Dict]]:
        """
        Helper function to call the Gemini API with error handling and optional JSON parsing.

        Args:
            prompt: The input prompt for the model.
            response_schema: Pydantic model for structured JSON output.
            temperature: Controls randomness (0.0-1.0).
            max_output_tokens: Maximum tokens in the response.
            use_search: Whether to enable the Google Search tool.

        Returns:
            A tuple containing:
            - Raw text response (str or None if error).
            - Parsed Pydantic model (BaseModel or None if not applicable or error).
            - Raw response dictionary (dict or None if error).
        """
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_output_tokens,
        }
        tools = [self.google_search_tool] if use_search else None

        if response_schema:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = response_schema # Pass the schema directly

        try:
            # Use the instance client
            response = await self.client.generate_content_async(
                contents=prompt,
                generation_config=types.GenerationConfig(**generation_config), # Pass config object
                tools=tools
            )

            raw_response_dict = GenerateContentResponse.to_dict(response) # Get raw dict if needed

            # Handle potential API denials or empty responses
            if not response.candidates:
                 print(f"Warning: API returned no candidates for prompt: {prompt[:100]}...")
                 return None, None, raw_response_dict

            # Extract text part safely
            text_response = ""
            if response.candidates[0].content.parts:
                 text_response = response.candidates[0].content.parts[0].text

            # Parse JSON if schema was provided
            parsed_model = None
            if response_schema:
                try:
                    # Gemini client might automatically parse if schema is provided and mime type is JSON
                    # Check if the response object has a direct way to access parsed data (might vary by SDK version)
                    # For now, attempt manual parsing from text, assuming it's valid JSON
                    if text_response:
                        json_data = json.loads(text_response)
                        parsed_model = response_schema(**json_data)
                    else:
                         print(f"Warning: JSON response expected but got empty text for prompt: {prompt[:100]}...")

                except (json.JSONDecodeError, ValidationError, AttributeError) as parse_error:
                    print(f"Error parsing JSON response: {parse_error}")
                    print(f"Raw text received: {text_response}")
                    # Return raw text even if parsing fails, might still be useful
                    return text_response, None, raw_response_dict
                except Exception as e: # Catch other potential parsing issues
                     print(f"Unexpected error during JSON parsing: {e}")
                     return text_response, None, raw_response_dict


            return text_response, parsed_model, raw_response_dict

        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            # Consider more specific error handling (e.g., rate limits, auth errors)
            return None, None, None


    def determine_research_breadth_and_depth(self, query: str) -> Dict[str, int | str]:
        """
        Determines initial research breadth and depth based on query complexity using Gemini.
        Uses synchronous call as it's a setup step.

        Args:
            query: The initial research query.

        Returns:
            A dictionary with 'breadth', 'depth', and 'explanation'.
        """
        user_prompt = f"""
        Analyze this research query and determine the appropriate initial breadth (number of parallel search queries)
        and depth (levels of follow-up questions) needed for thorough research, considering the current research mode '{self.mode}'.

        Query: {query}

        Mode '{self.mode}' implies:
        - fast: Focus on core aspects, less exploration.
        - balanced: Good coverage of main facets.
        - comprehensive: Aim for exhaustive detail and nuances.

        Consider:
        1. Complexity and ambiguity of the topic.
        2. Expected scope (broad overview vs. specific details).
        3. Potential for diverse sub-topics or perspectives.
        4. How the mode ('{self.mode}') should influence the scale.

        Return ONLY a JSON object matching this schema:
        {{
          "breadth": integer (1-10, adjusted for mode),
          "depth": integer (1-5, adjusted for mode),
          "explanation": "brief reasoning"
        }}
        """
        # Use a synchronous call for this setup step
        try:
            # Configure synchronous client for this specific call if needed, or use the main async client carefully
            # For simplicity, let's assume a sync call method exists or adapt the async one
            # NOTE: The provided SDK snippet primarily uses async. A dedicated sync call might look different.
            # Let's simulate the call structure using the async helper and running it synchronously.

             async def run_sync():
                 _, parsed_response, _ = await self._call_gemini_api(
                     prompt=user_prompt,
                     response_schema=ResearchParameters,
                     temperature=0.2, # Lower temp for deterministic analysis
                     max_output_tokens=512 # Smaller max tokens for this task
                 )
                 return parsed_response

             parsed_response = asyncio.run(run_sync()) # Run the async func synchronously

             if parsed_response and isinstance(parsed_response, ResearchParameters):
                 # Apply mode-based adjustments if needed (or let the model handle it via prompt)
                 # Example adjustment (optional, model might do this already):
                 # breadth = max(1, int(parsed_response.breadth * self.current_mode_params['breadth_factor']))
                 # depth = max(1, int(parsed_response.depth * self.current_mode_params['depth_factor']))

                 return {
                     "breadth": parsed_response.breadth, # Use model's suggestion directly
                     "depth": parsed_response.depth,
                     "explanation": parsed_response.explanation
                 }
             else:
                 print("Warning: Failed to get valid parameters from API.")
                 raise ValueError("Failed to determine research parameters.")


        except Exception as e:
            print(f"Error determining research parameters: {str(e)}. Using mode defaults.")
            # Default values based on mode
            defaults = {
                "fast": {"breadth": 3, "depth": 1},
                "balanced": {"breadth": 5, "depth": 2},
                "comprehensive": {"breadth": 7, "depth": 3}
            }
            params = defaults.get(self.mode, defaults["balanced"])
            params["explanation"] = f"Using default values for '{self.mode}' mode due to error."
            return params


    async def generate_queries(
        self,
        topic: str,
        num_queries: int,
        learnings: List[str] = [],
        parent_query: Optional[str] = None, # Context about the parent
        existing_queries: Optional[Set[str]] = None # Pass the global history
    ) -> Set[str]:
        """
        Generates diverse, specific search queries for a topic, considering context and history.

        Args:
            topic: The central topic or query to generate sub-queries for.
            num_queries: The desired number of queries.
            learnings: Key insights gathered so far related to the topic.
            parent_query: The query that led to this generation request (provides context).
            existing_queries: Set of all queries generated so far in the research process to avoid duplicates.

        Returns:
            A set of unique query strings.
        """
        if existing_queries is None:
            existing_queries = set()

        mode_guidance = {
            "fast": "Generate concise, highly relevant queries focusing on the core topic.",
            "balanced": "Generate a mix of queries covering key aspects and related sub-topics.",
            "comprehensive": "Generate diverse and in-depth queries exploring nuances, different angles, and potential gaps in knowledge."
        }

        learnings_text = "\n".join([f"- {l}" for l in learnings]) if learnings else "No specific learnings yet."
        # Include parent query context if available
        parent_context = f'These queries should follow up on the parent query: "{parent_query}"' if parent_query else ""

        # Filter existing queries relevant to the current topic/parent for the prompt context (optional optimization)
        # relevant_history = {q for q in existing_queries if topic.lower() in q.lower() or (parent_query and parent_query.lower() in q.lower())} # Simple relevance check
        # history_text = "\n".join(f"- {q}" for q in relevant_history) if relevant_history else "None relevant yet."
        # For simplicity, provide the full history size context
        history_info = f"We have already explored {len(existing_queries)} queries in total."


        user_prompt = f"""
        You are a research assistant exploring the topic: "{topic}"
        {parent_context}
        Current research mode: '{self.mode}'. {mode_guidance[self.mode]}

        Based on what we know:
        {learnings_text}

        {history_info} Avoid generating queries that are semantically identical or too similar to previous ones.

        Generate {num_queries} distinct, specific, and well-formed search engine queries to gather more information. Focus on uncovering new aspects or deeper details relevant to "{topic}".

        Return ONLY a JSON object matching this schema:
        {{
          "queries": ["query 1", "query 2", ...]
        }}
        """

        text_response, parsed_response, _ = await self._call_gemini_api(
            prompt=user_prompt,
            response_schema=QueryResponse,
            temperature=0.8, # Slightly higher temp for diversity
            max_output_tokens=1024
        )

        if parsed_response and isinstance(parsed_response, QueryResponse):
            generated_queries = set(parsed_response.queries)
            # Filter out queries that are identical to existing ones (case-insensitive)
            new_queries = {q for q in generated_queries if q.lower() not in {eq.lower() for eq in existing_queries}}

            # Optional: Add semantic similarity check if needed (can be slow)
            # final_queries = set()
            # for q in new_queries:
            #     is_similar = False
            #     for existing_q in existing_queries:
            #         if await self._are_queries_similar(q, existing_q):
            #             is_similar = True
            #             break
            #     if not is_similar:
            #         final_queries.add(q)
            # return final_queries

            # Limit to num_queries requested
            return set(list(new_queries)[:num_queries])

        elif text_response: # Fallback: Try to parse from raw text if JSON failed
             print("Warning: Failed to parse JSON for queries, attempting fallback parsing.")
             try:
                 # Basic line splitting and filtering
                 lines = text_response.strip().split('\n')
                 queries = set()
                 in_queries_list = False
                 for line in lines:
                     line = line.strip()
                     if '"queries": [' in line:
                         in_queries_list = True
                         continue
                     if in_queries_list:
                         if line.endswith(']'):
                             in_queries_list = False
                             line = line[:-1] # Remove trailing bracket
                         if line.startswith('"') and line.endswith('"'):
                              query = line[1:-1].replace('\\"', '"') # Handle escaped quotes
                              if query and query.lower() not in {eq.lower() for eq in existing_queries}:
                                   queries.add(query)
                         elif line.startswith('"') and line.endswith('",'):
                              query = line[1:-2].replace('\\"', '"')
                              if query and query.lower() not in {eq.lower() for eq in existing_queries}:
                                   queries.add(query)

                 if queries:
                      return set(list(queries)[:num_queries])
                 else:
                      print("Warning: Fallback query parsing failed.")

             except Exception as fallback_e:
                 print(f"Error during fallback query parsing: {fallback_e}")


        print(f"Warning: Query generation failed for '{topic}'. Returning empty set.")
        return set() # Return empty set on failure


    def _extract_sources_from_response(self, response_dict: Optional[Dict]) -> List[Dict[str, str]]:
        """Extracts source URLs and titles from Gemini API grounding metadata."""
        sources = []
        if not response_dict:
            return sources

        try:
            # Navigate through the dictionary structure safely
            candidates = response_dict.get('candidates', [])
            if not candidates: return sources

            # Check for grounding metadata (structure might vary slightly)
            grounding_metadata = candidates[0].get('grounding_metadata')
            if not grounding_metadata: return sources

            retrieval_queries = grounding_metadata.get('retrieval_queries', [])
            search_results = grounding_metadata.get('search_entry_point', {}).get('rendered_content') # Example path

            # Extract from web search results if available (preferred)
            web_search_queries = grounding_metadata.get('web_search_queries', [])
            citations = grounding_metadata.get('citations', []) # More direct source info

            if citations:
                 for citation in citations:
                      uri = citation.get('uri')
                      title = citation.get('title', 'Unknown Title') # Provide default title
                      if uri:
                           # Basic check to avoid duplicates by URL
                           if uri not in {s['url'] for s in sources}:
                                sources.append({"url": uri, "title": title})

            # Fallback: Try extracting from rendered content if citations are missing
            # This part is highly dependent on the exact API response structure and might need adjustment
            elif search_results:
                 # Example: Parse HTML if rendered_content is HTML
                 # This requires a robust HTML parser (e.g., BeautifulSoup) which is not imported here.
                 # For simplicity, we'll assume a simpler structure or skip this fallback.
                 print("Note: Citations metadata preferred but missing. Skipping extraction from rendered_content for now.")
                 pass # Add parsing logic here if needed


        except (AttributeError, KeyError, IndexError) as e:
            print(f"Error parsing sources from response: {e}. Response structure might have changed.")
        except Exception as e:
            print(f"Unexpected error extracting sources: {e}")

        return sources


    async def search(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Performs a search using the Gemini model with the Google Search tool enabled.

        Args:
            query: The search query string.

        Returns:
            A tuple containing:
            - The generated text answer from the model.
            - A list of source dictionaries [{'url': str, 'title': str}].
        """
        print(f"Performing search for: {query}")
        prompt = f"Please provide a comprehensive answer with sources for the query: {query}"

        # Use the search tool
        text_response, _, raw_response_dict = await self._call_gemini_api(
            prompt=prompt,
            use_search=True,
            temperature=0.5, # Lower temperature for factual search summarization
            max_output_tokens=4096 # Allow longer responses for search results
        )

        if text_response is None:
             text_response = f"Error performing search for: {query}" # Provide error message in text

        sources = self._extract_sources_from_response(raw_response_dict)

        # Basic formatting (optional): Add source list to the end of the text
        # formatted_text = text_response
        # if sources:
        #     formatted_text += "\n\n**Sources:**\n"
        #     for i, src in enumerate(sources):
        #         formatted_text += f"{i+1}. [{src.get('title', 'Source')}]({src.get('url', '#')})\n"

        # Return the raw text and the structured sources separately
        return text_response or "", sources


    async def process_result(
        self,
        query: str,
        result_text: str,
        num_learnings: int,
        num_follow_up: int,
        existing_queries: Set[str] # Pass history for context
    ) -> Dict[str, List[str]]:
        """
        Processes search result text to extract key learnings and generate relevant follow-up questions.

        Args:
            query: The original query for context.
            result_text: The text obtained from the search.
            num_learnings: Desired number of learnings.
            num_follow_up: Desired number of follow-up questions.
            existing_queries: Set of all queries generated so far.

        Returns:
            A dictionary with "learnings" and "follow_up_questions" lists.
        """
        user_prompt = f"""
        Analyze the following search result obtained for the query: "{query}"
        Research mode: '{self.mode}'.

        Search Result Text:
        ---
        {result_text[:6000]}
        ---
        (Result truncated if necessary)

        Based *only* on the text provided above:
        1. Extract the {num_learnings} most important and distinct learnings/insights directly supported by the text. Be concise.
        2. Generate {num_follow_up} specific and insightful follow-up questions that arise *from this text* and would logically extend the research on "{query}". Avoid generic questions. Ensure these questions are different from existing ones (total existing: {len(existing_queries)}).

        Return ONLY a JSON object matching this schema:
        {{
          "learnings": ["concise learning 1", ...],
          "follow_up_questions": ["specific follow-up question 1", ...]
        }}
        """

        _, parsed_response, _ = await self._call_gemini_api(
            prompt=user_prompt,
            response_schema=ProcessedResult,
            temperature=0.6, # Moderate temperature for extraction and generation
            max_output_tokens=1536 # Sufficient for learnings and questions
        )

        if parsed_response and isinstance(parsed_response, ProcessedResult):
            # Filter follow-up questions against history
            new_follow_ups = [
                 q for q in parsed_response.follow_up_questions
                 if q.lower() not in {eq.lower() for eq in existing_queries}
            ]
            return {
                "learnings": parsed_response.learnings[:num_learnings], # Ensure limit
                "follow_up_questions": new_follow_ups[:num_follow_up] # Ensure limit
            }
        else:
            print(f"Warning: Failed to process result for '{query}'. Generating basic follow-up.")
            # Fallback: Generate simple follow-up questions if processing fails
            fallback_follow_ups = await self.generate_queries(
                topic=query,
                num_queries=num_follow_up,
                learnings=[], # No learnings extracted
                parent_query=query,
                existing_queries=existing_queries
            )
            return {
                "learnings": [f"Could not automatically extract learnings for '{query}'."],
                "follow_up_questions": list(fallback_follow_ups)
            }


    async def _are_queries_similar(self, query1: str, query2: str) -> bool:
        """
        Checks if two queries are semantically similar using Gemini. (Use sparingly due to API calls)
        """
        # Simple checks first
        if query1.strip().lower() == query2.strip().lower():
            return True
        # Consider adding more efficient checks (e.g., Jaccard similarity on words) if needed

        # Use Gemini for semantic check
        user_prompt = f"""
        Are the following two search queries semantically similar? (i.e., would they likely retrieve very similar search results?)

        Query 1: "{query1}"
        Query 2: "{query2}"

        Return ONLY a JSON object matching this schema:
        {{
          "are_similar": boolean
        }}
        """
        _, parsed_response, _ = await self._call_gemini_api(
            prompt=user_prompt,
            response_schema=SimilarityResult,
            temperature=0.1, # Very low temp for deterministic comparison
            max_output_tokens=256
        )

        if parsed_response and isinstance(parsed_response, SimilarityResult):
            return parsed_response.are_similar
        else:
            print(f"Warning: Similarity check failed between '{query1}' and '{query2}'. Assuming not similar.")
            return False # Default to not similar on error to avoid discarding potentially unique queries


    async def _research_recursive(self, query: str, depth: int, breadth: int, parent_query: Optional[str] = None):
        """Recursive helper function to perform research at a given depth."""
        if depth <= 0:
            print(f"Reached max depth for query: {query}")
            return

        if not self.progress:
             print("Error: ResearchProgress tracker not initialized.")
             return

        # --- 1. Start and Search ---
        await self.progress.start_query(query, depth, parent_query)
        search_text, sources = await self.search(query)

        # Add sources found for this query
        if sources:
            await self.progress.add_sources(query, depth, sources)

        # --- 2. Process Results ---
        num_learnings_to_extract = self.current_mode_params['max_learnings']
        num_follow_ups_to_generate = breadth # Generate enough potential follow-ups for the breadth

        processed_result = await self.process_result(
            query=query,
            result_text=search_text,
            num_learnings=num_learnings_to_extract,
            num_follow_up=num_follow_ups_to_generate,
            existing_queries=self.query_history # Pass the global history
        )

        # Add extracted learnings
        for learning in processed_result.get("learnings", []):
            await self.progress.add_learning(query, depth, learning)

        # --- 3. Generate and Recurse on Follow-up Questions ---
        follow_up_questions = processed_result.get("follow_up_questions", [])

        # Filter follow-ups that haven't been explored yet
        new_unique_follow_ups = {
            q for q in follow_up_questions
            if q.lower() not in {qh.lower() for qh in self.query_history}
        }

        # Limit the number of follow-ups based on breadth for the *next* level
        next_level_queries = list(new_unique_follow_ups)[:breadth]

        # Update global history *before* recursing
        self.query_history.update(next_level_queries)

        if next_level_queries and depth > 1:
            print(f"Depth {depth} -> {depth-1}: Exploring {len(next_level_queries)} follow-up(s) for '{query}'...")

            # Calculate breadth for the next level (can decrease in deeper levels)
            # Example: Reduce breadth by 1 for each level deeper, minimum of 1
            next_breadth = max(1, breadth - 1) if self.mode != "comprehensive" else breadth # Keep breadth in comprehensive

            tasks = []
            for next_query in next_level_queries:
                 # Check again if query somehow got added to history concurrently (unlikely with current structure but safe)
                 if next_query.lower() not in {p_query.lower() for p_depth in self.progress.queries_by_depth for p_query in self.progress.queries_by_depth[p_depth]}:
                      tasks.append(
                          self._research_recursive(
                              query=next_query,
                              depth=depth - 1,
                              breadth=next_breadth,
                              parent_query=query # Pass current query as parent
                          )
                      )

            if tasks:
                 await asyncio.gather(*tasks)
            else:
                 print(f"No new, unique follow-up queries to explore for '{query}' at depth {depth-1}.")

        else:
             print(f"No further recursion needed for '{query}' at depth {depth}.")


        # --- 4. Mark Query as Complete ---
        # This query's processing is done (recursion for children handled above)
        await self.progress.complete_query(query, depth)


    async def run_research(self, initial_query: str) -> Dict:
        """
        Starts and manages the deep research process for the initial query.

        Args:
            initial_query: The starting query for the research.

        Returns:
            A dictionary containing the final research tree, all learnings, and all sources.
            Example: {"tree": {...}, "learnings": [...], "sources": [...]}
        """
        print(f"--- Starting Deep Research ---")
        print(f"Initial Query: {initial_query}")
        print(f"Mode: {self.mode}")

        # 1. Determine initial parameters
        params = self.determine_research_breadth_and_depth(initial_query)
        initial_breadth = params['breadth']
        initial_depth = params['depth']
        print(f"Determined Parameters: Breadth={initial_breadth}, Depth={initial_depth}")
        print(f"Reasoning: {params['explanation']}")

        # 2. Initialize Progress Tracker
        self.progress = ResearchProgress(depth=initial_depth, breadth=initial_breadth)
        self.query_history = {initial_query} # Start history with the root query

        # 3. Start Recursive Research
        print(f"\n--- Research Execution ---")
        start_time = time.time()
        try:
            await self._research_recursive(
                query=initial_query,
                depth=initial_depth,
                breadth=initial_breadth,
                parent_query=None # Root query has no parent
            )
        except Exception as e:
             print(f"\n--- Research Error ---")
             print(f"An error occurred during research: {e}")
             # Optionally log the full traceback
             import traceback
             traceback.print_exc()
        finally:
            end_time = time.time()
            print(f"\n--- Research Complete ---")
            print(f"Total execution time: {end_time - start_time:.2f} seconds")
            print(f"Total unique queries explored: {len(self.query_history)}")
            if self.progress:
                 print(f"Total learnings gathered: {len(self.progress.get_all_learnings())}")
                 print(f"Total sources found: {len(self.progress.get_all_sources())}")


        # 4. Consolidate and Return Results
        if self.progress:
            final_tree = self.progress._build_research_tree()
            all_learnings = self.progress.get_all_learnings()
            all_sources = self.progress.get_all_sources()

            # Save the final tree structure
            try:
                 tree_filename = f"research_tree_{initial_query[:20].replace(' ','_')}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
                 with open(tree_filename, "w", encoding='utf-8') as f:
                     json.dump(final_tree, f, indent=2, ensure_ascii=False)
                 print(f"Research tree saved to: {tree_filename}")
            except Exception as e:
                 print(f"Error saving research tree: {e}")


            return {
                "tree": final_tree,
                "learnings": all_learnings,
                "sources": all_sources
            }
        else:
            print("Error: Research progress tracker was not available.")
            return {"tree": {}, "learnings": [], "sources": []}


    async def generate_final_report(self, query: str, learnings: list[str], sources: list[dict[str, str]]) -> str:
        """
        Generates a final, creatively formatted report summarizing the research findings.

        Args:
            query: The initial research query.
            learnings: A list of all unique learnings gathered.
            sources: A list of all unique sources found.

        Returns:
            A formatted string containing the final report in Markdown.
        """
        print("\n--- Generating Final Report ---")
        if not learnings:
            return "No learnings were gathered during the research process to generate a report."

        # Format learnings and sources for the prompt
        learnings_text = "\n".join([f"- {learning}" for learning in learnings])
        sources_text = "\n".join([f"- [{src.get('title', 'Source')}]({src.get('url')})" for i, src in enumerate(sources)]) if sources else "No specific sources were cited."


        user_prompt = f"""
        You are a creative storyteller and research synthesizer. Your task is to transform the following research findings into an engaging and distinctive report about "{query}".

        Research Query: {query}

        Key Discoveries (Learnings):
        ---
        {learnings_text[:6000]}
        ---
        (Learnings truncated if necessary)

        Sources Consulted:
        ---
        {sources_text[:2000]}
        ---
        (Sources truncated if necessary)

        Craft a captivating report in Markdown format that:

        ## CREATIVE APPROACH & STRUCTURE
        1.  **Imaginative Opening:** Start with a hook that draws the reader into the world of "{query}". Avoid generic introductions.
        2.  **Narrative Flow:** Weave the key discoveries into a coherent narrative. Don't just list facts; connect them, show relationships, and build understanding. Use your unique voice.
        3.  **Insightful Synthesis:** Go beyond summarizing. Offer fresh perspectives, highlight surprising connections, or pose thought-provoking questions based on the learnings.
        4.  **Distinctive Style:** Experiment with tone and style (e.g., investigative, reflective, enthusiastic, critical) appropriate to the topic, but maintain clarity.
        5.  **Memorable Conclusion:** End with a strong concluding thought, a lingering question, or a call to further exploration that leaves an impact.

        ## MARKDOWN FORMATTING FOR IMPACT
        * **Evocative Headings:** Use `##` and `###` for sections with creative, descriptive titles.
        * **Emphasis:** Use **bold** and *italics* strategically to highlight key terms or ideas.
        * **Blockquotes:** Use `> ` for impactful quotes, contrasting points, or highlighting crucial learnings.
        * **Lists:** Use bullet (`* ` or `- `) or numbered lists for clarity when presenting multiple related points (e.g., steps, components, types).
        * **Horizontal Rules:** Use `---` to create clear separations between major sections or for dramatic effect.
        * **(Optional) Tables:** If appropriate, use Markdown tables to organize comparative data or structured information concisely.

        ## GUIDELINES
        * **Focus on Learnings:** Base the report primarily on the provided "Key Discoveries".
        * **Acknowledge Sources:** Briefly mention that the report is based on synthesized information from various sources (you don't need to cite inline unless specifically requested). Include the provided source list at the very end under a "Sources Consulted" heading.
        * **Be Creative, Be Clear:** Let your creativity shine, but ensure the report is easy to understand and logically structured. Avoid academic jargon unless the topic demands it.
        * **Length:** Aim for a comprehensive yet readable report. Don't be overly verbose or unnecessarily brief.

        Produce ONLY the Markdown report based on these instructions. Start directly with the report content (e.g., the first heading).
        """

        # Use the helper function for the API call
        report_text, _, _ = await self._call_gemini_api(
            prompt=user_prompt,
            temperature=0.8, # Higher temperature for creative writing
            max_output_tokens=8192 # Allow ample space for a detailed report
        )

        if report_text:
             # Append the source list manually if the model didn't include it reliably
             if "Sources Consulted" not in report_text[-500:]: # Check near the end
                  report_text += "\n\n## Sources Consulted\n" + sources_text
             print("Final report generated successfully.")
             return report_text
        else:
            print("Error: Failed to generate the final report.")
            # Fallback: Return a simple summary if generation fails
            return f"# Research Summary for: {query}\n\n" \
                   f"## Key Learnings:\n{learnings_text}\n\n" \
                   f"## Sources Consulted:\n{sources_text}\n\n" \
                   f"(Automated report generation failed)"


# --- Main Execution Block ---

async def main():
    """Main function to parse arguments and run the deep research."""
    parser = argparse.ArgumentParser(description="Perform deep research using Gemini API.")
    parser.add_argument("query", type=str, help="The initial research query.")
    parser.add_argument("-m", "--mode", type=str, default="balanced",
                        choices=["fast", "balanced", "comprehensive"],
                        help="Research mode (default: balanced).")
    parser.add_argument("-k", "--api-key", type=str, default=os.getenv("GOOGLE_API_KEY"),
                        help="Google API Key (reads from GOOGLE_API_KEY env var by default).")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Optional file path to save the final report (Markdown).")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: Google API Key is required. Set GOOGLE_API_KEY environment variable or use --api-key.")
        return

    # Load .env file if it exists (optional)
    load_dotenv()
    # Re-check API key from env var if not provided via argument after load_dotenv
    if not args.api_key:
         args.api_key = os.getenv("GOOGLE_API_KEY")
         if not args.api_key:
              print("Error: Google API Key is required. Set GOOGLE_API_KEY environment variable or use --api-key.")
              return


    # Initialize DeepSearch
    try:
        deep_search = DeepSearch(api_key=args.api_key, mode=args.mode)
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return
    except Exception as e:
         print(f"An unexpected error occurred during initialization: {e}")
         return


    # Run the research process
    research_results = await deep_search.run_research(args.query)

    # Generate the final report
    if research_results and research_results.get("learnings"):
        final_report = await deep_search.generate_final_report(
            query=args.query,
            learnings=research_results["learnings"],
            sources=research_results["sources"]
        )

        print("\n--- Final Report ---")
        print(final_report)

        # Save report to file if requested
        if args.output:
            try:
                with open(args.output, "w", encoding='utf-8') as f:
                    f.write(final_report)
                print(f"\nReport saved to: {args.output}")
            except IOError as e:
                print(f"\nError saving report to file: {e}")
    else:
        print("\nNo learnings gathered, skipping final report generation.")


if __name__ == "__main__":
    # Ensure an event loop is running for async operations
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nResearch interrupted by user.")
    except Exception as e:
         print(f"\nAn unexpected error occurred in the main execution: {e}")
         import traceback
         traceback.print_exc()
