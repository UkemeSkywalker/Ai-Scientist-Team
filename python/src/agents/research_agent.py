"""
Strands Research Agent for the AI Scientist Team
Handles literature search, hypothesis generation, and research synthesis
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

try:
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    Agent = None

from ..tools.research_tools import (
    arxiv_search_tool,
    pubmed_search_tool,
    hypothesis_generation_tool,
    literature_analysis_tool,
    research_synthesis_tool
)
from ..models.research import ResearchFindings
from ..core.shared_memory import SharedMemory
from ..integrations.strands_config import strands_config

logger = structlog.get_logger(__name__)

class ResearchAgent:
    """
    Strands-powered Research Agent that conducts literature search and hypothesis generation
    """
    
    def __init__(self, shared_memory: Optional[SharedMemory] = None):
        self.shared_memory = shared_memory or SharedMemory()
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the Strands Research Agent with research tools"""
        if not STRANDS_AVAILABLE:
            logger.warning("Strands SDK not available, Research Agent will use direct tool calls")
            return
        
        try:
            # Validate Strands configuration
            if not strands_config.validate_config():
                logger.error("Strands configuration validation failed")
                return
            
            # Import conversation manager for better token management
            from strands.agent.conversation_manager import SlidingWindowConversationManager
            
            # Create Research Agent with specialized system prompt and tools
            self.agent = Agent(
                model=strands_config.model_id,
                system_prompt=self._get_system_prompt(),
                tools=[
                    arxiv_search_tool,
                    pubmed_search_tool,
                    hypothesis_generation_tool,
                    literature_analysis_tool,
                    research_synthesis_tool
                ],
                name="Research Agent",
                description="Specialized agent for literature search, hypothesis generation, and research synthesis",
                agent_id="research_agent",
                conversation_manager=SlidingWindowConversationManager(window_size=20, should_truncate_results=True)
            )
            
            logger.info("Research Agent initialized successfully with Strands SDK")
            
        except Exception as e:
            logger.error(f"Failed to initialize Research Agent: {str(e)}")
            self.agent = None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Research Agent"""
        return """You are a Research Agent specialized in conducting comprehensive literature research.

Your tools work together in a specific sequence:
1. arxiv_search_tool and pubmed_search_tool - search for academic papers
2. literature_analysis_tool - analyze the collected papers (requires papers_data as JSON)
3. hypothesis_generation_tool - generate hypotheses (requires research_context as JSON)
4. research_synthesis_tool - combine all findings (requires all previous results as JSON)

IMPORTANT: Always pass the complete JSON output from one tool as input to the next tool that needs it.
Each tool returns structured JSON data that must be used by subsequent tools.

Focus on high-quality, recent research and generate specific, testable hypotheses."""

    async def execute_research(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Execute comprehensive research workflow for the given query
        
        Args:
            query: Research query to investigate
            session_id: Session identifier for context management
            
        Returns:
            Dictionary containing research findings and results
        """
        logger.info("Starting research execution", query=query, session_id=session_id)
        
        try:
            if self.agent and STRANDS_AVAILABLE:
                try:
                    # Use Strands agent for research
                    result = await self._execute_strands_research(query, session_id)
                except Exception as strands_error:
                    logger.warning(f"Strands agent failed, falling back to direct tools: {str(strands_error)}")
                    # Fallback to direct tool execution if Strands fails
                    result = await self._execute_direct_research(query, session_id)
            else:
                # Fallback to direct tool execution
                result = await self._execute_direct_research(query, session_id)
            
            # Store results in shared memory
            self.shared_memory.update_context(session_id, {
                "research_result": result,
                "research_timestamp": datetime.now().isoformat()
            })
            
            # Save references to files if research was successful
            if result.get("status") == "success" and result.get("references"):
                try:
                    saved_files = self.save_references_to_file(result, session_id)
                    result["saved_files"] = saved_files
                    logger.info("References automatically saved to files", session_id=session_id, file_count=len(saved_files))
                except Exception as e:
                    logger.warning("Failed to save references to files", error=str(e), session_id=session_id)
            
            logger.info("Research execution completed successfully", session_id=session_id)
            return result
            
        except Exception as e:
            error_msg = f"Research execution failed: {str(e)}"
            logger.error("Research execution failed", error=error_msg, session_id=session_id)
            
            error_result = {
                "query": query,
                "error": error_msg,
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store error in shared memory
            self.shared_memory.update_context(session_id, {
                "research_result": error_result,
                "research_error": error_msg
            })
            
            return error_result
    
    async def _execute_strands_research(self, query: str, session_id: str) -> Dict[str, Any]:
        """Execute research using Strands agent with proper data passing"""
        logger.info("Executing research with Strands agent", query=query)
        
        try:
            # Execute the complete workflow using direct tool calls through the agent
            # This ensures proper data passing between tools
            
            # Step 1: Search arXiv
            logger.info("Strands agent searching arXiv", query=query)
            arxiv_result = self.agent.tool.arxiv_search_tool(query=query, max_results=5)
            
            # Step 2: Search PubMed  
            logger.info("Strands agent searching PubMed", query=query)
            pubmed_result = self.agent.tool.pubmed_search_tool(query=query, max_results=5)
            
            # Step 3: Combine papers for analysis
            arxiv_data = json.loads(arxiv_result["content"][0]["text"])
            pubmed_data = json.loads(pubmed_result["content"][0]["text"])
            
            combined_papers = {"papers": []}
            if arxiv_data.get("papers"):
                combined_papers["papers"].extend(arxiv_data["papers"])
            if pubmed_data.get("papers"):
                combined_papers["papers"].extend(pubmed_data["papers"])
            
            # Step 4: Analyze literature
            logger.info("Strands agent analyzing literature", query=query)
            analysis_result = self.agent.tool.literature_analysis_tool(
                papers_data=json.dumps(combined_papers),
                query=query
            )
            
            # Step 5: Generate hypotheses
            logger.info("Strands agent generating hypotheses", query=query)
            research_context = json.dumps({
                "arxiv_results": arxiv_data,
                "pubmed_results": pubmed_data,
                "analysis": json.loads(analysis_result["content"][0]["text"])
            })
            hypotheses_result = self.agent.tool.hypothesis_generation_tool(
                research_context=research_context,
                query=query
            )
            
            # Step 6: Synthesize findings
            logger.info("Strands agent synthesizing findings", query=query)
            synthesis_result = self.agent.tool.research_synthesis_tool(
                arxiv_results=arxiv_result["content"][0]["text"],
                pubmed_results=pubmed_result["content"][0]["text"],
                hypotheses=hypotheses_result["content"][0]["text"],
                analysis=analysis_result["content"][0]["text"],
                query=query
            )
            
            # Parse final results
            final_results = json.loads(synthesis_result["content"][0]["text"])
            final_results["execution_method"] = "strands_agent_direct_tools"
            final_results["session_id"] = session_id
            
            return final_results
            
        except Exception as e:
            logger.error(f"Strands research execution failed: {str(e)}")
            # Fallback to direct execution
            return await self._execute_direct_research(query, session_id)
    
    def _create_analysis_agent(self):
        """Create a separate agent instance for analysis to avoid token buildup"""
        try:
            from strands.agent.conversation_manager import SlidingWindowConversationManager
            
            return Agent(
                model=strands_config.model_id,
                system_prompt="You are a research analysis agent. Analyze literature, generate hypotheses, and synthesize findings. Keep responses concise.",
                tools=[
                    literature_analysis_tool,
                    hypothesis_generation_tool,
                    research_synthesis_tool
                ],
                name="Research Analysis Agent",
                agent_id="research_analysis_agent",
                conversation_manager=SlidingWindowConversationManager(window_size=10, should_truncate_results=True)
            )
        except Exception as e:
            logger.error(f"Failed to create analysis agent: {str(e)}")
            return self.agent
    
    async def _execute_direct_research(self, query: str, session_id: str) -> Dict[str, Any]:
        """Execute research using direct tool calls (fallback)"""
        logger.info("Executing research with direct tool calls", query=query)
        
        try:
            # Step 1: Search arXiv
            logger.info("Searching arXiv", query=query)
            arxiv_results = arxiv_search_tool(query, max_results=15)
            
            # Step 2: Search PubMed
            logger.info("Searching PubMed", query=query)
            pubmed_results = pubmed_search_tool(query, max_results=15)
            
            # Step 3: Analyze literature
            logger.info("Analyzing literature", query=query)
            # Combine results for analysis
            combined_papers = {
                "papers": [],
                "sources": ["arXiv", "PubMed"]
            }
            
            arxiv_data = json.loads(arxiv_results)
            pubmed_data = json.loads(pubmed_results)
            
            if arxiv_data.get("papers"):
                combined_papers["papers"].extend(arxiv_data["papers"])
            if pubmed_data.get("papers"):
                combined_papers["papers"].extend(pubmed_data["papers"])
            
            analysis_results = literature_analysis_tool(json.dumps(combined_papers), query)
            
            # Step 4: Generate hypotheses
            logger.info("Generating hypotheses", query=query)
            research_context = json.dumps({
                "arxiv_results": arxiv_data,
                "pubmed_results": pubmed_data,
                "analysis": json.loads(analysis_results)
            })
            hypotheses_results = hypothesis_generation_tool(research_context, query)
            
            # Step 5: Synthesize findings
            logger.info("Synthesizing research findings", query=query)
            synthesis_results = research_synthesis_tool(
                arxiv_results, pubmed_results, hypotheses_results, analysis_results, query
            )
            
            # Parse final results
            final_results = json.loads(synthesis_results)
            final_results["execution_method"] = "direct_tools"
            final_results["session_id"] = session_id
            
            return final_results
            
        except Exception as e:
            logger.error(f"Direct research execution failed: {str(e)}")
            raise
    
    def _parse_strands_research_result(self, result_text: str, query: str, session_id: str) -> Dict[str, Any]:
        """Parse Strands agent result into structured research findings"""
        # This is a simplified parser - in a real implementation,
        # the Strands agent would return structured data through tool calls
        
        try:
            # Try to extract JSON from the result if present
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed_result = json.loads(json_match.group())
                parsed_result["execution_method"] = "strands_agent"
                parsed_result["session_id"] = session_id
                return parsed_result
        except:
            pass
        
        # Fallback: create structured result from text
        return {
            "query": query,
            "session_id": session_id,
            "execution_method": "strands_agent",
            "research_summary": result_text,
            "hypotheses": [
                {
                    "text": f"Research hypothesis derived from Strands analysis of {query}",
                    "confidence": 0.75,
                    "testable": True,
                    "variables": [query, "outcome_metrics"]
                }
            ],
            "literature_summary": {
                "total_papers_found": "Multiple papers analyzed by Strands agent",
                "sources": {"strands_analysis": 1}
            },
            "confidence_score": 0.75,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_research_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current research status for a session"""
        context = self.shared_memory.get_context(session_id)
        if context and "research_result" in context:
            return context["research_result"]
        return None
    
    def _sanitize_query_for_folder(self, query: str, max_length: int = 30) -> str:
        """
        Sanitize query for use in folder names
        
        Args:
            query: Original research query
            max_length: Maximum length for the sanitized query
            
        Returns:
            Sanitized query suitable for folder names
        """
        import re
        
        # Convert to lowercase and replace spaces with hyphens
        sanitized = query.lower().replace(" ", "-")
        
        # Remove special characters, keep only alphanumeric, hyphens, and underscores
        sanitized = re.sub(r'[^a-z0-9\-_]', '', sanitized)
        
        # Remove multiple consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Remove leading/trailing hyphens
        sanitized = sanitized.strip('-')
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip('-')
        
        return sanitized or "research"  # Fallback if sanitization results in empty string

    def save_references_to_file(self, research_results: Dict[str, Any], session_id: str, output_dir: str = "data/research_outputs") -> Dict[str, str]:
        """
        Save references and bibliography to files in organized folder structure
        
        Args:
            research_results: Research results containing references
            session_id: Session identifier for file naming
            output_dir: Base directory to save files
            
        Returns:
            Dictionary with file paths for each saved format
        """
        import os
        from pathlib import Path
        
        # Create query prefix for folder name
        query = research_results.get("query", "research")
        query_prefix = self._sanitize_query_for_folder(query)
        
        # Get timestamp from results or use current time
        timestamp = research_results.get("synthesis_timestamp", datetime.now().isoformat())
        # Extract just the date and time part for folder name
        timestamp_short = timestamp.split('T')[0].replace('-', '') + '_' + timestamp.split('T')[1][:8].replace(':', '')
        
        # Create folder name: query-prefix_session_id_timestamp
        folder_name = f"{query_prefix}_{session_id}_{timestamp_short}"
        session_output_dir = Path(output_dir) / folder_name
        
        # Create the session-specific directory
        session_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save references as JSON
        references_file = session_output_dir / "references.json"
        with open(references_file, 'w', encoding='utf-8') as f:
            json.dump({
                "query": research_results.get("query"),
                "timestamp": research_results.get("synthesis_timestamp"),
                "references": research_results.get("references", []),
                "key_references": research_results.get("key_references", []),
                "total_papers": len(research_results.get("references", [])),
                "confidence_score": research_results.get("confidence_score"),
                "session_id": session_id,
                "folder_name": folder_name
            }, f, indent=2, ensure_ascii=False)
        saved_files["references_json"] = str(references_file)
        
        # Save APA bibliography
        apa_file = session_output_dir / "bibliography_apa.txt"
        with open(apa_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_bibliography(research_results, "apa"))
        saved_files["bibliography_apa"] = str(apa_file)
        
        # Save MLA bibliography
        mla_file = session_output_dir / "bibliography_mla.txt"
        with open(mla_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_bibliography(research_results, "mla"))
        saved_files["bibliography_mla"] = str(mla_file)
        
        # Save paper links as CSV
        csv_file = session_output_dir / "paper_links.csv"
        paper_links = self.get_paper_links(research_results)
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Title,Authors,Year,Source,URL,Relevance_Score,Abstract\n")
            for paper in paper_links:
                # Escape commas and quotes in CSV
                title = paper['title'].replace('"', '""')
                authors = paper['authors'].replace('"', '""')
                abstract = paper['abstract'].replace('"', '""').replace('\n', ' ')
                f.write(f'"{title}","{authors}","{paper["year"]}","{paper["source"]}","{paper["url"]}",{paper["relevance_score"]},"{abstract}"\n')
        saved_files["paper_links_csv"] = str(csv_file)
        
        # Save complete research results
        results_file = session_output_dir / "complete_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(research_results, f, indent=2, ensure_ascii=False)
        saved_files["complete_results"] = str(results_file)
        
        # Create a README file with session information
        readme_file = session_output_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f"""# Research Session: {query}

## Session Information
- **Query**: {query}
- **Session ID**: {session_id}
- **Timestamp**: {timestamp}
- **Folder**: {folder_name}

## Files in this directory:
- `references.json` - Complete structured reference data
- `bibliography_apa.txt` - APA format bibliography
- `bibliography_mla.txt` - MLA format bibliography  
- `paper_links.csv` - CSV file with all paper links and metadata
- `complete_results.json` - Full research results including hypotheses and analysis
- `README.md` - This information file

## Research Summary
- **Total Papers Found**: {len(research_results.get('references', []))}
- **Key References**: {len(research_results.get('key_references', []))}
- **Confidence Score**: {research_results.get('confidence_score', 'N/A')}
- **Research Quality**: {research_results.get('research_quality_indicators', {}).get('literature_coverage', 'N/A')}

Generated by AI Scientist Team Research Agent
""")
        saved_files["readme"] = str(readme_file)
        
        logger.info("References saved to organized folder structure", 
                   folder=folder_name, 
                   files_saved=len(saved_files),
                   output_dir=str(session_output_dir))
        return saved_files

    def generate_bibliography(self, research_results: Dict[str, Any], format_style: str = "apa") -> str:
        """
        Generate a formatted bibliography from research results
        
        Args:
            research_results: Research results containing references
            format_style: Citation format style ("apa", "mla", "chicago", or "plain")
            
        Returns:
            Formatted bibliography as a string
        """
        references = research_results.get("references", [])
        if not references:
            return "No references available."
        
        bibliography_lines = []
        
        if format_style.lower() == "apa":
            bibliography_lines.append("# References (APA Style)\n")
            for ref in references:
                authors = ref.get("authors", ["Unknown Author"])
                author_str = ", ".join(authors[:3])  # Limit to first 3 authors
                if len(authors) > 3:
                    author_str += ", et al."
                
                year = ref.get("year", "n.d.")
                title = ref.get("title", "Unknown title")
                source = ref.get("source", "Unknown source")
                url = ref.get("url", "")
                
                citation = f"{author_str} ({year}). *{title}*. {source}."
                if url:
                    citation += f" Retrieved from {url}"
                
                bibliography_lines.append(f"{ref['id']}. {citation}")
                
        elif format_style.lower() == "mla":
            bibliography_lines.append("# Works Cited (MLA Style)\n")
            for ref in references:
                authors = ref.get("authors", ["Unknown Author"])
                if authors:
                    # MLA format: Last, First. 
                    author_parts = authors[0].split()
                    if len(author_parts) >= 2:
                        author_str = f"{author_parts[-1]}, {' '.join(author_parts[:-1])}"
                    else:
                        author_str = authors[0]
                else:
                    author_str = "Unknown Author"
                
                title = ref.get("title", "Unknown title")
                source = ref.get("source", "Unknown source")
                year = ref.get("year", "n.d.")
                url = ref.get("url", "")
                
                citation = f'{author_str}. "{title}." *{source}*, {year}.'
                if url:
                    citation += f" Web. {url}"
                
                bibliography_lines.append(f"{ref['id']}. {citation}")
                
        else:  # Plain format
            bibliography_lines.append("# References\n")
            for ref in references:
                citation = ref.get("citation", "Unknown reference")
                url = ref.get("url", "")
                relevance = ref.get("relevance_score", 0)
                
                line = f"{ref['id']}. {citation}"
                if url:
                    line += f"\n   URL: {url}"
                line += f"\n   Relevance Score: {relevance:.2f}"
                
                bibliography_lines.append(line)
        
        return "\n\n".join(bibliography_lines)
    
    def get_paper_links(self, research_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract paper links and metadata from research results
        
        Args:
            research_results: Research results containing references
            
        Returns:
            List of dictionaries with paper metadata and links
        """
        references = research_results.get("references", [])
        paper_links = []
        
        for ref in references:
            paper_info = {
                "title": ref.get("title", "Unknown Title"),
                "authors": ", ".join(ref.get("authors", ["Unknown Author"])),
                "year": ref.get("year", "Unknown Year"),
                "source": ref.get("source", "Unknown Source"),
                "url": ref.get("url", "No URL available"),
                "relevance_score": ref.get("relevance_score", 0),
                "abstract": ref.get("abstract", "No abstract available")[:200] + "..." if len(ref.get("abstract", "")) > 200 else ref.get("abstract", "No abstract available")
            }
            paper_links.append(paper_info)
        
        return paper_links
    
    def validate_research_quality(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of research results"""
        quality_metrics = {
            "has_hypotheses": bool(research_results.get("hypotheses")),
            "has_literature": bool(research_results.get("literature_summary")),
            "confidence_score": research_results.get("confidence_score", 0),
            "paper_count": 0,
            "source_diversity": 0
        }
        
        # Count papers and sources
        lit_summary = research_results.get("literature_summary", {})
        if isinstance(lit_summary.get("sources"), dict):
            quality_metrics["source_diversity"] = len(lit_summary["sources"])
            quality_metrics["paper_count"] = sum(lit_summary["sources"].values()) if all(isinstance(v, int) for v in lit_summary["sources"].values()) else 0
        
        # Calculate overall quality score
        quality_score = 0
        if quality_metrics["has_hypotheses"]:
            quality_score += 0.3
        if quality_metrics["has_literature"]:
            quality_score += 0.3
        if quality_metrics["confidence_score"] >= 0.7:
            quality_score += 0.2
        if quality_metrics["paper_count"] >= 5:
            quality_score += 0.1
        if quality_metrics["source_diversity"] >= 2:
            quality_score += 0.1
        
        quality_metrics["overall_quality"] = round(quality_score, 2)
        quality_metrics["quality_level"] = (
            "high" if quality_score >= 0.8 else
            "medium" if quality_score >= 0.6 else
            "low"
        )
        
        return quality_metrics

# Create a convenience function for easy agent creation
def create_research_agent(shared_memory: Optional[SharedMemory] = None) -> ResearchAgent:
    """Create and return a Research Agent instance"""
    return ResearchAgent(shared_memory)