"""
Research tools for the Strands Research Agent
Implements arXiv, PubMed search, hypothesis generation, and literature analysis
"""

import json
import requests
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

try:
    import strands
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    strands = None

from ..models.research import Hypothesis, LiteratureSource, ResearchFindings

logger = structlog.get_logger(__name__)

# Tool decorator - use strands.tool if available, otherwise create a simple wrapper
def tool_decorator(func):
    """Decorator for Strands tools with fallback"""
    if STRANDS_AVAILABLE and hasattr(strands, 'tool'):
        return strands.tool(func)
    else:
        # Simple fallback decorator that preserves function metadata
        func._is_strands_tool = True
        return func

@tool_decorator
def arxiv_search_tool(query: str, max_results: int = 10) -> str:
    """
    Search arXiv for academic papers related to the research query.
    
    Args:
        query: Research query to search for
        max_results: Maximum number of papers to return (default: 10)
        
    Returns:
        JSON string containing search results with paper details
    """
    logger.info("ArXiv search initiated", query=query, max_results=max_results)
    
    try:
        # Construct arXiv API query
        encoded_query = quote_plus(query)
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        
        # Make request to arXiv API
        response = requests.get(arxiv_url, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Extract paper information
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            try:
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                
                # Extract authors
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name = author.find('{http://www.w3.org/2005/Atom}name').text
                    authors.append(name)
                
                # Extract publication date
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                
                # Extract arXiv URL
                arxiv_url = entry.find('{http://www.w3.org/2005/Atom}id').text
                
                # Calculate relevance score (simplified)
                relevance_score = min(1.0, len([word for word in query.lower().split() 
                                              if word in title.lower() or word in summary.lower()]) / len(query.split()))
                
                paper = {
                    "title": title[:200] + "..." if len(title) > 200 else title,
                    "authors": authors[:3],  # Limit to first 3 authors
                    "publication_date": pub_date.isoformat(),
                    "source": "arXiv",
                    "url": arxiv_url,
                    "relevance_score": round(relevance_score, 2),
                    "abstract": summary[:200] + "..." if len(summary) > 200 else summary
                }
                papers.append(paper)
                
            except Exception as e:
                logger.warning("Failed to parse arXiv entry", error=str(e))
                continue
        
        result = {
            "query": query,
            "source": "arXiv",
            "papers_found": len(papers),
            "papers": papers,
            "search_timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        logger.info("ArXiv search completed", papers_found=len(papers))
        return json.dumps(result, indent=2)
        
    except requests.RequestException as e:
        error_msg = f"ArXiv API request failed: {str(e)}"
        logger.error("ArXiv search failed", error=error_msg)
        return json.dumps({
            "query": query,
            "source": "arXiv",
            "error": error_msg,
            "status": "error"
        })
    except Exception as e:
        error_msg = f"ArXiv search error: {str(e)}"
        logger.error("ArXiv search failed", error=error_msg)
        return json.dumps({
            "query": query,
            "source": "arXiv", 
            "error": error_msg,
            "status": "error"
        })

@tool_decorator
def pubmed_search_tool(query: str, max_results: int = 10) -> str:
    """
    Search PubMed for medical and life science research papers.
    
    Args:
        query: Research query to search for
        max_results: Maximum number of papers to return (default: 10)
        
    Returns:
        JSON string containing search results with paper details
    """
    logger.info("PubMed search initiated", query=query, max_results=max_results)
    
    try:
        # Use PubMed E-utilities API
        # First, search for paper IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=30)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        if not search_data.get("esearchresult", {}).get("idlist"):
            return json.dumps({
                "query": query,
                "source": "PubMed",
                "papers_found": 0,
                "papers": [],
                "message": "No papers found",
                "status": "success"
            })
        
        # Get paper details
        paper_ids = search_data["esearchresult"]["idlist"]
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(paper_ids),
            "retmode": "xml"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(fetch_response.content)
        
        papers = []
        for article in root.findall('.//PubmedArticle'):
            try:
                # Extract title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else "No title available"
                
                # Extract authors
                authors = []
                for author in article.findall('.//Author'):
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    if last_name is not None and first_name is not None:
                        authors.append(f"{first_name.text} {last_name.text}")
                
                # Extract abstract
                abstract_elem = article.find('.//Abstract/AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                
                # Extract publication date
                pub_date_elem = article.find('.//PubDate')
                pub_date = None
                if pub_date_elem is not None:
                    year = pub_date_elem.find('Year')
                    month = pub_date_elem.find('Month')
                    day = pub_date_elem.find('Day')
                    if year is not None:
                        try:
                            pub_date = datetime(
                                int(year.text),
                                int(month.text) if month is not None and month.text.isdigit() else 1,
                                int(day.text) if day is not None and day.text.isdigit() else 1
                            ).isoformat()
                        except (ValueError, TypeError):
                            pub_date = None
                
                # Extract PMID for URL
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else None
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
                
                # Calculate relevance score
                relevance_score = min(1.0, len([word for word in query.lower().split() 
                                              if word in title.lower() or word in abstract.lower()]) / len(query.split()))
                
                paper = {
                    "title": title[:200] + "..." if len(title) > 200 else title,
                    "authors": authors[:3],  # Limit to first 3 authors
                    "publication_date": pub_date,
                    "source": "PubMed",
                    "url": url,
                    "relevance_score": round(relevance_score, 2),
                    "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract
                }
                papers.append(paper)
                
            except Exception as e:
                logger.warning("Failed to parse PubMed entry", error=str(e))
                continue
        
        result = {
            "query": query,
            "source": "PubMed",
            "papers_found": len(papers),
            "papers": papers,
            "search_timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        logger.info("PubMed search completed", papers_found=len(papers))
        return json.dumps(result, indent=2)
        
    except requests.RequestException as e:
        error_msg = f"PubMed API request failed: {str(e)}"
        logger.error("PubMed search failed", error=error_msg)
        return json.dumps({
            "query": query,
            "source": "PubMed",
            "error": error_msg,
            "status": "error"
        })
    except Exception as e:
        error_msg = f"PubMed search error: {str(e)}"
        logger.error("PubMed search failed", error=error_msg)
        return json.dumps({
            "query": query,
            "source": "PubMed",
            "error": error_msg,
            "status": "error"
        })

@tool_decorator
def hypothesis_generation_tool(research_context: str, query: str) -> str:
    """
    Generate testable hypotheses based on research context and query.
    
    Args:
        research_context: Context from previous research (literature findings)
        query: Original research query
        
    Returns:
        JSON string containing generated hypotheses
    """
    logger.info("Hypothesis generation initiated", query=query)
    
    try:
        # Parse research context if it's JSON
        context_data = {}
        if research_context:
            try:
                context_data = json.loads(research_context) if isinstance(research_context, str) else research_context
            except json.JSONDecodeError:
                context_data = {"raw_context": research_context}
        
        # Generate hypotheses based on query and context
        hypotheses = []
        
        # Base hypothesis patterns (reduced to 3 for token efficiency)
        hypothesis_templates = [
            f"There is a significant positive correlation between {query} and performance outcomes",
            f"The effectiveness of {query} varies significantly across different domains or contexts",
            f"Implementation of {query} leads to measurable improvements in key metrics"
        ]
        
        # Generate hypotheses with confidence scores
        for i, template in enumerate(hypothesis_templates):
            # Calculate confidence based on available research context
            confidence = 0.7 + (0.1 * (len(context_data.get("papers", [])) / 10))  # Higher confidence with more papers
            confidence = min(0.95, confidence)  # Cap at 95%
            
            # Extract variables from the hypothesis
            variables = []
            if "correlation" in template:
                variables = [query, "performance_metrics"]
            elif "varies" in template:
                variables = [query, "domain_context"]
            elif "implementation" in template:
                variables = [query, "outcome_metrics"]
            elif "moderated" in template:
                variables = [query, "environmental_factors"]
            elif "patterns" in template:
                variables = [query, "population_groups"]
            
            hypothesis = {
                "text": template,
                "confidence": round(confidence, 2),
                "testable": True,
                "variables": variables,
                "expected_outcome": f"Measurable change in dependent variables when {query} is applied",
                "methodology_suggestion": f"Experimental or observational study design with {query} as independent variable"
            }
            hypotheses.append(hypothesis)
        
        # Add context-specific hypotheses if research papers are available
        if context_data.get("papers"):
            papers = context_data["papers"]
            if len(papers) > 0:
                # Generate hypothesis based on most relevant paper
                top_paper = max(papers, key=lambda p: p.get("relevance_score", 0))
                context_hypothesis = {
                    "text": f"Based on recent research findings, {query} demonstrates specific patterns that can be replicated and validated",
                    "confidence": 0.85,
                    "testable": True,
                    "variables": [query, "replication_outcomes"],
                    "expected_outcome": "Results consistent with existing literature",
                    "methodology_suggestion": f"Replication study based on methodology from: {top_paper.get('title', 'recent research')}",
                    "supporting_evidence": top_paper.get("title", "")
                }
                hypotheses.append(context_hypothesis)
        
        result = {
            "query": query,
            "research_context_used": bool(context_data),
            "hypotheses_generated": len(hypotheses),
            "hypotheses": hypotheses,
            "generation_timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        logger.info("Hypothesis generation completed", hypotheses_count=len(hypotheses))
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Hypothesis generation error: {str(e)}"
        logger.error("Hypothesis generation failed", error=error_msg)
        return json.dumps({
            "query": query,
            "error": error_msg,
            "status": "error"
        })

@tool_decorator
def literature_analysis_tool(papers_data: str, query: str) -> str:
    """
    Analyze literature search results for relevance, patterns, and research gaps.
    
    Args:
        papers_data: JSON string containing papers from search results
        query: Original research query
        
    Returns:
        JSON string containing literature analysis results
    """
    logger.info("Literature analysis initiated", query=query)
    
    try:
        # Parse papers data - require valid input
        if not papers_data or papers_data.strip() == "":
            return json.dumps({
                "query": query,
                "error": "No papers data provided for analysis",
                "status": "error"
            })
        
        try:
            papers_info = json.loads(papers_data) if isinstance(papers_data, str) else papers_data
        except json.JSONDecodeError as e:
            return json.dumps({
                "query": query,
                "error": f"Invalid JSON format in papers_data: {str(e)}",
                "status": "error"
            })
        
        papers = papers_info.get("papers", [])
        
        if not papers:
            return json.dumps({
                "query": query,
                "analysis": "No papers available for analysis",
                "status": "error"
            })
        
        # Analyze papers
        analysis = {
            "total_papers": len(papers),
            "sources": {},
            "relevance_distribution": {"high": 0, "medium": 0, "low": 0},
            "temporal_distribution": {},
            "key_themes": [],
            "research_gaps": [],
            "methodological_approaches": [],
            "confidence_assessment": 0.0
        }
        
        # Analyze sources
        for paper in papers:
            source = paper.get("source", "unknown")
            analysis["sources"][source] = analysis["sources"].get(source, 0) + 1
            
            # Analyze relevance
            relevance = paper.get("relevance_score", 0)
            if relevance >= 0.7:
                analysis["relevance_distribution"]["high"] += 1
            elif relevance >= 0.4:
                analysis["relevance_distribution"]["medium"] += 1
            else:
                analysis["relevance_distribution"]["low"] += 1
            
            # Analyze temporal distribution
            pub_date = paper.get("publication_date")
            if pub_date:
                try:
                    year = datetime.fromisoformat(pub_date.replace('Z', '+00:00')).year
                    analysis["temporal_distribution"][str(year)] = analysis["temporal_distribution"].get(str(year), 0) + 1
                except:
                    pass
        
        # Identify key themes (simplified keyword extraction)
        all_text = " ".join([
            paper.get("title", "") + " " + paper.get("abstract", "")
            for paper in papers
        ]).lower()
        
        # Common research keywords
        research_keywords = [
            "machine learning", "artificial intelligence", "deep learning", "neural network",
            "algorithm", "model", "prediction", "classification", "regression", "optimization",
            "data mining", "statistical", "experimental", "empirical", "methodology",
            "performance", "accuracy", "evaluation", "validation", "bias", "fairness"
        ]
        
        for keyword in research_keywords:
            if keyword in all_text and all_text.count(keyword) >= 2:
                analysis["key_themes"].append({
                    "theme": keyword,
                    "frequency": all_text.count(keyword),
                    "relevance": "high" if all_text.count(keyword) >= 5 else "medium"
                })
        
        # Identify research gaps (simplified)
        analysis["research_gaps"] = [
            f"Limited longitudinal studies on {query}",
            f"Need for more diverse datasets in {query} research",
            f"Lack of standardized evaluation metrics for {query}"
        ]
        
        # Identify methodological approaches
        methodology_keywords = {
            "experimental": ["experiment", "trial", "controlled", "randomized"],
            "observational": ["observational", "survey", "cohort", "longitudinal"],
            "computational": ["simulation", "modeling", "algorithm", "computational"],
            "meta-analysis": ["meta-analysis", "systematic review", "review"]
        }
        
        for method, keywords in methodology_keywords.items():
            count = sum(all_text.count(keyword) for keyword in keywords)
            if count > 0:
                analysis["methodological_approaches"].append({
                    "approach": method,
                    "frequency": count,
                    "papers_using": min(count, len(papers))
                })
        
        # Calculate overall confidence
        high_relevance_ratio = analysis["relevance_distribution"]["high"] / len(papers)
        source_diversity = len(analysis["sources"])
        temporal_spread = len(analysis["temporal_distribution"])
        
        confidence = (high_relevance_ratio * 0.4 + 
                     min(source_diversity / 2, 1) * 0.3 + 
                     min(temporal_spread / 5, 1) * 0.3)
        analysis["confidence_assessment"] = round(confidence, 2)
        
        result = {
            "query": query,
            "literature_analysis": analysis,
            "analysis_timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        logger.info("Literature analysis completed", 
                   papers_analyzed=len(papers), 
                   confidence=analysis["confidence_assessment"])
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Literature analysis error: {str(e)}"
        logger.error("Literature analysis failed", error=error_msg)
        return json.dumps({
            "query": query,
            "error": error_msg,
            "status": "error"
        })

@tool_decorator
def research_synthesis_tool(arxiv_results: str, pubmed_results: str, hypotheses: str, analysis: str, query: str) -> str:
    """
    Synthesize all research findings into structured research findings.
    
    Args:
        arxiv_results: Results from arXiv search
        pubmed_results: Results from PubMed search  
        hypotheses: Generated hypotheses
        analysis: Literature analysis results
        query: Original research query
        
    Returns:
        JSON string containing synthesized research findings
    """
    logger.info("Research synthesis initiated", query=query)
    
    try:
        # Parse all input data - require valid inputs
        def parse_required_json(data, data_name):
            if not data or data.strip() == "":
                raise ValueError(f"Missing required data: {data_name}")
            try:
                return json.loads(data) if isinstance(data, str) else data
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {data_name}: {str(e)}")
        
        try:
            arxiv_data = parse_required_json(arxiv_results, "arxiv_results")
            pubmed_data = parse_required_json(pubmed_results, "pubmed_results")
            hypotheses_data = parse_required_json(hypotheses, "hypotheses")
            analysis_data = parse_required_json(analysis, "analysis")
        except ValueError as e:
            return json.dumps({
                "query": query,
                "error": str(e),
                "status": "error"
            })
        
        # Combine all literature sources
        all_papers = []
        if arxiv_data.get("papers"):
            all_papers.extend(arxiv_data["papers"])
        if pubmed_data.get("papers"):
            all_papers.extend(pubmed_data["papers"])
        
        # Sort papers by relevance score and limit to top papers
        all_papers.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)
        all_papers = all_papers[:10]  # Limit to top 10 papers
        
        # Extract key concepts from analysis (limit to top 5)
        key_concepts = []
        if analysis_data.get("literature_analysis", {}).get("key_themes"):
            key_concepts = [theme["theme"] for theme in analysis_data["literature_analysis"]["key_themes"][:5]]
        
        # Extract research gaps (limit to top 3)
        research_gaps = analysis_data.get("literature_analysis", {}).get("research_gaps", [])[:3]
        
        # Extract methodology suggestions from hypotheses (limit to top 3)
        methodology_suggestions = []
        if hypotheses_data.get("hypotheses"):
            methodology_suggestions = [h.get("methodology_suggestion", "") for h in hypotheses_data["hypotheses"][:3] if h.get("methodology_suggestion")]
        
        # Calculate overall confidence score
        lit_confidence = analysis_data.get("literature_analysis", {}).get("confidence_assessment", 0.5)
        hyp_confidence = sum(h.get("confidence", 0.5) for h in hypotheses_data.get("hypotheses", [])) / max(len(hypotheses_data.get("hypotheses", [])), 1)
        overall_confidence = (lit_confidence + hyp_confidence) / 2
        
        # Create formatted references for all papers
        references = []
        for i, paper in enumerate(all_papers, 1):
            # Format authors
            authors_str = ", ".join(paper.get("authors", ["Unknown Author"]))
            if len(authors_str) > 100:  # Truncate very long author lists
                authors_str = authors_str[:100] + "..."
            
            # Format publication date
            pub_date = paper.get("publication_date")
            year = "Unknown Year"
            if pub_date:
                try:
                    year = str(datetime.fromisoformat(pub_date.replace('Z', '+00:00')).year)
                except:
                    year = "Unknown Year"
            
            # Create reference entry
            reference = {
                "id": i,
                "citation": f"{authors_str} ({year}). {paper.get('title', 'Unknown Title')}. {paper.get('source', 'Unknown Source')}.",
                "title": paper.get("title", "Unknown Title"),
                "authors": paper.get("authors", []),
                "year": year,
                "source": paper.get("source", "Unknown Source"),
                "url": paper.get("url"),
                "relevance_score": paper.get("relevance_score", 0),
                "abstract": paper.get("abstract", "No abstract available")
            }
            references.append(reference)
        
        # Create synthesized research findings
        research_findings = {
            "query": query,
            "synthesis_timestamp": datetime.now().isoformat(),
            "literature_summary": {
                "total_papers_found": len(all_papers),
                "sources": {
                    "arxiv": len(arxiv_data.get("papers", [])),
                    "pubmed": len(pubmed_data.get("papers", []))
                },
                "top_papers": all_papers[:3]  # Top 3 most relevant papers
            },
            "references": references,
            "key_references": references[:5],  # Top 5 most relevant papers as key references
            "hypotheses": hypotheses_data.get("hypotheses", []),
            "key_concepts": key_concepts,
            "research_gaps": research_gaps,
            "methodology_suggestions": methodology_suggestions,
            "confidence_score": round(overall_confidence, 2),
            "research_quality_indicators": {
                "literature_coverage": "high" if len(all_papers) >= 10 else "medium" if len(all_papers) >= 5 else "low",
                "source_diversity": "high" if len(set(p.get("source") for p in all_papers)) >= 2 else "low",
                "temporal_coverage": "recent" if any(
                    datetime.fromisoformat(p.get("publication_date", "2020-01-01").replace('Z', '+00:00')).year >= 2022 
                    for p in all_papers if p.get("publication_date")
                ) else "older",
                "hypothesis_strength": "strong" if hyp_confidence >= 0.8 else "moderate" if hyp_confidence >= 0.6 else "weak"
            },
            "next_steps": [
                "Proceed to data collection phase based on identified datasets and methodologies",
                "Focus experimental design on highest-confidence hypotheses",
                "Consider addressing identified research gaps in experimental approach",
                "Validate findings against top-ranked literature sources"
            ],
            "status": "success"
        }
        
        logger.info("Research synthesis completed", 
                   total_papers=len(all_papers),
                   hypotheses_count=len(hypotheses_data.get("hypotheses", [])),
                   confidence=overall_confidence)
        
        return json.dumps(research_findings, indent=2)
        
    except Exception as e:
        error_msg = f"Research synthesis error: {str(e)}"
        logger.error("Research synthesis failed", error=error_msg)
        return json.dumps({
            "query": query,
            "error": error_msg,
            "status": "error"
        })

# Export all tools for easy import
__all__ = [
    "arxiv_search_tool",
    "pubmed_search_tool", 
    "hypothesis_generation_tool",
    "literature_analysis_tool",
    "research_synthesis_tool"
]