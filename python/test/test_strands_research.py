#!/usr/bin/env python3
"""
Test script for the Strands Research Agent
Tests all research tools and agent interactions
"""

import asyncio
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.research_agent import ResearchAgent, create_research_agent
from src.tools.research_tools import (
    arxiv_search_tool,
    pubmed_search_tool,
    hypothesis_generation_tool,
    literature_analysis_tool,
    research_synthesis_tool
)
from src.core.shared_memory import SharedMemory
from src.core.logger import get_logger

logger = get_logger(__name__)

class ResearchAgentTester:
    """Test suite for Research Agent and tools"""
    
    def __init__(self):
        self.shared_memory = SharedMemory()
        self.research_agent = create_research_agent(self.shared_memory)
        self.test_results = []
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def test_arxiv_search_tool(self, query: str = "machine learning bias"):
        """Test arXiv search tool"""
        print("\n🔍 Testing arXiv Search Tool...")
        
        try:
            result = arxiv_search_tool(query, max_results=3)  # Reduced for token efficiency
            result_data = json.loads(result)
            
            # Validate result structure
            required_fields = ["query", "source", "papers_found", "papers", "status"]
            missing_fields = [field for field in required_fields if field not in result_data]
            
            if missing_fields:
                self.log_test_result("arXiv Search Tool", False, f"Missing fields: {missing_fields}")
                return False
            
            # Check if papers were found
            papers_found = result_data.get("papers_found", 0)
            papers = result_data.get("papers", [])
            
            if papers_found > 0 and len(papers) > 0:
                # Validate paper structure
                first_paper = papers[0]
                paper_fields = ["title", "authors", "source", "relevance_score"]
                missing_paper_fields = [field for field in paper_fields if field not in first_paper]
                
                if missing_paper_fields:
                    self.log_test_result("arXiv Search Tool", False, f"Missing paper fields: {missing_paper_fields}")
                    return False
                
                self.log_test_result("arXiv Search Tool", True, f"Found {papers_found} papers")
                return True
            else:
                self.log_test_result("arXiv Search Tool", True, "No papers found (acceptable for some queries)")
                return True
                
        except Exception as e:
            self.log_test_result("arXiv Search Tool", False, f"Exception: {str(e)}")
            return False
    
    def test_pubmed_search_tool(self, query: str = "machine learning bias"):
        """Test PubMed search tool"""
        print("\n🔍 Testing PubMed Search Tool...")
        
        try:
            result = pubmed_search_tool(query, max_results=3)  # Reduced for token efficiency
            result_data = json.loads(result)
            
            # Validate result structure
            required_fields = ["query", "source", "papers_found", "papers", "status"]
            missing_fields = [field for field in required_fields if field not in result_data]
            
            if missing_fields:
                self.log_test_result("PubMed Search Tool", False, f"Missing fields: {missing_fields}")
                return False
            
            # Check if papers were found
            papers_found = result_data.get("papers_found", 0)
            papers = result_data.get("papers", [])
            
            if papers_found > 0 and len(papers) > 0:
                # Validate paper structure
                first_paper = papers[0]
                paper_fields = ["title", "authors", "source", "relevance_score"]
                missing_paper_fields = [field for field in paper_fields if field not in first_paper]
                
                if missing_paper_fields:
                    self.log_test_result("PubMed Search Tool", False, f"Missing paper fields: {missing_paper_fields}")
                    return False
                
                self.log_test_result("PubMed Search Tool", True, f"Found {papers_found} papers")
                return True
            else:
                self.log_test_result("PubMed Search Tool", True, "No papers found (acceptable for some queries)")
                return True
                
        except Exception as e:
            self.log_test_result("PubMed Search Tool", False, f"Exception: {str(e)}")
            return False
    
    def test_hypothesis_generation_tool(self, query: str = "machine learning bias"):
        """Test hypothesis generation tool"""
        print("\n💡 Testing Hypothesis Generation Tool...")
        
        try:
            # Create mock research context
            research_context = json.dumps({
                "papers": [
                    {"title": "Test Paper", "relevance_score": 0.8},
                    {"title": "Another Paper", "relevance_score": 0.7}
                ]
            })
            
            result = hypothesis_generation_tool(research_context, query)
            result_data = json.loads(result)
            
            # Validate result structure
            required_fields = ["query", "hypotheses_generated", "hypotheses", "status"]
            missing_fields = [field for field in required_fields if field not in result_data]
            
            if missing_fields:
                self.log_test_result("Hypothesis Generation Tool", False, f"Missing fields: {missing_fields}")
                return False
            
            # Check hypotheses
            hypotheses = result_data.get("hypotheses", [])
            if len(hypotheses) == 0:
                self.log_test_result("Hypothesis Generation Tool", False, "No hypotheses generated")
                return False
            
            # Validate hypothesis structure
            first_hypothesis = hypotheses[0]
            hypothesis_fields = ["text", "confidence", "testable", "variables"]
            missing_hyp_fields = [field for field in hypothesis_fields if field not in first_hypothesis]
            
            if missing_hyp_fields:
                self.log_test_result("Hypothesis Generation Tool", False, f"Missing hypothesis fields: {missing_hyp_fields}")
                return False
            
            self.log_test_result("Hypothesis Generation Tool", True, f"Generated {len(hypotheses)} hypotheses")
            return True
            
        except Exception as e:
            self.log_test_result("Hypothesis Generation Tool", False, f"Exception: {str(e)}")
            return False
    
    def test_literature_analysis_tool(self, query: str = "machine learning bias"):
        """Test literature analysis tool"""
        print("\n📊 Testing Literature Analysis Tool...")
        
        try:
            # Create mock papers data
            papers_data = {
                "papers": [
                    {
                        "title": "Machine Learning Bias in Healthcare",
                        "authors": ["Smith, J.", "Doe, A."],
                        "source": "arXiv",
                        "relevance_score": 0.9,
                        "publication_date": "2023-01-15T00:00:00+00:00",
                        "abstract": "This paper examines bias in machine learning algorithms used in healthcare applications."
                    },
                    {
                        "title": "Fairness in AI Systems",
                        "authors": ["Johnson, B."],
                        "source": "PubMed",
                        "relevance_score": 0.8,
                        "publication_date": "2022-06-10T00:00:00+00:00",
                        "abstract": "An analysis of fairness considerations in artificial intelligence systems."
                    }
                ]
            }
            
            result = literature_analysis_tool(json.dumps(papers_data), query)
            result_data = json.loads(result)
            
            # Validate result structure
            required_fields = ["query", "literature_analysis", "status"]
            missing_fields = [field for field in required_fields if field not in result_data]
            
            if missing_fields:
                self.log_test_result("Literature Analysis Tool", False, f"Missing fields: {missing_fields}")
                return False
            
            # Check analysis structure
            analysis = result_data.get("literature_analysis", {})
            analysis_fields = ["total_papers", "sources", "relevance_distribution", "confidence_assessment"]
            missing_analysis_fields = [field for field in analysis_fields if field not in analysis]
            
            if missing_analysis_fields:
                self.log_test_result("Literature Analysis Tool", False, f"Missing analysis fields: {missing_analysis_fields}")
                return False
            
            self.log_test_result("Literature Analysis Tool", True, f"Analyzed {analysis['total_papers']} papers")
            return True
            
        except Exception as e:
            self.log_test_result("Literature Analysis Tool", False, f"Exception: {str(e)}")
            return False
    
    def test_research_synthesis_tool(self, query: str = "machine learning bias"):
        """Test research synthesis tool"""
        print("\n🔬 Testing Research Synthesis Tool...")
        
        try:
            # Create mock data for all inputs
            arxiv_results = json.dumps({
                "papers": [{"title": "ArXiv Paper", "relevance_score": 0.8}],
                "papers_found": 1
            })
            
            pubmed_results = json.dumps({
                "papers": [{"title": "PubMed Paper", "relevance_score": 0.7}],
                "papers_found": 1
            })
            
            hypotheses = json.dumps({
                "hypotheses": [
                    {"text": "Test hypothesis", "confidence": 0.8, "testable": True}
                ]
            })
            
            analysis = json.dumps({
                "literature_analysis": {
                    "confidence_assessment": 0.75,
                    "key_themes": [{"theme": "bias", "frequency": 3}],
                    "research_gaps": ["Gap 1", "Gap 2"]
                }
            })
            
            result = research_synthesis_tool(arxiv_results, pubmed_results, hypotheses, analysis, query)
            result_data = json.loads(result)
            
            # Validate result structure
            required_fields = ["query", "literature_summary", "hypotheses", "confidence_score", "status"]
            missing_fields = [field for field in required_fields if field not in result_data]
            
            if missing_fields:
                self.log_test_result("Research Synthesis Tool", False, f"Missing fields: {missing_fields}")
                return False
            
            # Check synthesis quality
            confidence = result_data.get("confidence_score", 0)
            if confidence < 0 or confidence > 1:
                self.log_test_result("Research Synthesis Tool", False, f"Invalid confidence score: {confidence}")
                return False
            
            self.log_test_result("Research Synthesis Tool", True, f"Synthesis completed with confidence: {confidence}")
            return True
            
        except Exception as e:
            self.log_test_result("Research Synthesis Tool", False, f"Exception: {str(e)}")
            return False
    
    async def test_research_agent_execution(self, query: str = "machine learning bias"):
        """Test full research agent execution"""
        print("\n🤖 Testing Research Agent Execution...")
        
        try:
            session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Execute research
            result = await self.research_agent.execute_research(query, session_id)
            
            # Validate result structure
            if not isinstance(result, dict):
                self.log_test_result("Research Agent Execution", False, "Result is not a dictionary")
                return False
            
            # Check for required fields
            if "status" not in result:
                self.log_test_result("Research Agent Execution", False, "Missing status field")
                return False
            
            if result.get("status") == "failed":
                self.log_test_result("Research Agent Execution", False, f"Agent execution failed: {result.get('error', 'Unknown error')}")
                return False
            
            # Check if results were stored in shared memory
            stored_context = self.shared_memory.get_context(session_id)
            if not stored_context or "research_result" not in stored_context:
                self.log_test_result("Research Agent Execution", False, "Results not stored in shared memory")
                return False
            
            # Validate research quality
            quality_metrics = self.research_agent.validate_research_quality(result)
            quality_level = quality_metrics.get("quality_level", "unknown")
            
            # Test reference functionality
            references_count = len(result.get("references", []))
            key_references_count = len(result.get("key_references", []))
            
            # Generate bibliography
            bibliography = self.research_agent.generate_bibliography(result, "plain")
            paper_links = self.research_agent.get_paper_links(result)
            
            details = f"Execution completed with {quality_level} quality. Found {references_count} references ({key_references_count} key references). Generated bibliography with {len(paper_links)} paper links."
            
            self.log_test_result("Research Agent Execution", True, details)
            return True
            
        except Exception as e:
            self.log_test_result("Research Agent Execution", False, f"Exception: {str(e)}")
            return False
    
    def test_reference_functionality(self, query: str = "machine learning bias"):
        """Test reference and bibliography generation functionality"""
        print("\n📚 Testing Reference Functionality...")
        
        try:
            # Create mock research results with references
            mock_results = {
                "query": query,
                "references": [
                    {
                        "id": 1,
                        "citation": "Smith, J., Doe, A. (2023). Machine Learning Bias in Healthcare. arXiv.",
                        "title": "Machine Learning Bias in Healthcare",
                        "authors": ["Smith, J.", "Doe, A."],
                        "year": "2023",
                        "source": "arXiv",
                        "url": "https://arxiv.org/abs/2301.12345",
                        "relevance_score": 0.95,
                        "abstract": "This paper examines bias in machine learning algorithms used in healthcare applications."
                    },
                    {
                        "id": 2,
                        "citation": "Johnson, B. (2022). Fairness in AI Systems. PubMed.",
                        "title": "Fairness in AI Systems",
                        "authors": ["Johnson, B."],
                        "year": "2022",
                        "source": "PubMed",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                        "relevance_score": 0.88,
                        "abstract": "An analysis of fairness considerations in artificial intelligence systems."
                    }
                ],
                "key_references": []
            }
            
            # Test bibliography generation
            bibliography_plain = self.research_agent.generate_bibliography(mock_results, "plain")
            bibliography_apa = self.research_agent.generate_bibliography(mock_results, "apa")
            bibliography_mla = self.research_agent.generate_bibliography(mock_results, "mla")
            
            if not bibliography_plain or len(bibliography_plain) < 50:
                self.log_test_result("Reference Functionality", False, "Plain bibliography generation failed")
                return False
            
            if not bibliography_apa or "References (APA Style)" not in bibliography_apa:
                self.log_test_result("Reference Functionality", False, "APA bibliography generation failed")
                return False
            
            if not bibliography_mla or "Works Cited (MLA Style)" not in bibliography_mla:
                self.log_test_result("Reference Functionality", False, "MLA bibliography generation failed")
                return False
            
            # Test paper links extraction
            paper_links = self.research_agent.get_paper_links(mock_results)
            
            if len(paper_links) != 2:
                self.log_test_result("Reference Functionality", False, f"Expected 2 paper links, got {len(paper_links)}")
                return False
            
            # Validate paper link structure
            first_link = paper_links[0]
            required_fields = ["title", "authors", "year", "source", "url", "relevance_score", "abstract"]
            missing_fields = [field for field in required_fields if field not in first_link]
            
            if missing_fields:
                self.log_test_result("Reference Functionality", False, f"Missing fields in paper links: {missing_fields}")
                return False
            
            self.log_test_result("Reference Functionality", True, f"Generated bibliographies in 3 formats and extracted {len(paper_links)} paper links")
            return True
            
        except Exception as e:
            self.log_test_result("Reference Functionality", False, f"Exception: {str(e)}")
            return False
    
    def test_shared_memory_integration(self):
        """Test shared memory integration"""
        print("\n💾 Testing Shared Memory Integration...")
        
        try:
            session_id = "test_memory_session"
            
            # Test storing research context
            test_data = {
                "research_result": {
                    "query": "test query",
                    "status": "success",
                    "confidence_score": 0.8
                }
            }
            
            self.shared_memory.update_context(session_id, test_data)
            
            # Test retrieving research context
            retrieved_data = self.shared_memory.get_context(session_id)
            
            if not retrieved_data:
                self.log_test_result("Shared Memory Integration", False, "Failed to retrieve data")
                return False
            
            if retrieved_data.get("research_result", {}).get("data", {}).get("query") != "test query":
                self.log_test_result("Shared Memory Integration", False, "Data integrity issue")
                return False
            
            # Test research status retrieval
            status = self.research_agent.get_research_status(session_id)
            if not status:
                self.log_test_result("Shared Memory Integration", False, "Research status retrieval failed")
                return False
            
            # Test raw context retrieval (with metadata)
            raw_context = self.shared_memory.get_context(session_id)
            if not raw_context or "research_result" not in raw_context:
                self.log_test_result("Shared Memory Integration", False, "Raw context retrieval failed")
                return False
            
            # Verify metadata structure
            research_data = raw_context["research_result"]
            if not isinstance(research_data, dict) or "data" not in research_data:
                self.log_test_result("Shared Memory Integration", False, "Metadata structure invalid")
                return False
            
            self.log_test_result("Shared Memory Integration", True, "All memory operations successful")
            return True
            
        except Exception as e:
            self.log_test_result("Shared Memory Integration", False, f"Exception: {str(e)}")
            return False
    
    async def run_all_tests(self, query: str = "machine learning bias"):
        """Run all tests"""
        print(f"🧪 Starting Research Agent Test Suite")
        print(f"📝 Test Query: '{query}'")
        print("=" * 60)
        
        # Run individual tool tests
        self.test_arxiv_search_tool(query)
        self.test_pubmed_search_tool(query)
        self.test_hypothesis_generation_tool(query)
        self.test_literature_analysis_tool(query)
        self.test_research_synthesis_tool(query)
        
        # Run integration tests
        await self.test_research_agent_execution(query)
        self.test_shared_memory_integration()
        self.test_reference_functionality(query)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")
        
        # Save detailed results
        results_file = f"test_results_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": (passed_tests/total_tests)*100,
                    "test_query": query
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return failed_tests == 0

async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description="Test Strands Research Agent")
    parser.add_argument("--query", default="machine learning bias", 
                       help="Research query to test with (default: 'machine learning bias')")
    parser.add_argument("--ai-ethics", action="store_true",
                       help="Use 'AI ethics' as test query")
    
    args = parser.parse_args()
    
    # Set query based on arguments
    if args.ai_ethics:
        query = "AI ethics"
    else:
        query = args.query
    
    # Run tests
    tester = ResearchAgentTester()
    success = await tester.run_all_tests(query)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())