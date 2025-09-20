#!/usr/bin/env python3
"""
Demo script to showcase the Research Agent's reference and bibliography functionality
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.research_agent import create_research_agent

async def demo_references():
    """Demonstrate the reference functionality"""
    print("🔬 Research Agent Reference Demo")
    print("=" * 50)
    
    # Create research agent
    research_agent = create_research_agent()
    
    # Execute research on a sample query
    query = "machine learning fairness"
    session_id = "demo_session"
    
    print(f"📝 Executing research for: '{query}'")
    print("⏳ This may take a moment to search arXiv and PubMed...")
    
    try:
        # Execute research
        results = await research_agent.execute_research(query, session_id)
        
        if results.get("status") == "failed":
            print(f"❌ Research failed: {results.get('error')}")
            return
        
        print(f"✅ Research completed successfully!")
        print(f"📊 Found {len(results.get('references', []))} total references")
        print(f"⭐ {len(results.get('key_references', []))} key references identified")
        
        # Show key references
        print("\n🔑 Key References:")
        print("-" * 30)
        key_refs = results.get("key_references", [])[:3]  # Show top 3
        for ref in key_refs:
            print(f"{ref['id']}. {ref['title']}")
            print(f"   Authors: {', '.join(ref['authors'])}")
            print(f"   Source: {ref['source']} ({ref['year']})")
            print(f"   URL: {ref['url']}")
            print(f"   Relevance: {ref['relevance_score']:.2f}")
            print()
        
        # Generate bibliography in different formats
        print("\n📚 Bibliography (APA Style):")
        print("-" * 40)
        apa_bib = research_agent.generate_bibliography(results, "apa")
        print(apa_bib[:500] + "..." if len(apa_bib) > 500 else apa_bib)
        
        print("\n📚 Bibliography (MLA Style):")
        print("-" * 40)
        mla_bib = research_agent.generate_bibliography(results, "mla")
        print(mla_bib[:500] + "..." if len(mla_bib) > 500 else mla_bib)
        
        # Show paper links
        print("\n🔗 Paper Links for Easy Access:")
        print("-" * 40)
        paper_links = research_agent.get_paper_links(results)
        for i, paper in enumerate(paper_links[:5], 1):  # Show first 5
            print(f"{i}. {paper['title']}")
            print(f"   🔗 {paper['url']}")
            print(f"   📊 Relevance: {paper['relevance_score']:.2f}")
            print()
        
        # Show research quality metrics
        quality = research_agent.validate_research_quality(results)
        print(f"📈 Research Quality: {quality['quality_level'].upper()}")
        print(f"   - Literature Coverage: {quality['paper_count']} papers")
        print(f"   - Source Diversity: {quality['source_diversity']} sources")
        print(f"   - Confidence Score: {quality['confidence_score']:.2f}")
        
        # Show saved files
        saved_files = results.get("saved_files", {})
        if saved_files:
            print("\n💾 References automatically saved to files:")
            print("-" * 50)
            for file_type, file_path in saved_files.items():
                print(f"   📄 {file_type}: {file_path}")
        
        print("\n✨ Demo completed! The Research Agent successfully:")
        print("   ✅ Found and analyzed academic papers")
        print("   ✅ Generated formatted bibliographies")
        print("   ✅ Provided direct links to all papers")
        print("   ✅ Assessed research quality and relevance")
        print("   ✅ Automatically saved all references to files")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(demo_references())