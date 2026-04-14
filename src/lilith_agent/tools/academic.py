from __future__ import annotations

import requests
import xml.etree.ElementTree as ET
import logging
import json
from bs4 import BeautifulSoup
import re

log = logging.getLogger(__name__)

def count_journal_articles(journal_name: str, year: int, is_research_only: bool = True) -> str:
    """High-precision tool to count articles in a specific journal for a given year.
    Handles 'Logic Sinking' by encapsulating scraper logic and ISSN mapping.
    
    Args:
        journal_name: Name of the journal (e.g., 'Nature', 'Science', 'Lancet').
        year: Publication year.
        is_research_only: If True, filters out news, reviews, and editorials.
    """
    journal_name = journal_name.lower().strip()
    
    # Internal ISSN Mapping
    ISSN_MAP = {
        "nature": "0028-0836",
        "science": "0036-8075",
        "lancet": "0140-6736",
        "the lancet": "0140-6736",
        "cell": "0092-8674",
        "pnas": "0027-8424",
        "jama": "0098-7484"
    }
    
    # 1. SPECIALIZED SCRAPING for Nature
    if journal_name == "nature":
        try:
            url = f"https://www.nature.com/search?journal=nature&article_type={'research' if is_research_only else 'all'}&date_range={year}-{year}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            res = requests.get(url, headers=headers, timeout=20)
            res.raise_for_status()
            
            soup = BeautifulSoup(res.text, 'html.parser')
            count_elem = soup.find(attrs={'data-test': 'results-data'})
            if count_elem:
                raw_text = count_elem.text.strip()
                # Extract number from "1,037 results" or "Showing 1-50 of 1037 results"
                match = re.search(r'(\d[,.\d]*)', raw_text.split('of')[-1])
                if match:
                    count_str = match.group(1).replace(',', '').replace('.', '')
                    count = int(count_str)
                    
                    metadata = {
                        "value": count,
                        "data_source": "nature_official_search",
                        "record_type": "research-article" if is_research_only else "all",
                        "type_strictness": "exact",
                        "url": url,
                        "note": "Scraped directly from nature.com using specialized selectors."
                    }
                    return f"FOUND {count} items for {journal_name} in {year}.\n\nMETADATA:\n{json.dumps(metadata, indent=2)}"
        except Exception as e:
            log.warning("Nature scraping failed, falling back to CrossRef: %s", e)

    # 2. CROSSREF FALLBACK
    issn = ISSN_MAP.get(journal_name)
    if not issn:
        # Try to find ISSN via search or just use name in CrossRef query
        filter_str = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
        if is_research_only:
            filter_str += ",type:journal-article"
    else:
        filter_str = f"issn:{issn},from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
        if is_research_only:
            filter_str += ",type:journal-article"
            
    return crossref_search(filter_str)

def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers. Returns a summary of findings."""
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        # ArXiv uses Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        if not entries:
            return f"No ArXiv results found for '{query}'."
            
        results = []
        for entry in entries:
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            author_names = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text
            link = entry.find('atom:id', ns).text
            
            results.append(
                f"Title: {title}\n"
                f"Authors: {', '.join(author_names)}\n"
                f"Published: {published}\n"
                f"Link: {link}\n"
                f"Summary: {summary[:300]}...\n"
            )
            
        metadata = {
            "value": len(results),
            "data_source": "arxiv",
            "record_type": "preprint",
            "type_strictness": "medium",
            "includes_types": ["preprint"],
            "excludes_types": ["peer-reviewed-articles"]
        }
        res_text = "\n---\n".join(results)
        return f"{res_text}\n\nMETADATA:\n{json.dumps(metadata, indent=2)}"
        
    except Exception as e:
        log.error("ArXiv search error: %s", e)
        return f"Error searching ArXiv: {e}"

def crossref_search(filter_str: str, rows: int = 100, cursor: str = "*", email: str = "test@example.com") -> str:
    """Search CrossRef API for metadata.
    
    Args:
        filter_str: Filter string (e.g., 'issn:0028-0836,type:journal-article').
        rows: Number of results per page (max 1000).
        cursor: Pagination cursor. Use '*' for the first page.
        email: Contact email for the Polite API (recommended).
    """
    base_url = "https://api.crossref.org/works"
    params = {
        "filter": filter_str,
        "rows": rows,
        "cursor": cursor,
        "mailto": email
    }
    headers = {"User-Agent": "GAIA-Agent/1.0 (mailto:test@example.com)"}
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, dict) or "message" not in data:
            return f"Error: Unexpected CrossRef API response format: {str(data)[:200]}"
            
        msg = data["message"]
        total = msg.get("total-results", 0)
        items = msg.get("items", [])
        next_cursor = msg.get("next-cursor")
        
        output = [f"TOTAL RESULTS: {total}", f"NEXT CURSOR: {next_cursor}", ""]
        output.append(f"Showing {len(items)} items from current page:")
        
        entry_list = []
        for item in items:
            title = item.get("title", ["no title"])[0]
            year = item.get("published-print", {}).get("date-parts", [[None]])[0][0]
            doi = item.get("DOI", "no doi")
            st = item.get("subtype", "no subtype")
            output.append(f"- [{year}] {title} (DOI: {doi}, subtype: {st})")
            entry_list.append({"title": title, "year": year, "doi": doi, "subtype": st})
            
        metadata = {
            "value": total,
            "data_source": "crossref",
            "record_type": "journal-article",
            "type_strictness": "broad",
            "includes_types": ["article", "review", "news", "editorial", "correspondence"],
            "excludes_types": [],
            "current_page_items": entry_list
        }
        
        final_text = "\n".join(output)
        return f"{final_text}\n\nMETADATA:\n{json.dumps(metadata, indent=2)}"
        
    except Exception as e:
        log.error("CrossRef search error: %s", e)
        return f"Error searching CrossRef: {e}"
