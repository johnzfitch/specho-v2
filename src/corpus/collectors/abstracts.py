"""
arXiv Abstract Collector

Source: arXiv API / bulk data
Contains: Millions of paper abstracts from 1991-present
License: Various (mostly permissive for metadata)
Pre-LLM: Verifiable by submission date

arXiv is ideal because:
1. Exact submission dates available
2. High-quality academic writing
3. Large volume
4. Multiple subject domains
"""

import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional
from datetime import date, datetime
from dataclasses import dataclass
import logging
import json

from ..schema import CorpusSample, Domain, Subdomain
from ..registry import BaseCollector, DomainRegistry

logger = logging.getLogger(__name__)


# arXiv category to general domain mapping
ARXIV_CATEGORIES = {
    # Physics (sample these less - very technical)
    'astro-ph': 'physics',
    'cond-mat': 'physics', 
    'gr-qc': 'physics',
    'hep-ex': 'physics',
    'hep-lat': 'physics',
    'hep-ph': 'physics',
    'hep-th': 'physics',
    'math-ph': 'physics',
    'nlin': 'physics',
    'nucl-ex': 'physics',
    'nucl-th': 'physics',
    'physics': 'physics',
    'quant-ph': 'physics',
    
    # Math
    'math': 'math',
    
    # CS - high value for LLM detection training
    'cs.AI': 'cs',
    'cs.CL': 'cs',  # Computational linguistics - very relevant
    'cs.CV': 'cs',
    'cs.LG': 'cs',  # Machine learning
    'cs.NE': 'cs',
    'cs.SE': 'cs',
    'cs.HC': 'cs',
    
    # Quantitative Biology
    'q-bio': 'biology',
    
    # Quantitative Finance
    'q-fin': 'finance',
    
    # Statistics
    'stat': 'statistics',
    
    # Electrical Engineering
    'eess': 'engineering',
    
    # Economics
    'econ': 'economics',
}


@DomainRegistry.register
class ArxivCollector(BaseCollector):
    """Collector for arXiv paper abstracts."""
    
    DOMAIN = Domain.ACADEMIC
    SUBDOMAIN = Subdomain.ABSTRACTS
    SOURCE_NAME = "arxiv"
    SOURCE_VERSION = "api_2024"
    LICENSE = "arxiv_license"
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # LLM cutoff - only collect papers submitted before this
    LLM_CUTOFF = date(2022, 11, 1)
    
    def __init__(
        self,
        categories: Optional[list[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Default to CS categories (most relevant for AI detection research)
        self.categories = categories or ['cs.CL', 'cs.AI', 'cs.LG', 'cs.HC']
        self.start_date = start_date or date(2015, 1, 1)
        self.end_date = end_date or date(2022, 10, 31)  # Before ChatGPT
        
        # Rate limit for arXiv API
        self.rate_limit = 3.0  # 3 seconds between requests (arXiv requirement)
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield abstracts from arXiv."""
        
        target = limit or self.config.target_samples
        collected = 0
        
        for category in self.categories:
            if collected >= target:
                break
            
            logger.info(f"Collecting from arXiv category: {category}")
            
            # Calculate samples per category
            per_category = (target - collected) // len([c for c in self.categories if c not in self._seen_hashes])
            
            for sample in self._query_category(category, per_category):
                if collected >= target:
                    break
                
                if self.validate_sample(sample):
                    self._seen_hashes.add(sample.content_hash)
                    collected += 1
                    yield sample
                    
                    if collected % 100 == 0:
                        logger.info(f"Collected {collected}/{target} arXiv abstracts")
        
        logger.info(f"arXiv collection complete: {collected} abstracts")
    
    def _query_category(
        self, 
        category: str, 
        max_results: int
    ) -> Iterator[CorpusSample]:
        """Query arXiv API for a specific category."""
        
        start = 0
        batch_size = 100  # arXiv max per request
        
        while start < max_results:
            # Build query
            date_range = f"submittedDate:[{self.start_date.strftime('%Y%m%d')}0000 TO {self.end_date.strftime('%Y%m%d')}2359]"
            query = f"cat:{category} AND {date_range}"
            
            params = {
                'search_query': query,
                'start': start,
                'max_results': min(batch_size, max_results - start),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending',
            }
            
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            
            try:
                # Rate limiting
                time.sleep(self.rate_limit)
                
                # Fetch
                with urllib.request.urlopen(url, timeout=30) as response:
                    xml_data = response.read().decode('utf-8')
                
                # Parse
                root = ET.fromstring(xml_data)
                
                # Namespace handling for Atom feed
                ns = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                entries = root.findall('atom:entry', ns)
                
                if not entries:
                    break
                
                for entry in entries:
                    sample = self._parse_entry(entry, ns, category)
                    if sample:
                        yield sample
                
                start += len(entries)
                
                if len(entries) < batch_size:
                    break  # No more results
                    
            except Exception as e:
                logger.error(f"Error querying arXiv: {e}")
                break
    
    def _parse_entry(self, entry, ns: dict, category: str) -> Optional[CorpusSample]:
        """Parse a single arXiv entry."""
        try:
            # Extract fields
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            title = entry.find('atom:title', ns).text or ''
            abstract = entry.find('atom:summary', ns).text or ''
            published = entry.find('atom:published', ns).text
            
            # Parse date
            pub_date = datetime.fromisoformat(published.replace('Z', '+00:00')).date()
            
            # Skip if after LLM cutoff
            if pub_date >= self.LLM_CUTOFF:
                return None
            
            # Clean text
            title = ' '.join(title.split())
            abstract = ' '.join(abstract.split())
            
            # Combine title + abstract for more context
            text = f"{title}\n\n{abstract}"
            
            # Get URL
            links = entry.findall('atom:link', ns)
            url = None
            for link in links:
                if link.get('type') == 'text/html':
                    url = link.get('href')
                    break
            
            return self.create_sample(
                text=text,
                original_id=arxiv_id,
                content_date=pub_date,
                url=url or f"https://arxiv.org/abs/{arxiv_id}",
            )
            
        except Exception as e:
            logger.warning(f"Error parsing arXiv entry: {e}")
            return None


@DomainRegistry.register
class SemanticScholarCollector(BaseCollector):
    """
    Alternative academic collector using Semantic Scholar API.
    Good for getting abstracts across multiple disciplines.
    """
    
    DOMAIN = Domain.ACADEMIC
    SUBDOMAIN = Subdomain.ABSTRACTS
    SOURCE_NAME = "semantic_scholar"
    SOURCE_VERSION = "api_2024"
    LICENSE = "semantic_scholar_api"
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        fields_of_study: Optional[list[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.fields = fields_of_study or [
            'Computer Science',
            'Linguistics', 
            'Psychology',
            'Sociology',
            'Political Science',
        ]
        self.rate_limit = 1.0  # S2 allows 100 req/5min without key
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield abstracts from Semantic Scholar."""
        
        target = limit or self.config.target_samples
        collected = 0
        
        for field in self.fields:
            if collected >= target:
                break
            
            logger.info(f"Collecting from Semantic Scholar field: {field}")
            
            # Query papers from 2015-2022
            params = {
                'query': '',
                'fieldsOfStudy': field,
                'year': '2015-2022',
                'fields': 'paperId,title,abstract,year,publicationDate,url',
                'limit': 100,
            }
            
            headers = {}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            offset = 0
            while collected < target:
                params['offset'] = offset
                
                try:
                    time.sleep(self.rate_limit)
                    
                    req = urllib.request.Request(
                        f"{self.BASE_URL}?{urllib.parse.urlencode(params)}",
                        headers=headers
                    )
                    
                    with urllib.request.urlopen(req, timeout=30) as response:
                        data = json.loads(response.read().decode('utf-8'))
                    
                    papers = data.get('data', [])
                    if not papers:
                        break
                    
                    for paper in papers:
                        if collected >= target:
                            break
                        
                        sample = self._parse_paper(paper)
                        if sample and self.validate_sample(sample):
                            self._seen_hashes.add(sample.content_hash)
                            collected += 1
                            yield sample
                    
                    offset += len(papers)
                    
                    if len(papers) < 100:
                        break
                        
                except Exception as e:
                    logger.error(f"Error querying Semantic Scholar: {e}")
                    break
        
        logger.info(f"Semantic Scholar collection complete: {collected} abstracts")
    
    def _parse_paper(self, paper: dict) -> Optional[CorpusSample]:
        """Parse a Semantic Scholar paper."""
        try:
            paper_id = paper.get('paperId', '')
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            if not abstract:
                return None
            
            # Parse date
            pub_date_str = paper.get('publicationDate')
            year = paper.get('year')
            
            if pub_date_str:
                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
            elif year:
                pub_date = date(int(year), 6, 15)  # Mid-year estimate
            else:
                return None
            
            # Skip if after LLM cutoff
            if pub_date >= date(2022, 11, 1):
                return None
            
            text = f"{title}\n\n{abstract}"
            
            return self.create_sample(
                text=' '.join(text.split()),
                original_id=paper_id,
                content_date=pub_date,
                url=paper.get('url'),
            )
            
        except Exception as e:
            logger.warning(f"Error parsing S2 paper: {e}")
            return None
