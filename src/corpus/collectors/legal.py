"""
Legal and Government Document Collectors

Sources:
- CourtListener (court filings, opinions)
- Congress.gov (legislation)
- CRS Reports (policy analysis)

Legal text provides extremely formal, structured writing that
represents one extreme of the formality spectrum.
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Iterator, Optional
from datetime import date, datetime
import logging
import re

from ..schema import CorpusSample, Domain, Subdomain
from ..registry import BaseCollector, DomainRegistry

logger = logging.getLogger(__name__)


@DomainRegistry.register
class CourtListenerCollector(BaseCollector):
    """
    Collector for court opinions from CourtListener.
    
    CourtListener provides free access to millions of legal opinions
    with excellent date metadata.
    
    API: https://www.courtlistener.com/api/rest-info/
    Bulk data: https://www.courtlistener.com/api/bulk-info/
    """
    
    DOMAIN = Domain.LEGAL
    SUBDOMAIN = Subdomain.COURT_FILINGS
    SOURCE_NAME = "courtlistener"
    SOURCE_VERSION = "api_2024"
    LICENSE = "public_domain"  # Court opinions are public domain
    
    BASE_URL = "https://www.courtlistener.com/api/rest/v3"
    
    # Court types to include (federal appellate courts have highest quality)
    DEFAULT_COURTS = [
        'scotus',     # Supreme Court
        'ca1', 'ca2', 'ca3', 'ca4', 'ca5', 'ca6', 'ca7', 'ca8', 'ca9', 'ca10', 'ca11', 'cadc', 'cafc',  # Circuit Courts
    ]
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        courts: Optional[list[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.api_token = api_token
        self.courts = courts or self.DEFAULT_COURTS
        self.start_date = start_date or date(2010, 1, 1)
        self.end_date = end_date or date(2022, 10, 31)
        
        # CourtListener rate limit: 5000 req/hour with token, 100/hour without
        self.rate_limit = 0.75 if api_token else 36.0
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield court opinions from CourtListener."""
        
        target = limit or self.config.target_samples
        collected = 0
        
        for court in self.courts:
            if collected >= target:
                break
            
            logger.info(f"Collecting from court: {court}")
            
            next_url = self._build_initial_url(court)
            
            while next_url and collected < target:
                try:
                    time.sleep(self.rate_limit)
                    
                    response = self._make_request(next_url)
                    
                    for opinion in response.get('results', []):
                        if collected >= target:
                            break
                        
                        sample = self._parse_opinion(opinion, court)
                        
                        if sample and self.validate_sample(sample):
                            self._seen_hashes.add(sample.content_hash)
                            collected += 1
                            yield sample
                            
                            if collected % 100 == 0:
                                logger.info(f"Collected {collected}/{target} court opinions")
                    
                    next_url = response.get('next')
                    
                except Exception as e:
                    logger.error(f"Error fetching from CourtListener: {e}")
                    break
        
        logger.info(f"CourtListener collection complete: {collected} opinions")
    
    def _build_initial_url(self, court: str) -> str:
        """Build initial API query URL."""
        params = {
            'court': court,
            'date_filed__gte': self.start_date.isoformat(),
            'date_filed__lte': self.end_date.isoformat(),
            'order_by': '-date_filed',
            'type': 'o',  # Opinions only
        }
        return f"{self.BASE_URL}/opinions/?{urllib.parse.urlencode(params)}"
    
    def _make_request(self, url: str) -> dict:
        """Make authenticated API request."""
        headers = {
            'User-Agent': 'PACT-Corpus-Collector/1.0 (research)',
        }
        if self.api_token:
            headers['Authorization'] = f'Token {self.api_token}'
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    
    def _parse_opinion(self, opinion: dict, court: str) -> Optional[CorpusSample]:
        """Parse a court opinion."""
        
        # Get the opinion text
        # CourtListener has multiple text fields - prefer plain_text
        text = (
            opinion.get('plain_text') or 
            opinion.get('html_with_citations') or
            opinion.get('html') or
            ''
        )
        
        if not text:
            return None
        
        # Clean HTML if present
        if '<' in text:
            text = self._clean_html(text)
        
        # Skip very short opinions
        if len(text.split()) < 100:
            return None
        
        # Parse date
        date_filed = opinion.get('date_filed')
        if not date_filed:
            return None
        
        try:
            content_date = datetime.fromisoformat(date_filed).date()
        except:
            return None
        
        # Skip post-LLM
        if content_date >= date(2022, 11, 1):
            return None
        
        opinion_id = str(opinion.get('id', ''))
        
        # Build URL
        absolute_url = opinion.get('absolute_url', '')
        url = f"https://www.courtlistener.com{absolute_url}" if absolute_url else None
        
        return self.create_sample(
            text=text,
            original_id=f"{court}_{opinion_id}",
            content_date=content_date,
            url=url,
        )
    
    def _clean_html(self, html: str) -> str:
        """Clean HTML from opinion text."""
        # Remove script/style
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove tags but keep content
        html = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode entities
        html = html.replace('&nbsp;', ' ').replace('&amp;', '&')
        html = html.replace('&lt;', '<').replace('&gt;', '>')
        html = html.replace('&quot;', '"').replace('&#39;', "'")
        
        # Normalize whitespace
        return ' '.join(html.split())


@DomainRegistry.register
class CongressCollector(BaseCollector):
    """
    Collector for Congressional bills and reports.
    
    Source: Congress.gov API
    """
    
    DOMAIN = Domain.LEGAL
    SUBDOMAIN = Subdomain.LEGISLATION
    SOURCE_NAME = "congress_gov"
    SOURCE_VERSION = "api_2024"
    LICENSE = "public_domain"
    
    BASE_URL = "https://api.congress.gov/v3"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        congresses: Optional[list[int]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.api_key = api_key
        # Congresses 110-117 cover 2007-2022 (pre-LLM period)
        self.congresses = congresses or [110, 111, 112, 113, 114, 115, 116, 117]
        
        self.rate_limit = 1.0  # Conservative for Congress API
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield bill summaries and text from Congress.gov."""
        
        if not self.api_key:
            logger.warning("Congress.gov API key not provided. Get one at: https://api.congress.gov/sign-up/")
            return
        
        target = limit or self.config.target_samples
        collected = 0
        
        for congress in self.congresses:
            if collected >= target:
                break
            
            logger.info(f"Collecting from Congress {congress}")
            
            for sample in self._collect_congress(congress, target - collected):
                if self.validate_sample(sample):
                    self._seen_hashes.add(sample.content_hash)
                    collected += 1
                    yield sample
                    
                    if collected % 50 == 0:
                        logger.info(f"Collected {collected}/{target} congressional items")
                
                if collected >= target:
                    break
        
        logger.info(f"Congress collection complete: {collected} items")
    
    def _collect_congress(self, congress: int, max_items: int) -> Iterator[CorpusSample]:
        """Collect items from a specific Congress."""
        
        offset = 0
        limit = 250  # Max per request
        
        while offset < max_items:
            url = (
                f"{self.BASE_URL}/bill/{congress}"
                f"?format=json&offset={offset}&limit={min(limit, max_items - offset)}"
                f"&api_key={self.api_key}"
            )
            
            try:
                time.sleep(self.rate_limit)
                
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))
                
                bills = data.get('bills', [])
                
                if not bills:
                    break
                
                for bill in bills:
                    sample = self._parse_bill(bill, congress)
                    if sample:
                        yield sample
                
                offset += len(bills)
                
                if len(bills) < limit:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching from Congress.gov: {e}")
                break
    
    def _parse_bill(self, bill: dict, congress: int) -> Optional[CorpusSample]:
        """Parse a bill record."""
        
        # Get bill details including summary
        bill_type = bill.get('type', '')
        bill_number = bill.get('number', '')
        title = bill.get('title', '')
        
        # We need the summary - make another request
        # For now, use title + any available text
        text = title
        
        # Parse date
        introduced_date = bill.get('introducedDate')
        if not introduced_date:
            return None
        
        try:
            content_date = datetime.fromisoformat(introduced_date).date()
        except:
            return None
        
        # Skip post-LLM
        if content_date >= date(2022, 11, 1):
            return None
        
        # Title alone is too short
        if len(text.split()) < 20:
            return None
        
        bill_id = f"{congress}-{bill_type}{bill_number}"
        url = bill.get('url', f"https://www.congress.gov/bill/{congress}th-congress/{bill_type.lower()}/{bill_number}")
        
        return self.create_sample(
            text=text,
            original_id=bill_id,
            content_date=content_date,
            url=url,
        )


@DomainRegistry.register
class CRSReportCollector(BaseCollector):
    """
    Collector for Congressional Research Service reports.
    
    CRS reports are high-quality policy analysis documents.
    Source: EveryCRSReport.com or crsreports.congress.gov
    """
    
    DOMAIN = Domain.LEGAL
    SUBDOMAIN = Subdomain.POLICY
    SOURCE_NAME = "crs_reports"
    SOURCE_VERSION = "1.0"
    LICENSE = "public_domain"
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_path = data_path or self.cache_dir / "crs_reports"
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield CRS reports."""
        
        if not self.data_path.exists():
            logger.error(f"CRS reports not found at {self.data_path}")
            logger.info("Download from: https://www.everycrsreport.com/")
            return
        
        target = limit or self.config.target_samples
        collected = 0
        
        # Process JSON files in directory
        for json_file in self.data_path.glob("*.json"):
            if collected >= target:
                break
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                sample = self._parse_report(report)
                
                if sample and self.validate_sample(sample):
                    self._seen_hashes.add(sample.content_hash)
                    collected += 1
                    yield sample
                    
            except Exception as e:
                logger.warning(f"Error parsing {json_file}: {e}")
                continue
        
        logger.info(f"CRS collection complete: {collected} reports")
    
    def _parse_report(self, report: dict) -> Optional[CorpusSample]:
        """Parse a CRS report."""
        
        title = report.get('title', '')
        summary = report.get('summary', '')
        
        # Combine title and summary
        text = f"{title}\n\n{summary}" if summary else title
        
        if len(text.split()) < 50:
            return None
        
        # Parse date
        date_str = report.get('date', report.get('latestPubDate', ''))
        if not date_str:
            return None
        
        try:
            content_date = datetime.fromisoformat(date_str[:10]).date()
        except:
            return None
        
        if content_date >= date(2022, 11, 1):
            return None
        
        report_id = report.get('number', report.get('id', ''))
        url = report.get('url', f"https://crsreports.congress.gov/{report_id}")
        
        return self.create_sample(
            text=' '.join(text.split()),
            original_id=report_id,
            content_date=content_date,
            url=url,
        )
