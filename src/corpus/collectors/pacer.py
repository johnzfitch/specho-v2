"""
PACER Document Collector

Integrates with the indepacer CLI tool to harvest federal court documents
for the PACT corpus.

PACER provides:
- Exact filing dates (authoritative timestamps)
- Multiple document types per case (complaints, motions, briefs, orders, opinions)
- Pre-LLM verification via filing date
- Diverse writing styles (plaintiffs, defendants, judges)

Requirements:
- indepacer CLI installed and configured (`pacer auth login`)
- PACER account with funds

Document Type Classification:
- COMPLAINT: Initial pleading, narrative style
- MOTION: Procedural requests, argumentative
- BRIEF: Legal arguments, persuasive/analytical
- ORDER: Court directives, terse/authoritative  
- OPINION: Judicial reasoning, analytical/precedent-heavy
- RESPONSE: Replies to motions, argumentative
- DECLARATION: Factual statements, formal/precise
- EXHIBIT: Supporting documents, varies
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Iterator, Optional, Literal
from datetime import date, datetime
from dataclasses import dataclass
import logging
import hashlib

from ..schema import CorpusSample, Domain, Subdomain, SourceMetadata
from ..registry import BaseCollector, DomainRegistry

logger = logging.getLogger(__name__)


# Document type patterns for classification
DOC_TYPE_PATTERNS = {
    'COMPLAINT': [
        r'\bcomplaint\b',
        r'\bpetition\b',
        r'\binitial\s+pleading\b',
    ],
    'MOTION': [
        r'\bmotion\s+(to|for)\b',
        r'\bmotion\s+in\s+limine\b',
        r'\bemergency\s+motion\b',
    ],
    'BRIEF': [
        r'\bbrief\b',
        r'\bmemorandum\s+(of|in)\s+(law|support|opposition)\b',
        r'\bopposition\b',
        r'\breply\b',
    ],
    'ORDER': [
        r'\border\b',
        r'\bjudgment\b',
        r'\bdecree\b',
        r'\bscheduling\s+order\b',
    ],
    'OPINION': [
        r'\bopinion\b',
        r'\bmemorandum\s+decision\b',
        r'\bfindings\s+of\s+fact\b',
    ],
    'RESPONSE': [
        r'\bresponse\b',
        r'\banswer\b',
        r'\breply\b',
    ],
    'DECLARATION': [
        r'\bdeclaration\b',
        r'\baffidavit\b',
        r'\baffirmation\b',
    ],
}


@dataclass
class PACERDocument:
    """Metadata for a PACER document."""
    court: str
    case_number: str
    doc_number: int
    description: str
    filed_date: date
    doc_type: str
    pages: int
    text: Optional[str] = None
    pdf_path: Optional[Path] = None


def classify_document_type(description: str) -> str:
    """Classify document type from docket description."""
    desc_lower = description.lower()
    
    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, desc_lower):
                return doc_type
    
    return 'OTHER'


@DomainRegistry.register
class PACERCollector(BaseCollector):
    """
    Collector for federal court documents via PACER/indepacer.
    
    Uses the indepacer CLI tool to:
    1. Search for cases via PCL (PACER Case Locator)
    2. Download dockets
    3. Download individual documents
    4. Extract text from PDFs
    
    Cost awareness:
    - PCL searches: $0.10 per search
    - Docket pages: $0.10 per page
    - Documents: $0.10 per page (max $3.00 per document)
    
    Pre-LLM Strategy:
    - Only collect documents filed before Nov 1, 2022
    - Filing dates are authoritative timestamps
    """
    
    DOMAIN = Domain.LEGAL
    SUBDOMAIN = Subdomain.COURT_FILINGS
    SOURCE_NAME = "pacer"
    SOURCE_VERSION = "indepacer_1.0"
    LICENSE = "public_record"
    
    # LLM cutoff
    LLM_CUTOFF = date(2022, 11, 1)
    
    # Courts to sample from (federal district courts)
    DEFAULT_COURTS = [
        'nysd',   # Southern District of New York
        'cacd',   # Central District of California  
        'txsd',   # Southern District of Texas
        'ilnd',   # Northern District of Illinois
        'mad',    # District of Massachusetts
        'dcd',    # District of Columbia
        'njd',    # District of New Jersey
        'paed',   # Eastern District of Pennsylvania
        'gand',   # Northern District of Georgia
        'wawd',   # Western District of Washington
    ]
    
    # Document types to prioritize (these have most text content)
    PRIORITY_DOC_TYPES = ['COMPLAINT', 'BRIEF', 'MOTION', 'OPINION', 'ORDER']
    
    def __init__(
        self,
        indepacer_path: str = "pacer",  # CLI command
        archive_dir: Optional[Path] = None,
        courts: Optional[list[str]] = None,
        case_types: Optional[list[str]] = None,  # cv, cr, bk, etc.
        date_start: Optional[date] = None,
        date_end: Optional[date] = None,
        doc_types: Optional[list[str]] = None,
        max_cost_per_session: float = 50.0,  # Safety limit
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.indepacer = indepacer_path
        self.archive_dir = archive_dir or Path.home() / ".pacer"
        self.courts = courts or self.DEFAULT_COURTS
        self.case_types = case_types or ['cv']  # Civil cases by default
        self.date_start = date_start or date(2018, 1, 1)
        self.date_end = date_end or date(2022, 10, 31)
        self.doc_types = doc_types or self.PRIORITY_DOC_TYPES
        self.max_cost = max_cost_per_session
        
        self._session_cost = 0.0
        self._verify_indepacer()
    
    def _verify_indepacer(self):
        """Verify indepacer CLI is available."""
        try:
            result = subprocess.run(
                [self.indepacer, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.warning("indepacer CLI not responding correctly")
        except FileNotFoundError:
            logger.error(
                f"indepacer CLI not found at '{self.indepacer}'. "
                "Install from your indepacer repository."
            )
        except Exception as e:
            logger.warning(f"Error checking indepacer: {e}")
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """
        Collect documents from PACER via indepacer.
        
        Strategy:
        1. Use PCL to find cases in date range
        2. Download dockets for selected cases
        3. Identify high-value documents (briefs, opinions, etc.)
        4. Download and extract text
        """
        target = limit or self.config.target_samples
        collected = 0
        
        for court in self.courts:
            if collected >= target:
                break
            
            if self._session_cost >= self.max_cost:
                logger.warning(f"Session cost limit reached: ${self._session_cost:.2f}")
                break
            
            logger.info(f"Collecting from court: {court}")
            
            # Find cases
            cases = self._search_cases(court, limit=(target - collected) // len(self.courts))
            
            for case in cases:
                if collected >= target or self._session_cost >= self.max_cost:
                    break
                
                # Get docket
                docket = self._get_docket(court, case['case_number'])
                if not docket:
                    continue
                
                # Find interesting documents
                docs = self._select_documents(docket)
                
                for doc in docs:
                    if collected >= target:
                        break
                    
                    # Download and extract text
                    sample = self._process_document(court, case, doc)
                    
                    if sample and self.validate_sample(sample):
                        self._seen_hashes.add(sample.content_hash)
                        collected += 1
                        yield sample
                        
                        if collected % 50 == 0:
                            logger.info(
                                f"Collected {collected}/{target} documents "
                                f"(session cost: ${self._session_cost:.2f})"
                            )
        
        logger.info(
            f"PACER collection complete: {collected} documents, "
            f"session cost: ${self._session_cost:.2f}"
        )
    
    def _search_cases(self, court: str, limit: int = 100) -> list[dict]:
        """Search for cases via PCL."""
        cases = []
        
        for case_type in self.case_types:
            try:
                # Use indepacer PCL search
                cmd = [
                    self.indepacer, "pcl", "cases",
                    "--court", court,
                    "--type", case_type,
                    "--filed-after", self.date_start.isoformat(),
                    "--filed-before", self.date_end.isoformat(),
                    "--limit", str(limit),
                    "--output", "json",
                    "-y",  # Skip confirmation
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    # Parse JSON output
                    data = json.loads(result.stdout)
                    cases.extend(data.get('cases', []))
                    
                    # Track cost (PCL search ~$0.10)
                    self._session_cost += 0.10
                else:
                    logger.warning(f"PCL search failed for {court}: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Error searching {court}: {e}")
        
        return cases[:limit]
    
    def _get_docket(self, court: str, case_number: str) -> Optional[list[dict]]:
        """Download and parse docket for a case."""
        try:
            # Check if already cached
            cache_path = self.archive_dir / court / case_number.replace(":", "-") / "docket.json"
            
            if cache_path.exists():
                with open(cache_path) as f:
                    return json.load(f)
            
            # Download via indepacer
            cmd = [
                self.indepacer, "docket",
                court, case_number,
                "-y",
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Track cost (estimate ~3 pages = $0.30)
                self._session_cost += 0.30
                
                # Parse the downloaded docket
                return self._parse_local_docket(court, case_number)
            else:
                logger.warning(f"Failed to download docket: {case_number}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting docket {case_number}: {e}")
            return None
    
    def _parse_local_docket(self, court: str, case_number: str) -> Optional[list[dict]]:
        """Parse a locally cached docket HTML."""
        docket_dir = self.archive_dir / court / case_number.replace(":", "-")
        html_path = docket_dir / "docket.html"
        
        if not html_path.exists():
            return None
        
        try:
            # Use indepacer's parser
            cmd = [
                self.indepacer, "parse", "file",
                str(html_path),
                "--output", "json",
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('entries', [])
            
        except Exception as e:
            logger.warning(f"Error parsing docket: {e}")
        
        return None
    
    def _select_documents(self, entries: list[dict], max_docs: int = 10) -> list[dict]:
        """Select high-value documents from docket entries."""
        selected = []
        
        for entry in entries:
            # Check date
            filed_str = entry.get('filed', entry.get('date', ''))
            if not filed_str:
                continue
            
            try:
                filed_date = datetime.strptime(filed_str, '%Y-%m-%d').date()
            except ValueError:
                try:
                    filed_date = datetime.strptime(filed_str, '%m/%d/%Y').date()
                except:
                    continue
            
            # Skip post-LLM
            if filed_date >= self.LLM_CUTOFF:
                continue
            
            # Classify document type
            description = entry.get('description', entry.get('text', ''))
            doc_type = classify_document_type(description)
            
            # Only select priority types
            if doc_type not in self.doc_types:
                continue
            
            # Check if document is available
            if not entry.get('doc_number') and not entry.get('document_number'):
                continue
            
            selected.append({
                'doc_number': entry.get('doc_number') or entry.get('document_number'),
                'description': description,
                'filed_date': filed_date,
                'doc_type': doc_type,
                'pages': entry.get('pages', 1),
            })
            
            if len(selected) >= max_docs:
                break
        
        return selected
    
    def _process_document(
        self, 
        court: str, 
        case: dict, 
        doc: dict
    ) -> Optional[CorpusSample]:
        """Download document and extract text."""
        case_number = case.get('case_number', case.get('case_id', ''))
        doc_number = doc['doc_number']
        
        try:
            # Download via indepacer
            cmd = [
                self.indepacer, "doc",
                str(doc_number),
                "-y",
            ]
            
            # Set context if needed
            context_cmd = [
                self.indepacer, "use", "case",
                court, case_number,
            ]
            subprocess.run(context_cmd, capture_output=True, timeout=10)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to download doc {doc_number}")
                return None
            
            # Track cost
            pages = min(doc.get('pages', 1), 30)  # Max $3.00
            self._session_cost += pages * 0.10
            
            # Find the PDF
            pdf_path = (
                self.archive_dir / court / 
                case_number.replace(":", "-") / 
                f"doc_{doc_number}.pdf"
            )
            
            if not pdf_path.exists():
                # Try alternate locations
                for p in self.archive_dir.glob(f"**/*{doc_number}*.pdf"):
                    pdf_path = p
                    break
            
            if not pdf_path.exists():
                logger.warning(f"PDF not found for doc {doc_number}")
                return None
            
            # Extract text
            text = self._extract_pdf_text(pdf_path)
            
            if not text or len(text) < 100:
                return None
            
            # Create sample
            source = SourceMetadata(
                corpus_name="pacer",
                corpus_version=f"{court}",
                original_id=f"{court}_{case_number}_{doc_number}",
                url=f"https://ecf.{court}.uscourts.gov/",
                license="public_record",
                content_date=doc['filed_date'],
                content_date_precision="day",
                verified_pre_llm=doc['filed_date'] < self.LLM_CUTOFF,
            )
            
            sample_id = CorpusSample.create_id(
                self.DOMAIN,
                self.SUBDOMAIN,
                "pacer",
                f"{court}_{case_number.replace(':', '_')}_{doc_number}"
            )
            
            return CorpusSample(
                sample_id=sample_id,
                domain=self.DOMAIN,
                subdomain=self.SUBDOMAIN,
                source=source,
                text=text,
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    def _extract_pdf_text(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a PDF file."""
        try:
            # Try pdftotext first (poppler)
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return self._clean_legal_text(result.stdout)
            
            # Fallback to indepacer's parser
            cmd = [
                self.indepacer, "parse", "text",
                str(pdf_path),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return self._clean_legal_text(result.stdout)
                
        except Exception as e:
            logger.warning(f"Error extracting text from {pdf_path}: {e}")
        
        return None
    
    def _clean_legal_text(self, text: str) -> str:
        """Clean extracted legal document text."""
        # Remove page numbers and headers
        text = re.sub(r'^\s*Page\s+\d+\s+of\s+\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
        
        # Remove case captions that repeat on every page
        text = re.sub(r'Case\s+\d+:\d+-\w+-\d+.*?Document\s+\d+.*?Filed\s+\d+/\d+/\d+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove form-feed characters
        text = text.replace('\x0c', '\n\n')
        
        return text.strip()


@DomainRegistry.register
class PACEROpinionsCollector(BaseCollector):
    """
    Specialized collector for judicial opinions from PACER.
    
    Opinions are the highest-quality legal writing and provide
    excellent samples of analytical reasoning.
    """
    
    DOMAIN = Domain.LEGAL
    SUBDOMAIN = Subdomain.COURT_FILINGS  # Could add OPINIONS subdomain
    SOURCE_NAME = "pacer_opinions"
    SOURCE_VERSION = "indepacer_1.0"
    LICENSE = "public_record"
    
    def __init__(self, **kwargs):
        # Force doc_types to opinions only
        kwargs['doc_types'] = ['OPINION', 'ORDER']
        super().__init__(**kwargs)
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Delegate to parent but filter for opinions."""
        # This is a placeholder - in practice would use the same
        # infrastructure but with different search parameters
        return super().collect(limit)


# Utility function for batch collection
def estimate_pacer_cost(
    num_cases: int,
    docs_per_case: int = 5,
    pages_per_doc: int = 10
) -> float:
    """
    Estimate PACER costs for collection run.
    
    Costs:
    - PCL search: $0.10 per search
    - Docket: $0.10 per page (~3 pages avg)
    - Document: $0.10 per page (max $3.00)
    """
    pcl_cost = 0.10 * (num_cases // 25 + 1)  # 25 results per search
    docket_cost = 0.30 * num_cases  # ~3 pages per docket
    doc_cost = min(pages_per_doc * 0.10, 3.00) * docs_per_case * num_cases
    
    total = pcl_cost + docket_cost + doc_cost
    
    return total


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect PACER documents")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--court", type=str, default="nysd")
    parser.add_argument("--max-cost", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    
    if args.dry_run:
        cost = estimate_pacer_cost(args.limit, docs_per_case=5)
        print(f"Estimated cost for {args.limit} documents: ${cost:.2f}")
    else:
        collector = PACERCollector(
            courts=[args.court],
            max_cost_per_session=args.max_cost,
        )
        
        for sample in collector.collect(limit=args.limit):
            print(f"Collected: {sample.sample_id} ({sample.word_count} words)")
