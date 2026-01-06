"""
Conversational Text Collectors

Sources: 
- Pushshift Reddit archives (pre-2023)
- Discord data dumps
- phpBB/vBulletin forum archives

These sources provide casual, conversational text that differs
significantly from academic/professional writing - critical for
cross-domain transfer validation.
"""

import json
import gzip
import zstandard as zstd
from pathlib import Path
from typing import Iterator, Optional
from datetime import date, datetime
from dataclasses import dataclass
import logging
import hashlib
import re

from ..schema import CorpusSample, Domain, Subdomain
from ..registry import BaseCollector, DomainRegistry

logger = logging.getLogger(__name__)


@DomainRegistry.register
class PushshiftRedditCollector(BaseCollector):
    """
    Collector for Reddit data from Pushshift archives.
    
    Pushshift provides historical Reddit data dumps going back to 2005.
    Use data from before Nov 2022 for guaranteed pre-LLM text.
    
    Data source: https://files.pushshift.io/reddit/
    
    Note: You need to download the compressed files first.
    Comments: RC_YYYY-MM.zst
    Submissions: RS_YYYY-MM.zst
    """
    
    DOMAIN = Domain.CONVERSATIONAL
    SUBDOMAIN = Subdomain.REDDIT
    SOURCE_NAME = "pushshift_reddit"
    SOURCE_VERSION = "2024_archive"
    LICENSE = "reddit_user_agreement"
    
    # Subreddits with high-quality discussion (avoid meme subs)
    DEFAULT_SUBREDDITS = {
        # Discussion-heavy
        'AskReddit', 'askscience', 'AskHistorians', 'explainlikeimfive',
        'changemyview', 'unpopularopinion', 'TrueOffMyChest',
        
        # Hobby/interest
        'books', 'movies', 'gaming', 'Music', 'Art',
        
        # Technical
        'programming', 'learnprogramming', 'webdev', 'sysadmin',
        'MachineLearning', 'datascience',
        
        # Lifestyle
        'personalfinance', 'relationships', 'Advice', 'careerguidance',
        
        # Local/community
        'sanfrancisco', 'bayarea', 'nyc', 'losangeles',
    }
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        subreddits: Optional[set[str]] = None,
        min_score: int = 5,  # Minimum upvotes
        content_type: str = "comments",  # "comments" or "submissions"
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.data_dir = data_dir or self.cache_dir / "pushshift"
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS
        self.min_score = min_score
        self.content_type = content_type
        
        if not self.data_dir.exists():
            logger.warning(
                f"Pushshift data not found at {self.data_dir}. "
                f"Download from: https://files.pushshift.io/reddit/"
            )
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield Reddit comments/submissions from Pushshift archives."""
        
        target = limit or self.config.target_samples
        collected = 0
        
        # Find archive files (RC_*.zst for comments, RS_*.zst for submissions)
        prefix = "RC_" if self.content_type == "comments" else "RS_"
        
        archive_files = sorted(
            self.data_dir.glob(f"{prefix}*.zst"),
            reverse=True  # Most recent first
        )
        
        # Also check for .gz files
        archive_files.extend(sorted(
            self.data_dir.glob(f"{prefix}*.gz"),
            reverse=True
        ))
        
        if not archive_files:
            logger.error(f"No archive files found in {self.data_dir}")
            logger.info("Expected files like: RC_2022-10.zst, RS_2021-06.zst")
            return
        
        for archive_path in archive_files:
            if collected >= target:
                break
            
            # Check date from filename (e.g., RC_2022-10.zst)
            try:
                date_str = archive_path.stem.split('_')[1]
                year, month = map(int, date_str.split('-'))
                archive_date = date(year, month, 15)
                
                # Skip if after LLM cutoff
                if archive_date >= date(2022, 11, 1):
                    logger.info(f"Skipping {archive_path.name} (post-LLM)")
                    continue
                    
            except (ValueError, IndexError):
                logger.warning(f"Could not parse date from {archive_path.name}")
                continue
            
            logger.info(f"Processing {archive_path.name}")
            
            for sample in self._process_archive(archive_path, target - collected):
                if self.validate_sample(sample):
                    self._seen_hashes.add(sample.content_hash)
                    collected += 1
                    yield sample
                    
                    if collected % 500 == 0:
                        logger.info(f"Collected {collected}/{target} Reddit items")
                
                if collected >= target:
                    break
        
        logger.info(f"Reddit collection complete: {collected} items")
    
    def _process_archive(
        self, 
        archive_path: Path, 
        max_items: int
    ) -> Iterator[CorpusSample]:
        """Process a single Pushshift archive file."""
        
        items_yielded = 0
        
        try:
            # Determine compression type
            if archive_path.suffix == '.zst':
                opener = self._open_zstd
            elif archive_path.suffix == '.gz':
                opener = gzip.open
            else:
                logger.warning(f"Unknown compression: {archive_path}")
                return
            
            with opener(archive_path, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if items_yielded >= max_items:
                        break
                    
                    try:
                        data = json.loads(line)
                        sample = self._parse_item(data)
                        
                        if sample:
                            items_yielded += 1
                            yield sample
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Error parsing item: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading archive {archive_path}: {e}")
    
    def _open_zstd(self, path: Path, mode: str, **kwargs):
        """Open a zstd compressed file."""
        dctx = zstd.ZstdDecompressor()
        f = open(path, 'rb')
        return dctx.stream_reader(f)
    
    def _parse_item(self, data: dict) -> Optional[CorpusSample]:
        """Parse a Reddit comment or submission."""
        
        # Filter by subreddit
        subreddit = data.get('subreddit', '')
        if self.subreddits and subreddit not in self.subreddits:
            return None
        
        # Filter by score
        score = data.get('score', 0)
        if score < self.min_score:
            return None
        
        # Get text content
        if self.content_type == "comments":
            text = data.get('body', '')
            item_id = data.get('id', '')
        else:
            # Submission: combine title and selftext
            title = data.get('title', '')
            selftext = data.get('selftext', '')
            text = f"{title}\n\n{selftext}" if selftext else title
            item_id = data.get('id', '')
        
        # Skip deleted/removed content
        if text in ['[deleted]', '[removed]', '']:
            return None
        
        # Skip very short content
        if len(text.split()) < 10:
            return None
        
        # Parse timestamp
        created_utc = data.get('created_utc', 0)
        if isinstance(created_utc, str):
            created_utc = int(created_utc)
        content_date = datetime.utcfromtimestamp(created_utc).date()
        
        # Skip if after LLM cutoff
        if content_date >= date(2022, 11, 1):
            return None
        
        # Clean text
        text = self._clean_reddit_text(text)
        
        # Hash author for privacy
        author = data.get('author', 'unknown')
        author_hash = hashlib.sha256(author.encode()).hexdigest()[:12]
        
        # Build permalink
        permalink = data.get('permalink', '')
        url = f"https://reddit.com{permalink}" if permalink else None
        
        return self.create_sample(
            text=text,
            original_id=item_id,
            content_date=content_date,
            url=url,
            author_id_hash=author_hash,
        )
    
    def _clean_reddit_text(self, text: str) -> str:
        """Clean Reddit-specific formatting."""
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove Reddit-specific markers
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&nbsp;', ' ', text)
        
        # Remove excessive quotes
        text = re.sub(r'^>+\s*', '', text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()


@DomainRegistry.register
class ForumCollector(BaseCollector):
    """
    Generic collector for phpBB/vBulletin/Invision forum archives.
    
    Forum data provides medium-length conversational text with
    topic-focused discussions - a middle ground between tweets
    and essays.
    """
    
    DOMAIN = Domain.CONVERSATIONAL
    SUBDOMAIN = Subdomain.FORUMS
    SOURCE_NAME = "forum_archive"
    SOURCE_VERSION = "1.0"
    LICENSE = "varies"
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        forum_name: str = "generic",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_path = data_path or self.cache_dir / "forum_export.json"
        self.forum_name = forum_name
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield forum posts from archive."""
        
        if not self.data_path.exists():
            logger.error(f"Forum data not found at {self.data_path}")
            return
        
        target = limit or self.config.target_samples
        collected = 0
        
        # Support both JSON and JSONL formats
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if collected >= target:
                        break
                    
                    try:
                        post = json.loads(line)
                        sample = self._parse_post(post)
                        
                        if sample and self.validate_sample(sample):
                            self._seen_hashes.add(sample.content_hash)
                            collected += 1
                            yield sample
                    except:
                        continue
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                posts = data if isinstance(data, list) else data.get('posts', [])
                
                for post in posts:
                    if collected >= target:
                        break
                    
                    sample = self._parse_post(post)
                    if sample and self.validate_sample(sample):
                        self._seen_hashes.add(sample.content_hash)
                        collected += 1
                        yield sample
        
        logger.info(f"Forum collection complete: {collected} posts")
    
    def _parse_post(self, post: dict) -> Optional[CorpusSample]:
        """Parse a forum post."""
        text = post.get('content', post.get('body', post.get('text', '')))
        post_id = str(post.get('id', post.get('post_id', '')))
        
        if not text or len(text.split()) < 10:
            return None
        
        # Parse date
        date_str = post.get('date', post.get('created_at', post.get('timestamp', '')))
        try:
            if isinstance(date_str, int):
                content_date = datetime.utcfromtimestamp(date_str).date()
            elif isinstance(date_str, str):
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%B %d, %Y']:
                    try:
                        content_date = datetime.strptime(date_str[:10], fmt).date()
                        break
                    except:
                        continue
                else:
                    return None
            else:
                return None
        except:
            return None
        
        # Skip post-LLM
        if content_date >= date(2022, 11, 1):
            return None
        
        # Clean BBCode
        text = self._clean_bbcode(text)
        
        return self.create_sample(
            text=text,
            original_id=post_id,
            content_date=content_date,
            url=post.get('url'),
            author_id_hash=hashlib.sha256(
                str(post.get('author', '')).encode()
            ).hexdigest()[:12],
        )
    
    def _clean_bbcode(self, text: str) -> str:
        """Remove BBCode formatting."""
        # Remove BBCode tags
        text = re.sub(r'\[/?[a-zA-Z]+[^\]]*\]', '', text)
        
        # Remove quotes
        text = re.sub(r'\[quote[^\]]*\].*?\[/quote\]', '', text, flags=re.DOTALL)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()


@DomainRegistry.register
class StackOverflowCollector(BaseCollector):
    """
    Collector for Stack Overflow posts from data dumps.
    
    Technical Q&A with clear temporal markers.
    Source: https://archive.org/details/stackexchange
    """
    
    DOMAIN = Domain.TECHNICAL
    SUBDOMAIN = Subdomain.TUTORIALS  # Close enough
    SOURCE_NAME = "stackoverflow"
    SOURCE_VERSION = "dump_2022"
    LICENSE = "cc-by-sa-4.0"
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        min_score: int = 10,
        content_type: str = "answers",  # "questions" or "answers"
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_path = data_path or self.cache_dir / "stackoverflow" / "Posts.xml"
        self.min_score = min_score
        self.content_type = content_type
        
        # PostTypeId: 1 = Question, 2 = Answer
        self.post_type_id = "2" if content_type == "answers" else "1"
    
    def collect(self, limit: Optional[int] = None) -> Iterator[CorpusSample]:
        """Yield Stack Overflow posts."""
        
        if not self.data_path.exists():
            logger.error(f"SO data not found at {self.data_path}")
            logger.info("Download from: https://archive.org/details/stackexchange")
            return
        
        target = limit or self.config.target_samples
        collected = 0
        
        # Parse large XML incrementally
        import xml.etree.ElementTree as ET
        
        context = ET.iterparse(self.data_path, events=('end',))
        
        for event, elem in context:
            if collected >= target:
                break
            
            if elem.tag == 'row':
                sample = self._parse_post(elem)
                
                if sample and self.validate_sample(sample):
                    self._seen_hashes.add(sample.content_hash)
                    collected += 1
                    yield sample
                    
                    if collected % 500 == 0:
                        logger.info(f"Collected {collected}/{target} SO posts")
                
                # Clear element to save memory
                elem.clear()
        
        logger.info(f"Stack Overflow collection complete: {collected} posts")
    
    def _parse_post(self, elem) -> Optional[CorpusSample]:
        """Parse a Stack Overflow post element."""
        
        # Filter by post type
        if elem.get('PostTypeId') != self.post_type_id:
            return None
        
        # Filter by score
        score = int(elem.get('Score', 0))
        if score < self.min_score:
            return None
        
        # Get content
        body = elem.get('Body', '')
        if not body:
            return None
        
        # Parse date
        date_str = elem.get('CreationDate', '')
        try:
            content_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
        except:
            return None
        
        # Skip post-LLM
        if content_date >= date(2022, 11, 1):
            return None
        
        # Clean HTML
        text = self._clean_html(body)
        
        if len(text.split()) < 20:
            return None
        
        post_id = elem.get('Id', '')
        
        return self.create_sample(
            text=text,
            original_id=post_id,
            content_date=content_date,
            url=f"https://stackoverflow.com/a/{post_id}" if self.content_type == "answers" 
                else f"https://stackoverflow.com/q/{post_id}",
        )
    
    def _clean_html(self, html: str) -> str:
        """Strip HTML tags and clean text."""
        # Remove code blocks (preserve for analysis but separate)
        html = re.sub(r'<code>.*?</code>', '[CODE]', html, flags=re.DOTALL)
        html = re.sub(r'<pre>.*?</pre>', '[CODE_BLOCK]', html, flags=re.DOTALL)
        
        # Remove remaining HTML tags
        html = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode entities
        html = html.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        html = html.replace('&quot;', '"').replace('&#39;', "'")
        
        # Normalize whitespace
        return ' '.join(html.split())
