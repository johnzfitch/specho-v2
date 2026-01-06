"""
Text formatters for realistic user presentation.

Formats corpus text in ways users actually present content to AI models:
- News articles: Markdown with headlines
- Academic: Paper-style formatting
- Code: Syntax-highlighted snippets
- Social: Plaintext posts
- Legal: Document headers
"""

from typing import Optional
from .schema import Domain, Subdomain


class TextFormatter:
    """Base formatter for realistic text presentation."""

    @staticmethod
    def format_news_article(text: str, title: Optional[str] = None, url: Optional[str] = None) -> str:
        """
        Format as markdown news article (how users copy-paste from web).

        Example output:
            # Article Headline

            Source: https://example.com/article

            Article text here...
        """
        lines = []

        # Extract title if embedded in text
        if not title and text.startswith(('# ', 'Supported by\n')):
            first_line = text.split('\n')[0].strip('# ').strip()
            if len(first_line) < 200:  # Reasonable title length
                title = first_line
                text = '\n'.join(text.split('\n')[1:]).strip()

        # Add title as markdown header
        if title:
            lines.append(f"# {title}\n")

        # Add source URL if available
        if url:
            lines.append(f"*Source: {url}*\n")

        lines.append(text)

        return '\n'.join(lines)

    @staticmethod
    def format_academic_essay(text: str, title: Optional[str] = None) -> str:
        """
        Format as student essay submission (typical format users paste).

        Example output:
            Essay Title

            Essay content here...
        """
        lines = []

        if title:
            lines.append(f"{title}\n")

        lines.append(text)

        return '\n'.join(lines)

    @staticmethod
    def format_social_post(text: str, platform: str = "reddit") -> str:
        """
        Format as social media post (plaintext, as users paste).

        Just returns text as-is since users typically paste plaintext.
        """
        return text.strip()

    @staticmethod
    def format_code_snippet(text: str, language: str = "python") -> str:
        """
        Format as code snippet (with markdown code fences).

        Example output:
            ```python
            code here...
            ```
        """
        return f"```{language}\n{text}\n```"

    @staticmethod
    def format_legal_document(
        text: str,
        case_name: Optional[str] = None,
        court: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """
        Format as legal document (with header).

        Example output:
            Case: Smith v. Jones
            Court: U.S. Supreme Court
            Date: 2022-05-15

            ---

            Document text here...
        """
        lines = []

        if case_name:
            lines.append(f"**Case:** {case_name}")
        if court:
            lines.append(f"**Court:** {court}")
        if date:
            lines.append(f"**Date:** {date}")

        if lines:
            lines.append("\n---\n")

        lines.append(text)

        return '\n'.join(lines)

    @staticmethod
    def format_for_domain(
        text: str,
        domain: Domain,
        subdomain: Subdomain,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Automatically format text based on domain.

        Args:
            text: Raw text content
            domain: Domain enum
            subdomain: Subdomain enum
            metadata: Optional dict with title, url, etc.

        Returns:
            Formatted text as users would present it to AI
        """
        metadata = metadata or {}

        if domain == Domain.JOURNALISTIC:
            return TextFormatter.format_news_article(
                text,
                title=metadata.get('title'),
                url=metadata.get('url')
            )

        elif domain == Domain.ACADEMIC:
            if subdomain == Subdomain.ESSAYS:
                return TextFormatter.format_academic_essay(
                    text,
                    title=metadata.get('title')
                )
            else:
                # Abstracts, papers - just plaintext
                return text.strip()

        elif domain == Domain.CONVERSATIONAL:
            return TextFormatter.format_social_post(text)

        elif domain == Domain.TECHNICAL:
            # Check if it looks like code
            if any(lang in text[:100].lower() for lang in ['def ', 'function ', 'class ', 'import ', '#include']):
                return TextFormatter.format_code_snippet(text)
            return text.strip()

        elif domain == Domain.LEGAL:
            return TextFormatter.format_legal_document(
                text,
                case_name=metadata.get('case_name'),
                court=metadata.get('court'),
                date=metadata.get('date')
            )

        else:
            # Default: plaintext
            return text.strip()


# Export main formatter
format_text = TextFormatter.format_for_domain
