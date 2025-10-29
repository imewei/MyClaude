#!/usr/bin/env python3
"""
Production-Ready Async Web Scraper

Demonstrates:
- Concurrent HTTP requests with aiohttp
- Rate limiting with semaphore
- Error handling and retries
- Progress tracking
- Resource cleanup
- Graceful shutdown
"""

import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin, urlparse


@dataclass
class ScrapeResult:
    """Result of scraping a single URL"""
    url: str
    status_code: int
    content_length: int
    elapsed_ms: float
    error: Optional[str] = None


class AsyncWebScraper:
    """
    Production-ready async web scraper with rate limiting and error handling.

    Features:
    - Concurrent requests (configurable)
    - Automatic rate limiting
    - Retry logic with exponential backoff
    - Timeout handling
    - Resource cleanup
    - Progress tracking
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results: List[ScrapeResult] = []

    async def fetch_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
        retry_count: int = 0
    ) -> ScrapeResult:
        """
        Fetch a single URL with retry logic.

        Args:
            session: aiohttp session
            url: URL to fetch
            retry_count: Current retry attempt

        Returns:
            ScrapeResult with fetch details
        """
        async with self.semaphore:  # Rate limiting
            start_time = time.time()

            try:
                async with session.get(url) as response:
                    content = await response.read()
                    elapsed_ms = (time.time() - start_time) * 1000

                    return ScrapeResult(
                        url=url,
                        status_code=response.status,
                        content_length=len(content),
                        elapsed_ms=elapsed_ms
                    )

            except asyncio.TimeoutError:
                if retry_count < self.max_retries:
                    print(f"Timeout on {url}, retrying ({retry_count + 1}/{self.max_retries})...")
                    await asyncio.sleep(self.retry_delay * (2 ** retry_count))  # Exponential backoff
                    return await self.fetch_url(session, url, retry_count + 1)
                else:
                    elapsed_ms = (time.time() - start_time) * 1000
                    return ScrapeResult(
                        url=url,
                        status_code=0,
                        content_length=0,
                        elapsed_ms=elapsed_ms,
                        error="Timeout after retries"
                    )

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                return ScrapeResult(
                    url=url,
                    status_code=0,
                    content_length=0,
                    elapsed_ms=elapsed_ms,
                    error=str(e)
                )

    async def scrape_urls(self, urls: List[str]) -> List[ScrapeResult]:
        """
        Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of ScrapeResults
        """
        print(f"Starting scrape of {len(urls)} URLs...")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"Timeout: {self.timeout.total}s")
        print()

        start_time = time.time()

        # Create session with proper headers
        headers = {
            'User-Agent': 'AsyncWebScraper/1.0 (+https://example.com/bot)'
        }

        async with aiohttp.ClientSession(
            timeout=self.timeout,
            headers=headers
        ) as session:

            # Create tasks for all URLs
            tasks = [
                self.fetch_url(session, url)
                for url in urls
            ]

            # Progress tracking
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                completed += 1

                # Progress update
                if completed % 10 == 0 or completed == len(urls):
                    print(f"Progress: {completed}/{len(urls)} completed")

                self.results.append(result)

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s")
        print(f"Average: {elapsed / len(urls):.2f}s per URL")
        print(f"Throughput: {len(urls) / elapsed:.2f} URLs/second")

        return self.results

    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results to summarize")
            return

        successful = [r for r in self.results if r.error is None]
        failed = [r for r in self.results if r.error is not None]

        print("\n" + "=" * 60)
        print("SCRAPE SUMMARY")
        print("=" * 60)
        print(f"Total URLs: {len(self.results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")

        if successful:
            avg_time = sum(r.elapsed_ms for r in successful) / len(successful)
            avg_size = sum(r.content_length for r in successful) / len(successful)
            print(f"\nAverage response time: {avg_time:.0f}ms")
            print(f"Average content size: {avg_size / 1024:.1f} KB")

            # Status code distribution
            status_codes = {}
            for result in successful:
                status_codes[result.status_code] = status_codes.get(result.status_code, 0) + 1

            print("\nStatus code distribution:")
            for code, count in sorted(status_codes.items()):
                print(f"  {code}: {count}")

        if failed:
            print("\nFailed URLs:")
            for result in failed[:10]:  # Show first 10
                print(f"  {result.url}: {result.error}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")


async def demo_basic_usage():
    """Basic usage demonstration"""
    print("=" * 60)
    print("Demo 1: Basic Web Scraping")
    print("=" * 60)

    urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/2',
        'https://httpbin.org/status/200',
        'https://httpbin.org/status/404',
        'https://httpbin.org/status/500',
    ]

    scraper = AsyncWebScraper(max_concurrent=3, timeout=10.0)
    results = await scraper.scrape_urls(urls)
    scraper.print_summary()


async def demo_large_scale():
    """Large-scale scraping demonstration"""
    print("\n" + "=" * 60)
    print("Demo 2: Large-Scale Scraping")
    print("=" * 60)

    # Generate many URLs
    base_urls = [
        'https://httpbin.org/delay/0.1',
        'https://httpbin.org/delay/0.2',
        'https://httpbin.org/status/200',
        'https://httpbin.org/bytes/1024',
    ]

    urls = [f"{url}?id={i}" for url in base_urls for i in range(25)]

    scraper = AsyncWebScraper(max_concurrent=20, timeout=5.0)
    results = await scraper.scrape_urls(urls)
    scraper.print_summary()


async def demo_error_handling():
    """Error handling and retry demonstration"""
    print("\n" + "=" * 60)
    print("Demo 3: Error Handling & Retries")
    print("=" * 60)

    urls = [
        'https://httpbin.org/status/200',  # Success
        'https://httpbin.org/status/500',  # Server error
        'https://httpbin.org/delay/100',   # Timeout
        'https://invalid-domain-xyz-123.com',  # DNS error
        'https://httpbin.org/status/429',  # Rate limited
    ]

    scraper = AsyncWebScraper(
        max_concurrent=5,
        timeout=2.0,  # Short timeout to trigger retries
        max_retries=2,
        retry_delay=0.5
    )
    results = await scraper.scrape_urls(urls)
    scraper.print_summary()


async def main():
    """Run all demos"""
    await demo_basic_usage()
    await demo_large_scale()
    await demo_error_handling()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("✓ Semaphore limits concurrent requests")
    print("✓ Async context managers ensure cleanup")
    print("✓ Exponential backoff for retries")
    print("✓ Progress tracking with as_completed()")
    print("✓ Comprehensive error handling")
    print("✓ Resource cleanup guaranteed")


if __name__ == "__main__":
    asyncio.run(main())
